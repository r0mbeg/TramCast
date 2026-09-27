// Package forecasts_worker runs the PostgreSQL prediction job queue inside the
// HTTP server process: it claims a job, computes it on the ML server while
// renewing the lease, then publishes the result or records the failure.
package forecasts_worker

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
)

const (
	// probeTimeout bounds the readiness probe; its route never starts the model.
	probeTimeout = 10 * time.Second
	// minProbeRetry is the shortest wait between probes while claims are
	// paused, so a server that is down is not called every poll interval.
	minProbeRetry = 15 * time.Second
)

// Error codes of failures the worker finds itself; the ML client adds ml_*.
const (
	codeRouteNotEnabled     = "route_not_enabled"
	codePublicationConflict = "publication_conflict"
	codePublicationFailed   = "publication_failed"
	codeInternal            = "internal_error"
)

// Causes of a cancelled job call. Either one means the worker writes nothing
// more for the attempt: the lease expires and recovery requeues the job.
var (
	errShutdown  = errors.New("worker shutdown grace expired")
	errLeaseLost = errors.New("prediction job lease lost")
)

// Jobs is implemented by *forecasts_service.Queue.
type Jobs interface {
	Claim(ctx context.Context) (forecasts_service.ClaimedJob, bool, error)
	Request(job forecasts_service.ClaimedJob) forecasts_predictor_grpc.Request
	RenewLease(ctx context.Context, job forecasts_service.ClaimedJob) error
	Publish(ctx context.Context, job forecasts_service.ClaimedJob, points []forecasts_predictor_grpc.Point) error
	FailAttempt(ctx context.Context, job forecasts_service.ClaimedJob, code string, retryable bool, cause error) (forecasts_sqlc.PredictionJob, error)
	Recover(ctx context.Context) (int, int, error)
	ActiveVersion(ctx context.Context) (forecasts_sqlc.ForecastVersion, error)
}

// Predictor is implemented by *forecasts_predictor_grpc.Client; its errors are
// *forecasts_predictor_grpc.Error.
type Predictor interface {
	Predict(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error)
}

// Config holds validated core_config.PredictionJobConfig values.
type Config struct {
	// Workers is the number of claim loops.
	Workers int
	// PollInterval is the wait after an empty or failed claim.
	PollInterval time.Duration
	// LeaseDuration is the lease the queue grants: it is renewed every third
	// and expired leases are recovered every half of it.
	LeaseDuration time.Duration
	// ShutdownTimeout is how long a running job may still finish after
	// shutdown starts.
	ShutdownTimeout time.Duration
}

type Worker struct {
	jobs      Jobs
	predictor Predictor
	cfg       Config
	log       *slog.Logger
}

func New(jobs Jobs, predictor Predictor, cfg Config, log *slog.Logger) *Worker {
	return &Worker{jobs: jobs, predictor: predictor, cfg: cfg, log: log}
}

// Run recovers expired leases and runs cfg.Workers claim loops until ctx is
// done. Claims wait until the ML server serves the active version, and wait
// again when another version becomes active. A job that is running when ctx
// ends may finish within ShutdownTimeout; then its call is cancelled and
// nothing is written. Run returns once every goroutine exited.
func (w *Worker) Run(ctx context.Context) {
	g := &gate{probing: make(chan struct{}, 1)}
	var wg sync.WaitGroup
	wg.Go(func() { w.recoverLoop(ctx) })
	for range w.cfg.Workers {
		wg.Go(func() { w.claimLoop(ctx, g) })
	}
	wg.Wait()
}

// recoverLoop requeues or fails jobs whose lease expired, at start and every
// half lease. It ignores the gate: it only touches the database.
func (w *Worker) recoverLoop(ctx context.Context) {
	for ctx.Err() == nil {
		recovered, exhausted, err := w.jobs.Recover(ctx)
		if recovered > 0 || exhausted > 0 {
			w.log.Info("prediction jobs recovered", "recovered", recovered, "exhausted", exhausted)
		}
		if err != nil && ctx.Err() == nil {
			w.log.Error("recover prediction jobs", "error", err)
		}
		sleep(ctx, w.cfg.LeaseDuration/2)
	}
}

func (w *Worker) claimLoop(ctx context.Context, g *gate) {
	for w.awaitGate(ctx, g) {
		if !w.probedIsActive(ctx, g) {
			continue
		}
		// Taken before the claim, so the local lease estimate never outlives
		// the lease in the database.
		leaseStart := time.Now()
		job, found, err := w.jobs.Claim(ctx)
		if err != nil && ctx.Err() == nil {
			w.log.Error("claim a prediction job", "error", err)
		}
		if !found {
			sleep(ctx, w.cfg.PollInterval)
			continue
		}
		w.process(ctx, g, job, leaseStart)
	}
}

// process computes one claimed job and records the outcome while the attempt
// still owns it.
func (w *Worker) process(ctx context.Context, g *gate, job forecasts_service.ClaimedJob, leaseStart time.Time) {
	started := time.Now()
	log := w.log.With("job_id", job.Job.ID.String(), "attempt", job.Job.AttemptCount,
		"forecast_version_id", job.Version.ID.String(), "route_number", job.RouteNumber)

	// The job outlives ctx by at most ShutdownTimeout, so a call that is about
	// to finish still publishes.
	jobCtx, cancelJob := context.WithCancelCause(context.WithoutCancel(ctx))
	defer cancelJob(nil)
	var helpers sync.WaitGroup
	defer helpers.Wait()
	finished := make(chan struct{})
	defer close(finished)
	helpers.Go(func() { shutdownGrace(ctx, finished, w.cfg.ShutdownTimeout, cancelJob) })

	if !job.RouteEnabled {
		w.fail(jobCtx, log, job, codeRouteNotEnabled, false,
			errors.New("forecasts were disabled for the route after admission"), started)
		return
	}

	renewCtx, stopRenewal := context.WithCancel(jobCtx)
	renewed := make(chan struct{})
	go func() {
		defer close(renewed)
		w.renew(renewCtx, log, job, leaseStart, cancelJob)
	}()
	result, err := w.predictor.Predict(jobCtx, w.jobs.Request(job))
	// Renewal stops before any write: one waiting behind the publication lock
	// would see a finished job and report the lease lost.
	stopRenewal()
	<-renewed

	if cause := context.Cause(jobCtx); errors.Is(cause, errShutdown) || errors.Is(cause, errLeaseLost) {
		log.Warn("prediction job abandoned; recovery requeues it once the lease expires",
			"reason", cause, "duration", time.Since(started))
		return
	}
	if err != nil {
		w.predictionFailed(jobCtx, g, log, job, err, started)
		return
	}

	err = w.jobs.Publish(jobCtx, job, result.Points)
	switch {
	case err == nil:
		log.Info("prediction job succeeded", "status", "succeeded", "points", len(result.Points),
			"duration", time.Since(started))
	case context.Cause(jobCtx) != nil:
		// The shutdown grace ran out: the transaction rolled back.
		log.Warn("prediction job abandoned during publication; recovery requeues it once the lease expires",
			"reason", context.Cause(jobCtx), "error", err, "duration", time.Since(started))
	case errors.Is(err, forecasts_service.ErrLeaseLost):
		log.Warn("prediction job lost before publication; nothing was written",
			"error", err, "duration", time.Since(started))
	case errors.Is(err, forecasts_service.ErrPublicationConflict):
		w.fail(jobCtx, log, job, codePublicationConflict, false, err, started)
	default:
		w.fail(jobCtx, log, job, codePublicationFailed, true, err, started)
	}
}

// predictionFailed records a failed call by its classification and pauses
// claims when the failure would repeat for every job.
func (w *Worker) predictionFailed(ctx context.Context, g *gate, log *slog.Logger, job forecasts_service.ClaimedJob, err error, started time.Time) {
	var predictErr *forecasts_predictor_grpc.Error
	if !errors.As(err, &predictErr) {
		w.fail(ctx, log, job, codeInternal, false, err, started)
		return
	}
	w.pauseClaims(ctx, g, log, job, predictErr.Code)
	// ml_cancelled reaches here only while the attempt owns the job, and the
	// client marks it terminal.
	w.fail(ctx, log, job, predictErr.Code, predictErr.Retryable, err, started)
}

// pauseClaims closes the gate when the ML server is down or slow, or serves
// other artifacts than the active version. A mismatch of another version fails
// only that version's jobs.
func (w *Worker) pauseClaims(ctx context.Context, g *gate, log *slog.Logger, job forecasts_service.ClaimedJob, code string) {
	switch code {
	case "ml_unavailable", "ml_deadline_exceeded":
	case "ml_version_mismatch":
		// Read now: the version may have been activated after the probe. A
		// failed read pauses too; the probe then decides.
		active, err := w.jobs.ActiveVersion(ctx)
		if errors.Is(err, forecasts_service.ErrNoActiveVersion) || (err == nil && active.ID != job.Version.ID) {
			return
		}
	default:
		return
	}
	if g.pause("paused " + code) {
		log.Warn("prediction job claims paused until the ML server passes the probe", "error_code", code)
	}
}

// fail records the failed attempt; the stored status tells whether the job is
// queued for a retry or failed for good.
func (w *Worker) fail(ctx context.Context, log *slog.Logger, job forecasts_service.ClaimedJob, code string, retryable bool, cause error, started time.Time) {
	stored, err := w.jobs.FailAttempt(ctx, job, code, retryable, cause)
	switch {
	case errors.Is(err, forecasts_service.ErrLeaseLost):
		log.Warn("prediction job lost before its failure was recorded",
			"error_code", code, "cause", cause, "error", err, "duration", time.Since(started))
	case err != nil:
		// The lease expires and recovery requeues or fails the job.
		log.Error("record the prediction job failure",
			"error_code", code, "cause", cause, "error", err, "duration", time.Since(started))
	default:
		level := slog.LevelWarn
		if stored.Status == "failed" {
			level = slog.LevelError
		}
		log.Log(ctx, level, "prediction job attempt failed", "status", stored.Status,
			"error_code", code, "retryable", retryable, "error", cause, "duration", time.Since(started))
	}
}

// renew extends the lease every third of its duration until ctx is done. A lost
// lease cancels the job at once. A failing database is retried until the lease
// may expire within a renewal interval; then the job is cancelled as well,
// since the publication would be refused anyway.
func (w *Worker) renew(ctx context.Context, log *slog.Logger, job forecasts_service.ClaimedJob, renewedAt time.Time, cancelJob context.CancelCauseFunc) {
	interval := w.cfg.LeaseDuration / 3
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		attempted := time.Now()
		callCtx, cancel := context.WithTimeout(ctx, interval)
		err := w.jobs.RenewLease(callCtx, job)
		cancel()
		switch {
		case err == nil:
			renewedAt = attempted
		case errors.Is(err, forecasts_service.ErrLeaseLost):
			log.Warn("prediction job lease lost; cancelling the call", "error", err)
			cancelJob(errLeaseLost)
			return
		case ctx.Err() != nil:
			return
		case !time.Now().Before(renewedAt.Add(w.cfg.LeaseDuration - interval)):
			log.Error("prediction job lease presumed lost after failed renewals; cancelling the call", "error", err)
			cancelJob(errLeaseLost)
			return
		default:
			log.Warn("renew the prediction job lease; retrying", "error", err)
		}
	}
}

// shutdownGrace cancels the job ShutdownTimeout after ctx is done, unless it
// has finished by then.
func shutdownGrace(ctx context.Context, finished <-chan struct{}, grace time.Duration, cancelJob context.CancelCauseFunc) {
	select {
	case <-finished:
		return
	case <-ctx.Done():
	}
	timer := time.NewTimer(grace)
	defer timer.Stop()
	select {
	case <-finished:
	case <-timer.C:
		cancelJob(errShutdown)
	}
}

// gate pauses the claims of every loop until the ML server serves the active
// version. It starts closed and opens for the version it probed. One loop
// probes at a time and keeps the token while it waits to retry, so the probe
// rate does not grow with the loops.
type gate struct {
	probing chan struct{}
	mu      sync.Mutex
	open    bool
	// probed is the version the gate opened for.
	probed pgtype.UUID
	// opened counts openings, so a loop holding an older read of the active
	// version does not close a gate another loop has just opened.
	opened uint64
	// state is the last logged outcome, so repeated outcomes are logged once.
	state string
}

// openings returns the opening count to pass to follow.
func (g *gate) openings() uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.opened
}

func (g *gate) isOpen() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.open
}

// openFor opens the gate for the version the ML server passed the probe for.
func (g *gate) openFor(version pgtype.UUID) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.open, g.probed, g.state = true, version, "ready"
	g.opened++
}

// pause closes the gate and reports whether state differs from the last one.
func (g *gate) pause(state string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	changed := g.state != state
	g.open, g.state = false, state
	return changed
}

// follow reports whether the gate is open for the active version, the zero
// UUID when there is none, read after openings returned opened. A gate open for
// another version closes with state, and closed reports that; a gate that is
// already closed, or opened again since that read, stays as it is.
func (g *gate) follow(active pgtype.UUID, opened uint64, state string) (open, closed bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.open || g.probed == active || g.opened != opened {
		return g.open, false
	}
	g.open, g.state = false, state
	return false, true
}

// awaitGate reports true once claims are open, or false when ctx is done.
func (w *Worker) awaitGate(ctx context.Context, g *gate) bool {
	for !g.isOpen() {
		select {
		case g.probing <- struct{}{}:
		case <-ctx.Done():
			return false
		}
		// Another loop may have opened the gate meanwhile.
		if !g.isOpen() && !w.probe(ctx, g) {
			sleep(ctx, max(w.cfg.PollInterval, minProbeRetry))
		}
		<-g.probing
		if ctx.Err() != nil {
			return false
		}
	}
	return ctx.Err() == nil
}

// probedIsActive reports whether the version the gate opened for is still the
// active one. A new active version, or none, closes the gate, so the probe
// runs for the new version before the next claim. An unreadable version skips
// the claim without closing: the database, not the ML server, is failing.
//
// ponytail: only the version ID is compared. An ML artifact swap under the
// same active version is still found by the first job that fails with
// ml_version_mismatch, which fails that job for good; probing before every
// claim would find it at the cost of one more RPC per job. The same holds for
// a version activated, and its job admitted, between this read and the claim;
// checking the claimed job's version against the probed one inside the claim
// transaction would close that window if it ever matters.
func (w *Worker) probedIsActive(ctx context.Context, g *gate) bool {
	opened := g.openings()
	active, err := w.jobs.ActiveVersion(ctx)
	switch {
	case ctx.Err() != nil:
		return false
	case errors.Is(err, forecasts_service.ErrNoActiveVersion):
		// The probe that runs next logs the missing version once.
		open, _ := g.follow(pgtype.UUID{}, opened, "active version gone")
		return open
	case err != nil:
		w.log.Error("read the active forecast version before a claim", "error", err)
		sleep(ctx, w.cfg.PollInterval)
		return false
	}
	open, closed := g.follow(active.ID, opened, "active version changed")
	if closed {
		w.log.Info("active forecast version changed; probing the ML server",
			"forecast_version_id", active.ID.String())
	}
	return open
}

// probe asks the ML server for the probe route of the active version, which it
// answers without the model, and opens the gate on success: the server is up
// and serves exactly the active artifacts and period.
func (w *Worker) probe(ctx context.Context, g *gate) bool {
	version, err := w.jobs.ActiveVersion(ctx)
	switch {
	case ctx.Err() != nil:
		return false
	case errors.Is(err, forecasts_service.ErrNoActiveVersion):
		if g.pause("no active version") {
			w.log.Warn("no active forecast version; prediction jobs wait for one")
		}
		return false
	case err != nil:
		if g.pause("active version unreadable") {
			w.log.Error("read the active forecast version; prediction jobs wait", "error", err)
		}
		return false
	}
	probeCtx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	result, err := w.predictor.Predict(probeCtx, forecasts_service.ProbeRequest(version))
	switch {
	case ctx.Err() != nil:
		return false
	case err != nil:
		code := errorCode(err)
		if g.pause("probe failed " + version.ID.String() + " " + code) {
			w.log.Warn("ML server does not serve the active forecast version; prediction jobs wait",
				"forecast_version_id", version.ID.String(), "error_code", code, "error", err)
		}
		return false
	}
	g.openFor(version.ID)
	w.log.Info("ML server ready; claiming prediction jobs", "forecast_version_id", version.ID.String(),
		"model_version", result.ModelVersion, "dataset_version", result.DatasetVersion)
	return true
}

func errorCode(err error) string {
	var predictErr *forecasts_predictor_grpc.Error
	if errors.As(err, &predictErr) {
		return predictErr.Code
	}
	return codeInternal
}

// sleep waits for d or until ctx is done.
func sleep(ctx context.Context, d time.Duration) {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
	}
}
