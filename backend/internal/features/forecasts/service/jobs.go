package forecasts_service

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

const (
	// recoveryBatchSize bounds one recovery statement; Recover repeats it
	// until a batch comes back smaller.
	recoveryBatchSize = 100
	// maxRetryDelay caps the growing delay between attempts of one job.
	maxRetryDelay = time.Hour
	// maxErrorMessageRunes bounds last_error_message, which may hold a server
	// message of any length.
	maxErrorMessageRunes = 1000
	// commitCheckTimeout bounds the re-read after a publication commit with
	// an unknown outcome; it runs even when the caller is shutting down.
	commitCheckTimeout = 5 * time.Second
)

var (
	// ErrQueueFull rejects a new job while queued plus running jobs fill the
	// capacity; an existing job of the pair is still returned.
	ErrQueueFull        = errors.New("prediction job queue is full")
	ErrRouteNotFound    = errors.New("route not found")
	ErrForecastDisabled = errors.New("forecasts are disabled for the route")
	// ErrLeaseLost means the attempt no longer owns its job: the lease
	// expired, another attempt took it or it finished. The caller must write
	// nothing more for this attempt.
	ErrLeaseLost       = errors.New("prediction job attempt lost ownership")
	ErrNoActiveVersion = errors.New("no active forecast version")
	// ErrPublicationConflict means the points break a key or check of
	// validation_predictions; computing them again would not help.
	ErrPublicationConflict = errors.New("predictions conflict with the stored data")
)

// JobPolicy holds the queue limits from the configuration, which validates
// them: the SQL reads a non-positive lease or attempt limit as nothing to
// claim and a negative delay as lost ownership.
type JobPolicy struct {
	QueueCapacity int64
	MaxAttempts   int32
	// LeaseDuration is how long a claim owns a job without renewal, in whole
	// seconds.
	LeaseDuration time.Duration
	// RetryDelay is the base delay: failed attempt n waits RetryDelay*4^(n-1),
	// at most an hour. Recovery of an expired lease waits RetryDelay.
	RetryDelay time.Duration
}

// ClaimedJob is a job owned by one attempt.
type ClaimedJob struct {
	// Job is the row as claimed; Job.AttemptCount is the ownership token of
	// every later write.
	Job forecasts_sqlc.PredictionJob
	// Version is the version the job is pinned to, not the active one.
	Version     forecasts_sqlc.ForecastVersion
	RouteNumber int16
	// RouteEnabled is false when forecasts were disabled after admission.
	RouteEnabled bool
}

// jobQueries and routeReader are the parts of the generated sqlc packages the
// queue uses; their *Queries types satisfy them directly.
type jobQueries interface {
	GetForecastVersion(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error)
	GetActiveForecastVersion(ctx context.Context) (forecasts_sqlc.ForecastVersion, error)
	LockPredictionJobAdmission(ctx context.Context) error
	GetPredictionJob(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.PredictionJob, error)
	GetPredictionJobByRouteAndVersion(ctx context.Context, arg forecasts_sqlc.GetPredictionJobByRouteAndVersionParams) (forecasts_sqlc.PredictionJob, error)
	CountPendingPredictionJobs(ctx context.Context) (int64, error)
	CreatePredictionJob(ctx context.Context, arg forecasts_sqlc.CreatePredictionJobParams) (forecasts_sqlc.PredictionJob, error)
	ClaimNextPredictionJob(ctx context.Context, arg forecasts_sqlc.ClaimNextPredictionJobParams) (forecasts_sqlc.PredictionJob, error)
	RenewPredictionJobLease(ctx context.Context, arg forecasts_sqlc.RenewPredictionJobLeaseParams) (int64, error)
	LockPredictionJobForPublication(ctx context.Context, arg forecasts_sqlc.LockPredictionJobForPublicationParams) (forecasts_sqlc.LockPredictionJobForPublicationRow, error)
	CopyValidationPredictions(ctx context.Context, arg []forecasts_sqlc.CopyValidationPredictionsParams) (int64, error)
	CompletePredictionJob(ctx context.Context, arg forecasts_sqlc.CompletePredictionJobParams) (int64, error)
	FinishPredictionJobWithError(ctx context.Context, arg forecasts_sqlc.FinishPredictionJobWithErrorParams) (forecasts_sqlc.PredictionJob, error)
	RecoverExpiredPredictionJobs(ctx context.Context, arg forecasts_sqlc.RecoverExpiredPredictionJobsParams) ([]forecasts_sqlc.RecoverExpiredPredictionJobsRow, error)
	FailExhaustedQueuedPredictionJobs(ctx context.Context, arg forecasts_sqlc.FailExhaustedQueuedPredictionJobsParams) ([]forecasts_sqlc.FailExhaustedQueuedPredictionJobsRow, error)
}

type routeReader interface {
	GetRouteByID(ctx context.Context, routeID int64) (routes_sqlc.Route, error)
	ListRoutes(ctx context.Context) ([]routes_sqlc.Route, error)
}

// queueStore groups the storage of one transaction.
type queueStore struct {
	jobs   jobQueries
	routes routeReader
}

func newQueueStore(tx pgx.Tx) queueStore {
	return queueStore{jobs: forecasts_sqlc.New(tx), routes: routes_sqlc.New(tx)}
}

// Queue runs the prediction job lifecycle on PostgreSQL: admission, claim,
// lease renewal, publication, failure and recovery. Every statement runs in a
// short READ COMMITTED transaction; no network call happens inside one.
type Queue struct {
	db TxBeginner
	// queries binds storage to a transaction; tests replace it.
	queries func(pgx.Tx) queueStore
	policy  JobPolicy
}

func NewQueue(db TxBeginner, policy JobPolicy) *Queue {
	return &Queue{db: db, queries: newQueueStore, policy: policy}
}

// Admit returns the job of the version and route, creating it when there is
// none and the queue has room. created reports an insert. An existing job is
// returned in any status, even with a full queue: failed is never restarted.
// A missing version or route fails with ErrVersionNotFound or ErrRouteNotFound,
// a route without forecasts with ErrForecastDisabled, a full queue with
// ErrQueueFull; none of them writes.
func (q *Queue) Admit(ctx context.Context, versionID pgtype.UUID, routeID int64) (forecasts_sqlc.PredictionJob, bool, error) {
	// The SQL checks neither: without these reads a missing row would fail
	// as a foreign key violation inside the admission lock.
	if err := q.readOnly(ctx, func(s queueStore) error {
		if _, err := getVersion(ctx, s, versionID); err != nil {
			return err
		}
		route, err := s.routes.GetRouteByID(ctx, routeID)
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			return fmt.Errorf("%w: %d", ErrRouteNotFound, routeID)
		case err != nil:
			return fmt.Errorf("get route: %w", err)
		case !route.ForecastEnabled:
			return fmt.Errorf("%w: route %d", ErrForecastDisabled, route.RouteNumber)
		}
		return nil
	}); err != nil {
		return forecasts_sqlc.PredictionJob{}, false, err
	}

	tx, err := q.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("begin job admission: %w", err)
	}
	// After Commit this is a no-op; otherwise it releases the lock and undoes
	// the insert.
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	s := q.queries(tx).jobs
	// The table lock is the first statement: it serializes admissions, so the
	// count and the insert below see every committed job.
	if err := s.LockPredictionJobAdmission(ctx); err != nil {
		return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("lock job admission: %w", err)
	}
	pair := forecasts_sqlc.GetPredictionJobByRouteAndVersionParams{ForecastVersionID: versionID, RouteID: routeID}
	existing, err := s.GetPredictionJobByRouteAndVersion(ctx, pair)
	switch {
	case err == nil:
		return existing, false, nil
	case !errors.Is(err, pgx.ErrNoRows):
		return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("find prediction job: %w", err)
	}
	pending, err := s.CountPendingPredictionJobs(ctx)
	if err != nil {
		return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("count pending prediction jobs: %w", err)
	}
	if pending >= q.policy.QueueCapacity {
		return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("%w: %d of %d jobs pending", ErrQueueFull, pending, q.policy.QueueCapacity)
	}
	job, err := s.CreatePredictionJob(ctx, forecasts_sqlc.CreatePredictionJobParams{
		ID: newUUID(), ForecastVersionID: versionID, RouteID: routeID,
	})
	created := err == nil
	if errors.Is(err, pgx.ErrNoRows) {
		// The pair conflicted. Under READ COMMITTED this new statement sees
		// the committed row.
		if job, err = s.GetPredictionJobByRouteAndVersion(ctx, pair); err != nil {
			return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("find the admitted prediction job: %w", err)
		}
	} else if err != nil {
		return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("create prediction job: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return forecasts_sqlc.PredictionJob{}, false, fmt.Errorf("commit job admission: %w", err)
	}
	return job, created, nil
}

// Lookup returns the job of the version and route without writing; found is
// false when there is none.
func (q *Queue) Lookup(ctx context.Context, versionID pgtype.UUID, routeID int64) (forecasts_sqlc.PredictionJob, bool, error) {
	var job forecasts_sqlc.PredictionJob
	found := false
	err := q.readOnly(ctx, func(s queueStore) error {
		var err error
		job, err = s.jobs.GetPredictionJobByRouteAndVersion(ctx, forecasts_sqlc.GetPredictionJobByRouteAndVersionParams{
			ForecastVersionID: versionID, RouteID: routeID,
		})
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			return nil
		case err != nil:
			return fmt.Errorf("find prediction job: %w", err)
		}
		found = true
		return nil
	})
	return job, found, err
}

// ForecastRoutes returns the routes with forecasts enabled by route number.
func (q *Queue) ForecastRoutes(ctx context.Context) ([]routes_sqlc.Route, error) {
	var enabled []routes_sqlc.Route
	err := q.readOnly(ctx, func(s queueStore) error {
		routes, err := s.routes.ListRoutes(ctx)
		if err != nil {
			return fmt.Errorf("read the route catalog: %w", err)
		}
		for _, route := range routes {
			if route.ForecastEnabled {
				enabled = append(enabled, route)
			}
		}
		return nil
	})
	return enabled, err
}

// ActiveVersion returns the active forecast version or ErrNoActiveVersion.
func (q *Queue) ActiveVersion(ctx context.Context) (forecasts_sqlc.ForecastVersion, error) {
	var version forecasts_sqlc.ForecastVersion
	err := q.readOnly(ctx, func(s queueStore) error {
		var err error
		version, err = s.jobs.GetActiveForecastVersion(ctx)
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			return ErrNoActiveVersion
		case err != nil:
			return fmt.Errorf("get the active forecast version: %w", err)
		}
		return nil
	})
	return version, err
}

// Version returns the forecast version or ErrVersionNotFound.
func (q *Queue) Version(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error) {
	var version forecasts_sqlc.ForecastVersion
	err := q.readOnly(ctx, func(s queueStore) error {
		var err error
		version, err = getVersion(ctx, s, id)
		return err
	})
	return version, err
}

// Claim takes the next claimable job for a new attempt and reads its pinned
// version and route in the same short transaction; found is false when no job
// is due. A failed read rolls the claim back, so no attempt is used.
func (q *Queue) Claim(ctx context.Context) (ClaimedJob, bool, error) {
	tx, err := q.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return ClaimedJob{}, false, fmt.Errorf("begin job claim: %w", err)
	}
	// After Commit this is a no-op; otherwise it undoes the claim.
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	s := q.queries(tx)
	job, err := s.jobs.ClaimNextPredictionJob(ctx, forecasts_sqlc.ClaimNextPredictionJobParams{
		LeaseSeconds: q.policy.leaseSeconds(), MaxAttempts: q.policy.MaxAttempts,
	})
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return ClaimedJob{}, false, nil
	case err != nil:
		return ClaimedJob{}, false, fmt.Errorf("claim prediction job: %w", err)
	}
	version, err := s.jobs.GetForecastVersion(ctx, job.ForecastVersionID)
	if err != nil {
		return ClaimedJob{}, false, fmt.Errorf("get the version of prediction job %s: %w", job.ID, err)
	}
	route, err := s.routes.GetRouteByID(ctx, job.RouteID)
	if err != nil {
		return ClaimedJob{}, false, fmt.Errorf("get the route of prediction job %s: %w", job.ID, err)
	}
	if err := tx.Commit(ctx); err != nil {
		return ClaimedJob{}, false, fmt.Errorf("commit job claim: %w", err)
	}
	return ClaimedJob{Job: job, Version: version, RouteNumber: route.RouteNumber, RouteEnabled: route.ForecastEnabled}, true, nil
}

// Request asks for the full horizon of the job's pinned version with its
// artifact IDs, so the client rejects a server that serves other artifacts.
func (q *Queue) Request(job ClaimedJob) forecasts_predictor_grpc.Request {
	return versionRequest(job.Version, job.RouteNumber)
}

// ProbeRequest asks for route 5 over the full horizon of version: a structural
// zero the ML server answers without the model, which proves that it serves
// exactly these artifacts and this period.
func ProbeRequest(version forecasts_sqlc.ForecastVersion) forecasts_predictor_grpc.Request {
	return versionRequest(version, verifyRouteNumber)
}

func versionRequest(version forecasts_sqlc.ForecastVersion, routeNumber int16) forecasts_predictor_grpc.Request {
	return forecasts_predictor_grpc.Request{
		RouteNumber:    routeNumber,
		From:           version.ForecastFrom.Time,
		To:             version.ForecastTo.Time,
		ModelVersion:   version.ModelVersion,
		DatasetVersion: version.DatasetVersion,
	}
}

// RenewLease extends the lease of the attempt; ErrLeaseLost means it no longer
// owns the job. A lease that has already expired is never revived.
func (q *Queue) RenewLease(ctx context.Context, job ClaimedJob) error {
	var renewed int64
	err := q.inTx(ctx, func(s queueStore) error {
		var err error
		renewed, err = s.jobs.RenewPredictionJobLease(ctx, forecasts_sqlc.RenewPredictionJobLeaseParams{
			LeaseSeconds: q.policy.leaseSeconds(), AttemptNumber: job.Job.AttemptCount, ID: job.Job.ID,
		})
		return err
	})
	switch {
	case err != nil:
		return fmt.Errorf("renew the lease of prediction job %s: %w", job.Job.ID, err)
	case renewed == 0:
		return fmt.Errorf("renew the lease of prediction job %s attempt %d: %w", job.Job.ID, job.Job.AttemptCount, ErrLeaseLost)
	}
	return nil
}

// Publish stores the full horizon of the job and marks it succeeded in one
// transaction, only while the attempt still owns the job. Stop renewing the
// lease first: a renewal waiting behind this transaction would then see a
// finished job. ErrLeaseLost means nothing was written and nothing more may
// be; ErrPublicationConflict means the points break the table rules. After a
// commit with an unknown outcome the job is read again: success returns nil.
func (q *Queue) Publish(ctx context.Context, job ClaimedJob, points []forecasts_predictor_grpc.Point) error {
	rows, err := predictionRows(job, points)
	if err != nil {
		return err
	}
	tx, err := q.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return fmt.Errorf("begin prediction publication: %w", err)
	}
	// After Commit this is a no-op; otherwise it undoes every row.
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	s := q.queries(tx).jobs
	// The row lock keeps recovery and other attempts away until commit, so the
	// lease is not checked again below.
	locked, err := s.LockPredictionJobForPublication(ctx, forecasts_sqlc.LockPredictionJobForPublicationParams{
		AttemptNumber: job.Job.AttemptCount, ID: job.Job.ID,
	})
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return fmt.Errorf("lock prediction job %s attempt %d: %w", job.Job.ID, job.Job.AttemptCount, ErrLeaseLost)
	case err != nil:
		return fmt.Errorf("lock prediction job %s: %w", job.Job.ID, err)
	case locked.ForecastVersionID != job.Job.ForecastVersionID || locked.RouteID != job.Job.RouteID:
		return fmt.Errorf("locked prediction job %s belongs to version %s and route %d, the claim to %s and %d",
			job.Job.ID, locked.ForecastVersionID, locked.RouteID, job.Job.ForecastVersionID, job.Job.RouteID)
	}
	copied, err := s.CopyValidationPredictions(ctx, rows)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && (pgErr.Code == "23505" || pgErr.Code == "23514") {
			return fmt.Errorf("copy predictions: %w: %w", ErrPublicationConflict, err)
		}
		return fmt.Errorf("copy predictions: %w", err)
	}
	if copied != int64(len(rows)) {
		return fmt.Errorf("copied %d predictions, want %d", copied, len(rows))
	}
	completed, err := s.CompletePredictionJob(ctx, forecasts_sqlc.CompletePredictionJobParams{
		ID: job.Job.ID, AttemptNumber: job.Job.AttemptCount,
	})
	switch {
	case err != nil:
		return fmt.Errorf("complete prediction job %s: %w", job.Job.ID, err)
	case completed != 1:
		return fmt.Errorf("completing prediction job %s changed %d rows, want 1", job.Job.ID, completed)
	}
	if err := tx.Commit(ctx); err != nil {
		return q.checkCommit(ctx, job, err)
	}
	return nil
}

// checkCommit reads the job after a failed publication commit, whose outcome
// may be unknown: succeeded with this attempt means the commit went through;
// still running with it means nothing was published and the attempt may
// record the failure; anything else means the attempt lost the job.
func (q *Queue) checkCommit(ctx context.Context, job ClaimedJob, commitErr error) error {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), commitCheckTimeout)
	defer cancel()
	var stored forecasts_sqlc.PredictionJob
	err := q.readOnly(ctx, func(s queueStore) error {
		var err error
		stored, err = s.jobs.GetPredictionJob(ctx, job.Job.ID)
		return err
	})
	if err != nil {
		return fmt.Errorf("commit prediction publication: %w; its outcome is unknown: %w", commitErr, err)
	}
	if stored.AttemptCount == job.Job.AttemptCount {
		switch stored.Status {
		case "succeeded":
			return nil
		case "running":
			return fmt.Errorf("commit prediction publication: %w", commitErr)
		}
	}
	return fmt.Errorf("commit prediction publication: %w; job %s is %s at attempt %d: %w",
		commitErr, job.Job.ID, stored.Status, stored.AttemptCount, ErrLeaseLost)
}

// predictionRows turns the checked response into rows of the job: version and
// route come from the claim, the date and hour from Moscow time. The client
// has checked the exact hourly grid; a duplicate hour still fails the primary
// key as ErrPublicationConflict and is never overwritten.
func predictionRows(job ClaimedJob, points []forecasts_predictor_grpc.Point) ([]forecasts_sqlc.CopyValidationPredictionsParams, error) {
	if job.Version.ID != job.Job.ForecastVersionID {
		return nil, fmt.Errorf("prediction job %s is pinned to version %s, not %s", job.Job.ID, job.Job.ForecastVersionID, job.Version.ID)
	}
	hours := int64(job.Version.ForecastTo.Time.Sub(job.Version.ForecastFrom.Time) / time.Hour)
	switch {
	case hours <= 0:
		// Otherwise an empty batch would mark the job succeeded.
		return nil, fmt.Errorf("version %s of prediction job %s has no forecast hours", job.Version.ID, job.Job.ID)
	case int64(len(points)) != hours:
		return nil, fmt.Errorf("got %d points for prediction job %s, its version has %d hours", len(points), job.Job.ID, hours)
	}
	rows := make([]forecasts_sqlc.CopyValidationPredictionsParams, len(points))
	for i, point := range points {
		// ponytail: Moscow keeps a fixed offset, so hours map to distinct
		// date/hour pairs; a zone with DST would need the repeated hour.
		local := point.HourStart.In(core_domain.Moscow)
		year, month, day := local.Date()
		rows[i] = forecasts_sqlc.CopyValidationPredictionsParams{
			ForecastVersionID: job.Job.ForecastVersionID,
			RouteID:           job.Job.RouteID,
			// pgx encodes the year, month and day of the time; UTC midnight
			// also equals the values it decodes.
			Date:      pgtype.Date{Time: time.Date(year, month, day, 0, 0, 0, 0, time.UTC), Valid: true},
			Hour:      int16(local.Hour()),
			Boardings: point.Boardings,
		}
	}
	return rows, nil
}

// FailAttempt records a failed attempt while it still owns the job: a
// retryable failure with attempts left requeues it after the backoff delay,
// anything else fails it for good. It returns the job as stored, so its status
// tells which. code goes to last_error_code; the text of cause, bounded, to
// last_error_message. ErrLeaseLost means the attempt no longer owns the job
// and nothing was written.
func (q *Queue) FailAttempt(ctx context.Context, job ClaimedJob, code string, retryable bool, cause error) (forecasts_sqlc.PredictionJob, error) {
	var finished forecasts_sqlc.PredictionJob
	err := q.inTx(ctx, func(s queueStore) error {
		var err error
		finished, err = s.jobs.FinishPredictionJobWithError(ctx, forecasts_sqlc.FinishPredictionJobWithErrorParams{
			Retryable:         retryable,
			MaxAttempts:       q.policy.MaxAttempts,
			RetryDelaySeconds: q.policy.retryDelaySeconds(job.Job.AttemptCount),
			ErrorCode:         code,
			ErrorMessage:      errorMessage(code, cause),
			AttemptNumber:     job.Job.AttemptCount,
			ID:                job.Job.ID,
		})
		return err
	})
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return forecasts_sqlc.PredictionJob{}, fmt.Errorf("record the failure of prediction job %s attempt %d: %w",
			job.Job.ID, job.Job.AttemptCount, ErrLeaseLost)
	case err != nil:
		return forecasts_sqlc.PredictionJob{}, fmt.Errorf("record the failure of prediction job %s: %w", job.Job.ID, err)
	}
	return finished, nil
}

// errorMessage bounds the text of cause and keeps it storable as PostgreSQL
// text, which rejects NUL and invalid UTF-8.
func errorMessage(code string, cause error) string {
	if cause == nil {
		return code
	}
	message := strings.ToValidUTF8(strings.ReplaceAll(cause.Error(), "\x00", ""), string(utf8.RuneError))
	if runes := []rune(message); len(runes) > maxErrorMessageRunes {
		message = string(runes[:maxErrorMessageRunes])
	}
	return message
}

// Recover requeues or fails running jobs whose lease expired, then fails
// queued jobs without attempts left, in bounded batches until none remain.
// recovered counts expired leases, exhausted the jobs failed for the attempt
// limit. Each batch commits on its own, so the counts are kept on error.
func (q *Queue) Recover(ctx context.Context) (recovered, exhausted int, err error) {
	for {
		var rows []forecasts_sqlc.RecoverExpiredPredictionJobsRow
		if err := q.inTx(ctx, func(s queueStore) error {
			var err error
			rows, err = s.jobs.RecoverExpiredPredictionJobs(ctx, forecasts_sqlc.RecoverExpiredPredictionJobsParams{
				MaxAttempts:       q.policy.MaxAttempts,
				RetryDelaySeconds: q.policy.retryDelaySeconds(1),
				BatchSize:         recoveryBatchSize,
			})
			return err
		}); err != nil {
			return recovered, exhausted, fmt.Errorf("recover expired prediction jobs: %w", err)
		}
		recovered += len(rows)
		if len(rows) < recoveryBatchSize {
			break
		}
	}
	for {
		var rows []forecasts_sqlc.FailExhaustedQueuedPredictionJobsRow
		if err := q.inTx(ctx, func(s queueStore) error {
			var err error
			rows, err = s.jobs.FailExhaustedQueuedPredictionJobs(ctx, forecasts_sqlc.FailExhaustedQueuedPredictionJobsParams{
				MaxAttempts: q.policy.MaxAttempts, BatchSize: recoveryBatchSize,
			})
			return err
		}); err != nil {
			return recovered, exhausted, fmt.Errorf("fail exhausted prediction jobs: %w", err)
		}
		exhausted += len(rows)
		if len(rows) < recoveryBatchSize {
			break
		}
	}
	return recovered, exhausted, nil
}

func (p JobPolicy) leaseSeconds() int32 {
	return int32(p.LeaseDuration / time.Second)
}

// retryDelaySeconds is the delay after failed attempt n: RetryDelay*4^(n-1),
// at most maxRetryDelay. Multiplying only below the cap cannot overflow.
func (p JobPolicy) retryDelaySeconds(attempt int32) int32 {
	delay := max(p.RetryDelay, 0)
	for n := int32(1); n < attempt && delay > 0 && delay < maxRetryDelay; n++ {
		delay *= 4
	}
	return int32(min(delay, maxRetryDelay) / time.Second)
}

func getVersion(ctx context.Context, s queueStore, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error) {
	version, err := s.jobs.GetForecastVersion(ctx, id)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return forecasts_sqlc.ForecastVersion{}, fmt.Errorf("%w: %s", ErrVersionNotFound, id)
	case err != nil:
		return forecasts_sqlc.ForecastVersion{}, fmt.Errorf("get forecast version: %w", err)
	}
	return version, nil
}

// inTx runs fn in a READ COMMITTED transaction and commits it when fn
// succeeds; the single statements of the queue use it like autocommit.
func (q *Queue) inTx(ctx context.Context, fn func(queueStore) error) error {
	tx, err := q.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if err := fn(q.queries(tx)); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	return nil
}

// readOnly runs fn in a READ ONLY transaction, which is always rolled back.
func (q *Queue) readOnly(ctx context.Context, fn func(queueStore) error) error {
	tx, err := q.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted, AccessMode: pgx.ReadOnly})
	if err != nil {
		return fmt.Errorf("begin read-only transaction: %w", err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	return fn(q.queries(tx))
}
