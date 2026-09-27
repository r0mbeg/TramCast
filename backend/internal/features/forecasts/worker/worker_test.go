package forecasts_worker

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
)

// Every test runs in a synctest bubble, so its clock is fake: the timings
// below are exact and cost no real time.
var testConfig = Config{Workers: 1, PollInterval: 2 * time.Second, LeaseDuration: time.Minute, ShutdownTimeout: 5 * time.Second}

// probeRoute is the route of forecasts_service.ProbeRequest; test jobs use
// other routes, so the fake predictor tells the probe from a job call.
const probeRoute = 5

var activeVersion = testVersion(1)

func testVersion(n byte) forecasts_sqlc.ForecastVersion {
	return forecasts_sqlc.ForecastVersion{
		ID:             pgtype.UUID{Bytes: [16]byte{n}, Valid: true},
		ModelVersion:   fmt.Sprintf("model-%d", n),
		DatasetVersion: fmt.Sprintf("dataset-%d", n),
		ForecastFrom:   pgtype.Timestamptz{Time: time.Date(2025, 11, 1, 0, 0, 0, 0, core_domain.Moscow), Valid: true},
		ForecastTo:     pgtype.Timestamptz{Time: time.Date(2026, 1, 1, 0, 0, 0, 0, core_domain.Moscow), Valid: true},
		Timezone:       core_domain.Timezone,
		IsActive:       n == 1,
	}
}

func testJob(route int16) forecasts_service.ClaimedJob {
	return testJobOf(route, activeVersion)
}

func testJobOf(route int16, version forecasts_sqlc.ForecastVersion) forecasts_service.ClaimedJob {
	return forecasts_service.ClaimedJob{
		Job: forecasts_sqlc.PredictionJob{
			ID:                pgtype.UUID{Bytes: [16]byte{0xa, byte(route)}, Valid: true},
			ForecastVersionID: version.ID,
			RouteID:           int64(route) + 1000,
			Status:            "running",
			AttemptCount:      1,
		},
		Version:      version,
		RouteNumber:  route,
		RouteEnabled: true,
	}
}

// answer is a response the client would accept for r.
func answer(r forecasts_predictor_grpc.Request) forecasts_predictor_grpc.Result {
	return forecasts_predictor_grpc.Result{
		ModelVersion:   r.ModelVersion,
		DatasetVersion: r.DatasetVersion,
		Points:         []forecasts_predictor_grpc.Point{{HourStart: r.From, Boardings: int64(r.RouteNumber)}},
	}
}

func mlError(code string, retryable bool) *forecasts_predictor_grpc.Error {
	return &forecasts_predictor_grpc.Error{Code: code, Retryable: retryable, Err: errors.New("server says " + code)}
}

// recorder keeps the calls of both fakes in order with their fake time since
// the start of the test.
type recorder struct {
	mu     sync.Mutex
	start  time.Time
	events []event
}

type event struct {
	name string
	at   time.Duration
}

// add needs r.mu held.
func (r *recorder) add(format string, args ...any) {
	r.events = append(r.events, event{fmt.Sprintf(format, args...), time.Since(r.start)})
}

func (r *recorder) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	names := make([]string, len(r.events))
	for i, e := range r.events {
		names[i] = e.name
	}
	return names
}

func (r *recorder) times(name string) []time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	var times []time.Duration
	for _, e := range r.events {
		if e.name == name {
			times = append(times, e.at)
		}
	}
	return times
}

// fakeJobs hands out queue, then reports no job due. The hooks run without
// the lock and may block.
type fakeJobs struct {
	*recorder
	queue     []forecasts_service.ClaimedJob
	claimErrs []error // returned by the first claims
	claims    []time.Duration
	recovers  []time.Duration
	published [][]forecasts_predictor_grpc.Point
	causes    []error

	active    func() (forecasts_sqlc.ForecastVersion, error)
	renew     func(ctx context.Context) error
	publish   func(ctx context.Context) error
	recoverFn func() (int, int, error)
	failErr   error
}

func (f *fakeJobs) Claim(ctx context.Context) (forecasts_service.ClaimedJob, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.claims = append(f.claims, time.Since(f.start))
	switch {
	case ctx.Err() != nil:
		return forecasts_service.ClaimedJob{}, false, ctx.Err()
	case len(f.claimErrs) > 0:
		err := f.claimErrs[0]
		f.claimErrs = f.claimErrs[1:]
		return forecasts_service.ClaimedJob{}, false, err
	case len(f.queue) == 0:
		return forecasts_service.ClaimedJob{}, false, nil
	}
	job := f.queue[0]
	f.queue = f.queue[1:]
	f.add("claim %d", job.RouteNumber)
	return job, true, nil
}

func (f *fakeJobs) Request(job forecasts_service.ClaimedJob) forecasts_predictor_grpc.Request {
	return forecasts_predictor_grpc.Request{
		RouteNumber:    job.RouteNumber,
		From:           job.Version.ForecastFrom.Time,
		To:             job.Version.ForecastTo.Time,
		ModelVersion:   job.Version.ModelVersion,
		DatasetVersion: job.Version.DatasetVersion,
	}
}

func (f *fakeJobs) RenewLease(ctx context.Context, job forecasts_service.ClaimedJob) error {
	f.mu.Lock()
	f.add("renew %d", job.RouteNumber)
	renew := f.renew
	f.mu.Unlock()
	if renew == nil {
		return nil
	}
	return renew(ctx)
}

func (f *fakeJobs) Publish(ctx context.Context, job forecasts_service.ClaimedJob, points []forecasts_predictor_grpc.Point) error {
	f.mu.Lock()
	f.add("publish %d", job.RouteNumber)
	f.published = append(f.published, points)
	publish := f.publish
	f.mu.Unlock()
	if publish == nil {
		return nil
	}
	return publish(ctx)
}

func (f *fakeJobs) FailAttempt(ctx context.Context, job forecasts_service.ClaimedJob, code string, retryable bool, cause error) (forecasts_sqlc.PredictionJob, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.add("fail %d %s %t", job.RouteNumber, code, retryable)
	f.causes = append(f.causes, cause)
	if err := ctx.Err(); err != nil {
		return forecasts_sqlc.PredictionJob{}, err
	}
	if f.failErr != nil {
		return forecasts_sqlc.PredictionJob{}, f.failErr
	}
	stored := job.Job
	stored.Status = "failed"
	if retryable {
		stored.Status = "queued"
	}
	return stored, nil
}

func (f *fakeJobs) Recover(context.Context) (int, int, error) {
	f.mu.Lock()
	f.recovers = append(f.recovers, time.Since(f.start))
	recoverFn := f.recoverFn
	f.mu.Unlock()
	if recoverFn == nil {
		return 0, 0, nil
	}
	return recoverFn()
}

func (f *fakeJobs) ActiveVersion(context.Context) (forecasts_sqlc.ForecastVersion, error) {
	f.mu.Lock()
	f.add("active")
	active := f.active
	f.mu.Unlock()
	if active == nil {
		return activeVersion, nil
	}
	return active()
}

// fakePredictor answers every call unless its hooks say otherwise.
type fakePredictor struct {
	*recorder
	probes   []forecasts_predictor_grpc.Request
	requests []forecasts_predictor_grpc.Request

	probe func(ctx context.Context) error
	job   func(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error)
}

func (p *fakePredictor) Predict(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
	p.mu.Lock()
	if r.RouteNumber == probeRoute {
		p.add("probe")
		p.probes = append(p.probes, r)
		probe := p.probe
		p.mu.Unlock()
		if probe != nil {
			if err := probe(ctx); err != nil {
				return forecasts_predictor_grpc.Result{}, err
			}
		}
		return answer(r), nil
	}
	p.add("predict %d", r.RouteNumber)
	p.requests = append(p.requests, r)
	job := p.job
	p.mu.Unlock()
	if job == nil {
		return answer(r), nil
	}
	return job(ctx, r)
}

type logBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *logBuffer) count(fragment string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return strings.Count(l.buf.String(), fragment)
}

func (l *logBuffer) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}

type harness struct {
	*recorder
	jobs      *fakeJobs
	predictor *fakePredictor
	logs      *logBuffer
}

// newHarness must run inside the bubble, like everything the worker touches.
func newHarness(queue ...forecasts_service.ClaimedJob) *harness {
	r := &recorder{start: time.Now()}
	return &harness{
		recorder:  r,
		jobs:      &fakeJobs{recorder: r, queue: queue},
		predictor: &fakePredictor{recorder: r},
		logs:      &logBuffer{},
	}
}

// at runs change under the lock of the fakes at d since the start.
func (h *harness) at(d time.Duration, change func()) {
	go func() {
		time.Sleep(d - time.Since(h.start))
		h.mu.Lock()
		defer h.mu.Unlock()
		change()
	}()
}

// run starts the worker. stop waits until every goroutine of it is blocked,
// cancels its context and returns how long Run took to return.
func (h *harness) run(cfg Config) (stop func() time.Duration) {
	ctx, cancel := context.WithCancel(context.Background())
	returned := make(chan time.Time, 1)
	w := New(h.jobs, h.predictor, cfg, slog.New(slog.NewTextHandler(h.logs, nil)))
	go func() {
		w.Run(ctx)
		returned <- time.Now()
	}()
	return func() time.Duration {
		synctest.Wait()
		cancelled := time.Now()
		cancel()
		return (<-returned).Sub(cancelled)
	}
}

// requireEvents compares the calls up to the last one that is not an active
// version read: an idle loop reads the version before every empty claim.
func requireEvents(t *testing.T, h *harness, want ...string) {
	t.Helper()
	got := h.names()
	for len(got) > 0 && got[len(got)-1] == "active" {
		got = got[:len(got)-1]
	}
	if !slices.Equal(got, want) {
		t.Fatalf("calls = %q\nwant    %q", got, want)
	}
}

func requireTimes(t *testing.T, name string, got []time.Duration, want ...time.Duration) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("%s at %v, want %v", name, got, want)
	}
}

func TestRunPublishesWhileRenewingTheLease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7))
		h.predictor.job = func(_ context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
			time.Sleep(70 * time.Second)
			return answer(r), nil
		}
		var publishCtxErr error
		renewalsDuringPublish := -1
		h.jobs.publish = func(ctx context.Context) error {
			before := len(h.times("renew 7"))
			// A renewer still running would renew during this wait.
			time.Sleep(time.Minute)
			renewalsDuringPublish = len(h.times("renew 7")) - before
			publishCtxErr = ctx.Err()
			return nil
		}
		stop := h.run(testConfig)
		time.Sleep(3*time.Minute + time.Second/2)
		stop()

		requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "renew 7", "renew 7", "renew 7", "publish 7")
		requireTimes(t, "renewals", h.times("renew 7"), 20*time.Second, 40*time.Second, 60*time.Second)
		requireTimes(t, "publication", h.times("publish 7"), 70*time.Second)
		if renewalsDuringPublish != 0 || publishCtxErr != nil {
			t.Errorf("during publication: %d renewals, context error %v; want none", renewalsDuringPublish, publishCtxErr)
		}
		if got, want := h.predictor.requests[0], h.jobs.Request(testJob(7)); got != want {
			t.Errorf("request = %+v, want %+v", got, want)
		}
		if points := h.jobs.published[0]; len(points) != 1 || points[0].Boardings != 7 {
			t.Errorf("published points = %+v, want the response points", points)
		}
		logs := h.logs.String()
		for _, fragment := range []string{`msg="prediction job succeeded"`, "status=succeeded", "attempt=1", "route_number=7",
			"job_id=" + testJob(7).Job.ID.String(), "forecast_version_id=" + activeVersion.ID.String(), "duration=2m10s"} {
			if !strings.Contains(logs, fragment) {
				t.Errorf("logs lack %q:\n%s", fragment, logs)
			}
		}
	})
}

func TestRunAbandonsJobWhenLeaseIsLost(t *testing.T) {
	for name, honoursCancel := range map[string]bool{"cancelled call": true, "late success": false} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				h := newHarness(testJob(7))
				h.jobs.renew = func(context.Context) error {
					return fmt.Errorf("renew: %w", forecasts_service.ErrLeaseLost)
				}
				var cause error
				var cancelledAt time.Duration
				h.predictor.job = func(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
					if !honoursCancel {
						time.Sleep(70 * time.Second)
						return answer(r), nil
					}
					<-ctx.Done()
					cause, cancelledAt = context.Cause(ctx), time.Since(h.start)
					// The client turns its caller's cancellation into ml_cancelled.
					return forecasts_predictor_grpc.Result{}, &forecasts_predictor_grpc.Error{Code: "ml_cancelled", Err: ctx.Err()}
				}
				stop := h.run(testConfig)
				time.Sleep(2*time.Minute + time.Second/2)
				stop()

				requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "renew 7")
				if honoursCancel && (!errors.Is(cause, errLeaseLost) || cancelledAt != 20*time.Second) {
					t.Errorf("call cancelled at %s by %v, want at 20s by errLeaseLost", cancelledAt, cause)
				}
				if h.logs.count("prediction job abandoned") != 1 {
					t.Errorf("logs lack the abandoned job:\n%s", h.logs)
				}
			})
		})
	}
}

func TestRunRetriesRenewalUntilTheLeaseMayExpire(t *testing.T) {
	t.Run("renewals keep failing", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness(testJob(7))
			h.jobs.renew = func(context.Context) error { return errors.New("connection refused") }
			var cause error
			var cancelledAt time.Duration
			h.predictor.job = func(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
				<-ctx.Done()
				cause, cancelledAt = context.Cause(ctx), time.Since(h.start)
				return forecasts_predictor_grpc.Result{}, &forecasts_predictor_grpc.Error{Code: "ml_cancelled", Err: ctx.Err()}
			}
			stop := h.run(testConfig)
			time.Sleep(2*time.Minute + time.Second/2)
			stop()

			// The lease of 60 s from the claim may expire within the next
			// renewal interval after the failure at 40 s.
			requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "renew 7", "renew 7")
			if !errors.Is(cause, errLeaseLost) || cancelledAt != 40*time.Second {
				t.Errorf("call cancelled at %s by %v, want at 40s by errLeaseLost", cancelledAt, cause)
			}
			if h.logs.count("retrying") != 1 || h.logs.count("lease presumed lost") != 1 {
				t.Errorf("logs lack the retried and the given-up renewal:\n%s", h.logs)
			}
		})
	})
	t.Run("a successful renewal moves the deadline", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness(testJob(7))
			renewals := 0
			h.jobs.renew = func(context.Context) error {
				if renewals++; renewals == 2 {
					return nil
				}
				return errors.New("connection refused")
			}
			var cause error
			var cancelledAt time.Duration
			h.predictor.job = func(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
				<-ctx.Done()
				cause, cancelledAt = context.Cause(ctx), time.Since(h.start)
				return forecasts_predictor_grpc.Result{}, &forecasts_predictor_grpc.Error{Code: "ml_cancelled", Err: ctx.Err()}
			}
			stop := h.run(testConfig)
			time.Sleep(2*time.Minute + time.Second/2)
			stop()

			// Failed at 20 s, renewed at 40 s, failed at 60 s and 80 s: the
			// lease from 40 s may expire within the interval after 80 s.
			requireTimes(t, "renewals", h.times("renew 7"), 20*time.Second, 40*time.Second, 60*time.Second, 80*time.Second)
			if !errors.Is(cause, errLeaseLost) || cancelledAt != 80*time.Second {
				t.Errorf("call cancelled at %s by %v, want at 80s by errLeaseLost", cancelledAt, cause)
			}
		})
	})
	t.Run("renewal timeout", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness(testJob(7))
			var deadlines []time.Duration
			h.jobs.renew = func(ctx context.Context) error {
				deadline, _ := ctx.Deadline()
				deadlines = append(deadlines, time.Until(deadline))
				return nil
			}
			h.predictor.job = func(_ context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
				time.Sleep(30 * time.Second)
				return answer(r), nil
			}
			stop := h.run(testConfig)
			time.Sleep(time.Minute + time.Second/2)
			stop()

			requireTimes(t, "renewal timeouts", deadlines, 20*time.Second)
		})
	})
}

func TestRunMapsPredictionErrors(t *testing.T) {
	unexpected := errors.New("unexpected")
	for _, test := range []struct {
		err  error
		want string
	}{
		{mlError("ml_unavailable", true), "fail 7 ml_unavailable true"},
		{mlError("ml_deadline_exceeded", true), "fail 7 ml_deadline_exceeded true"},
		{mlError("ml_invalid_response", false), "fail 7 ml_invalid_response false"},
		// Not cancelled by the worker: the attempt still owns the job.
		{mlError("ml_cancelled", false), "fail 7 ml_cancelled false"},
		{fmt.Errorf("predict: %w", mlError("ml_not_found", false)), "fail 7 ml_not_found false"},
		{unexpected, "fail 7 internal_error false"},
	} {
		t.Run(test.want, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				h := newHarness(testJob(7))
				h.predictor.job = func(context.Context, forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
					return forecasts_predictor_grpc.Result{}, test.err
				}
				stop := h.run(testConfig)
				time.Sleep(time.Minute + time.Second/2)
				stop()

				names := h.names()
				if !slices.Contains(names, test.want) || slices.Contains(names, "publish 7") {
					t.Fatalf("calls = %q, want %q and no publication", names, test.want)
				}
				if len(h.jobs.causes) != 1 || h.jobs.causes[0] != test.err {
					t.Errorf("failure causes = %v, want %v", h.jobs.causes, test.err)
				}
			})
		})
	}
}

func TestRunMapsPublicationErrors(t *testing.T) {
	for _, test := range []struct {
		name string
		err  error
		want []string // calls after the publication
		log  string
	}{
		{"published", nil, nil, "prediction job succeeded"},
		{"lease lost", fmt.Errorf("lock: %w", forecasts_service.ErrLeaseLost), nil, "lost before publication"},
		{"conflict", fmt.Errorf("copy: %w", forecasts_service.ErrPublicationConflict), []string{"fail 7 publication_conflict false"}, "status=failed"},
		{"database error", errors.New("connection reset"), []string{"fail 7 publication_failed true"}, "status=queued"},
	} {
		t.Run(test.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				h := newHarness(testJob(7))
				h.jobs.publish = func(context.Context) error { return test.err }
				stop := h.run(testConfig)
				time.Sleep(time.Minute + time.Second/2)
				stop()

				requireEvents(t, h, append([]string{"active", "probe", "active", "claim 7", "predict 7", "publish 7"}, test.want...)...)
				if len(test.want) > 0 && h.jobs.causes[0] != test.err {
					t.Errorf("failure cause = %v, want %v", h.jobs.causes[0], test.err)
				}
				if h.logs.count(test.log) != 1 {
					t.Errorf("logs lack %q:\n%s", test.log, h.logs)
				}
			})
		})
	}
}

func TestRunOnlyLogsFailureOfLostJob(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7), testJob(11))
		h.jobs.failErr = fmt.Errorf("finish: %w", forecasts_service.ErrLeaseLost)
		h.predictor.job = func(_ context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
			if r.RouteNumber == 7 {
				return forecasts_predictor_grpc.Result{}, mlError("ml_invalid_response", false)
			}
			return answer(r), nil
		}
		stop := h.run(testConfig)
		time.Sleep(time.Minute + time.Second/2)
		stop()

		requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "fail 7 ml_invalid_response false",
			"active", "claim 11", "predict 11", "publish 11")
		if h.logs.count("lost before its failure was recorded") != 1 {
			t.Errorf("logs lack the lost job:\n%s", h.logs)
		}
	})
}

func TestRunFailsJobOfDisabledRoute(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		job := testJob(7)
		job.RouteEnabled = false
		h := newHarness(job)
		stop := h.run(testConfig)
		time.Sleep(time.Minute + time.Second/2)
		stop()

		requireEvents(t, h, "active", "probe", "active", "claim 7", "fail 7 route_not_enabled false")
	})
}

func TestRunShutdownCancelsSlowCallAfterGrace(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7), testJob(11))
		var cause error
		var cancelledAt time.Duration
		h.predictor.job = func(ctx context.Context, _ forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
			<-ctx.Done()
			cause, cancelledAt = context.Cause(ctx), time.Since(h.start)
			return forecasts_predictor_grpc.Result{}, &forecasts_predictor_grpc.Error{Code: "ml_cancelled", Err: ctx.Err()}
		}
		stop := h.run(testConfig)
		time.Sleep(38*time.Second + time.Second/2)
		took := stop()

		if took != testConfig.ShutdownTimeout {
			t.Errorf("Run returned %s after the shutdown, want %s", took, testConfig.ShutdownTimeout)
		}
		if !errors.Is(cause, errShutdown) || cancelledAt != 43*time.Second+time.Second/2 {
			t.Errorf("call cancelled at %s by %v, want at 43.5s by errShutdown", cancelledAt, cause)
		}
		// The lease is still renewed during the grace; nothing is written and
		// no other job is claimed.
		requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "renew 7", "renew 7")
		requireTimes(t, "claims", h.jobs.claims, 0)
		if h.logs.count("prediction job abandoned") != 1 {
			t.Errorf("logs lack the abandoned job:\n%s", h.logs)
		}
	})
}

func TestRunShutdownLetsFastCallPublish(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7), testJob(11))
		var publishCtxErr error
		h.predictor.job = func(_ context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
			time.Sleep(8 * time.Second)
			return answer(r), nil
		}
		h.jobs.publish = func(ctx context.Context) error {
			publishCtxErr = ctx.Err()
			return nil
		}
		stop := h.run(testConfig)
		time.Sleep(3*time.Second + time.Second/2)
		took := stop()

		if took != 4*time.Second+time.Second/2 {
			t.Errorf("Run returned %s after the shutdown, want 4.5s", took)
		}
		requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "publish 7")
		requireTimes(t, "claims", h.jobs.claims, 0)
		if publishCtxErr != nil {
			t.Errorf("publication context error = %v, want none within the grace", publishCtxErr)
		}
	})
}

func TestRunShutdownAbandonsPublicationAfterGrace(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7))
		h.jobs.publish = func(ctx context.Context) error {
			<-ctx.Done()
			return fmt.Errorf("copy predictions: %w", ctx.Err())
		}
		stop := h.run(testConfig)
		time.Sleep(3*time.Second + time.Second/2)
		took := stop()

		if took != testConfig.ShutdownTimeout {
			t.Errorf("Run returned %s after the shutdown, want %s", took, testConfig.ShutdownTimeout)
		}
		requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "publish 7")
		if h.logs.count("abandoned during publication") != 1 {
			t.Errorf("logs lack the abandoned publication:\n%s", h.logs)
		}
	})
}

func TestRunWithoutActiveVersionClaimsNothing(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7))
		h.jobs.active = func() (forecasts_sqlc.ForecastVersion, error) {
			return forecasts_sqlc.ForecastVersion{}, forecasts_service.ErrNoActiveVersion
		}
		stop := h.run(testConfig)
		time.Sleep(2*time.Minute + time.Second/2)
		if took := stop(); took != 0 {
			t.Errorf("Run returned %s after the shutdown, want at once", took)
		}

		requireTimes(t, "claims", h.jobs.claims)
		// The gate retries every max(PollInterval, 15s); recovery ignores it.
		s := time.Second
		requireTimes(t, "active version reads", h.times("active"), 0, 15*s, 30*s, 45*s, 60*s, 75*s, 90*s, 105*s, 120*s)
		requireTimes(t, "recoveries", h.jobs.recovers, 0, 30*s, 60*s, 90*s, 120*s)
		if len(h.predictor.probes) != 0 {
			t.Errorf("probes = %d, want none", len(h.predictor.probes))
		}
		if n := h.logs.count("no active forecast version"); n != 1 {
			t.Errorf("logged the missing version %d times, want once:\n%s", n, h.logs)
		}
	})
}

func TestRunWaitsForTheProbe(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7))
		probes := 0
		var timeouts []time.Duration
		h.predictor.probe = func(ctx context.Context) error {
			deadline, _ := ctx.Deadline()
			timeouts = append(timeouts, time.Until(deadline))
			switch probes++; probes {
			case 1:
				<-ctx.Done()
				return mlError("ml_deadline_exceeded", true)
			case 2, 3:
				return mlError("ml_unavailable", true)
			}
			return nil
		}
		stop := h.run(testConfig)
		time.Sleep(time.Minute + time.Second/2)
		stop()

		s := time.Second
		requireTimes(t, "probes", h.times("probe"), 0, 25*s, 40*s, 55*s)
		requireTimes(t, "probe timeouts", timeouts, probeTimeout, probeTimeout, probeTimeout, probeTimeout)
		requireTimes(t, "job claims", h.times("claim 7"), 55*s)
		requireTimes(t, "recoveries", h.jobs.recovers, 0, 30*s, 60*s)
		for _, probe := range h.predictor.probes {
			if want := forecasts_service.ProbeRequest(activeVersion); probe != want {
				t.Fatalf("probe = %+v, want %+v", probe, want)
			}
		}
		// Two failure codes, each logged once, then readiness.
		for fragment, want := range map[string]int{
			"ML server does not serve the active forecast version": 2,
			"error_code=ml_unavailable":                            1,
			"ML server ready":                                      1,
		} {
			if n := h.logs.count(fragment); n != want {
				t.Errorf("logged %q %d times, want %d:\n%s", fragment, n, want, h.logs)
			}
		}
	})
}

func TestRunProbesOnceForAllLoops(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7))
		h.predictor.probe = func(context.Context) error { return mlError("ml_unavailable", true) }
		cfg := testConfig
		cfg.Workers = 3
		stop := h.run(cfg)
		time.Sleep(time.Minute + time.Second/2)
		stop()

		s := time.Second
		requireTimes(t, "probes", h.times("probe"), 0, 15*s, 30*s, 45*s, 60*s)
		requireTimes(t, "claims", h.jobs.claims)
		if n := h.logs.count("ML server does not serve"); n != 1 {
			t.Errorf("logged the probe failure %d times, want once:\n%s", n, h.logs)
		}
	})
}

func TestRunPausesClaimsAfterUnavailableServer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7), testJob(11))
		probes := 0
		h.predictor.probe = func(context.Context) error {
			if probes++; probes == 2 {
				return mlError("ml_unavailable", true)
			}
			return nil
		}
		h.predictor.job = func(_ context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
			if r.RouteNumber == 7 {
				return forecasts_predictor_grpc.Result{}, mlError("ml_unavailable", true)
			}
			return answer(r), nil
		}
		stop := h.run(testConfig)
		time.Sleep(time.Minute + time.Second/2)
		stop()

		requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "fail 7 ml_unavailable true",
			"active", "probe", "active", "probe", "active", "claim 11", "predict 11", "publish 11")
		requireTimes(t, "second claim", h.times("claim 11"), 15*time.Second)
		if h.logs.count("claims paused") != 1 {
			t.Errorf("logs lack the paused claims:\n%s", h.logs)
		}
	})
}

func TestRunVersionMismatch(t *testing.T) {
	t.Run("active version pauses claims", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness(testJob(7), testJob(11))
			probes := 0
			h.predictor.probe = func(context.Context) error {
				if probes++; probes > 1 {
					return mlError("ml_version_mismatch", false)
				}
				return nil
			}
			h.predictor.job = func(context.Context, forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
				return forecasts_predictor_grpc.Result{}, mlError("ml_version_mismatch", false)
			}
			stop := h.run(testConfig)
			time.Sleep(20 * time.Second)
			stop()

			requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "active", "fail 7 ml_version_mismatch false",
				"active", "probe", "active", "probe")
		})
	})
	t.Run("another version fails only its job", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness(testJobOf(7, testVersion(2)), testJob(11))
			h.predictor.job = func(_ context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
				if r.RouteNumber == 7 {
					return forecasts_predictor_grpc.Result{}, mlError("ml_version_mismatch", false)
				}
				return answer(r), nil
			}
			stop := h.run(testConfig)
			time.Sleep(20 * time.Second)
			stop()

			requireEvents(t, h, "active", "probe", "active", "claim 7", "predict 7", "active", "fail 7 ml_version_mismatch false",
				"active", "claim 11", "predict 11", "publish 11")
		})
	})
}

func TestRunFollowsTheActiveVersion(t *testing.T) {
	s := time.Second
	next := testVersion(2)
	// The active version changes off the 2 s poll grid, so the loop reads the
	// change before the claim at 12 s.
	const switchAt = 11 * time.Second
	claimsBetween := func(h *harness, from, to time.Duration) []time.Duration {
		var between []time.Duration
		for _, at := range h.jobs.claims {
			if at > from && at < to {
				between = append(between, at)
			}
		}
		return between
	}
	requireLogs := func(t *testing.T, h *harness, counts map[string]int) {
		t.Helper()
		for fragment, want := range counts {
			if n := h.logs.count(fragment); n != want {
				t.Errorf("logged %q %d times, want %d:\n%s", fragment, n, want, h.logs)
			}
		}
	}
	// switchVersion makes next active at switchAt and queues a job of it.
	switchVersion := func(h *harness) {
		current := activeVersion
		h.jobs.active = func() (forecasts_sqlc.ForecastVersion, error) {
			h.mu.Lock()
			defer h.mu.Unlock()
			return current, nil
		}
		h.at(switchAt, func() {
			current = next
			h.jobs.queue = append(h.jobs.queue, testJobOf(7, next))
		})
	}

	t.Run("a new active version waits for its probe", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness()
			switchVersion(h)
			// The ML server serves the new version from the third probe on.
			probes := 0
			h.predictor.probe = func(context.Context) error {
				if probes++; probes == 2 {
					return mlError("ml_version_mismatch", false)
				}
				return nil
			}
			stop := h.run(testConfig)
			time.Sleep(time.Minute + s/2)
			stop()

			requireTimes(t, "probes", h.times("probe"), 0, 12*s, 27*s)
			requireTimes(t, "claims while the gate is closed", claimsBetween(h, switchAt, 27*s))
			requireTimes(t, "job claims", h.times("claim 7"), 27*s)
			requireTimes(t, "publications", h.times("publish 7"), 27*s)
			want := []forecasts_predictor_grpc.Request{forecasts_service.ProbeRequest(activeVersion),
				forecasts_service.ProbeRequest(next), forecasts_service.ProbeRequest(next)}
			if !slices.Equal(h.predictor.probes, want) {
				t.Errorf("probes = %+v, want %+v", h.predictor.probes, want)
			}
			if got := h.predictor.requests; len(got) != 1 || got[0] != h.jobs.Request(testJobOf(7, next)) {
				t.Errorf("job requests = %+v, want the job of the new version", got)
			}
			requireLogs(t, h, map[string]int{
				"active forecast version changed; probing the ML server": 1,
				"ML server does not serve the active forecast version":   1,
				"ML server ready": 2,
			})
		})
	})
	t.Run("a failed probe of the new version claims nothing", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness()
			switchVersion(h)
			probes := 0
			h.predictor.probe = func(context.Context) error {
				if probes++; probes > 1 {
					return mlError("ml_version_mismatch", false)
				}
				return nil
			}
			stop := h.run(testConfig)
			time.Sleep(time.Minute + s/2)
			stop()

			requireTimes(t, "probes", h.times("probe"), 0, 12*s, 27*s, 42*s, 57*s)
			requireTimes(t, "claims after the switch", claimsBetween(h, switchAt, time.Hour))
			if len(h.predictor.requests) != 0 {
				t.Errorf("job requests = %+v, want none", h.predictor.requests)
			}
			requireLogs(t, h, map[string]int{
				"active forecast version changed; probing the ML server": 1,
				"ML server does not serve the active forecast version":   1,
				"ML server ready": 1,
			})
		})
	})
	t.Run("no active version closes the gate", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness()
			gone := false
			h.jobs.active = func() (forecasts_sqlc.ForecastVersion, error) {
				h.mu.Lock()
				defer h.mu.Unlock()
				if gone {
					return forecasts_sqlc.ForecastVersion{}, forecasts_service.ErrNoActiveVersion
				}
				return activeVersion, nil
			}
			h.at(switchAt, func() {
				gone = true
				h.jobs.queue = append(h.jobs.queue, testJob(7))
			})
			h.at(35*s, func() { gone = false })
			stop := h.run(testConfig)
			time.Sleep(time.Minute + s/2)
			stop()

			// Closed at 12 s, the gate retries at 27 s and finds the version
			// again at 42 s.
			requireTimes(t, "probes", h.times("probe"), 0, 42*s)
			requireTimes(t, "claims while the gate is closed", claimsBetween(h, switchAt, 42*s))
			requireTimes(t, "job claims", h.times("claim 7"), 42*s)
			requireLogs(t, h, map[string]int{
				"no active forecast version":      1,
				"active forecast version changed": 0,
				"ML server ready":                 2,
			})
		})
	})
	t.Run("an unreadable version skips the claim", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHarness(testJob(7))
			reads := 0
			h.jobs.active = func() (forecasts_sqlc.ForecastVersion, error) {
				// The probe reads first, the check before the first claim second.
				if reads++; reads == 2 {
					return forecasts_sqlc.ForecastVersion{}, errors.New("connection refused")
				}
				return activeVersion, nil
			}
			stop := h.run(testConfig)
			time.Sleep(10*s + s/2)
			stop()

			// The gate stays open: the claims start one poll interval later
			// without another probe.
			requireTimes(t, "probes", h.times("probe"), 0)
			requireTimes(t, "claims", h.jobs.claims, 2*s, 2*s, 4*s, 6*s, 8*s, 10*s)
			requireTimes(t, "job claims", h.times("claim 7"), 2*s)
			requireLogs(t, h, map[string]int{"read the active forecast version before a claim": 1})
		})
	})
}

func TestRunRetriesFailedClaims(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(testJob(7))
		h.jobs.claimErrs = []error{errors.New("connection refused")}
		stop := h.run(testConfig)
		time.Sleep(10*time.Second + time.Second/2)
		if took := stop(); took != 0 {
			t.Errorf("Run returned %s after the shutdown, want at once", took)
		}

		s := time.Second
		// A found job is followed by another claim at once; an empty or failed
		// one by the poll interval.
		requireTimes(t, "claims", h.jobs.claims, 0, 2*s, 2*s, 4*s, 6*s, 8*s, 10*s)
		requireTimes(t, "job claims", h.times("claim 7"), 2*s)
		if h.logs.count("claim a prediction job") != 1 {
			t.Errorf("logs lack the failed claim:\n%s", h.logs)
		}
	})
}

func TestRunRecoversAtStartAndEveryHalfLease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness()
		recoveries := 0
		h.jobs.recoverFn = func() (int, int, error) {
			switch recoveries++; recoveries {
			case 1:
				return 2, 1, nil
			case 2:
				return 0, 0, errors.New("connection refused")
			case 3:
				return 0, 3, nil
			}
			return 0, 0, nil
		}
		stop := h.run(testConfig)
		time.Sleep(90*time.Second + time.Second/2)
		stop()

		requireTimes(t, "recoveries", h.jobs.recovers, 0, 30*time.Second, time.Minute, 90*time.Second)
		for fragment, want := range map[string]int{
			"prediction jobs recovered": 2,
			"recovered=2 exhausted=1":   1,
			"recovered=0 exhausted=3":   1,
			"recover prediction jobs\"": 1,
		} {
			if n := h.logs.count(fragment); n != want {
				t.Errorf("logged %q %d times, want %d:\n%s", fragment, n, want, h.logs)
			}
		}
	})
}

func TestGateFollowIgnoresReadsFromBeforeAnOpening(t *testing.T) {
	v1, v2 := pgtype.UUID{Bytes: [16]byte{1}, Valid: true}, pgtype.UUID{Bytes: [16]byte{2}, Valid: true}
	g := &gate{probing: make(chan struct{}, 1)}
	g.openFor(v1)
	stale := g.openings()
	// Another loop sees v2 active, closes the gate and opens it for v2.
	if open, closed := g.follow(v2, g.openings(), "active version changed"); open || !closed {
		t.Fatalf("follow(v2) = %t, %t; want the gate closed for the new version", open, closed)
	}
	g.openFor(v2)
	// A read of v1 taken before that opening must not close the gate again.
	if open, closed := g.follow(v1, stale, "active version changed"); !open || closed {
		t.Fatalf("stale follow(v1) = %t, %t; want the gate kept open for v2", open, closed)
	}
	// A current read of another version still closes it.
	if open, closed := g.follow(v1, g.openings(), "active version changed"); open || !closed {
		t.Fatalf("follow(v1) = %t, %t; want the gate closed", open, closed)
	}
}
