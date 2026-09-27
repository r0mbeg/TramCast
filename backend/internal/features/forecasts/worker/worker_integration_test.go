package forecasts_worker_test

import (
	"context"
	"crypto/rand"
	"fmt"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	core_postgres_testdb "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres/testdb"
	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
	forecasts_worker "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/worker"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

// These tests run the worker against the queue on a migrated schema of their
// own and skip unless TRAMCAST_TEST_DATABASE_URL is set (see
// core_postgres_testdb). The ML server is a fake that answers at once.

var contestRoutes = []int16{1, 5, 7, 11, 12, 17, 25, 26, 28, 50}

const horizonHours = 1464

// boardings is distinct per route and hour; route 5 is the structural zero.
func boardings(route int16, hour int) int64 {
	if route == 5 {
		return 0
	}
	return int64(route)*100 + int64(hour%24)
}

// gridPredictor answers every request with the full grid the client would
// accept, after the delay of its route.
type gridPredictor struct {
	delay map[int16]time.Duration
	mu    sync.Mutex
	calls map[int16]int
}

func newGridPredictor(delay map[int16]time.Duration) *gridPredictor {
	return &gridPredictor{delay: delay, calls: map[int16]int{}}
}

func (p *gridPredictor) Predict(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
	p.mu.Lock()
	p.calls[r.RouteNumber]++
	p.mu.Unlock()
	timer := time.NewTimer(p.delay[r.RouteNumber])
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
		// The real client reports the caller's cancellation so.
		return forecasts_predictor_grpc.Result{}, &forecasts_predictor_grpc.Error{Code: "ml_cancelled", Err: context.Cause(ctx)}
	}
	points := make([]forecasts_predictor_grpc.Point, int(r.To.Sub(r.From)/time.Hour))
	for i := range points {
		points[i] = forecasts_predictor_grpc.Point{
			HourStart: r.From.Add(time.Duration(i) * time.Hour).In(core_domain.Moscow),
			Boardings: boardings(r.RouteNumber, i),
		}
	}
	return forecasts_predictor_grpc.Result{ModelVersion: r.ModelVersion, DatasetVersion: r.DatasetVersion, Points: points}, nil
}

func (p *gridPredictor) callsOf(route int16) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls[route]
}

// servingPredictor is an ML server loaded with the artifacts of one version:
// a request for another version fails the way the client reports it.
type servingPredictor struct {
	grid   *gridPredictor
	mu     sync.Mutex
	served forecasts_sqlc.ForecastVersion
	// probes counts the calls of the probe route by requested model.
	probes map[string]int
}

func newServingPredictor(served forecasts_sqlc.ForecastVersion) *servingPredictor {
	return &servingPredictor{grid: newGridPredictor(nil), served: served, probes: map[string]int{}}
}

func (p *servingPredictor) serve(version forecasts_sqlc.ForecastVersion) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.served = version
}

func (p *servingPredictor) Predict(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
	p.mu.Lock()
	served := p.served
	if r.RouteNumber == 5 {
		p.probes[r.ModelVersion]++
	}
	p.mu.Unlock()
	if r.ModelVersion != served.ModelVersion || r.DatasetVersion != served.DatasetVersion {
		return forecasts_predictor_grpc.Result{}, &forecasts_predictor_grpc.Error{Code: "ml_version_mismatch",
			Err: fmt.Errorf("server artifacts are model %q, the forecast version expects %q", served.ModelVersion, r.ModelVersion)}
	}
	return p.grid.Predict(ctx, r)
}

func (p *servingPredictor) probesOf(model string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.probes[model]
}

type workerFixture struct {
	pool    *pgxpool.Pool
	queue   *forecasts_service.Queue
	version forecasts_sqlc.ForecastVersion
}

// newWorkerFixture migrates a fresh schema with the active contest version
// and one admitted job per route.
func newWorkerFixture(t *testing.T, policy forecasts_service.JobPolicy, routeNumbers ...int16) workerFixture {
	t.Helper()
	f := workerFixture{pool: core_postgres_testdb.New(t)}
	f.queue = forecasts_service.NewQueue(f.pool, policy)
	f.version = f.activate(t, "model-it")
	for _, number := range routeNumbers {
		f.admit(t, f.version, number)
	}
	return f
}

// activate registers a contest version of the model and makes it active.
func (f workerFixture) activate(t *testing.T, model string) forecasts_sqlc.ForecastVersion {
	t.Helper()
	id := pgtype.UUID{Valid: true}
	_, _ = rand.Read(id.Bytes[:])
	from := time.Date(2025, 11, 1, 0, 0, 0, 0, core_domain.Moscow)
	if _, err := forecasts_sqlc.New(f.pool).CreateForecastVersion(t.Context(), forecasts_sqlc.CreateForecastVersionParams{
		ID: id, ModelVersion: model, DatasetVersion: "dataset-it",
		HistoryEnd:   pgtype.Timestamptz{Time: from, Valid: true},
		ForecastFrom: pgtype.Timestamptz{Time: from, Valid: true},
		ForecastTo:   pgtype.Timestamptz{Time: time.Date(2026, 1, 1, 0, 0, 0, 0, core_domain.Moscow), Valid: true},
	}); err != nil {
		t.Fatal(err)
	}
	// Activation needs no ML call.
	version, activated, err := forecasts_service.NewService(f.pool, nil).Activate(t.Context(), id)
	if err != nil || !activated {
		t.Fatalf("activate %s: activated %t, %v", model, activated, err)
	}
	return version
}

// admit enables forecasts for the route and admits its job of the version.
func (f workerFixture) admit(t *testing.T, version forecasts_sqlc.ForecastVersion, number int16) {
	t.Helper()
	ctx := t.Context()
	routes := routes_sqlc.New(f.pool)
	if _, err := routes.EnsureRoute(ctx, number); err != nil {
		t.Fatal(err)
	}
	route, err := routes.GetRouteByNumber(ctx, number)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := routes.SetRouteForecastEnabled(ctx, routes_sqlc.SetRouteForecastEnabledParams{ForecastEnabled: true, RouteID: route.ID}); err != nil {
		t.Fatal(err)
	}
	if _, created, err := f.queue.Admit(ctx, version.ID, route.ID); err != nil || !created {
		t.Fatalf("admit route %d: created %t, %v", number, created, err)
	}
}

func (f workerFixture) count(t *testing.T, sql string, args ...any) int64 {
	t.Helper()
	var n int64
	if err := f.pool.QueryRow(t.Context(), sql, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// start runs the worker; stop cancels it and waits until Run returns.
func start(t *testing.T, f workerFixture, predictor forecasts_worker.Predictor, cfg forecasts_worker.Config) (stop func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	returned := make(chan struct{})
	w := forecasts_worker.New(f.queue, predictor, cfg, slog.New(slog.NewTextHandler(t.Output(), nil)))
	go func() {
		defer close(returned)
		w.Run(ctx)
	}()
	var once sync.Once
	stop = func() {
		once.Do(func() {
			cancel()
			select {
			case <-returned:
			case <-time.After(30 * time.Second):
				t.Fatal("Run did not return within 30 s of the shutdown")
			}
		})
	}
	// Runs before the pool of the fixture is closed.
	t.Cleanup(stop)
	return stop
}

func eventually(t *testing.T, what string, timeout time.Duration, done func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !done() {
		if time.Now().After(deadline) {
			t.Fatalf("%s: not within %s", what, timeout)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestWorkerPublishesEveryContestRoute(t *testing.T) {
	policy := forecasts_service.JobPolicy{QueueCapacity: 20, MaxAttempts: 3, LeaseDuration: 3 * time.Second, RetryDelay: time.Second}
	f := newWorkerFixture(t, policy, contestRoutes...)
	// Route 1 computes longer than a lease: only renewals keep it owned.
	predictor := newGridPredictor(map[int16]time.Duration{1: 4 * time.Second})
	stop := start(t, f, predictor, forecasts_worker.Config{
		Workers: 2, PollInterval: 50 * time.Millisecond, LeaseDuration: policy.LeaseDuration, ShutdownTimeout: 5 * time.Second,
	})
	eventually(t, "every job finishes", time.Minute, func() bool {
		return f.count(t, "SELECT count(*) FROM prediction_jobs WHERE status IN ('succeeded', 'failed')") == int64(len(contestRoutes))
	})
	stop()

	rows, err := f.pool.Query(t.Context(), `
		SELECT r.route_number, j.status, j.attempt_count, j.last_error_code,
			count(p.hour), coalesce(sum(p.boardings), 0)::bigint, count(DISTINCT p.date), min(p.date), max(p.date)
		FROM prediction_jobs j
		JOIN routes r ON r.id = j.route_id
		LEFT JOIN validation_predictions p ON p.forecast_version_id = j.forecast_version_id AND p.route_id = j.route_id
		WHERE j.forecast_version_id = $1
		GROUP BY r.route_number, j.status, j.attempt_count, j.last_error_code
		ORDER BY r.route_number`, f.version.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	seen := 0
	for rows.Next() {
		var (
			route             int16
			status            string
			attempt           int32
			code              pgtype.Text
			points, sum, days int64
			first, last       pgtype.Date
		)
		if err := rows.Scan(&route, &status, &attempt, &code, &points, &sum, &days, &first, &last); err != nil {
			t.Fatal(err)
		}
		seen++
		var want int64
		for i := range horizonHours {
			want += boardings(route, i)
		}
		// Attempt 1 everywhere: no lease ran out, route 1 included.
		if status != "succeeded" || attempt != 1 || code.Valid || points != horizonHours || sum != want || days != 61 ||
			first.Time.Format(time.DateOnly) != "2025-11-01" || last.Time.Format(time.DateOnly) != "2025-12-31" {
			t.Errorf("route %d: %s attempt %d, error %q, %d points summing to %d over %d days %s to %s; want %d points summing to %d",
				route, status, attempt, code.String, points, sum, days, first.Time.Format(time.DateOnly),
				last.Time.Format(time.DateOnly), horizonHours, want)
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if seen != len(contestRoutes) {
		t.Errorf("%d jobs, want %d", seen, len(contestRoutes))
	}
	if n := f.count(t, "SELECT count(*) FROM validation_predictions WHERE forecast_version_id = $1", f.version.ID); n != 14640 {
		t.Errorf("%d predictions, want 14 640", n)
	}
	// Route 5 is also the readiness probe.
	for _, route := range contestRoutes {
		want := 1
		if route == 5 {
			want = 2
		}
		if got := predictor.callsOf(route); got != want {
			t.Errorf("route %d: %d calls, want %d", route, got, want)
		}
	}
}

func TestWorkerProbesANewlyActivatedVersion(t *testing.T) {
	policy := forecasts_service.JobPolicy{QueueCapacity: 20, MaxAttempts: 3, LeaseDuration: 3 * time.Second, RetryDelay: time.Second}
	f := newWorkerFixture(t, policy, 1)
	predictor := newServingPredictor(f.version)
	stop := start(t, f, predictor, forecasts_worker.Config{
		Workers: 1, PollInterval: 50 * time.Millisecond, LeaseDuration: policy.LeaseDuration, ShutdownTimeout: 5 * time.Second,
	})
	succeeded := func(version forecasts_sqlc.ForecastVersion) func() bool {
		return func() bool {
			return f.count(t, "SELECT count(*) FROM prediction_jobs WHERE forecast_version_id = $1 AND status = 'succeeded'",
				version.ID) == 1
		}
	}
	eventually(t, "the job of the first version succeeds", 30*time.Second, succeeded(f.version))

	// The gate is open for the first version. Activate one the ML server does
	// not serve yet; its job is queued once the worker has probed it, so no
	// claim can race the switch.
	next := f.activate(t, "model-it-next")
	eventually(t, "the probe of the new version", 30*time.Second, func() bool { return predictor.probesOf(next.ModelVersion) == 1 })
	f.admit(t, next, 7)
	// Twenty poll intervals: the gate stays closed and the job keeps its attempts.
	time.Sleep(time.Second)
	var status string
	var attempts int32
	if err := f.pool.QueryRow(t.Context(), "SELECT status, attempt_count FROM prediction_jobs WHERE forecast_version_id = $1",
		next.ID).Scan(&status, &attempts); err != nil {
		t.Fatal(err)
	}
	if status != "queued" || attempts != 0 {
		t.Fatalf("job of the new version before its probe passed: %s attempt %d, want queued attempt 0", status, attempts)
	}

	// The retry of the probe, 15 s after the failed one, opens the gate.
	predictor.serve(next)
	eventually(t, "the job of the new version succeeds", time.Minute, succeeded(next))
	stop()

	var code pgtype.Text
	if err := f.pool.QueryRow(t.Context(), "SELECT attempt_count, last_error_code FROM prediction_jobs WHERE forecast_version_id = $1",
		next.ID).Scan(&attempts, &code); err != nil {
		t.Fatal(err)
	}
	if attempts != 1 || code.Valid {
		t.Errorf("job of the new version: attempt %d, error %q; want attempt 1 without an error", attempts, code.String)
	}
	for _, version := range []forecasts_sqlc.ForecastVersion{f.version, next} {
		if n := f.count(t, "SELECT count(*) FROM validation_predictions WHERE forecast_version_id = $1", version.ID); n != horizonHours {
			t.Errorf("version %s: %d predictions, want %d", version.ModelVersion, n, horizonHours)
		}
	}
	if n := predictor.probesOf(next.ModelVersion); n != 2 {
		t.Errorf("%d probes of the new version, want the failed one and the one that opened the gate", n)
	}
}

func TestWorkerShutdownLeavesTheJobToRecovery(t *testing.T) {
	policy := forecasts_service.JobPolicy{QueueCapacity: 20, MaxAttempts: 3, LeaseDuration: time.Minute}
	f := newWorkerFixture(t, policy, 1)
	predictor := newGridPredictor(map[int16]time.Duration{1: time.Hour})
	stop := start(t, f, predictor, forecasts_worker.Config{
		Workers: 1, PollInterval: 50 * time.Millisecond, LeaseDuration: policy.LeaseDuration, ShutdownTimeout: 200 * time.Millisecond,
	})
	eventually(t, "the job call starts", 30*time.Second, func() bool { return predictor.callsOf(1) == 1 })
	stopped := time.Now()
	stop()
	if d := time.Since(stopped); d > 10*time.Second {
		t.Errorf("Run returned %s after the shutdown", d)
	}

	// The cancelled call writes nothing: the job keeps its unexpired lease.
	job, err := forecasts_sqlc.New(f.pool).GetPredictionJob(t.Context(), mustJobID(t, f))
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != "running" || job.AttemptCount != 1 || job.LastErrorCode.Valid || job.FinishedAt.Valid ||
		f.count(t, "SELECT count(*) FROM prediction_jobs WHERE lease_until > clock_timestamp()") != 1 {
		t.Fatalf("job after the shutdown: %s attempt %d, error %q, finished %t", job.Status, job.AttemptCount,
			job.LastErrorCode.String, job.FinishedAt.Valid)
	}
	if n := f.count(t, "SELECT count(*) FROM validation_predictions"); n != 0 {
		t.Fatalf("%d predictions after the shutdown", n)
	}

	// Once the lease runs out, recovery requeues it for the next attempt.
	if _, err := f.pool.Exec(t.Context(), `UPDATE prediction_jobs SET lease_until = clock_timestamp() - interval '1 second'`); err != nil {
		t.Fatal(err)
	}
	if recovered, _, err := f.queue.Recover(t.Context()); err != nil || recovered != 1 {
		t.Fatalf("Recover = %d, %v", recovered, err)
	}
	job, err = forecasts_sqlc.New(f.pool).GetPredictionJob(t.Context(), job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != "queued" || job.AttemptCount != 1 || job.LastErrorCode.String != "lease_expired" {
		t.Errorf("recovered job: %s attempt %d, error %q", job.Status, job.AttemptCount, job.LastErrorCode.String)
	}
}

// mustJobID returns the ID of the only job.
func mustJobID(t *testing.T, f workerFixture) pgtype.UUID {
	t.Helper()
	var id pgtype.UUID
	if err := f.pool.QueryRow(t.Context(), "SELECT id FROM prediction_jobs").Scan(&id); err != nil {
		t.Fatal(err)
	}
	return id
}
