package forecasts_service_test

import (
	"context"
	"crypto/rand"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	core_postgres_testdb "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres/testdb"
	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

// These tests run the queue on a migrated schema of their own and skip unless
// TRAMCAST_TEST_DATABASE_URL is set (see core_postgres_testdb).

const horizonHours = 1464

var (
	horizonFrom = time.Date(2025, 11, 1, 0, 0, 0, 0, core_domain.Moscow)
	horizonTo   = time.Date(2026, 1, 1, 0, 0, 0, 0, core_domain.Moscow)
	// dbPolicy requeues at once, so a scenario can claim again without waiting.
	dbPolicy = forecasts_service.JobPolicy{QueueCapacity: 20, MaxAttempts: 3, LeaseDuration: time.Minute}
)

type dbFixture struct {
	pool    *pgxpool.Pool
	version forecasts_sqlc.ForecastVersion
	// routes maps route numbers to routes.id.
	routes map[int16]int64
}

// newDBFixture migrates a fresh schema with an inactive contest version and
// the given routes with forecasts enabled.
func newDBFixture(t *testing.T, routeNumbers ...int16) dbFixture {
	t.Helper()
	f := dbFixture{pool: core_postgres_testdb.New(t), routes: map[int16]int64{}}
	for _, number := range routeNumbers {
		f.routes[number] = addRoute(t, f.pool, number, true)
	}
	var err error
	f.version, err = forecasts_sqlc.New(f.pool).CreateForecastVersion(t.Context(), forecasts_sqlc.CreateForecastVersionParams{
		ID: newID(), ModelVersion: "model-it", DatasetVersion: "dataset-it",
		HistoryEnd:   pgtype.Timestamptz{Time: horizonFrom, Valid: true},
		ForecastFrom: pgtype.Timestamptz{Time: horizonFrom, Valid: true},
		ForecastTo:   pgtype.Timestamptz{Time: horizonTo, Valid: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func addRoute(t *testing.T, pool *pgxpool.Pool, number int16, enabled bool) int64 {
	t.Helper()
	ctx := t.Context()
	q := routes_sqlc.New(pool)
	if _, err := q.EnsureRoute(ctx, number); err != nil {
		t.Fatal(err)
	}
	route, err := q.GetRouteByNumber(ctx, number)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.SetRouteForecastEnabled(ctx, routes_sqlc.SetRouteForecastEnabledParams{ForecastEnabled: enabled, RouteID: route.ID}); err != nil {
		t.Fatal(err)
	}
	return route.ID
}

func newID() pgtype.UUID {
	id := pgtype.UUID{Valid: true}
	_, _ = rand.Read(id.Bytes[:])
	id.Bytes[6] = id.Bytes[6]&0x0f | 0x40
	id.Bytes[8] = id.Bytes[8]&0x3f | 0x80
	return id
}

// horizon is a response the client would accept for the fixture version:
// every hour of November and December 2025, with boardings i at hour i.
func horizon() []forecasts_predictor_grpc.Point {
	points := make([]forecasts_predictor_grpc.Point, horizonHours)
	for i := range points {
		points[i] = forecasts_predictor_grpc.Point{HourStart: horizonFrom.Add(time.Duration(i) * time.Hour), Boardings: int64(i)}
	}
	return points
}

func (f dbFixture) queue(policy forecasts_service.JobPolicy) *forecasts_service.Queue {
	return forecasts_service.NewQueue(f.pool, policy)
}

func (f dbFixture) admit(t *testing.T, queue *forecasts_service.Queue, number int16) forecasts_sqlc.PredictionJob {
	t.Helper()
	job, created, err := queue.Admit(t.Context(), f.version.ID, f.routes[number])
	if err != nil || !created {
		t.Fatalf("admit route %d: created %t, %v", number, created, err)
	}
	return job
}

func claim(t *testing.T, queue *forecasts_service.Queue) forecasts_service.ClaimedJob {
	t.Helper()
	job, found, err := queue.Claim(t.Context())
	if err != nil || !found {
		t.Fatalf("claim: found %t, %v", found, err)
	}
	return job
}

func requireNoClaim(t *testing.T, queue *forecasts_service.Queue) {
	t.Helper()
	if job, found, err := queue.Claim(t.Context()); err != nil || found {
		t.Fatalf("claim: found %t (job %s attempt %d), %v", found, job.Job.ID, job.Job.AttemptCount, err)
	}
}

func (f dbFixture) job(t *testing.T, id pgtype.UUID) forecasts_sqlc.PredictionJob {
	t.Helper()
	job, err := forecasts_sqlc.New(f.pool).GetPredictionJob(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	return job
}

func (f dbFixture) exec(t *testing.T, sql string, args ...any) int64 {
	t.Helper()
	tag, err := f.pool.Exec(t.Context(), sql, args...)
	if err != nil {
		t.Fatal(err)
	}
	return tag.RowsAffected()
}

func (f dbFixture) count(t *testing.T, sql string, args ...any) int64 {
	t.Helper()
	var n int64
	if err := f.pool.QueryRow(t.Context(), sql, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// predictions counts the stored rows of the fixture version and route.
func (f dbFixture) predictions(t *testing.T, number int16) int64 {
	t.Helper()
	return f.count(t, `SELECT count(*) FROM validation_predictions WHERE forecast_version_id = $1 AND route_id = $2`,
		f.version.ID, f.routes[number])
}

// until returns the time from the database clock to a timestamp column of
// the job, which must be set.
func (f dbFixture) until(t *testing.T, column string, id pgtype.UUID) time.Duration {
	t.Helper()
	var seconds float64
	if err := f.pool.QueryRow(t.Context(),
		"SELECT extract(epoch FROM "+column+" - clock_timestamp())::float8 FROM prediction_jobs WHERE id = $1", id,
	).Scan(&seconds); err != nil {
		t.Fatal(err)
	}
	return time.Duration(seconds * float64(time.Second))
}

// expireLease stands in for a lease running out.
func (f dbFixture) expireLease(t *testing.T, id pgtype.UUID) {
	t.Helper()
	if n := f.exec(t, `UPDATE prediction_jobs SET lease_until = clock_timestamp() - interval '1 second'
		WHERE id = $1 AND status = 'running'`, id); n != 1 {
		t.Fatalf("expired %d leases, want 1", n)
	}
}

// makeDue stands in for a retry delay running out.
func (f dbFixture) makeDue(t *testing.T, id pgtype.UUID) {
	t.Helper()
	if n := f.exec(t, `UPDATE prediction_jobs SET run_after = clock_timestamp() WHERE id = $1 AND status = 'queued'`, id); n != 1 {
		t.Fatalf("made %d jobs due, want 1", n)
	}
}

func TestQueueAdmitsOnePairOnceUnderConcurrency(t *testing.T) {
	f := newDBFixture(t, 1)
	queue := f.queue(dbPolicy)
	type outcome struct {
		job     forecasts_sqlc.PredictionJob
		created bool
		err     error
	}
	outcomes := make([]outcome, 16)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range outcomes {
		wg.Go(func() {
			<-start
			o := &outcomes[i]
			o.job, o.created, o.err = queue.Admit(t.Context(), f.version.ID, f.routes[1])
		})
	}
	close(start)
	wg.Wait()

	created := 0
	for i, o := range outcomes {
		if o.err != nil {
			t.Fatalf("caller %d: %v", i, o.err)
		}
		if o.created {
			created++
		}
		if o.job.ID != outcomes[0].job.ID {
			t.Errorf("caller %d got job %s, caller 0 %s", i, o.job.ID, outcomes[0].job.ID)
		}
	}
	if created != 1 {
		t.Errorf("%d callers created the job, want 1", created)
	}
	if n := f.count(t, "SELECT count(*) FROM prediction_jobs"); n != 1 {
		t.Errorf("%d jobs stored, want 1", n)
	}
	job, found, err := queue.Lookup(t.Context(), f.version.ID, f.routes[1])
	if err != nil || !found || job.ID != outcomes[0].job.ID || job.Status != "queued" || job.AttemptCount != 0 {
		t.Errorf("Lookup = %s %s attempt %d, found %t, %v", job.ID, job.Status, job.AttemptCount, found, err)
	}
}

func TestQueueCapacityHoldsUnderConcurrentAdmission(t *testing.T) {
	numbers := []int16{1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
	f := newDBFixture(t, numbers...)
	policy := dbPolicy
	policy.QueueCapacity = 3
	queue := f.queue(policy)
	ctx := t.Context()

	created := make([]bool, len(numbers))
	errs := make([]error, len(numbers))
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, number := range numbers {
		wg.Go(func() {
			<-start
			_, created[i], errs[i] = queue.Admit(ctx, f.version.ID, f.routes[number])
		})
	}
	close(start)
	wg.Wait()

	var admitted, rejected []int16
	for i, number := range numbers {
		switch {
		case errs[i] == nil && created[i]:
			admitted = append(admitted, number)
		case errors.Is(errs[i], forecasts_service.ErrQueueFull) && !created[i]:
			rejected = append(rejected, number)
		default:
			t.Fatalf("route %d: created %t, %v", number, created[i], errs[i])
		}
	}
	if len(admitted) != 3 || len(rejected) != 7 {
		t.Fatalf("admitted %v, rejected %v; want 3 and 7", admitted, rejected)
	}
	if n := f.count(t, "SELECT count(*) FROM prediction_jobs"); n != 3 {
		t.Fatalf("%d jobs stored, want 3", n)
	}

	// A full queue still returns an existing job and creates no new one.
	if job, created, err := queue.Admit(ctx, f.version.ID, f.routes[admitted[0]]); err != nil || created || job.Status != "queued" {
		t.Errorf("admit an existing job at capacity: %s created %t, %v", job.Status, created, err)
	}
	if _, _, err := queue.Admit(ctx, f.version.ID, f.routes[rejected[0]]); !errors.Is(err, forecasts_service.ErrQueueFull) {
		t.Errorf("admit a new job at capacity: %v", err)
	}
	// A succeeded job no longer takes a place.
	if err := queue.Publish(ctx, claim(t, queue), horizon()); err != nil {
		t.Fatal(err)
	}
	if _, created, err := queue.Admit(ctx, f.version.ID, f.routes[rejected[0]]); err != nil || !created {
		t.Errorf("admit after a publication: created %t, %v", created, err)
	}
}

func TestQueueConcurrentClaimersTakeDistinctJobs(t *testing.T) {
	numbers := []int16{1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
	f := newDBFixture(t, numbers...)
	queue := f.queue(dbPolicy)
	for _, number := range numbers {
		f.admit(t, queue, number)
	}

	var mu sync.Mutex
	var claimed []forecasts_service.ClaimedJob
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			<-start
			for {
				job, found, err := queue.Claim(t.Context())
				if err != nil {
					t.Error(err)
					return
				}
				if !found {
					return
				}
				mu.Lock()
				claimed = append(claimed, job)
				mu.Unlock()
			}
		})
	}
	close(start)
	wg.Wait()

	ids := map[pgtype.UUID]bool{}
	for _, c := range claimed {
		ids[c.Job.ID] = true
		if c.Job.Status != "running" || c.Job.AttemptCount != 1 || !c.Job.LeaseUntil.Valid ||
			c.Version.ID != f.version.ID || f.routes[c.RouteNumber] != c.Job.RouteID || !c.RouteEnabled {
			t.Errorf("claimed job %s: %s attempt %d, version %s, route %d (id %d), enabled %t", c.Job.ID, c.Job.Status,
				c.Job.AttemptCount, c.Version.ID, c.RouteNumber, c.Job.RouteID, c.RouteEnabled)
		}
	}
	if len(claimed) != len(numbers) || len(ids) != len(numbers) {
		t.Errorf("%d claims of %d distinct jobs, want %d of %d", len(claimed), len(ids), len(numbers), len(numbers))
	}
	if n := f.count(t, "SELECT count(*) FROM prediction_jobs WHERE status = 'running' AND attempt_count = 1"); n != int64(len(numbers)) {
		t.Errorf("%d jobs running at attempt 1, want %d", n, len(numbers))
	}
}

func TestQueueClaimSkipsALockedJob(t *testing.T) {
	f := newDBFixture(t, 1, 7)
	queue := f.queue(dbPolicy)
	ctx := t.Context()
	locked := f.admit(t, queue, 1)
	free := f.admit(t, queue, 7)

	tx, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, "SELECT id FROM prediction_jobs WHERE id = $1 FOR UPDATE", locked.ID); err != nil {
		t.Fatal(err)
	}
	// A claim that waited for the lock would run into the timeout.
	claimCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	job, found, err := queue.Claim(claimCtx)
	if err != nil || !found || job.Job.ID != free.ID {
		t.Fatalf("claim next to a locked job: %s found %t, %v; want %s", job.Job.ID, found, err, free.ID)
	}
	if _, found, err := queue.Claim(claimCtx); err != nil || found {
		t.Fatalf("claim with only the locked job left: found %t, %v", found, err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if job := claim(t, queue); job.Job.ID != locked.ID {
		t.Errorf("claim after the lock: %s, want %s", job.Job.ID, locked.ID)
	}
}

func TestQueueRecoversAnExpiredLease(t *testing.T) {
	f := newDBFixture(t, 1)
	policy := dbPolicy
	policy.RetryDelay = 30 * time.Second
	queue := f.queue(policy)
	ctx := t.Context()
	f.admit(t, queue, 1)
	c := claim(t, queue)
	id := c.Job.ID

	before := f.until(t, "lease_until", id)
	if err := queue.RenewLease(ctx, c); err != nil {
		t.Fatal(err)
	}
	if after := f.until(t, "lease_until", id); after < before-time.Second || after > time.Minute {
		t.Errorf("lease left after renewal %s, before %s", after, before)
	}
	f.expireLease(t, id)
	if err := queue.RenewLease(ctx, c); !errors.Is(err, forecasts_service.ErrLeaseLost) {
		t.Fatalf("renew an expired lease: %v", err)
	}
	if left := f.until(t, "lease_until", id); left >= 0 {
		t.Fatalf("the expired lease was revived: %s left", left)
	}

	recovered, exhausted, err := queue.Recover(ctx)
	if err != nil || recovered != 1 || exhausted != 0 {
		t.Fatalf("Recover = %d, %d, %v; want 1, 0", recovered, exhausted, err)
	}
	job := f.job(t, id)
	if job.Status != "queued" || job.AttemptCount != 1 || job.LeaseUntil.Valid || job.FinishedAt.Valid ||
		job.LastErrorCode.String != "lease_expired" {
		t.Fatalf("recovered job: %s attempt %d, lease %t, finished %t, error %q", job.Status, job.AttemptCount,
			job.LeaseUntil.Valid, job.FinishedAt.Valid, job.LastErrorCode.String)
	}
	if due := f.until(t, "run_after", id); due <= 25*time.Second || due > 30*time.Second {
		t.Errorf("recovered job is due in %s, want the base delay of 30s", due)
	}
	requireNoClaim(t, queue)
	f.makeDue(t, id)
	if c := claim(t, queue); c.Job.ID != id || c.Job.AttemptCount != 2 {
		t.Errorf("claim after recovery: %s attempt %d, want %s attempt 2", c.Job.ID, c.Job.AttemptCount, id)
	}
}

func TestQueueRejectsAStaleAttempt(t *testing.T) {
	f := newDBFixture(t, 1)
	queue := f.queue(dbPolicy)
	ctx := t.Context()
	f.admit(t, queue, 1)
	first := claim(t, queue)
	f.expireLease(t, first.Job.ID)

	// Before recovery the expired lease alone refuses the publication.
	if err := queue.Publish(ctx, first, horizon()); !errors.Is(err, forecasts_service.ErrLeaseLost) {
		t.Fatalf("publish with an expired lease: %v", err)
	}
	if recovered, _, err := queue.Recover(ctx); err != nil || recovered != 1 {
		t.Fatalf("Recover = %d, %v", recovered, err)
	}
	second := claim(t, queue)
	if second.Job.ID != first.Job.ID || second.Job.AttemptCount != 2 {
		t.Fatalf("second claim: %s attempt %d", second.Job.ID, second.Job.AttemptCount)
	}

	// Attempt 1 may write nothing while attempt 2 owns the job.
	if err := queue.Publish(ctx, first, horizon()); !errors.Is(err, forecasts_service.ErrLeaseLost) {
		t.Errorf("publish as attempt 1: %v", err)
	}
	if err := queue.RenewLease(ctx, first); !errors.Is(err, forecasts_service.ErrLeaseLost) {
		t.Errorf("renew as attempt 1: %v", err)
	}
	if _, err := queue.FailAttempt(ctx, first, "ml_internal", false, errors.New("late failure")); !errors.Is(err, forecasts_service.ErrLeaseLost) {
		t.Errorf("fail as attempt 1: %v", err)
	}
	if n := f.predictions(t, 1); n != 0 {
		t.Fatalf("attempt 1 stored %d predictions", n)
	}
	job := f.job(t, first.Job.ID)
	if job.Status != "running" || job.AttemptCount != 2 || job.LastErrorCode.String != "lease_expired" {
		t.Fatalf("job after stale writes: %s attempt %d, error %q", job.Status, job.AttemptCount, job.LastErrorCode.String)
	}

	if err := queue.Publish(ctx, second, horizon()); err != nil {
		t.Fatal(err)
	}
	if job := f.job(t, first.Job.ID); job.Status != "succeeded" || job.AttemptCount != 2 || f.predictions(t, 1) != horizonHours {
		t.Errorf("after attempt 2 published: %s attempt %d, %d predictions", job.Status, job.AttemptCount, f.predictions(t, 1))
	}
}

func TestQueuePublishesTheHorizonAtomically(t *testing.T) {
	f := newDBFixture(t, 1)
	queue := f.queue(dbPolicy)
	ctx := t.Context()
	f.admit(t, queue, 1)
	c := claim(t, queue)

	// One statement reads the status and the row count from one snapshot while
	// the publication runs: it may see the job before or after, never between.
	type observation struct {
		status string
		rows   int64
	}
	seen := map[observation]int{}
	var observeErr error
	stop := make(chan struct{})
	var observer sync.WaitGroup
	observer.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			var o observation
			if err := f.pool.QueryRow(ctx, `SELECT j.status, (SELECT count(*) FROM validation_predictions p
				WHERE p.forecast_version_id = j.forecast_version_id AND p.route_id = j.route_id)
				FROM prediction_jobs j WHERE j.id = $1`, c.Job.ID).Scan(&o.status, &o.rows); err != nil {
				observeErr = err
				return
			}
			seen[o]++
		}
	})
	err := queue.Publish(ctx, c, horizon())
	close(stop)
	observer.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if observeErr != nil {
		t.Fatal(observeErr)
	}
	for o, n := range seen {
		if o != (observation{"running", 0}) && o != (observation{"succeeded", horizonHours}) {
			t.Errorf("seen %d times: job %s with %d predictions", n, o.status, o.rows)
		}
	}
	t.Logf("observations during the publication: %v", seen)

	job := f.job(t, c.Job.ID)
	if job.Status != "succeeded" || job.AttemptCount != 1 || !job.FinishedAt.Valid || job.LeaseUntil.Valid ||
		job.LastErrorCode.Valid || job.LastErrorMessage.Valid {
		t.Errorf("published job: %s attempt %d, finished %t, lease %t, error %q", job.Status, job.AttemptCount,
			job.FinishedAt.Valid, job.LeaseUntil.Valid, job.LastErrorCode.String)
	}
	rows, err := forecasts_sqlc.New(f.pool).ListValidationPredictions(ctx, forecasts_sqlc.ListValidationPredictionsParams{
		ForecastVersionID: f.version.ID, RouteID: f.routes[1],
		DateFrom: pgtype.Date{Time: time.Date(2025, 11, 1, 0, 0, 0, 0, time.UTC), Valid: true},
		DateTo:   pgtype.Date{Time: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), Valid: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != horizonHours {
		t.Fatalf("%d predictions over the horizon, want %d", len(rows), horizonHours)
	}
	// The first row is 2025-11-01 hour 0, the last 2025-12-31 hour 23.
	for i, row := range rows {
		want := horizonFrom.Add(time.Duration(i) * time.Hour)
		if row.Date.Time.Format(time.DateOnly) != want.Format(time.DateOnly) || int(row.Hour) != want.Hour() || row.Boardings != int64(i) {
			t.Fatalf("row %d: %s hour %d boardings %d, want %s hour %d boardings %d", i,
				row.Date.Time.Format(time.DateOnly), row.Hour, row.Boardings, want.Format(time.DateOnly), want.Hour(), i)
		}
	}
	// created_at defaults to the transaction start: one value, one transaction.
	if n := f.count(t, "SELECT count(DISTINCT created_at) FROM validation_predictions"); n != 1 {
		t.Errorf("predictions carry %d creation times, want 1", n)
	}
	if err := queue.Publish(ctx, c, horizon()); !errors.Is(err, forecasts_service.ErrLeaseLost) {
		t.Errorf("publish the finished attempt again: %v", err)
	}
	if n := f.predictions(t, 1); n != horizonHours {
		t.Errorf("%d predictions after the second publication, want %d", n, horizonHours)
	}
}

func TestQueuePublishFailureRollsBack(t *testing.T) {
	f := newDBFixture(t, 1)
	queue := f.queue(dbPolicy)
	ctx := t.Context()
	f.admit(t, queue, 1)
	c := claim(t, queue)
	requireOwned := func(after string) {
		t.Helper()
		job := f.job(t, c.Job.ID)
		if n := f.predictions(t, 1); n != 0 || job.Status != "running" || job.AttemptCount != 1 || f.until(t, "lease_until", c.Job.ID) <= 0 {
			t.Fatalf("after %s: %d predictions, job %s at attempt %d", after, n, job.Status, job.AttemptCount)
		}
	}

	duplicate := horizon()
	duplicate[1].HourStart = duplicate[0].HourStart
	if err := queue.Publish(ctx, c, duplicate); !errors.Is(err, forecasts_service.ErrPublicationConflict) {
		t.Fatalf("publish a duplicate hour: %v", err)
	}
	requireOwned("a duplicate hour")

	negative := horizon()
	negative[100].Boardings = -1
	if err := queue.Publish(ctx, c, negative); !errors.Is(err, forecasts_service.ErrPublicationConflict) {
		t.Fatalf("publish negative boardings: %v", err)
	}
	requireOwned("negative boardings")

	// A deferred trigger refuses the commit; Publish reads the job back and
	// finds it still owned, so the attempt may record the failure.
	f.exec(t, `
		CREATE FUNCTION refuse_commit() RETURNS trigger LANGUAGE plpgsql AS $$
		BEGIN
			RAISE EXCEPTION 'commit refused by the test';
		END
		$$;
		CREATE CONSTRAINT TRIGGER refuse_success AFTER UPDATE ON prediction_jobs
			DEFERRABLE INITIALLY DEFERRED
			FOR EACH ROW WHEN (NEW.status = 'succeeded')
			EXECUTE FUNCTION refuse_commit();`)
	err := queue.Publish(ctx, c, horizon())
	if err == nil || errors.Is(err, forecasts_service.ErrLeaseLost) || errors.Is(err, forecasts_service.ErrPublicationConflict) {
		t.Fatalf("publish with a refused commit: %v", err)
	}
	requireOwned("a refused commit")
	f.exec(t, "DROP TRIGGER refuse_success ON prediction_jobs")

	// Nothing of the failed batches is left to conflict with.
	if err := queue.Publish(ctx, c, horizon()); err != nil {
		t.Fatal(err)
	}
	if n := f.predictions(t, 1); n != horizonHours {
		t.Errorf("%d predictions, want %d", n, horizonHours)
	}
}

func TestQueueFailAttemptRetriesThenFails(t *testing.T) {
	f := newDBFixture(t, 1, 7)
	policy := dbPolicy
	policy.RetryDelay = 30 * time.Second
	queue := f.queue(policy)
	ctx := t.Context()

	retried := f.admit(t, queue, 1)
	unavailable := errors.New("ML server unavailable")
	// Attempts 1 and 2 wait 30 s and 2 min; attempt 3 is the last.
	for _, step := range []struct {
		attempt int32
		status  string
		delay   time.Duration
	}{{1, "queued", 30 * time.Second}, {2, "queued", 2 * time.Minute}, {3, "failed", 0}} {
		c := claim(t, queue)
		if c.Job.ID != retried.ID || c.Job.AttemptCount != step.attempt {
			t.Fatalf("claimed %s attempt %d, want %s attempt %d", c.Job.ID, c.Job.AttemptCount, retried.ID, step.attempt)
		}
		stored, err := queue.FailAttempt(ctx, c, "ml_unavailable", true, unavailable)
		if err != nil || stored.Status != step.status {
			t.Fatalf("attempt %d: FailAttempt = %s, %v; want %s", step.attempt, stored.Status, err, step.status)
		}
		job := f.job(t, retried.ID)
		if job.Status != step.status || job.AttemptCount != step.attempt || job.LeaseUntil.Valid || !job.StartedAt.Valid ||
			job.FinishedAt.Valid != (step.status == "failed") ||
			job.LastErrorCode.String != "ml_unavailable" || job.LastErrorMessage.String != unavailable.Error() {
			t.Fatalf("attempt %d stored: %s attempt %d, lease %t, finished %t, error %q %q", step.attempt, job.Status,
				job.AttemptCount, job.LeaseUntil.Valid, job.FinishedAt.Valid, job.LastErrorCode.String, job.LastErrorMessage.String)
		}
		if _, err := queue.FailAttempt(ctx, c, "ml_unavailable", true, unavailable); !errors.Is(err, forecasts_service.ErrLeaseLost) {
			t.Fatalf("attempt %d failed twice: %v", step.attempt, err)
		}
		if step.status == "queued" {
			if due := f.until(t, "run_after", retried.ID); due <= step.delay-5*time.Second || due > step.delay {
				t.Errorf("attempt %d: due in %s, want %s", step.attempt, due, step.delay)
			}
			requireNoClaim(t, queue)
			f.makeDue(t, retried.ID)
		}
	}
	// Admission returns the failed job instead of restarting it.
	if job, created, err := queue.Admit(ctx, f.version.ID, f.routes[1]); err != nil || created || job.Status != "failed" {
		t.Errorf("admit a failed pair: %s created %t, %v", job.Status, created, err)
	}

	terminal := f.admit(t, queue, 7)
	c := claim(t, queue)
	// PostgreSQL text rejects NUL; the message is also bounded.
	cause := errors.New("bad\x00" + strings.Repeat("я", 1500))
	if stored, err := queue.FailAttempt(ctx, c, "ml_internal", false, cause); err != nil || stored.Status != "failed" {
		t.Fatalf("terminal FailAttempt = %s, %v", stored.Status, err)
	}
	job := f.job(t, terminal.ID)
	message := job.LastErrorMessage.String
	if job.Status != "failed" || job.AttemptCount != 1 || !job.FinishedAt.Valid || job.LastErrorCode.String != "ml_internal" ||
		strings.ContainsRune(message, 0) || utf8.RuneCountInString(message) != 1000 || !strings.HasPrefix(message, "badя") {
		t.Errorf("terminal failure stored: %s attempt %d, finished %t, error %q, message of %d runes", job.Status,
			job.AttemptCount, job.FinishedAt.Valid, job.LastErrorCode.String, utf8.RuneCountInString(message))
	}
}

func TestQueueFailsExhaustedJobs(t *testing.T) {
	f := newDBFixture(t, 1, 7)
	ctx := t.Context()
	generous := dbPolicy
	generous.RetryDelay = time.Hour
	strict := dbPolicy
	strict.MaxAttempts = 1
	queue3, queue1 := f.queue(generous), f.queue(strict)

	waiting := f.admit(t, queue3, 1)
	if stored, err := queue3.FailAttempt(ctx, claim(t, queue3), "ml_unavailable", true, errors.New("down")); err != nil || stored.Status != "queued" {
		t.Fatalf("FailAttempt = %s, %v", stored.Status, err)
	}
	expired := f.admit(t, queue1, 7)
	if c := claim(t, queue1); c.Job.ID != expired.ID {
		t.Fatalf("claimed %s, want %s", c.Job.ID, expired.ID)
	}
	f.expireLease(t, expired.ID)

	// With the limit lowered to one attempt, neither job has one left.
	f.makeDue(t, waiting.ID)
	requireNoClaim(t, queue1)
	recovered, exhausted, err := queue1.Recover(ctx)
	if err != nil || recovered != 1 || exhausted != 1 {
		t.Fatalf("Recover = %d, %d, %v; want 1, 1", recovered, exhausted, err)
	}
	for id, code := range map[pgtype.UUID]string{waiting.ID: "attempt_limit_exceeded", expired.ID: "lease_expired"} {
		job := f.job(t, id)
		if job.Status != "failed" || job.AttemptCount != 1 || !job.FinishedAt.Valid || job.LeaseUntil.Valid || job.LastErrorCode.String != code {
			t.Errorf("job %s: %s attempt %d, finished %t, error %q; want failed with %s", id, job.Status, job.AttemptCount,
				job.FinishedAt.Valid, job.LastErrorCode.String, code)
		}
	}
	if recovered, exhausted, err := queue1.Recover(ctx); err != nil || recovered != 0 || exhausted != 0 {
		t.Errorf("second Recover = %d, %d, %v", recovered, exhausted, err)
	}
}

func TestQueueRecoverDrainsEveryBatch(t *testing.T) {
	f := newDBFixture(t)
	queue := f.queue(dbPolicy)
	f.exec(t, `INSERT INTO routes (route_number, forecast_enabled) SELECT n, true FROM generate_series(1, 270) AS n`)
	// 150 expired leases and 120 queued jobs without attempts left: two and two
	// batches of at most 100.
	f.exec(t, `INSERT INTO prediction_jobs (id, forecast_version_id, route_id, status, attempt_count, started_at, lease_until)
		SELECT gen_random_uuid(), $1, id, 'running', 1, clock_timestamp(), clock_timestamp() - interval '1 second'
		FROM routes WHERE route_number <= 150`, f.version.ID)
	f.exec(t, `INSERT INTO prediction_jobs (id, forecast_version_id, route_id, attempt_count)
		SELECT gen_random_uuid(), $1, id, 3 FROM routes WHERE route_number > 150`, f.version.ID)

	recovered, exhausted, err := queue.Recover(t.Context())
	if err != nil || recovered != 150 || exhausted != 120 {
		t.Fatalf("Recover = %d, %d, %v; want 150, 120", recovered, exhausted, err)
	}
	for _, want := range []struct {
		status, code string
		n            int64
	}{{"queued", "lease_expired", 150}, {"failed", "attempt_limit_exceeded", 120}} {
		if n := f.count(t, "SELECT count(*) FROM prediction_jobs WHERE status = $1 AND last_error_code = $2", want.status, want.code); n != want.n {
			t.Errorf("%d jobs %s with %s, want %d", n, want.status, want.code, want.n)
		}
	}
}

func TestQueueReadsCatalogAndVersions(t *testing.T) {
	f := newDBFixture(t, 7, 1)
	disabled := addRoute(t, f.pool, 99, false)
	queue := f.queue(dbPolicy)
	ctx := t.Context()

	routes, err := queue.ForecastRoutes(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var numbers []int16
	for _, route := range routes {
		numbers = append(numbers, route.RouteNumber)
	}
	if !slices.Equal(numbers, []int16{1, 7}) {
		t.Errorf("ForecastRoutes = %v, want [1 7]", numbers)
	}

	if _, err := queue.ActiveVersion(ctx); !errors.Is(err, forecasts_service.ErrNoActiveVersion) {
		t.Errorf("ActiveVersion without one: %v", err)
	}
	if _, err := forecasts_sqlc.New(f.pool).ActivateForecastVersion(ctx, f.version.ID); err != nil {
		t.Fatal(err)
	}
	if active, err := queue.ActiveVersion(ctx); err != nil || active.ID != f.version.ID {
		t.Errorf("ActiveVersion = %s, %v", active.ID, err)
	}
	unknown := newID()
	if _, err := queue.Version(ctx, unknown); !errors.Is(err, forecasts_service.ErrVersionNotFound) {
		t.Errorf("Version of an unknown ID: %v", err)
	}
	if version, err := queue.Version(ctx, f.version.ID); err != nil || version.ModelVersion != "model-it" {
		t.Errorf("Version = %q, %v", version.ModelVersion, err)
	}

	for name, tc := range map[string]struct {
		version pgtype.UUID
		route   int64
		want    error
	}{
		"unknown version": {unknown, f.routes[1], forecasts_service.ErrVersionNotFound},
		"unknown route":   {f.version.ID, disabled + 1000, forecasts_service.ErrRouteNotFound},
		"disabled route":  {f.version.ID, disabled, forecasts_service.ErrForecastDisabled},
	} {
		if _, created, err := queue.Admit(ctx, tc.version, tc.route); created || !errors.Is(err, tc.want) {
			t.Errorf("%s: created %t, %v", name, created, err)
		}
	}
	if n := f.count(t, "SELECT count(*) FROM prediction_jobs"); n != 0 {
		t.Errorf("rejected admissions stored %d jobs", n)
	}
	if _, found, err := queue.Lookup(ctx, f.version.ID, f.routes[1]); err != nil || found {
		t.Errorf("Lookup of a pair without a job: found %t, %v", found, err)
	}
}
