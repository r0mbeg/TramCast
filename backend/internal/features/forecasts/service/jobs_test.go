package forecasts_service

import (
	"context"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
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

// queueNow is the database clock of fakeQueue.
var queueNow = time.Date(2026, 9, 27, 12, 0, 0, 0, time.UTC)

var testPolicy = JobPolicy{QueueCapacity: 20, MaxAttempts: 3, LeaseDuration: time.Minute, RetryDelay: 30 * time.Second}

// fakeQueue mirrors the SQL semantics of the queue queries over rows in
// memory: the unique version and route pair, ownership by attempt and lease,
// retries and recovery. It logs every call together with the transaction
// boundaries of fakeQueueDB.
type fakeQueue struct {
	versions    []forecasts_sqlc.ForecastVersion
	routes      []routes_sqlc.Route
	jobs        []forecasts_sqlc.PredictionJob
	predictions []forecasts_sqlc.CopyValidationPredictionsParams
	calls       []string
	// fail makes the named call return the error; before runs first, for
	// example to act as a concurrent writer.
	fail   map[string]error
	before map[string]func()
	// shortCopy reports one copied row less; noComplete changes no row.
	shortCopy, noComplete bool
	// readOnly is set inside a READ ONLY transaction, where writes fail.
	readOnly bool

	claimArgs   []forecasts_sqlc.ClaimNextPredictionJobParams
	renewArgs   []forecasts_sqlc.RenewPredictionJobLeaseParams
	finishArgs  []forecasts_sqlc.FinishPredictionJobWithErrorParams
	recoverArgs []forecasts_sqlc.RecoverExpiredPredictionJobsParams
	exhaustArgs []forecasts_sqlc.FailExhaustedQueuedPredictionJobsParams
	listArgs    []forecasts_sqlc.ListValidationPredictionsParams
}

func (s *fakeQueue) call(name string) error {
	s.calls = append(s.calls, name)
	if hook := s.before[name]; hook != nil {
		hook()
	}
	return s.fail[name]
}

// write is call for statements PostgreSQL refuses in a READ ONLY transaction.
func (s *fakeQueue) write(name string) error {
	if err := s.call(name); err != nil {
		return err
	}
	if s.readOnly {
		return errors.New(name + " in a read-only transaction")
	}
	return nil
}

func (s *fakeQueue) job(id pgtype.UUID) *forecasts_sqlc.PredictionJob {
	for i := range s.jobs {
		if s.jobs[i].ID == id {
			return &s.jobs[i]
		}
	}
	return nil
}

// owns is the ownership rule of renewal, publication and failure.
func owns(job *forecasts_sqlc.PredictionJob, attempt int32) bool {
	return job != nil && job.Status == "running" && job.AttemptCount == attempt &&
		job.LeaseUntil.Valid && job.LeaseUntil.Time.After(queueNow)
}

func seconds(n int32) time.Duration { return time.Duration(n) * time.Second }

func text(s string) pgtype.Text { return pgtype.Text{String: s, Valid: true} }

func (s *fakeQueue) GetForecastVersion(_ context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error) {
	if err := s.call("get version"); err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	for _, version := range s.versions {
		if version.ID == id {
			return version, nil
		}
	}
	return forecasts_sqlc.ForecastVersion{}, pgx.ErrNoRows
}

func (s *fakeQueue) GetActiveForecastVersion(context.Context) (forecasts_sqlc.ForecastVersion, error) {
	if err := s.call("get active version"); err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	for _, version := range s.versions {
		if version.IsActive {
			return version, nil
		}
	}
	return forecasts_sqlc.ForecastVersion{}, pgx.ErrNoRows
}

func (s *fakeQueue) LockPredictionJobAdmission(context.Context) error {
	return s.write("lock admission")
}

func (s *fakeQueue) GetPredictionJob(_ context.Context, id pgtype.UUID) (forecasts_sqlc.PredictionJob, error) {
	if err := s.call("get job"); err != nil {
		return forecasts_sqlc.PredictionJob{}, err
	}
	if job := s.job(id); job != nil {
		return *job, nil
	}
	return forecasts_sqlc.PredictionJob{}, pgx.ErrNoRows
}

func (s *fakeQueue) GetPredictionJobByRouteAndVersion(_ context.Context, arg forecasts_sqlc.GetPredictionJobByRouteAndVersionParams) (forecasts_sqlc.PredictionJob, error) {
	if err := s.call("find job"); err != nil {
		return forecasts_sqlc.PredictionJob{}, err
	}
	for _, job := range s.jobs {
		if job.ForecastVersionID == arg.ForecastVersionID && job.RouteID == arg.RouteID {
			return job, nil
		}
	}
	return forecasts_sqlc.PredictionJob{}, pgx.ErrNoRows
}

func (s *fakeQueue) CountPendingPredictionJobs(context.Context) (int64, error) {
	if err := s.call("count pending"); err != nil {
		return 0, err
	}
	var pending int64
	for _, job := range s.jobs {
		if job.Status == "queued" || job.Status == "running" {
			pending++
		}
	}
	return pending, nil
}

func (s *fakeQueue) CreatePredictionJob(_ context.Context, arg forecasts_sqlc.CreatePredictionJobParams) (forecasts_sqlc.PredictionJob, error) {
	if err := s.write("create job"); err != nil {
		return forecasts_sqlc.PredictionJob{}, err
	}
	for _, job := range s.jobs {
		if job.ForecastVersionID == arg.ForecastVersionID && job.RouteID == arg.RouteID {
			return forecasts_sqlc.PredictionJob{}, pgx.ErrNoRows // ON CONFLICT DO NOTHING
		}
	}
	job := forecasts_sqlc.PredictionJob{
		ID: arg.ID, ForecastVersionID: arg.ForecastVersionID, RouteID: arg.RouteID, Status: "queued",
		RunAfter: timestamptz(queueNow), CreatedAt: timestamptz(queueNow),
	}
	s.jobs = append(s.jobs, job)
	return job, nil
}

func (s *fakeQueue) ClaimNextPredictionJob(_ context.Context, arg forecasts_sqlc.ClaimNextPredictionJobParams) (forecasts_sqlc.PredictionJob, error) {
	if err := s.write("claim"); err != nil {
		return forecasts_sqlc.PredictionJob{}, err
	}
	s.claimArgs = append(s.claimArgs, arg)
	if arg.LeaseSeconds <= 0 {
		return forecasts_sqlc.PredictionJob{}, pgx.ErrNoRows
	}
	for i := range s.jobs {
		job := &s.jobs[i]
		if job.Status != "queued" || job.RunAfter.Time.After(queueNow) || job.AttemptCount >= arg.MaxAttempts {
			continue
		}
		job.Status, job.AttemptCount = "running", job.AttemptCount+1
		job.StartedAt = timestamptz(queueNow)
		job.LeaseUntil = timestamptz(queueNow.Add(seconds(arg.LeaseSeconds)))
		job.FinishedAt = pgtype.Timestamptz{}
		return *job, nil
	}
	return forecasts_sqlc.PredictionJob{}, pgx.ErrNoRows
}

func (s *fakeQueue) RenewPredictionJobLease(_ context.Context, arg forecasts_sqlc.RenewPredictionJobLeaseParams) (int64, error) {
	if err := s.write("renew"); err != nil {
		return 0, err
	}
	s.renewArgs = append(s.renewArgs, arg)
	job := s.job(arg.ID)
	if !owns(job, arg.AttemptNumber) || arg.LeaseSeconds <= 0 {
		return 0, nil
	}
	if next := queueNow.Add(seconds(arg.LeaseSeconds)); next.After(job.LeaseUntil.Time) {
		job.LeaseUntil = timestamptz(next)
	}
	return 1, nil
}

func (s *fakeQueue) LockPredictionJobForPublication(_ context.Context, arg forecasts_sqlc.LockPredictionJobForPublicationParams) (forecasts_sqlc.LockPredictionJobForPublicationRow, error) {
	if err := s.write("lock publication"); err != nil {
		return forecasts_sqlc.LockPredictionJobForPublicationRow{}, err
	}
	job := s.job(arg.ID)
	if !owns(job, arg.AttemptNumber) {
		return forecasts_sqlc.LockPredictionJobForPublicationRow{}, pgx.ErrNoRows
	}
	return forecasts_sqlc.LockPredictionJobForPublicationRow(*job), nil
}

type predictionKey struct {
	version pgtype.UUID
	route   int64
	date    time.Time
	hour    int16
}

func keyOf(row forecasts_sqlc.CopyValidationPredictionsParams) predictionKey {
	return predictionKey{row.ForecastVersionID, row.RouteID, row.Date.Time, row.Hour}
}

// CopyValidationPredictions applies the primary key and the boardings check,
// all or nothing like COPY.
func (s *fakeQueue) CopyValidationPredictions(_ context.Context, rows []forecasts_sqlc.CopyValidationPredictionsParams) (int64, error) {
	if err := s.write("copy"); err != nil {
		return 0, err
	}
	seen := map[predictionKey]bool{}
	for _, row := range s.predictions {
		seen[keyOf(row)] = true
	}
	for _, row := range rows {
		switch {
		case row.Boardings < 0:
			return 0, &pgconn.PgError{Code: "23514", Message: "violates check constraint"}
		case seen[keyOf(row)]:
			return 0, &pgconn.PgError{Code: "23505", Message: "duplicate key value violates unique constraint"}
		}
		seen[keyOf(row)] = true
	}
	s.predictions = append(s.predictions, rows...)
	copied := int64(len(rows))
	if s.shortCopy {
		copied--
	}
	return copied, nil
}

func (s *fakeQueue) CompletePredictionJob(_ context.Context, arg forecasts_sqlc.CompletePredictionJobParams) (int64, error) {
	if err := s.write("complete"); err != nil {
		return 0, err
	}
	job := s.job(arg.ID)
	if s.noComplete || job == nil || job.Status != "running" || job.AttemptCount != arg.AttemptNumber {
		return 0, nil
	}
	job.Status, job.FinishedAt, job.LeaseUntil = "succeeded", timestamptz(queueNow), pgtype.Timestamptz{}
	job.LastErrorCode, job.LastErrorMessage = pgtype.Text{}, pgtype.Text{}
	return 1, nil
}

func (s *fakeQueue) FinishPredictionJobWithError(_ context.Context, arg forecasts_sqlc.FinishPredictionJobWithErrorParams) (forecasts_sqlc.PredictionJob, error) {
	if err := s.write("finish"); err != nil {
		return forecasts_sqlc.PredictionJob{}, err
	}
	s.finishArgs = append(s.finishArgs, arg)
	job := s.job(arg.ID)
	if !owns(job, arg.AttemptNumber) || arg.MaxAttempts <= 0 || arg.RetryDelaySeconds < 0 {
		return forecasts_sqlc.PredictionJob{}, pgx.ErrNoRows
	}
	if arg.Retryable && job.AttemptCount < arg.MaxAttempts {
		job.Status, job.FinishedAt = "queued", pgtype.Timestamptz{}
		job.RunAfter = timestamptz(queueNow.Add(seconds(arg.RetryDelaySeconds)))
	} else {
		job.Status, job.FinishedAt = "failed", timestamptz(queueNow)
	}
	job.LeaseUntil = pgtype.Timestamptz{}
	job.LastErrorCode, job.LastErrorMessage = text(arg.ErrorCode), text(arg.ErrorMessage)
	return *job, nil
}

func (s *fakeQueue) RecoverExpiredPredictionJobs(_ context.Context, arg forecasts_sqlc.RecoverExpiredPredictionJobsParams) ([]forecasts_sqlc.RecoverExpiredPredictionJobsRow, error) {
	if err := s.write("recover"); err != nil {
		return nil, err
	}
	s.recoverArgs = append(s.recoverArgs, arg)
	rows := []forecasts_sqlc.RecoverExpiredPredictionJobsRow{}
	if arg.MaxAttempts <= 0 || arg.RetryDelaySeconds < 0 {
		return rows, nil
	}
	for i := range s.jobs {
		job := &s.jobs[i]
		if int32(len(rows)) == arg.BatchSize {
			break
		}
		if job.Status != "running" || job.LeaseUntil.Time.After(queueNow) {
			continue
		}
		if job.AttemptCount < arg.MaxAttempts {
			job.Status, job.FinishedAt = "queued", pgtype.Timestamptz{}
			job.RunAfter = timestamptz(queueNow.Add(seconds(arg.RetryDelaySeconds)))
		} else {
			job.Status, job.FinishedAt = "failed", timestamptz(queueNow)
		}
		job.LeaseUntil = pgtype.Timestamptz{}
		job.LastErrorCode, job.LastErrorMessage = text("lease_expired"), text("The worker lease expired before completion")
		rows = append(rows, forecasts_sqlc.RecoverExpiredPredictionJobsRow{ID: job.ID, Status: job.Status, AttemptCount: job.AttemptCount})
	}
	return rows, nil
}

func (s *fakeQueue) FailExhaustedQueuedPredictionJobs(_ context.Context, arg forecasts_sqlc.FailExhaustedQueuedPredictionJobsParams) ([]forecasts_sqlc.FailExhaustedQueuedPredictionJobsRow, error) {
	if err := s.write("fail exhausted"); err != nil {
		return nil, err
	}
	s.exhaustArgs = append(s.exhaustArgs, arg)
	rows := []forecasts_sqlc.FailExhaustedQueuedPredictionJobsRow{}
	for i := range s.jobs {
		job := &s.jobs[i]
		if int32(len(rows)) == arg.BatchSize {
			break
		}
		if job.Status != "queued" || job.AttemptCount < arg.MaxAttempts || arg.MaxAttempts <= 0 {
			continue
		}
		job.Status, job.FinishedAt, job.LeaseUntil = "failed", timestamptz(queueNow), pgtype.Timestamptz{}
		job.LastErrorCode = text("attempt_limit_exceeded")
		job.LastErrorMessage = text("The configured attempt limit has been reached")
		rows = append(rows, forecasts_sqlc.FailExhaustedQueuedPredictionJobsRow{ID: job.ID, Status: job.Status, AttemptCount: job.AttemptCount})
	}
	return rows, nil
}

func (s *fakeQueue) GetRouteByID(_ context.Context, id int64) (routes_sqlc.Route, error) {
	if err := s.call("get route"); err != nil {
		return routes_sqlc.Route{}, err
	}
	for _, route := range s.routes {
		if route.ID == id {
			return route, nil
		}
	}
	return routes_sqlc.Route{}, pgx.ErrNoRows
}

func (s *fakeQueue) ListRoutes(context.Context) ([]routes_sqlc.Route, error) {
	if err := s.call("list routes"); err != nil {
		return nil, err
	}
	return slices.Clone(s.routes), nil
}

// fakeQueueDB starts transactions over the store. A rollback, or a commit
// that does not apply, restores the rows seen at BEGIN; either ends the
// transaction, as in pgx.
type fakeQueueDB struct {
	t        *testing.T
	store    *fakeQueue
	tx       *fakeQueueTx
	beginErr error
	// commit, when set, decides every commit: whether it applies and what it
	// returns. An applied commit with an error is an unknown outcome.
	commit func(ctx context.Context) (apply bool, err error)
}

func (db *fakeQueueDB) BeginTx(ctx context.Context, options pgx.TxOptions) (pgx.Tx, error) {
	if options.IsoLevel != pgx.ReadCommitted {
		db.t.Errorf("isolation = %s, want read committed", options.IsoLevel)
	}
	// pgx cannot start a transaction on a cancelled context.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if db.beginErr != nil {
		return nil, db.beginErr
	}
	readOnly := options.AccessMode == pgx.ReadOnly
	if readOnly {
		db.store.calls = append(db.store.calls, "begin read-only")
	} else {
		db.store.calls = append(db.store.calls, "begin")
	}
	db.store.readOnly = readOnly
	db.tx = &fakeQueueTx{db: db, jobs: slices.Clone(db.store.jobs), predictions: slices.Clone(db.store.predictions)}
	return db.tx, nil
}

type fakeQueueTx struct {
	pgx.Tx
	db          *fakeQueueDB
	jobs        []forecasts_sqlc.PredictionJob
	predictions []forecasts_sqlc.CopyValidationPredictionsParams
	closed      bool
}

func (tx *fakeQueueTx) Commit(ctx context.Context) error {
	store := tx.db.store
	store.calls = append(store.calls, "commit")
	apply, err := store.fail["commit"] == nil, store.fail["commit"]
	if tx.db.commit != nil {
		apply, err = tx.db.commit(ctx)
	}
	if !apply {
		tx.restore()
	}
	tx.closed, store.readOnly = true, false
	return err
}

func (tx *fakeQueueTx) Rollback(ctx context.Context) error {
	if ctx.Err() != nil {
		tx.db.t.Error("rollback must remain possible after caller cancellation")
	}
	if tx.closed {
		return pgx.ErrTxClosed
	}
	tx.db.store.calls = append(tx.db.store.calls, "rollback")
	tx.restore()
	tx.closed, tx.db.store.readOnly = true, false
	return nil
}

func (tx *fakeQueueTx) restore() {
	tx.db.store.jobs, tx.db.store.predictions = tx.jobs, tx.predictions
}

// newTestQueue binds the queue to the store and checks that every query runs
// in the transaction just started.
func newTestQueue(t *testing.T, store *fakeQueue, policy JobPolicy) (*Queue, *fakeQueueDB) {
	t.Helper()
	db := &fakeQueueDB{t: t, store: store}
	queue := NewQueue(db, policy)
	queue.queries = func(tx pgx.Tx) queueStore {
		if tx != db.tx {
			t.Error("queries must use the current transaction")
		}
		return queueStore{jobs: store, routes: store}
	}
	return queue, db
}

// queueFixture has the active contest version (ID 1), the ten target routes
// with forecasts (IDs 100-109; 100 is route 1, 101 route 5) and route 3
// without them (ID 200).
func queueFixture() *fakeQueue {
	routes := enabledRoutes(targetRoutes...)
	routes = slices.Insert(routes, 1, routes_sqlc.Route{ID: 200, RouteNumber: 3})
	return &fakeQueue{versions: []forecasts_sqlc.ForecastVersion{storedVersion(1, true, contestSpec())}, routes: routes}
}

func jobID(n int) pgtype.UUID {
	id := pgtype.UUID{Bytes: [16]byte{0xab}, Valid: true}
	id.Bytes[14], id.Bytes[15] = byte(n>>8), byte(n)
	return id
}

// seed adds job n of the active version. A running job owns a lease that is
// still valid; tests expire it explicitly.
func (s *fakeQueue) seed(n int, routeID int64, status string, attempt int32) forecasts_sqlc.PredictionJob {
	job := forecasts_sqlc.PredictionJob{
		ID: jobID(n), ForecastVersionID: s.versions[0].ID, RouteID: routeID, Status: status, AttemptCount: attempt,
		RunAfter: timestamptz(queueNow.Add(-time.Minute)), CreatedAt: timestamptz(queueNow.Add(-time.Hour)),
	}
	switch status {
	case "running":
		job.StartedAt, job.LeaseUntil = timestamptz(queueNow.Add(-time.Minute)), timestamptz(queueNow.Add(30*time.Second))
	case "succeeded", "failed":
		job.FinishedAt = timestamptz(queueNow.Add(-time.Minute))
	}
	s.jobs = append(s.jobs, job)
	return job
}

// claimed is the claim of job as Claim would return it.
func (s *fakeQueue) claimed(job forecasts_sqlc.PredictionJob) ClaimedJob {
	claim := ClaimedJob{Job: job}
	for _, version := range s.versions {
		if version.ID == job.ForecastVersionID {
			claim.Version = version
		}
	}
	for _, route := range s.routes {
		if route.ID == job.RouteID {
			claim.RouteNumber, claim.RouteEnabled = route.RouteNumber, route.ForecastEnabled
		}
	}
	return claim
}

// horizonPoints is a valid response for version as the client returns it:
// every hour in Moscow time, with zeros among the values.
func horizonPoints(version forecasts_sqlc.ForecastVersion) []forecasts_predictor_grpc.Point {
	var points []forecasts_predictor_grpc.Point
	for hour := version.ForecastFrom.Time; hour.Before(version.ForecastTo.Time); hour = hour.Add(time.Hour) {
		points = append(points, forecasts_predictor_grpc.Point{HourStart: hour.In(core_domain.Moscow), Boardings: int64(len(points) % 7)})
	}
	return points
}

func queueCalls(t *testing.T, store *fakeQueue, want ...string) {
	t.Helper()
	if !reflect.DeepEqual(store.calls, want) {
		t.Fatalf("calls = %v, want %v", store.calls, want)
	}
}

// admissionChecks are the reads Admit does before its transaction.
var admissionChecks = []string{"begin read-only", "get version", "get route", "rollback"}

func TestAdmitCreatesJob(t *testing.T) {
	store := queueFixture()
	queue, _ := newTestQueue(t, store, testPolicy)
	version := store.versions[0]
	job, created, err := queue.Admit(context.Background(), version.ID, 100)
	if err != nil || !created {
		t.Fatalf("created = %t, error = %v; want a new job", created, err)
	}
	if !job.ID.Valid || job.ForecastVersionID != version.ID || job.RouteID != 100 || job.Status != "queued" || job.AttemptCount != 0 {
		t.Fatalf("job = %+v, want a queued job of version 1 and route 100", job)
	}
	if len(store.jobs) != 1 || store.jobs[0] != job {
		t.Fatalf("stored jobs = %+v", store.jobs)
	}
	queueCalls(t, store, slices.Concat(admissionChecks,
		[]string{"begin", "lock admission", "find job", "count pending", "create job", "commit"})...)
}

func TestAdmitReturnsExistingJobEvenWhenFull(t *testing.T) {
	for _, status := range []string{"queued", "running", "succeeded", "failed"} {
		t.Run(status, func(t *testing.T) {
			store := queueFixture()
			existing := store.seed(1, 100, status, 1)
			store.seed(2, 101, "queued", 0)
			policy := testPolicy
			policy.QueueCapacity = 1
			queue, _ := newTestQueue(t, store, policy)

			job, created, err := queue.Admit(context.Background(), store.versions[0].ID, 100)
			if err != nil || created || job != existing {
				t.Fatalf("job = %+v, created = %t, error = %v; want the existing job", job, created, err)
			}
			if len(store.jobs) != 2 {
				t.Fatalf("stored jobs = %d, want no new job", len(store.jobs))
			}
			// Neither polling nor admission restarts a failed job.
			queueCalls(t, store, slices.Concat(admissionChecks, []string{"begin", "lock admission", "find job", "rollback"})...)
		})
	}
}

func TestAdmitRejectsNewJobWhenFull(t *testing.T) {
	store := queueFixture()
	store.seed(1, 101, "queued", 0)
	store.seed(2, 102, "running", 1)
	store.seed(3, 103, "succeeded", 1)
	store.seed(4, 104, "failed", 3)
	policy := testPolicy
	policy.QueueCapacity = 2
	queue, _ := newTestQueue(t, store, policy)

	_, created, err := queue.Admit(context.Background(), store.versions[0].ID, 100)
	if !errors.Is(err, ErrQueueFull) || created {
		t.Fatalf("created = %t, error = %v; want ErrQueueFull", created, err)
	}
	if len(store.jobs) != 4 {
		t.Fatalf("stored jobs = %d, want 4", len(store.jobs))
	}
	queueCalls(t, store, slices.Concat(admissionChecks,
		[]string{"begin", "lock admission", "find job", "count pending", "rollback"})...)

	// Finished jobs do not take capacity.
	policy.QueueCapacity = 3
	queue, _ = newTestQueue(t, store, policy)
	if _, created, err := queue.Admit(context.Background(), store.versions[0].ID, 100); err != nil || !created {
		t.Fatalf("created = %t, error = %v; want a new job below capacity", created, err)
	}
}

func TestAdmitConflictReadsTheCommittedJob(t *testing.T) {
	store := queueFixture()
	var concurrent forecasts_sqlc.PredictionJob
	// Another admission commits the pair just before this insert.
	store.before = map[string]func(){"create job": func() { concurrent = store.seed(9, 100, "queued", 0) }}
	queue, _ := newTestQueue(t, store, testPolicy)

	job, created, err := queue.Admit(context.Background(), store.versions[0].ID, 100)
	if err != nil || created || job != concurrent {
		t.Fatalf("job = %+v, created = %t, error = %v; want the concurrent job", job, created, err)
	}
	if len(store.jobs) != 1 {
		t.Fatalf("stored jobs = %d, want one per pair", len(store.jobs))
	}
	queueCalls(t, store, slices.Concat(admissionChecks,
		[]string{"begin", "lock admission", "find job", "count pending", "create job", "find job", "commit"})...)
}

func TestAdmitRejectsUnknownOrDisabledTargets(t *testing.T) {
	for _, tt := range []struct {
		name      string
		version   pgtype.UUID
		routeID   int64
		want      error
		wantCalls []string
	}{
		{"unknown version", jobID(99), 100, ErrVersionNotFound, []string{"begin read-only", "get version", "rollback"}},
		{"unknown route", pgtype.UUID{Bytes: [16]byte{1}, Valid: true}, 999, ErrRouteNotFound, admissionChecks},
		{"forecasts disabled", pgtype.UUID{Bytes: [16]byte{1}, Valid: true}, 200, ErrForecastDisabled, admissionChecks},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			queue, _ := newTestQueue(t, store, testPolicy)
			if _, created, err := queue.Admit(context.Background(), tt.version, tt.routeID); !errors.Is(err, tt.want) || created {
				t.Fatalf("created = %t, error = %v; want %v", created, err, tt.want)
			}
			if len(store.jobs) != 0 {
				t.Fatalf("stored jobs = %+v, want none", store.jobs)
			}
			queueCalls(t, store, tt.wantCalls...)
		})
	}
}

func TestAdmitFailuresRollBack(t *testing.T) {
	failure := errors.New("database unavailable")
	admission := []string{"begin", "lock admission", "find job", "count pending", "create job"}
	for _, tt := range []struct {
		fail      string
		conflict  bool
		wantCalls []string
	}{
		{"get version", false, []string{"begin read-only", "get version", "rollback"}},
		{"get route", false, admissionChecks},
		{"lock admission", false, slices.Concat(admissionChecks, admission[:2], []string{"rollback"})},
		{"find job", false, slices.Concat(admissionChecks, admission[:3], []string{"rollback"})},
		{"count pending", false, slices.Concat(admissionChecks, admission[:4], []string{"rollback"})},
		{"create job", false, slices.Concat(admissionChecks, admission, []string{"rollback"})},
		{"find job", true, slices.Concat(admissionChecks, admission, []string{"find job", "rollback"})},
		// A failed commit ends the transaction.
		{"commit", false, slices.Concat(admissionChecks, admission, []string{"commit"})},
	} {
		t.Run(tt.fail, func(t *testing.T) {
			store := queueFixture()
			if tt.conflict {
				store.before = map[string]func(){"create job": func() {
					store.seed(9, 100, "queued", 0)
					store.fail = map[string]error{"find job": failure}
				}}
			} else {
				store.fail = map[string]error{tt.fail: failure}
			}
			queue, _ := newTestQueue(t, store, testPolicy)
			if _, created, err := queue.Admit(context.Background(), store.versions[0].ID, 100); !errors.Is(err, failure) || created {
				t.Fatalf("created = %t, error = %v; want the failure", created, err)
			}
			queueCalls(t, store, tt.wantCalls...)
			if len(store.jobs) != 0 {
				t.Fatalf("stored jobs = %+v, want the insert undone", store.jobs)
			}
		})
	}

	store := queueFixture()
	queue, db := newTestQueue(t, store, testPolicy)
	db.beginErr = failure
	if _, _, err := queue.Admit(context.Background(), store.versions[0].ID, 100); !errors.Is(err, failure) {
		t.Fatalf("error = %v, want the begin failure", err)
	}
	queueCalls(t, store)
}

func TestClaimTakesJobOfItsPinnedVersion(t *testing.T) {
	store := queueFixture()
	pinned := storedVersion(2, false, contestSpec())
	pinned.ModelVersion = "tabpfn030-previous"
	store.versions = append(store.versions, pinned)
	store.seed(1, 101, "queued", 1)
	store.jobs[0].ForecastVersionID = pinned.ID
	queue, _ := newTestQueue(t, store, testPolicy)

	claimed, found, err := queue.Claim(context.Background())
	if err != nil || !found {
		t.Fatalf("found = %t, error = %v; want a claim", found, err)
	}
	job := claimed.Job
	if job.ID != jobID(1) || job.Status != "running" || job.AttemptCount != 2 || !job.LeaseUntil.Time.Equal(queueNow.Add(time.Minute)) {
		t.Fatalf("job = %+v, want attempt 2 running with a 60 s lease", job)
	}
	if claimed.Version != pinned || claimed.RouteNumber != 5 || !claimed.RouteEnabled {
		t.Fatalf("claim = %+v, want the pinned version and route 5", claimed)
	}
	if store.jobs[0] != job {
		t.Fatalf("stored job = %+v, want the claimed row", store.jobs[0])
	}
	if want := []forecasts_sqlc.ClaimNextPredictionJobParams{{LeaseSeconds: 60, MaxAttempts: 3}}; !reflect.DeepEqual(store.claimArgs, want) {
		t.Fatalf("claim params = %+v, want %+v", store.claimArgs, want)
	}
	queueCalls(t, store, "begin", "claim", "get version", "get route", "commit")
}

func TestClaimFindsNothingDue(t *testing.T) {
	for name, seed := range map[string]func(*fakeQueue){
		"empty queue": func(*fakeQueue) {},
		"attempts exhausted": func(s *fakeQueue) {
			s.seed(1, 100, "queued", 3)
		},
		"retry not due": func(s *fakeQueue) {
			s.seed(1, 100, "queued", 1)
			s.jobs[0].RunAfter = timestamptz(queueNow.Add(time.Second))
		},
		"no queued job": func(s *fakeQueue) {
			s.seed(1, 100, "running", 1)
			s.seed(2, 101, "succeeded", 1)
			s.seed(3, 102, "failed", 1)
		},
	} {
		t.Run(name, func(t *testing.T) {
			store := queueFixture()
			seed(store)
			before := slices.Clone(store.jobs)
			queue, _ := newTestQueue(t, store, testPolicy)
			claimed, found, err := queue.Claim(context.Background())
			if err != nil || found || claimed != (ClaimedJob{}) {
				t.Fatalf("claim = %+v, found = %t, error = %v; want nothing", claimed, found, err)
			}
			if !slices.Equal(store.jobs, before) {
				t.Fatalf("jobs = %+v, want them unchanged", store.jobs)
			}
			queueCalls(t, store, "begin", "claim", "rollback")
		})
	}
}

func TestClaimFailuresUseNoAttempt(t *testing.T) {
	failure := errors.New("database unavailable")
	for _, tt := range []struct {
		fail      string
		wantCalls []string
	}{
		{"claim", []string{"begin", "claim", "rollback"}},
		{"get version", []string{"begin", "claim", "get version", "rollback"}},
		{"get route", []string{"begin", "claim", "get version", "get route", "rollback"}},
		{"commit", []string{"begin", "claim", "get version", "get route", "commit"}},
	} {
		t.Run(tt.fail, func(t *testing.T) {
			store := queueFixture()
			store.seed(1, 100, "queued", 0)
			store.fail = map[string]error{tt.fail: failure}
			queue, _ := newTestQueue(t, store, testPolicy)
			if _, found, err := queue.Claim(context.Background()); !errors.Is(err, failure) || found {
				t.Fatalf("found = %t, error = %v; want the failure", found, err)
			}
			if job := store.jobs[0]; job.Status != "queued" || job.AttemptCount != 0 || job.LeaseUntil.Valid {
				t.Fatalf("job = %+v, want it queued with no attempt used", job)
			}
			queueCalls(t, store, tt.wantCalls...)
		})
	}
}

func TestClaimReportsDisabledRoute(t *testing.T) {
	store := queueFixture()
	store.seed(1, 200, "queued", 0)
	queue, _ := newTestQueue(t, store, testPolicy)
	claimed, found, err := queue.Claim(context.Background())
	if err != nil || !found || claimed.RouteEnabled || claimed.RouteNumber != 3 {
		t.Fatalf("claim = %+v, found = %t, error = %v; want route 3 reported as disabled", claimed, found, err)
	}
}

func TestRequestsUseThePinnedVersion(t *testing.T) {
	version := storedVersion(2, false, contestSpec())
	version.ModelVersion, version.DatasetVersion = "tabpfn030-previous", "prepared-previous"
	queue := NewQueue(nil, testPolicy)

	want := forecasts_predictor_grpc.Request{
		RouteNumber: 50, From: version.ForecastFrom.Time, To: version.ForecastTo.Time,
		ModelVersion: "tabpfn030-previous", DatasetVersion: "prepared-previous",
	}
	if got := queue.Request(ClaimedJob{Version: version, RouteNumber: 50}); got != want {
		t.Fatalf("request = %+v, want %+v", got, want)
	}
	want.RouteNumber = 5
	if got := ProbeRequest(version); got != want {
		t.Fatalf("probe = %+v, want %+v", got, want)
	}
}

func TestRenewLeaseExtendsOwnedLease(t *testing.T) {
	store := queueFixture()
	store.seed(1, 100, "running", 2)
	store.jobs[0].LeaseUntil = timestamptz(queueNow.Add(10 * time.Second))
	queue, _ := newTestQueue(t, store, testPolicy)
	if err := queue.RenewLease(context.Background(), store.claimed(store.jobs[0])); err != nil {
		t.Fatal(err)
	}
	if got := store.jobs[0].LeaseUntil.Time; !got.Equal(queueNow.Add(time.Minute)) {
		t.Fatalf("lease until %s, want %s", got, queueNow.Add(time.Minute))
	}
	want := []forecasts_sqlc.RenewPredictionJobLeaseParams{{LeaseSeconds: 60, AttemptNumber: 2, ID: jobID(1)}}
	if !reflect.DeepEqual(store.renewArgs, want) {
		t.Fatalf("renew params = %+v, want %+v", store.renewArgs, want)
	}
	queueCalls(t, store, "begin", "renew", "commit")
}

func TestRenewLeaseLost(t *testing.T) {
	for name, change := range map[string]func(*forecasts_sqlc.PredictionJob){
		"expired lease": func(j *forecasts_sqlc.PredictionJob) { j.LeaseUntil = timestamptz(queueNow.Add(-time.Second)) },
		"newer attempt": func(j *forecasts_sqlc.PredictionJob) { j.AttemptCount = 2 },
		"requeued":      func(j *forecasts_sqlc.PredictionJob) { j.Status, j.LeaseUntil = "queued", pgtype.Timestamptz{} },
		"succeeded":     func(j *forecasts_sqlc.PredictionJob) { j.Status, j.LeaseUntil = "succeeded", pgtype.Timestamptz{} },
		"unknown job":   func(j *forecasts_sqlc.PredictionJob) { j.ID = jobID(2) },
	} {
		t.Run(name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 100, "running", 1))
			change(&store.jobs[0])
			before := store.jobs[0]
			queue, _ := newTestQueue(t, store, testPolicy)
			if err := queue.RenewLease(context.Background(), claimed); !errors.Is(err, ErrLeaseLost) {
				t.Fatalf("error = %v, want ErrLeaseLost", err)
			}
			if store.jobs[0] != before {
				t.Fatalf("job = %+v, want it unchanged", store.jobs[0])
			}
		})
	}
}

func TestRenewLeaseDatabaseError(t *testing.T) {
	failure := errors.New("database unavailable")
	store := queueFixture()
	claimed := store.claimed(store.seed(1, 100, "running", 1))
	store.fail = map[string]error{"renew": failure}
	queue, _ := newTestQueue(t, store, testPolicy)
	// A database error is not a lost lease: the worker keeps trying.
	if err := queue.RenewLease(context.Background(), claimed); !errors.Is(err, failure) || errors.Is(err, ErrLeaseLost) {
		t.Fatalf("error = %v, want the failure only", err)
	}
	queueCalls(t, store, "begin", "renew", "rollback")
}

// publication is the call sequence of a publication up to its commit.
var publication = []string{"begin", "lock publication", "copy", "complete", "commit"}

func TestPublishStoresTheFullHorizon(t *testing.T) {
	store := queueFixture()
	claimed := store.claimed(store.seed(1, 101, "running", 1))
	queue, _ := newTestQueue(t, store, testPolicy)
	if err := queue.Publish(context.Background(), claimed, horizonPoints(claimed.Version)); err != nil {
		t.Fatal(err)
	}
	if len(store.predictions) != 1464 {
		t.Fatalf("predictions = %d, want 1464", len(store.predictions))
	}
	if job := store.jobs[0]; job.Status != "succeeded" || job.LeaseUntil.Valid || !job.FinishedAt.Valid {
		t.Fatalf("job = %+v, want it succeeded", job)
	}
	queueCalls(t, store, publication...)
}

func TestPredictionRows(t *testing.T) {
	version := storedVersion(1, true, contestSpec())
	job := ClaimedJob{
		Job:     forecasts_sqlc.PredictionJob{ID: jobID(1), ForecastVersionID: version.ID, RouteID: 101},
		Version: version,
	}
	points := horizonPoints(version)
	for i := range points {
		// Rows follow Moscow time whatever the location of the point.
		points[i].HourStart = points[i].HourStart.UTC()
	}
	if first := points[0].HourStart; !first.Equal(time.Date(2025, 10, 31, 21, 0, 0, 0, time.UTC)) {
		t.Fatalf("first point starts at %s", first)
	}
	rows, err := predictionRows(job, points)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1464 {
		t.Fatalf("rows = %d, want 1464", len(rows))
	}
	date := func(year int, month time.Month, day int) pgtype.Date {
		return pgtype.Date{Time: time.Date(year, month, day, 0, 0, 0, 0, time.UTC), Valid: true}
	}
	if want := (forecasts_sqlc.CopyValidationPredictionsParams{
		ForecastVersionID: version.ID, RouteID: 101, Date: date(2025, 11, 1), Hour: 0, Boardings: 0,
	}); rows[0] != want {
		t.Fatalf("first row = %+v, want %+v", rows[0], want)
	}
	if last := rows[1463]; last.Date != date(2025, 12, 31) || last.Hour != 23 {
		t.Fatalf("last row = %+v, want 2025-12-31 hour 23", last)
	}
	for i, row := range rows {
		want := forecasts_sqlc.CopyValidationPredictionsParams{
			ForecastVersionID: version.ID, RouteID: 101,
			Date: date(2025, 11, 1+i/24), Hour: int16(i % 24), Boardings: points[i].Boardings,
		}
		if row != want || row.Date.Time.Location() != time.UTC {
			t.Fatalf("row %d = %+v, want %+v at UTC midnight", i, row, want)
		}
	}
}

func TestPublishRejectsBadBatchBeforeTheTransaction(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*ClaimedJob, []forecasts_predictor_grpc.Point) []forecasts_predictor_grpc.Point
		want   string
	}{
		{"missing hour", func(_ *ClaimedJob, p []forecasts_predictor_grpc.Point) []forecasts_predictor_grpc.Point {
			return p[:len(p)-1]
		}, "got 1463 points"},
		{"extra hour", func(_ *ClaimedJob, p []forecasts_predictor_grpc.Point) []forecasts_predictor_grpc.Point {
			return append(p, p[0])
		}, "got 1465 points"},
		{"no points", func(*ClaimedJob, []forecasts_predictor_grpc.Point) []forecasts_predictor_grpc.Point {
			return nil
		}, "got 0 points"},
		{"other version", func(c *ClaimedJob, p []forecasts_predictor_grpc.Point) []forecasts_predictor_grpc.Point {
			c.Version.ID = pgtype.UUID{Bytes: [16]byte{2}, Valid: true}
			return p
		}, "is pinned to version"},
		{"empty horizon", func(c *ClaimedJob, _ []forecasts_predictor_grpc.Point) []forecasts_predictor_grpc.Point {
			c.Version.ForecastTo = c.Version.ForecastFrom
			return nil
		}, "no forecast hours"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 101, "running", 1))
			points := tt.change(&claimed, horizonPoints(store.versions[0]))
			queue, _ := newTestQueue(t, store, testPolicy)
			if err := queue.Publish(context.Background(), claimed, points); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
			queueCalls(t, store)
		})
	}
}

func TestPublishWithoutOwnershipWritesNothing(t *testing.T) {
	for name, change := range map[string]func(*forecasts_sqlc.PredictionJob){
		"expired lease": func(j *forecasts_sqlc.PredictionJob) { j.LeaseUntil = timestamptz(queueNow.Add(-time.Second)) },
		"newer attempt": func(j *forecasts_sqlc.PredictionJob) { j.AttemptCount = 2 },
		"recovered":     func(j *forecasts_sqlc.PredictionJob) { j.Status, j.LeaseUntil = "queued", pgtype.Timestamptz{} },
		"succeeded":     func(j *forecasts_sqlc.PredictionJob) { j.Status, j.LeaseUntil = "succeeded", pgtype.Timestamptz{} },
	} {
		t.Run(name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 101, "running", 1))
			change(&store.jobs[0])
			before := store.jobs[0]
			queue, _ := newTestQueue(t, store, testPolicy)
			if err := queue.Publish(context.Background(), claimed, horizonPoints(claimed.Version)); !errors.Is(err, ErrLeaseLost) {
				t.Fatalf("error = %v, want ErrLeaseLost", err)
			}
			if len(store.predictions) != 0 || store.jobs[0] != before {
				t.Fatalf("predictions = %d, job = %+v; want nothing written", len(store.predictions), store.jobs[0])
			}
			queueCalls(t, store, "begin", "lock publication", "rollback")
		})
	}
}

func TestPublishChecksTheLockedJob(t *testing.T) {
	other := storedVersion(2, false, contestSpec())
	for name, change := range map[string]func(*ClaimedJob){
		"other route": func(c *ClaimedJob) { c.Job.RouteID = 102 },
		"other version": func(c *ClaimedJob) {
			c.Job.ForecastVersionID, c.Version = other.ID, other
		},
	} {
		t.Run(name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 101, "running", 1))
			change(&claimed)
			queue, _ := newTestQueue(t, store, testPolicy)
			err := queue.Publish(context.Background(), claimed, horizonPoints(claimed.Version))
			if err == nil || !strings.Contains(err.Error(), "belongs to version") || errors.Is(err, ErrLeaseLost) {
				t.Fatalf("error = %v, want the mismatch", err)
			}
			if len(store.predictions) != 0 || store.jobs[0].Status != "running" {
				t.Fatalf("predictions = %d, job = %+v; want nothing written", len(store.predictions), store.jobs[0])
			}
			queueCalls(t, store, "begin", "lock publication", "rollback")
		})
	}
}

func TestPublishFailuresRollBack(t *testing.T) {
	failure := errors.New("database unavailable")
	for _, tt := range []struct {
		name      string
		setup     func(*fakeQueue)
		wantErr   error
		wantCalls []string
	}{
		{"lock fails", func(s *fakeQueue) { s.fail = map[string]error{"lock publication": failure} },
			failure, []string{"begin", "lock publication", "rollback"}},
		{"copy fails", func(s *fakeQueue) { s.fail = map[string]error{"copy": failure} },
			failure, []string{"begin", "lock publication", "copy", "rollback"}},
		{"copy count differs", func(s *fakeQueue) { s.shortCopy = true },
			nil, []string{"begin", "lock publication", "copy", "rollback"}},
		{"complete fails", func(s *fakeQueue) { s.fail = map[string]error{"complete": failure} },
			failure, []string{"begin", "lock publication", "copy", "complete", "rollback"}},
		{"complete changes no row", func(s *fakeQueue) { s.noComplete = true },
			nil, []string{"begin", "lock publication", "copy", "complete", "rollback"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 101, "running", 1))
			before := store.jobs[0]
			tt.setup(store)
			queue, _ := newTestQueue(t, store, testPolicy)
			err := queue.Publish(context.Background(), claimed, horizonPoints(claimed.Version))
			if err == nil || (tt.wantErr != nil && !errors.Is(err, tt.wantErr)) ||
				errors.Is(err, ErrLeaseLost) || errors.Is(err, ErrPublicationConflict) {
				t.Fatalf("error = %v, want a retryable publication failure", err)
			}
			if len(store.predictions) != 0 || store.jobs[0] != before {
				t.Fatalf("predictions = %d, job = %+v; want the batch undone", len(store.predictions), store.jobs[0])
			}
			queueCalls(t, store, tt.wantCalls...)
		})
	}
}

func TestPublishConflictsWithStoredRows(t *testing.T) {
	for _, tt := range []struct {
		name  string
		setup func(*fakeQueue, ClaimedJob, []forecasts_predictor_grpc.Point)
		code  string
	}{
		{"hour already stored", func(s *fakeQueue, c ClaimedJob, p []forecasts_predictor_grpc.Point) {
			rows, err := predictionRows(c, p)
			if err != nil {
				t.Fatal(err)
			}
			s.predictions = rows[100:101]
		}, "23505"},
		{"duplicate hour", func(_ *fakeQueue, _ ClaimedJob, p []forecasts_predictor_grpc.Point) {
			p[1].HourStart = p[0].HourStart
		}, "23505"},
		{"negative boardings", func(_ *fakeQueue, _ ClaimedJob, p []forecasts_predictor_grpc.Point) {
			p[5].Boardings = -1
		}, "23514"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 101, "running", 1))
			points := horizonPoints(claimed.Version)
			tt.setup(store, claimed, points)
			stored := slices.Clone(store.predictions)
			queue, _ := newTestQueue(t, store, testPolicy)
			err := queue.Publish(context.Background(), claimed, points)
			var pgErr *pgconn.PgError
			if !errors.Is(err, ErrPublicationConflict) || !errors.As(err, &pgErr) || pgErr.Code != tt.code {
				t.Fatalf("error = %v, want ErrPublicationConflict with SQLSTATE %s", err, tt.code)
			}
			if !slices.Equal(store.predictions, stored) || store.jobs[0].Status != "running" {
				t.Fatalf("predictions = %d, job = %+v; want nothing overwritten", len(store.predictions), store.jobs[0])
			}
			queueCalls(t, store, "begin", "lock publication", "copy", "rollback")
		})
	}
}

func TestPublishCommitOutcome(t *testing.T) {
	commitFailure := errors.New("connection reset during commit")
	checkFailure := errors.New("database unavailable")
	check := []string{"begin read-only", "get job", "rollback"}
	for _, tt := range []struct {
		name string
		// applied is whether the failed commit went through anyway.
		applied bool
		setup   func(*fakeQueue)
		// wantErr lists the errors the result must wrap; none means success.
		wantErr    []error
		wantStatus string
		wantRows   int
	}{
		{name: "rolled back", wantErr: []error{commitFailure}, wantStatus: "running"},
		{name: "committed", applied: true, wantStatus: "succeeded", wantRows: 1464},
		{name: "recovered meanwhile", setup: func(s *fakeQueue) {
			s.before = map[string]func(){"get job": func() { s.jobs[0].Status, s.jobs[0].LeaseUntil = "queued", pgtype.Timestamptz{} }}
		}, wantErr: []error{commitFailure, ErrLeaseLost}},
		{name: "claimed by a newer attempt", setup: func(s *fakeQueue) {
			s.before = map[string]func(){"get job": func() { s.jobs[0].AttemptCount = 2 }}
		}, wantErr: []error{commitFailure, ErrLeaseLost}},
		{name: "check fails", setup: func(s *fakeQueue) {
			s.fail = map[string]error{"get job": checkFailure}
		}, wantErr: []error{commitFailure, checkFailure}, wantStatus: "running"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 101, "running", 1))
			if tt.setup != nil {
				tt.setup(store)
			}
			queue, db := newTestQueue(t, store, testPolicy)
			db.commit = func(context.Context) (bool, error) { return tt.applied, commitFailure }

			err := queue.Publish(context.Background(), claimed, horizonPoints(claimed.Version))
			if len(tt.wantErr) == 0 && err != nil {
				t.Fatalf("error = %v, want success", err)
			}
			for _, want := range tt.wantErr {
				if !errors.Is(err, want) {
					t.Fatalf("error = %v, want it to wrap %v", err, want)
				}
			}
			if slices.Contains(tt.wantErr, checkFailure) && !strings.Contains(err.Error(), "outcome is unknown") {
				t.Fatalf("error = %v, want the unknown outcome reported", err)
			}
			if !slices.Contains(tt.wantErr, ErrLeaseLost) && errors.Is(err, ErrLeaseLost) {
				t.Fatalf("error = %v, want no ErrLeaseLost", err)
			}
			if tt.wantStatus != "" && store.jobs[0].Status != tt.wantStatus || len(store.predictions) != tt.wantRows {
				t.Fatalf("job = %s, predictions = %d; want %s and %d", store.jobs[0].Status, len(store.predictions), tt.wantStatus, tt.wantRows)
			}
			queueCalls(t, store, slices.Concat(publication, check)...)
		})
	}

	// Shutdown may cancel the context during the commit: the check still runs.
	store := queueFixture()
	claimed := store.claimed(store.seed(1, 101, "running", 1))
	queue, db := newTestQueue(t, store, testPolicy)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	db.commit = func(context.Context) (bool, error) {
		cancel()
		return true, context.Canceled
	}
	if err := queue.Publish(ctx, claimed, horizonPoints(claimed.Version)); err != nil {
		t.Fatalf("error = %v, want the committed publication confirmed", err)
	}
	queueCalls(t, store, slices.Concat(publication, check)...)
}

func TestFailAttemptRequeuesOrFails(t *testing.T) {
	for _, tt := range []struct {
		name       string
		attempt    int32
		retryable  bool
		wantStatus string
		wantDelay  int32
	}{
		{"first retryable attempt", 1, true, "queued", 30},
		{"second retryable attempt", 2, true, "queued", 120},
		{"last attempt", 3, true, "failed", 480},
		{"terminal error", 1, false, "failed", 30},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 100, "running", tt.attempt))
			queue, _ := newTestQueue(t, store, testPolicy)
			cause := &forecasts_predictor_grpc.Error{Code: "ml_unavailable", Retryable: true, Err: errors.New("connection refused")}

			finished, err := queue.FailAttempt(context.Background(), claimed, "ml_unavailable", tt.retryable, cause)
			if err != nil {
				t.Fatal(err)
			}
			if finished != store.jobs[0] || finished.Status != tt.wantStatus || finished.LeaseUntil.Valid ||
				finished.LastErrorCode != text("ml_unavailable") || finished.LastErrorMessage != text("ml_unavailable: connection refused") {
				t.Fatalf("job = %+v, want it %s with the error recorded", finished, tt.wantStatus)
			}
			if tt.wantStatus == "queued" && (!finished.RunAfter.Time.Equal(queueNow.Add(seconds(tt.wantDelay))) || finished.FinishedAt.Valid) {
				t.Fatalf("job = %+v, want a retry after %d s", finished, tt.wantDelay)
			}
			if tt.wantStatus == "failed" && !finished.FinishedAt.Valid {
				t.Fatalf("job = %+v, want it finished", finished)
			}
			want := []forecasts_sqlc.FinishPredictionJobWithErrorParams{{
				Retryable: tt.retryable, MaxAttempts: 3, RetryDelaySeconds: tt.wantDelay,
				ErrorCode: "ml_unavailable", ErrorMessage: "ml_unavailable: connection refused",
				AttemptNumber: tt.attempt, ID: jobID(1),
			}}
			if !reflect.DeepEqual(store.finishArgs, want) {
				t.Fatalf("finish params = %+v, want %+v", store.finishArgs, want)
			}
			queueCalls(t, store, "begin", "finish", "commit")
		})
	}
}

func TestFailAttemptWithoutOwnershipWritesNothing(t *testing.T) {
	for name, change := range map[string]func(*forecasts_sqlc.PredictionJob){
		"expired lease": func(j *forecasts_sqlc.PredictionJob) { j.LeaseUntil = timestamptz(queueNow.Add(-time.Second)) },
		"newer attempt": func(j *forecasts_sqlc.PredictionJob) { j.AttemptCount = 2 },
		"succeeded":     func(j *forecasts_sqlc.PredictionJob) { j.Status, j.LeaseUntil = "succeeded", pgtype.Timestamptz{} },
	} {
		t.Run(name, func(t *testing.T) {
			store := queueFixture()
			claimed := store.claimed(store.seed(1, 100, "running", 1))
			change(&store.jobs[0])
			before := store.jobs[0]
			queue, _ := newTestQueue(t, store, testPolicy)
			_, err := queue.FailAttempt(context.Background(), claimed, "ml_internal", false, errors.New("boom"))
			if !errors.Is(err, ErrLeaseLost) {
				t.Fatalf("error = %v, want ErrLeaseLost", err)
			}
			if store.jobs[0] != before {
				t.Fatalf("job = %+v, want it unchanged", store.jobs[0])
			}
			queueCalls(t, store, "begin", "finish", "rollback")
		})
	}

	failure := errors.New("database unavailable")
	store := queueFixture()
	claimed := store.claimed(store.seed(1, 100, "running", 1))
	store.fail = map[string]error{"finish": failure}
	queue, _ := newTestQueue(t, store, testPolicy)
	if _, err := queue.FailAttempt(context.Background(), claimed, "ml_internal", false, nil); !errors.Is(err, failure) || errors.Is(err, ErrLeaseLost) {
		t.Fatalf("error = %v, want the failure only", err)
	}
}

func TestErrorMessage(t *testing.T) {
	long := strings.Repeat("я", 1500)
	for _, tt := range []struct {
		name  string
		cause error
		want  string
	}{
		{"no cause", nil, "publication_failed"},
		{"short", errors.New("connection refused"), "connection refused"},
		{"long", errors.New(long), strings.Repeat("я", 1000)},
		{"not storable as text", errors.New("a\x00b\xffc"), "ab" + string(utf8.RuneError) + "c"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := errorMessage("publication_failed", tt.cause)
			if got != tt.want || !utf8.ValidString(got) {
				t.Fatalf("message = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestRetryDelaySeconds(t *testing.T) {
	for _, tt := range []struct {
		base    time.Duration
		attempt int32
		want    int32
	}{
		{30 * time.Second, 1, 30},
		{30 * time.Second, 2, 120},
		{30 * time.Second, 3, 480},
		{30 * time.Second, 4, 1920},
		{30 * time.Second, 5, 3600}, // 7680 s is capped at an hour
		{30 * time.Second, 100, 3600},
		{30 * time.Second, math.MaxInt32, 3600},
		{30 * time.Second, 0, 30},
		{time.Second, 7, 3600},
		{time.Hour, 1, 3600},
		{time.Hour, 50, 3600},
		{0, 1, 0},
		{0, math.MaxInt32, 0},
		{-time.Second, 2, 0},
	} {
		policy := JobPolicy{RetryDelay: tt.base}
		if got := policy.retryDelaySeconds(tt.attempt); got != tt.want {
			t.Errorf("delay after attempt %d with base %s = %d s, want %d s", tt.attempt, tt.base, got, tt.want)
		}
	}
}

// seedMany adds count jobs from job n on, each for its own route.
func (s *fakeQueue) seedMany(n *int, count int, status string, attempt int32, expired bool) {
	for range count {
		*n++
		s.seed(*n, int64(1000+*n), status, attempt)
		if expired {
			s.jobs[len(s.jobs)-1].LeaseUntil = timestamptz(queueNow.Add(-time.Second))
		}
	}
}

func TestRecoverRunsBatchesUntilDone(t *testing.T) {
	store := queueFixture()
	n := 0
	store.seedMany(&n, 150, "running", 1, true) // requeued
	store.seedMany(&n, 20, "running", 3, true)  // failed: no attempt left
	store.seedMany(&n, 101, "queued", 3, false) // exhausted
	store.seedMany(&n, 5, "running", 1, false)  // owned: untouched
	store.seedMany(&n, 5, "queued", 1, false)   // waiting: untouched
	queue, _ := newTestQueue(t, store, testPolicy)

	recovered, exhausted, err := queue.Recover(context.Background())
	if err != nil || recovered != 170 || exhausted != 101 {
		t.Fatalf("recovered = %d, exhausted = %d, error = %v; want 170 and 101", recovered, exhausted, err)
	}
	queueCalls(t, store, "begin", "recover", "commit", "begin", "recover", "commit",
		"begin", "fail exhausted", "commit", "begin", "fail exhausted", "commit")
	for _, arg := range store.recoverArgs {
		if want := (forecasts_sqlc.RecoverExpiredPredictionJobsParams{MaxAttempts: 3, RetryDelaySeconds: 30, BatchSize: 100}); arg != want {
			t.Fatalf("recover params = %+v, want %+v", arg, want)
		}
	}
	for _, arg := range store.exhaustArgs {
		if want := (forecasts_sqlc.FailExhaustedQueuedPredictionJobsParams{MaxAttempts: 3, BatchSize: 100}); arg != want {
			t.Fatalf("fail exhausted params = %+v, want %+v", arg, want)
		}
	}
	counts := map[string]int{}
	for _, job := range store.jobs {
		counts[job.Status+" "+job.LastErrorCode.String]++
	}
	want := map[string]int{
		"queued lease_expired": 150, "failed lease_expired": 20, "failed attempt_limit_exceeded": 101,
		"running ": 5, "queued ": 5,
	}
	if !reflect.DeepEqual(counts, want) {
		t.Fatalf("jobs by status and error = %v, want %v", counts, want)
	}
}

func TestRecoverExactBatchReadsOnce(t *testing.T) {
	store := queueFixture()
	n := 0
	store.seedMany(&n, 100, "running", 1, true)
	queue, _ := newTestQueue(t, store, testPolicy)
	recovered, exhausted, err := queue.Recover(context.Background())
	if err != nil || recovered != 100 || exhausted != 0 {
		t.Fatalf("recovered = %d, exhausted = %d, error = %v", recovered, exhausted, err)
	}
	// A full batch may have more behind it; the empty one ends the loop.
	queueCalls(t, store, "begin", "recover", "commit", "begin", "recover", "commit", "begin", "fail exhausted", "commit")
}

func TestRecoverFailureKeepsCommittedBatches(t *testing.T) {
	failure := errors.New("database unavailable")

	store := queueFixture()
	n := 0
	store.seedMany(&n, 150, "running", 1, true)
	calls := 0
	store.before = map[string]func(){"recover": func() {
		if calls++; calls == 2 {
			store.fail = map[string]error{"recover": failure}
		}
	}}
	queue, _ := newTestQueue(t, store, testPolicy)
	recovered, exhausted, err := queue.Recover(context.Background())
	if !errors.Is(err, failure) || recovered != 100 || exhausted != 0 {
		t.Fatalf("recovered = %d, exhausted = %d, error = %v; want the first batch and the failure", recovered, exhausted, err)
	}
	queueCalls(t, store, "begin", "recover", "commit", "begin", "recover", "rollback")

	store = queueFixture()
	store.fail = map[string]error{"fail exhausted": failure}
	queue, _ = newTestQueue(t, store, testPolicy)
	if _, _, err := queue.Recover(context.Background()); !errors.Is(err, failure) {
		t.Fatalf("error = %v, want the failure", err)
	}
	queueCalls(t, store, "begin", "recover", "commit", "begin", "fail exhausted", "rollback")
}

func TestForecastRoutes(t *testing.T) {
	store := queueFixture()
	queue, _ := newTestQueue(t, store, testPolicy)
	routes, err := queue.ForecastRoutes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if want := enabledRoutes(targetRoutes...); !reflect.DeepEqual(routes, want) {
		t.Fatalf("routes = %+v, want the ten target routes in order", routes)
	}
	queueCalls(t, store, "begin read-only", "list routes", "rollback")

	failure := errors.New("database unavailable")
	store.fail = map[string]error{"list routes": failure}
	if _, err := queue.ForecastRoutes(context.Background()); !errors.Is(err, failure) {
		t.Fatalf("error = %v, want the failure", err)
	}
}

func TestActiveVersion(t *testing.T) {
	store := queueFixture()
	queue, _ := newTestQueue(t, store, testPolicy)
	version, err := queue.ActiveVersion(context.Background())
	if err != nil || version != store.versions[0] {
		t.Fatalf("version = %+v, error = %v; want the active version", version, err)
	}
	queueCalls(t, store, "begin read-only", "get active version", "rollback")

	store.versions[0].IsActive = false
	if _, err := queue.ActiveVersion(context.Background()); !errors.Is(err, ErrNoActiveVersion) {
		t.Fatalf("error = %v, want ErrNoActiveVersion", err)
	}
	failure := errors.New("database unavailable")
	store.fail = map[string]error{"get active version": failure}
	if _, err := queue.ActiveVersion(context.Background()); !errors.Is(err, failure) || errors.Is(err, ErrNoActiveVersion) {
		t.Fatalf("error = %v, want the failure only", err)
	}
}

func TestVersion(t *testing.T) {
	store := queueFixture()
	queue, _ := newTestQueue(t, store, testPolicy)
	if version, err := queue.Version(context.Background(), store.versions[0].ID); err != nil || version != store.versions[0] {
		t.Fatalf("version = %+v, error = %v", version, err)
	}
	if _, err := queue.Version(context.Background(), jobID(1)); !errors.Is(err, ErrVersionNotFound) {
		t.Fatalf("error = %v, want ErrVersionNotFound", err)
	}
	queueCalls(t, store, "begin read-only", "get version", "rollback", "begin read-only", "get version", "rollback")
}

func TestLookup(t *testing.T) {
	store := queueFixture()
	failed := store.seed(1, 100, "failed", 3)
	queue, _ := newTestQueue(t, store, testPolicy)
	versionID := store.versions[0].ID

	job, found, err := queue.Lookup(context.Background(), versionID, 100)
	if err != nil || !found || job != failed {
		t.Fatalf("job = %+v, found = %t, error = %v; want the failed job", job, found, err)
	}
	if _, found, err := queue.Lookup(context.Background(), versionID, 101); err != nil || found {
		t.Fatalf("found = %t, error = %v; want no job", found, err)
	}
	queueCalls(t, store, "begin read-only", "find job", "rollback", "begin read-only", "find job", "rollback")

	failure := errors.New("database unavailable")
	store.fail = map[string]error{"find job": failure}
	if _, found, err := queue.Lookup(context.Background(), versionID, 100); !errors.Is(err, failure) || found {
		t.Fatalf("found = %t, error = %v; want the failure", found, err)
	}
}
