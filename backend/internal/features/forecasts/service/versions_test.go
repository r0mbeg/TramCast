package forecasts_service

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

// fakeStore mirrors the SQL semantics of the version queries: the artifacts
// key, at most one active version and rows returned only on success. It logs
// every call together with the transaction boundaries of fakeDB.
type fakeStore struct {
	versions []forecasts_sqlc.ForecastVersion
	routes   []routes_sqlc.Route
	calls    []string
	// fail makes the named call return the error.
	fail map[string]error
}

func (s *fakeStore) call(name string) error {
	s.calls = append(s.calls, name)
	return s.fail[name]
}

func (s *fakeStore) find(match func(forecasts_sqlc.ForecastVersion) bool) (int, error) {
	for i, version := range s.versions {
		if match(version) {
			return i, nil
		}
	}
	return 0, pgx.ErrNoRows
}

func sameArtifacts(version forecasts_sqlc.ForecastVersion, arg forecasts_sqlc.GetForecastVersionByArtifactsParams) bool {
	return version.ModelVersion == arg.ModelVersion && version.DatasetVersion == arg.DatasetVersion &&
		version.HistoryEnd.Time.Equal(arg.HistoryEnd.Time) && version.ForecastFrom.Time.Equal(arg.ForecastFrom.Time) &&
		version.ForecastTo.Time.Equal(arg.ForecastTo.Time)
}

func (s *fakeStore) CreateForecastVersion(_ context.Context, arg forecasts_sqlc.CreateForecastVersionParams) (forecasts_sqlc.ForecastVersion, error) {
	if err := s.call("create"); err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	if _, err := s.find(func(v forecasts_sqlc.ForecastVersion) bool {
		return sameArtifacts(v, forecasts_sqlc.GetForecastVersionByArtifactsParams{
			ModelVersion: arg.ModelVersion, DatasetVersion: arg.DatasetVersion,
			HistoryEnd: arg.HistoryEnd, ForecastFrom: arg.ForecastFrom, ForecastTo: arg.ForecastTo,
		})
	}); err == nil {
		return forecasts_sqlc.ForecastVersion{}, pgx.ErrNoRows // ON CONFLICT DO NOTHING
	}
	version := forecasts_sqlc.ForecastVersion{
		ID: arg.ID, ModelVersion: arg.ModelVersion, DatasetVersion: arg.DatasetVersion,
		HistoryEnd: arg.HistoryEnd, ForecastFrom: arg.ForecastFrom, ForecastTo: arg.ForecastTo,
		Timezone: "Europe/Moscow", CreatedAt: timestamptz(time.Now()),
	}
	s.versions = append(s.versions, version)
	return version, nil
}

func (s *fakeStore) GetForecastVersion(_ context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error) {
	if err := s.call("get"); err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	i, err := s.find(func(v forecasts_sqlc.ForecastVersion) bool { return v.ID == id })
	if err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	return s.versions[i], nil
}

func (s *fakeStore) GetForecastVersionByArtifacts(_ context.Context, arg forecasts_sqlc.GetForecastVersionByArtifactsParams) (forecasts_sqlc.ForecastVersion, error) {
	if err := s.call("find"); err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	i, err := s.find(func(v forecasts_sqlc.ForecastVersion) bool { return sameArtifacts(v, arg) })
	if err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	return s.versions[i], nil
}

func (s *fakeStore) LockForecastVersionsForActivation(context.Context) error {
	return s.call("lock")
}

func (s *fakeStore) DeactivateForecastVersions(context.Context) error {
	if err := s.call("deactivate"); err != nil {
		return err
	}
	for i := range s.versions {
		s.versions[i].IsActive = false
	}
	return nil
}

func (s *fakeStore) ActivateForecastVersion(_ context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error) {
	if err := s.call("activate"); err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	if _, err := s.find(func(v forecasts_sqlc.ForecastVersion) bool { return v.IsActive && v.ID != id }); err == nil {
		return forecasts_sqlc.ForecastVersion{}, errors.New("unique violation: forecast_versions_one_active_idx")
	}
	i, err := s.find(func(v forecasts_sqlc.ForecastVersion) bool { return v.ID == id })
	if err != nil {
		return forecasts_sqlc.ForecastVersion{}, err
	}
	s.versions[i].IsActive = true
	return s.versions[i], nil
}

func (s *fakeStore) ListRoutes(context.Context) ([]routes_sqlc.Route, error) {
	if err := s.call("list routes"); err != nil {
		return nil, err
	}
	return slices.Clone(s.routes), nil
}

// fakeDB starts transactions over the store. Rollback restores the versions
// seen at BEGIN, so a failed transaction leaves no trace.
type fakeDB struct {
	t        *testing.T
	store    *fakeStore
	tx       *fakeTx
	options  []pgx.TxOptions
	beginErr error
}

func (db *fakeDB) BeginTx(_ context.Context, options pgx.TxOptions) (pgx.Tx, error) {
	db.options = append(db.options, options)
	if db.beginErr != nil {
		return nil, db.beginErr
	}
	db.store.calls = append(db.store.calls, "begin")
	db.tx = &fakeTx{t: db.t, store: db.store, saved: slices.Clone(db.store.versions)}
	return db.tx, nil
}

type fakeTx struct {
	pgx.Tx
	t         *testing.T
	store     *fakeStore
	saved     []forecasts_sqlc.ForecastVersion
	committed bool
}

func (tx *fakeTx) Commit(context.Context) error {
	if err := tx.store.call("commit"); err != nil {
		return err
	}
	tx.committed = true
	return nil
}

func (tx *fakeTx) Rollback(ctx context.Context) error {
	if ctx.Err() != nil {
		tx.t.Error("rollback must remain possible after caller cancellation")
	}
	if tx.committed {
		return pgx.ErrTxClosed
	}
	tx.store.calls = append(tx.store.calls, "rollback")
	tx.store.versions = tx.saved
	return nil
}

// newTestService binds the service to the store and checks that every query
// runs in the transaction just started.
func newTestService(t *testing.T, store *fakeStore, predictor Predictor) (*Service, *fakeDB) {
	t.Helper()
	db := &fakeDB{t: t, store: store}
	service := NewService(db, predictor)
	service.queries = func(tx pgx.Tx) queries {
		if tx != db.tx {
			t.Error("queries must use the current transaction")
		}
		return queries{versions: store, routes: store}
	}
	return service, db
}

type fakePredictor struct {
	requests []forecasts_predictor_grpc.Request
	err      error
}

func (p *fakePredictor) Predict(_ context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error) {
	p.requests = append(p.requests, r)
	if p.err != nil {
		return forecasts_predictor_grpc.Result{}, p.err
	}
	return forecasts_predictor_grpc.Result{ModelVersion: r.ModelVersion, DatasetVersion: r.DatasetVersion}, nil
}

func contestSpec() VersionSpec {
	return VersionSpec{
		ModelVersion:   "tabpfn030-0123456789abcdef",
		DatasetVersion: "prepared-844b17f7",
		Timezone:       "Europe/Moscow",
		HistoryEnd:     novemberStart,
		ForecastFrom:   novemberStart,
		ForecastTo:     januaryStart,
		RouteNumbers:   slices.Clone(targetRoutes),
	}
}

func enabledRoutes(numbers ...int16) []routes_sqlc.Route {
	routes := make([]routes_sqlc.Route, len(numbers))
	for i, number := range numbers {
		routes[i] = routes_sqlc.Route{ID: int64(100 + i), RouteNumber: number, ForecastEnabled: true}
	}
	return routes
}

func storedVersion(id byte, active bool, spec VersionSpec) forecasts_sqlc.ForecastVersion {
	return forecasts_sqlc.ForecastVersion{
		ID:           pgtype.UUID{Bytes: [16]byte{id}, Valid: true},
		ModelVersion: spec.ModelVersion, DatasetVersion: spec.DatasetVersion,
		HistoryEnd: timestamptz(spec.HistoryEnd), ForecastFrom: timestamptz(spec.ForecastFrom), ForecastTo: timestamptz(spec.ForecastTo),
		Timezone: "Europe/Moscow", IsActive: active,
	}
}

func checkCalls(t *testing.T, store *fakeStore, want ...string) {
	t.Helper()
	if !reflect.DeepEqual(store.calls, want) {
		t.Fatalf("calls = %v, want %v", store.calls, want)
	}
}

var (
	readCommitted = pgx.TxOptions{IsoLevel: pgx.ReadCommitted}
	readOnly      = pgx.TxOptions{IsoLevel: pgx.ReadCommitted, AccessMode: pgx.ReadOnly}
)

func checkOptions(t *testing.T, db *fakeDB, want ...pgx.TxOptions) {
	t.Helper()
	if !reflect.DeepEqual(db.options, want) {
		t.Fatalf("transaction options = %+v, want %+v", db.options, want)
	}
}

func TestCheckSpecAcceptsContestVersion(t *testing.T) {
	for name, spec := range map[string]VersionSpec{
		"contest": contestSpec(),
		// The same instants written in UTC; alignment is checked in Moscow.
		"UTC offsets": func() VersionSpec {
			spec := contestSpec()
			spec.HistoryEnd, spec.ForecastFrom, spec.ForecastTo = novemberStart.UTC(), novemberStart.UTC(), januaryStart.UTC()
			return spec
		}(),
		"history before the forecast": func() VersionSpec {
			spec := contestSpec()
			spec.HistoryEnd = spec.ForecastFrom.Add(-24 * time.Hour)
			return spec
		}(),
		"one hour": func() VersionSpec {
			spec := contestSpec()
			spec.ForecastTo = spec.ForecastFrom.Add(time.Hour)
			return spec
		}(),
	} {
		t.Run(name, func(t *testing.T) {
			if err := CheckSpec(spec, false); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCheckSpecRejectsInvalidSpec(t *testing.T) {
	moscowHalfHour := time.Date(2025, 11, 1, 0, 30, 0, 0, time.UTC).Add(-3 * time.Hour)
	for _, tt := range []struct {
		name   string
		change func(*VersionSpec)
		want   string
	}{
		{"UTC timezone", func(s *VersionSpec) { s.Timezone = "UTC" }, "timezone"},
		{"missing timezone", func(s *VersionSpec) { s.Timezone = "" }, "timezone"},
		{"blank model", func(s *VersionSpec) { s.ModelVersion = " " }, "model_version"},
		{"empty model", func(s *VersionSpec) { s.ModelVersion = "" }, "model_version"},
		{"model with spaces", func(s *VersionSpec) { s.ModelVersion = "tabpfn030-x\n" }, "model_version"},
		{"blank dataset", func(s *VersionSpec) { s.DatasetVersion = "\t" }, "dataset_version"},
		{"dataset with spaces", func(s *VersionSpec) { s.DatasetVersion = " prepared-x" }, "dataset_version"},
		{"zero history end", func(s *VersionSpec) { s.HistoryEnd = time.Time{} }, "must be set"},
		{"zero forecast from", func(s *VersionSpec) { s.ForecastFrom = time.Time{} }, "must be set"},
		{"zero forecast to", func(s *VersionSpec) { s.ForecastTo = time.Time{} }, "must be set"},
		{"history after forecast", func(s *VersionSpec) { s.HistoryEnd = s.ForecastFrom.Add(time.Hour) }, "history_end <= forecast_from"},
		{"empty horizon", func(s *VersionSpec) { s.ForecastTo = s.ForecastFrom }, "forecast_from < forecast_to"},
		{"reversed horizon", func(s *VersionSpec) { s.ForecastTo = s.ForecastFrom.Add(-time.Hour) }, "forecast_from < forecast_to"},
		{"unaligned from", func(s *VersionSpec) { s.HistoryEnd, s.ForecastFrom = moscowHalfHour, moscowHalfHour }, "whole Moscow hours"},
		{"unaligned to", func(s *VersionSpec) { s.ForecastTo = s.ForecastTo.Add(time.Minute) }, "whole Moscow hours"},
		{"nanoseconds", func(s *VersionSpec) { s.ForecastTo = s.ForecastTo.Add(time.Nanosecond) }, "whole Moscow hours"},
		{
			// A whole hour at UTC+05:30 is half past in Moscow.
			"whole hour elsewhere",
			func(s *VersionSpec) {
				s.ForecastTo = time.Date(2025, 12, 31, 0, 0, 0, 0, time.FixedZone("UTC+05:30", 5*3600+1800))
			},
			"whole Moscow hours",
		},
		{"horizon over 1464 hours", func(s *VersionSpec) { s.ForecastTo = s.ForecastTo.Add(time.Hour) }, "exceeds 1464 hours"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			spec := contestSpec()
			tt.change(&spec)
			err := CheckSpec(spec, true)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want one mentioning %q", err, tt.want)
			}
		})
	}
}

func TestCheckSpecReplayNeedsOptIn(t *testing.T) {
	spec := contestSpec()
	spec.Replay = true
	if err := CheckSpec(spec, false); !errors.Is(err, ErrReplayNotAllowed) {
		t.Fatalf("error = %v, want ErrReplayNotAllowed", err)
	}
	if err := CheckSpec(spec, true); err != nil {
		t.Fatalf("allowed replay: %v", err)
	}
	// The opt-in does not relax the other rules.
	spec.Timezone = "UTC"
	if err := CheckSpec(spec, true); err == nil || errors.Is(err, ErrReplayNotAllowed) {
		t.Fatalf("error = %v, want the timezone rule", err)
	}
}

func TestCheckCoverage(t *testing.T) {
	routes := append(enabledRoutes(1, 5, 17, 50), routes_sqlc.Route{ID: 9, RouteNumber: 3})
	disabled := []routes_sqlc.Route{{ID: 9, RouteNumber: 3}}
	for _, tt := range []struct {
		name    string
		routes  []routes_sqlc.Route
		numbers []int16
		want    string
	}{
		{"all enabled routes", routes, []int16{50, 17, 5, 1}, ""},
		{"extra routes", routes, slices.Clone(targetRoutes), ""},
		// Route 3 has forecasts disabled, so the ML service need not serve it.
		{"missing enabled routes", routes, []int16{1, 5}, "lack forecast-enabled routes 17, 50"},
		{"no routes served", routes, nil, "lack forecast-enabled routes 1, 5, 17, 50"},
		{"empty catalog", nil, slices.Clone(targetRoutes), "no forecast-enabled routes: import the catalog first"},
		{"no enabled routes", disabled, slices.Clone(targetRoutes), "no forecast-enabled routes: import the catalog first"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			spec := contestSpec()
			spec.RouteNumbers = tt.numbers
			store := &fakeStore{routes: tt.routes}
			service, db := newTestService(t, store, nil)
			err := service.checkCoverage(context.Background(), spec)
			if (tt.want == "" && err != nil) || (tt.want != "" && (err == nil || !strings.Contains(err.Error(), tt.want))) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
			checkCalls(t, store, "begin", "list routes", "rollback")
			checkOptions(t, db, readOnly)
		})
	}
}

func TestCheckCoverageDatabaseFailures(t *testing.T) {
	failure := errors.New("database unavailable")

	store := &fakeStore{fail: map[string]error{"list routes": failure}}
	service, _ := newTestService(t, store, nil)
	err := service.checkCoverage(context.Background(), contestSpec())
	// A database failure must not read as bad metadata.
	if !errors.Is(err, failure) || err.Error() != "read the route catalog: database unavailable" {
		t.Fatalf("error = %v, want the wrapped list failure", err)
	}
	checkCalls(t, store, "begin", "list routes", "rollback")

	store = &fakeStore{}
	service, db := newTestService(t, store, nil)
	db.beginErr = failure
	if err := service.checkCoverage(context.Background(), contestSpec()); !errors.Is(err, failure) ||
		!strings.HasPrefix(err.Error(), "begin read-only transaction: ") {
		t.Fatalf("error = %v, want the wrapped begin failure", err)
	}
	checkCalls(t, store)
}

func TestVerifyCallsRouteFiveOverTheHorizon(t *testing.T) {
	predictor := &fakePredictor{}
	service, _ := newTestService(t, &fakeStore{}, predictor)
	spec := contestSpec()
	if err := service.Verify(context.Background(), spec); err != nil {
		t.Fatal(err)
	}
	want := []forecasts_predictor_grpc.Request{{
		RouteNumber: 5, From: spec.ForecastFrom, To: spec.ForecastTo,
		ModelVersion: spec.ModelVersion, DatasetVersion: spec.DatasetVersion,
	}}
	if !reflect.DeepEqual(predictor.requests, want) {
		t.Fatalf("requests = %+v, want %+v", predictor.requests, want)
	}
}

func TestVerifyFailures(t *testing.T) {
	// The client reports a mismatch of the served artifacts as an error.
	for _, failure := range []*forecasts_predictor_grpc.Error{
		{Code: "ml_version_mismatch", Err: errors.New("server artifacts differ")},
		{Code: "ml_failed_precondition", Err: errors.New("period outside the loaded bundle")},
		{Code: "ml_unavailable", Retryable: true, Err: errors.New("connection refused")},
	} {
		t.Run(failure.Code, func(t *testing.T) {
			predictor := &fakePredictor{err: failure}
			service, _ := newTestService(t, &fakeStore{}, predictor)
			err := service.Verify(context.Background(), contestSpec())
			var predictErr *forecasts_predictor_grpc.Error
			if !errors.As(err, &predictErr) || predictErr.Code != failure.Code {
				t.Fatalf("error = %v, want code %s", err, failure.Code)
			}
			if len(predictor.requests) != 1 {
				t.Fatalf("calls = %d, want 1", len(predictor.requests))
			}
		})
	}

	t.Run("route 5 not served", func(t *testing.T) {
		predictor := &fakePredictor{}
		service, _ := newTestService(t, &fakeStore{}, predictor)
		spec := contestSpec()
		spec.RouteNumbers = []int16{1, 7}
		if err := service.Verify(context.Background(), spec); err == nil || !strings.Contains(err.Error(), "route 5") {
			t.Fatalf("error = %v, want the missing verification route", err)
		}
		if len(predictor.requests) != 0 {
			t.Fatal("the server must not be called")
		}
	})

	t.Run("no ML client", func(t *testing.T) {
		service, _ := newTestService(t, &fakeStore{}, nil)
		if err := service.Verify(context.Background(), contestSpec()); err == nil {
			t.Fatal("verification without a client must fail")
		}
	})
}

func TestRegisterCreatesVersion(t *testing.T) {
	store := &fakeStore{}
	service, db := newTestService(t, store, nil)
	spec := contestSpec()
	registered, created, err := service.Register(context.Background(), spec, false)
	if err != nil || !created {
		t.Fatalf("created = %t, error = %v; want a new version", created, err)
	}
	if !registered.ID.Valid || registered.IsActive || registered.ModelVersion != spec.ModelVersion ||
		registered.DatasetVersion != spec.DatasetVersion || !registered.ForecastTo.Time.Equal(spec.ForecastTo) {
		t.Fatalf("version = %+v, want the inactive spec", registered)
	}
	if len(store.versions) != 1 || store.versions[0].ID != registered.ID {
		t.Fatalf("stored versions = %+v", store.versions)
	}
	checkCalls(t, store, "begin", "create", "commit")
	checkOptions(t, db, readCommitted)
}

func TestRegisterReturnsExistingVersion(t *testing.T) {
	spec := contestSpec()
	existing := storedVersion(1, true, spec)
	store := &fakeStore{versions: []forecasts_sqlc.ForecastVersion{existing}}
	service, _ := newTestService(t, store, nil)

	// The same instants in another zone are the same artifacts key.
	spec.HistoryEnd, spec.ForecastFrom, spec.ForecastTo = spec.HistoryEnd.UTC(), spec.ForecastFrom.UTC(), spec.ForecastTo.UTC()
	registered, created, err := service.Register(context.Background(), spec, false)
	if err != nil || created || registered != existing {
		t.Fatalf("version = %+v, created = %t, error = %v; want the existing version", registered, created, err)
	}
	if len(store.versions) != 1 {
		t.Fatalf("stored versions = %+v, want no duplicate", store.versions)
	}
	checkCalls(t, store, "begin", "create", "find", "commit")
}

func TestRegisterIsIdempotent(t *testing.T) {
	store := &fakeStore{}
	service, _ := newTestService(t, store, nil)
	first, created, err := service.Register(context.Background(), contestSpec(), false)
	if err != nil || !created {
		t.Fatalf("first: created = %t, error = %v", created, err)
	}
	second, created, err := service.Register(context.Background(), contestSpec(), false)
	if err != nil || created || second.ID != first.ID {
		t.Fatalf("second: version = %+v, created = %t, error = %v; want the first", second, created, err)
	}

	// Another model is a new version for the same period.
	other := contestSpec()
	other.ModelVersion = "tabpfn030-fedcba9876543210"
	third, created, err := service.Register(context.Background(), other, false)
	if err != nil || !created || third.ID == first.ID || len(store.versions) != 2 {
		t.Fatalf("other model: version = %+v, created = %t, error = %v", third, created, err)
	}
}

func TestRegisterDryRunDoesNotWrite(t *testing.T) {
	spec := contestSpec()

	store := &fakeStore{}
	service, db := newTestService(t, store, nil)
	registered, created, err := service.Register(context.Background(), spec, true)
	if err != nil || created || registered.ID.Valid {
		t.Fatalf("version = %+v, created = %t, error = %v; want a zero version", registered, created, err)
	}
	checkCalls(t, store, "begin", "find", "rollback")
	checkOptions(t, db, readOnly)

	existing := storedVersion(1, false, spec)
	store = &fakeStore{versions: []forecasts_sqlc.ForecastVersion{existing}}
	service, _ = newTestService(t, store, nil)
	registered, created, err = service.Register(context.Background(), spec, true)
	if err != nil || created || registered != existing {
		t.Fatalf("version = %+v, created = %t, error = %v; want the existing version", registered, created, err)
	}
	checkCalls(t, store, "begin", "find", "rollback")
}

func TestRegisterFailuresRollBack(t *testing.T) {
	failure := errors.New("database unavailable")
	spec := contestSpec()
	for _, tt := range []struct {
		name      string
		existing  bool
		fail      string
		dryRun    bool
		wantCalls []string
	}{
		{"create", false, "create", false, []string{"begin", "create", "rollback"}},
		{"find after conflict", true, "find", false, []string{"begin", "create", "find", "rollback"}},
		{"commit", false, "commit", false, []string{"begin", "create", "commit", "rollback"}},
		{"dry run find", false, "find", true, []string{"begin", "find", "rollback"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := &fakeStore{fail: map[string]error{tt.fail: failure}}
			if tt.existing {
				store.versions = []forecasts_sqlc.ForecastVersion{storedVersion(1, false, spec)}
			}
			before := fmt.Sprintf("%+v", store.versions)
			service, _ := newTestService(t, store, nil)
			if _, _, err := service.Register(context.Background(), spec, tt.dryRun); !errors.Is(err, failure) {
				t.Fatalf("error = %v, want the failure", err)
			}
			checkCalls(t, store, tt.wantCalls...)
			if after := fmt.Sprintf("%+v", store.versions); after != before {
				t.Fatalf("versions = %s, want %s", after, before)
			}
		})
	}

	store := &fakeStore{}
	service, db := newTestService(t, store, nil)
	db.beginErr = failure
	if _, _, err := service.Register(context.Background(), spec, false); !errors.Is(err, failure) {
		t.Fatalf("error = %v, want the begin failure", err)
	}
	checkCalls(t, store)
}

func TestActivateSwitchesVersion(t *testing.T) {
	spec := contestSpec()
	old := storedVersion(1, true, spec)
	next := storedVersion(2, false, spec)
	next.ModelVersion = "tabpfn030-next"
	store := &fakeStore{versions: []forecasts_sqlc.ForecastVersion{old, next}}
	service, db := newTestService(t, store, nil)

	activated, changed, err := service.Activate(context.Background(), next.ID)
	if err != nil || !changed || activated.ID != next.ID || !activated.IsActive {
		t.Fatalf("version = %+v, changed = %t, error = %v; want the next version active", activated, changed, err)
	}
	if store.versions[0].IsActive || !store.versions[1].IsActive {
		t.Fatalf("versions = %+v, want only the next one active", store.versions)
	}
	checkCalls(t, store, "begin", "lock", "get", "deactivate", "activate", "commit")
	checkOptions(t, db, readCommitted)
}

func TestActivateFirstVersion(t *testing.T) {
	first := storedVersion(1, false, contestSpec())
	store := &fakeStore{versions: []forecasts_sqlc.ForecastVersion{first}}
	service, _ := newTestService(t, store, nil)
	activated, changed, err := service.Activate(context.Background(), first.ID)
	if err != nil || !changed || !activated.IsActive {
		t.Fatalf("version = %+v, changed = %t, error = %v", activated, changed, err)
	}
	checkCalls(t, store, "begin", "lock", "get", "deactivate", "activate", "commit")
}

func TestActivateAlreadyActiveDoesNotWrite(t *testing.T) {
	active := storedVersion(1, true, contestSpec())
	store := &fakeStore{versions: []forecasts_sqlc.ForecastVersion{active}}
	service, _ := newTestService(t, store, nil)
	got, changed, err := service.Activate(context.Background(), active.ID)
	if err != nil || changed || got != active {
		t.Fatalf("version = %+v, changed = %t, error = %v; want no change", got, changed, err)
	}
	checkCalls(t, store, "begin", "lock", "get", "rollback")
}

func TestActivateMissingVersionKeepsActive(t *testing.T) {
	active := storedVersion(1, true, contestSpec())
	store := &fakeStore{versions: []forecasts_sqlc.ForecastVersion{active}}
	service, _ := newTestService(t, store, nil)
	_, changed, err := service.Activate(context.Background(), pgtype.UUID{Bytes: [16]byte{9}, Valid: true})
	if !errors.Is(err, ErrVersionNotFound) || changed {
		t.Fatalf("changed = %t, error = %v; want ErrVersionNotFound", changed, err)
	}
	if !store.versions[0].IsActive {
		t.Fatal("the active version must stay active")
	}
	checkCalls(t, store, "begin", "lock", "get", "rollback")
}

func TestActivateFailuresRollBack(t *testing.T) {
	failure := errors.New("database unavailable")
	for _, tt := range []struct {
		fail      string
		wantCalls []string
	}{
		{"lock", []string{"begin", "lock", "rollback"}},
		{"get", []string{"begin", "lock", "get", "rollback"}},
		{"deactivate", []string{"begin", "lock", "get", "deactivate", "rollback"}},
		{"activate", []string{"begin", "lock", "get", "deactivate", "activate", "rollback"}},
		{"commit", []string{"begin", "lock", "get", "deactivate", "activate", "commit", "rollback"}},
	} {
		t.Run(tt.fail, func(t *testing.T) {
			spec := contestSpec()
			old := storedVersion(1, true, spec)
			next := storedVersion(2, false, spec)
			store := &fakeStore{versions: []forecasts_sqlc.ForecastVersion{old, next}, fail: map[string]error{tt.fail: failure}}
			service, _ := newTestService(t, store, nil)
			if _, changed, err := service.Activate(context.Background(), next.ID); !errors.Is(err, failure) || changed {
				t.Fatalf("changed = %t, error = %v; want the failure", changed, err)
			}
			checkCalls(t, store, tt.wantCalls...)
			if !store.versions[0].IsActive || store.versions[1].IsActive {
				t.Fatalf("versions = %+v, want the old version still active", store.versions)
			}
		})
	}

	store := &fakeStore{}
	service, db := newTestService(t, store, nil)
	db.beginErr = failure
	if _, _, err := service.Activate(context.Background(), pgtype.UUID{Valid: true}); !errors.Is(err, failure) {
		t.Fatalf("error = %v, want the begin failure", err)
	}
	checkCalls(t, store)
}

func TestActivateCancellationStillRollsBack(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	store := &fakeStore{fail: map[string]error{"lock": context.Canceled}}
	service, _ := newTestService(t, store, nil)
	if _, _, err := service.Activate(ctx, pgtype.UUID{Valid: true}); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want cancellation", err)
	}
	checkCalls(t, store, "begin", "lock", "rollback")
}

func TestApplyOrdersSteps(t *testing.T) {
	spec := contestSpec()
	catalog := enabledRoutes(targetRoutes...)
	var (
		coverage      = []string{"begin", "list routes", "rollback"}
		register      = []string{"begin", "create", "commit"}
		registerAgain = []string{"begin", "create", "find", "commit"}
		lookup        = []string{"begin", "find", "rollback"}
		activate      = []string{"begin", "lock", "get", "deactivate", "activate", "commit"}
		alreadyActive = []string{"begin", "lock", "get", "rollback"}
		verifyAll     = ApplyOptions{Verify: true, DryRun: true, Activate: true}
		verifyAndSave = ApplyOptions{Verify: true, Activate: true}
	)
	for _, tt := range []struct {
		name       string
		existing   []forecasts_sqlc.ForecastVersion
		routes     []routes_sqlc.Route
		predictErr error
		noClient   bool
		fail       string // store call that fails
		opts       ApplyOptions
		// wantErr is empty for success; a failure must leave Outcome zero.
		wantErr      string
		wantCalls    []string
		wantPredicts int
		// wantVersion is whether Outcome.Version is set, wantActive its state.
		wantVersion, wantActive, wantCreated, wantActivated bool
	}{
		{
			name: "register", routes: catalog, opts: ApplyOptions{Verify: true},
			wantCalls: slices.Concat(coverage, register), wantPredicts: 1, wantVersion: true, wantCreated: true,
		},
		{
			name: "register without verification", routes: catalog,
			wantCalls: slices.Concat(coverage, register), wantVersion: true, wantCreated: true,
		},
		{
			name: "register and activate", routes: catalog, opts: verifyAndSave,
			wantCalls: slices.Concat(coverage, register, activate), wantPredicts: 1,
			wantVersion: true, wantActive: true, wantCreated: true, wantActivated: true,
		},
		{
			name: "already registered", existing: []forecasts_sqlc.ForecastVersion{storedVersion(1, false, spec)},
			routes: catalog, opts: verifyAndSave,
			wantCalls: slices.Concat(coverage, registerAgain, activate), wantPredicts: 1,
			wantVersion: true, wantActive: true, wantActivated: true,
		},
		{
			name: "already active", existing: []forecasts_sqlc.ForecastVersion{storedVersion(1, true, spec)},
			routes: catalog, opts: verifyAndSave,
			wantCalls: slices.Concat(coverage, registerAgain, alreadyActive), wantPredicts: 1,
			wantVersion: true, wantActive: true,
		},
		// A dry run only looks the version up, even with Activate.
		{
			name: "dry run", routes: catalog, opts: verifyAll,
			wantCalls: slices.Concat(coverage, lookup), wantPredicts: 1,
		},
		{
			name: "dry run of a registered version", existing: []forecasts_sqlc.ForecastVersion{storedVersion(1, false, spec)},
			routes: catalog, opts: verifyAll,
			wantCalls: slices.Concat(coverage, lookup), wantPredicts: 1, wantVersion: true,
		},
		// Failed checks stop Apply before any write.
		{
			name: "verification fails", routes: catalog, opts: verifyAndSave,
			predictErr: &forecasts_predictor_grpc.Error{Code: "ml_version_mismatch", Err: errors.New("server artifacts differ")},
			wantErr:    "ml_version_mismatch", wantCalls: coverage, wantPredicts: 1,
		},
		{
			name: "no ML client", routes: catalog, noClient: true, opts: verifyAndSave,
			wantErr: "needs an ML client", wantCalls: coverage,
		},
		{
			name: "routes not covered", routes: enabledRoutes(1, 99), opts: verifyAndSave,
			wantErr: "lack forecast-enabled routes 99", wantCalls: coverage,
		},
		{
			name: "catalog not imported", opts: verifyAndSave,
			wantErr: "import the catalog first", wantCalls: coverage,
		},
		// A failed write stops Apply; activation never runs without a version.
		{
			name: "registration fails", routes: catalog, fail: "create", opts: verifyAndSave,
			wantErr: "database unavailable", wantCalls: slices.Concat(coverage, []string{"begin", "create", "rollback"}), wantPredicts: 1,
		},
		{
			name: "activation fails", routes: catalog, fail: "lock", opts: verifyAndSave,
			wantErr:   "is registered but not activated: lock forecast versions: database unavailable",
			wantCalls: slices.Concat(coverage, register, []string{"begin", "lock", "rollback"}), wantPredicts: 1,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := &fakeStore{versions: slices.Clone(tt.existing), routes: tt.routes}
			if tt.fail != "" {
				store.fail = map[string]error{tt.fail: errors.New("database unavailable")}
			}
			predictor := &fakePredictor{err: tt.predictErr}
			var client Predictor = predictor
			if tt.noClient {
				client = nil
			}
			service, _ := newTestService(t, store, client)

			outcome, err := service.Apply(context.Background(), spec, tt.opts)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) || outcome != (Outcome{}) {
					t.Fatalf("outcome = %+v, error = %v; want a zero outcome and %q", outcome, err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			checkCalls(t, store, tt.wantCalls...)
			if len(predictor.requests) != tt.wantPredicts {
				t.Fatalf("Predict calls = %d, want %d", len(predictor.requests), tt.wantPredicts)
			}
			version := outcome.Version
			if version.ID.Valid != tt.wantVersion || version.IsActive != tt.wantActive ||
				outcome.Created != tt.wantCreated || outcome.Activated != tt.wantActivated {
				t.Fatalf("outcome = %+v, want version %t, active %t, created %t, activated %t",
					outcome, tt.wantVersion, tt.wantActive, tt.wantCreated, tt.wantActivated)
			}
			if tt.wantVersion && (version.ModelVersion != spec.ModelVersion || !version.ForecastTo.Time.Equal(spec.ForecastTo)) {
				t.Fatalf("version = %+v, want the spec", version)
			}
			// A failed activation keeps the committed registration, inactive.
			if tt.fail == "lock" && (len(store.versions) != 1 || store.versions[0].IsActive) {
				t.Fatalf("versions = %+v, want the registered version, inactive", store.versions)
			}
		})
	}
}

func TestNewUUID(t *testing.T) {
	first, second := newUUID(), newUUID()
	if !first.Valid || first == second {
		t.Fatalf("ids = %s and %s, want two random UUIDs", first, second)
	}
	for _, id := range []pgtype.UUID{first, second} {
		if id.Bytes[6]>>4 != 4 || id.Bytes[8]>>6 != 0b10 {
			t.Fatalf("id %s is not a version 4 RFC 9562 UUID", id)
		}
	}
}
