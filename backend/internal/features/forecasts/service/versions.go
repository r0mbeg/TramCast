// Package forecasts_service registers forecast versions, switches the active
// one, runs the prediction job queue and serves the published forecasts.
package forecasts_service

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

// verifyRouteNumber is a structural zero in the ML contract: the server answers
// it without running the model, so the probe is cheap on a GPU host too.
const verifyRouteNumber int16 = 5

var (
	ErrVersionNotFound = errors.New("forecast version not found")
	// ErrReplayNotAllowed rejects replay metadata without an explicit opt-in:
	// a replay bundle serves a saved result and is not a live model version.
	ErrReplayNotAllowed = errors.New("metadata comes from a replay bundle")
)

// VersionSpec is a forecast version as the ML service describes it.
type VersionSpec struct {
	ModelVersion, DatasetVersion, Timezone string
	// HistoryEnd excludes the forecast; [ForecastFrom, ForecastTo) is the
	// horizon every job of the version computes.
	HistoryEnd, ForecastFrom, ForecastTo time.Time
	// RouteNumbers are the routes the ML service serves.
	RouteNumbers []int16
	// Replay means the metadata came from a replay bundle: it has
	// serving_mode or prediction_file.
	Replay bool
}

// Predictor is satisfied by *forecasts_predictor_grpc.Client, which fails
// unless the server reports exactly the requested model and dataset versions
// and the full hourly grid.
type Predictor interface {
	Predict(ctx context.Context, r forecasts_predictor_grpc.Request) (forecasts_predictor_grpc.Result, error)
}

// TxBeginner is satisfied by *pgxpool.Pool.
type TxBeginner interface {
	BeginTx(ctx context.Context, options pgx.TxOptions) (pgx.Tx, error)
}

// versionQueries and routeQueries are the parts of the generated sqlc packages
// used here; their *Queries types satisfy them directly.
type versionQueries interface {
	CreateForecastVersion(ctx context.Context, arg forecasts_sqlc.CreateForecastVersionParams) (forecasts_sqlc.ForecastVersion, error)
	GetForecastVersion(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error)
	GetForecastVersionByArtifacts(ctx context.Context, arg forecasts_sqlc.GetForecastVersionByArtifactsParams) (forecasts_sqlc.ForecastVersion, error)
	LockForecastVersionsForActivation(ctx context.Context) error
	DeactivateForecastVersions(ctx context.Context) error
	ActivateForecastVersion(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error)
}

type routeQueries interface {
	ListRoutes(ctx context.Context) ([]routes_sqlc.Route, error)
}

// queries groups the storage of one transaction.
type queries struct {
	versions versionQueries
	routes   routeQueries
}

func newQueries(tx pgx.Tx) queries {
	return queries{versions: forecasts_sqlc.New(tx), routes: routes_sqlc.New(tx)}
}

type Service struct {
	db        TxBeginner
	predictor Predictor
	// queries binds storage to a transaction; tests replace it.
	queries func(pgx.Tx) queries
}

// NewService returns the version service. predictor may be nil when Verify is
// not used.
func NewService(db TxBeginner, predictor Predictor) *Service {
	return &Service{db: db, predictor: predictor, queries: newQueries}
}

// ApplyOptions select the steps of Apply.
type ApplyOptions struct {
	// Verify calls the ML server before any write; the service needs a
	// Predictor for it.
	Verify bool
	// DryRun only looks the version up: nothing is written or activated.
	DryRun bool
	// Activate makes the version active for new requests unless DryRun is set.
	Activate bool
}

// Outcome is what Apply did.
type Outcome struct {
	// Version is the registered version in its final state. After a dry run it
	// is the existing version, or a zero one (ID.Valid false) when spec is not
	// registered yet.
	Version forecasts_sqlc.ForecastVersion
	// Created and Activated report whether this call inserted the version and
	// changed the active one.
	Created, Activated bool
}

// CheckSpec checks spec without the database: the forecast_versions schema
// rules and the one-call horizon of the ML server. Replay metadata fails with
// ErrReplayNotAllowed unless allowReplay is set. Call it before Apply.
func CheckSpec(spec VersionSpec, allowReplay bool) error {
	if err := spec.check(); err != nil {
		return err
	}
	if spec.Replay && !allowReplay {
		return ErrReplayNotAllowed
	}
	return nil
}

// Apply registers a spec that passed CheckSpec. Each step runs only after the
// previous one succeeded: route coverage, then the ML server check if
// opts.Verify, then registration, then activation if opts.Activate. So nothing
// is written for a version the checks reject, and a dry run writes nothing.
func (s *Service) Apply(ctx context.Context, spec VersionSpec, opts ApplyOptions) (Outcome, error) {
	if err := s.checkCoverage(ctx, spec); err != nil {
		return Outcome{}, err
	}
	if opts.Verify {
		if err := s.Verify(ctx, spec); err != nil {
			return Outcome{}, err
		}
	}
	var outcome Outcome
	var err error
	if outcome.Version, outcome.Created, err = s.Register(ctx, spec, opts.DryRun); err != nil {
		return Outcome{}, err
	}
	if opts.Activate && !opts.DryRun {
		id := outcome.Version.ID
		if outcome.Version, outcome.Activated, err = s.Activate(ctx, id); err != nil {
			// Register has already committed: say so, or the log reads as if
			// nothing was written.
			return Outcome{}, fmt.Errorf("version %s is registered but not activated: %w", id, err)
		}
	}
	return outcome, nil
}

// checkCoverage checks that the ML service serves every route with forecasts
// enabled, since Go requests each of them. Without such routes the catalog is
// not imported yet and coverage cannot be checked, so it fails too.
func (s *Service) checkCoverage(ctx context.Context, spec VersionSpec) error {
	enabled := 0
	var missing []string
	err := s.readOnly(ctx, func(q queries) error {
		routes, err := q.routes.ListRoutes(ctx)
		if err != nil {
			return fmt.Errorf("read the route catalog: %w", err)
		}
		for _, route := range routes {
			if !route.ForecastEnabled {
				continue
			}
			enabled++
			if !slices.Contains(spec.RouteNumbers, route.RouteNumber) {
				missing = append(missing, strconv.Itoa(int(route.RouteNumber)))
			}
		}
		return nil
	})
	switch {
	case err != nil:
		return err
	case enabled == 0:
		return errors.New("no forecast-enabled routes: import the catalog first")
	case len(missing) > 0:
		return fmt.Errorf("metadata route_numbers lack forecast-enabled routes %s: the ML service must serve every route Go requests",
			strings.Join(missing, ", "))
	}
	return nil
}

func (spec VersionSpec) check() error {
	if spec.Timezone != core_domain.Timezone {
		return fmt.Errorf("timezone %q must be %s", spec.Timezone, core_domain.Timezone)
	}
	// The gRPC client rejects IDs with surrounding spaces in a response, so
	// such a version could never be computed.
	for _, id := range []struct{ name, value string }{
		{"model_version", spec.ModelVersion},
		{"dataset_version", spec.DatasetVersion},
	} {
		if id.value == "" || strings.TrimSpace(id.value) != id.value {
			return fmt.Errorf("%s %q must be non-blank without surrounding spaces", id.name, id.value)
		}
	}
	switch {
	case spec.HistoryEnd.IsZero() || spec.ForecastFrom.IsZero() || spec.ForecastTo.IsZero():
		return errors.New("history_end, forecast_from and forecast_to must be set")
	case spec.HistoryEnd.After(spec.ForecastFrom) || !spec.ForecastFrom.Before(spec.ForecastTo):
		return fmt.Errorf("period must satisfy history_end <= forecast_from < forecast_to, got %s, %s and %s",
			spec.HistoryEnd.Format(time.RFC3339), spec.ForecastFrom.Format(time.RFC3339), spec.ForecastTo.Format(time.RFC3339))
	case !core_domain.IsHourAligned(spec.ForecastFrom) || !core_domain.IsHourAligned(spec.ForecastTo):
		return errors.New("forecast_from and forecast_to must start whole Moscow hours")
	case spec.ForecastTo.Sub(spec.ForecastFrom) > forecasts_predictor_grpc.MaxHours*time.Hour:
		// A job computes the whole horizon of its version in one call.
		return fmt.Errorf("horizon of %s exceeds %d hours", spec.ForecastTo.Sub(spec.ForecastFrom), forecasts_predictor_grpc.MaxHours)
	}
	return nil
}

// Verify asks the ML server for route 5 over the whole horizon with the spec
// versions. The Predictor fails unless the server serves this period with
// exactly these artifacts, so a version the running server cannot compute is
// not activated: every job of it would fail terminally.
func (s *Service) Verify(ctx context.Context, spec VersionSpec) error {
	if s.predictor == nil {
		return errors.New("verification needs an ML client")
	}
	if !slices.Contains(spec.RouteNumbers, verifyRouteNumber) {
		return fmt.Errorf("metadata route_numbers lack route %d, the verification route", verifyRouteNumber)
	}
	// ponytail: route 5 proves the IDs and the period, not that the model
	// runs; add a model probe if activation must prove the GPU works.
	if _, err := s.predictor.Predict(ctx, forecasts_predictor_grpc.Request{
		RouteNumber:    verifyRouteNumber,
		From:           spec.ForecastFrom,
		To:             spec.ForecastTo,
		ModelVersion:   spec.ModelVersion,
		DatasetVersion: spec.DatasetVersion,
	}); err != nil {
		return fmt.Errorf("verify the ML server with route %d: %w", verifyRouteNumber, err)
	}
	return nil
}

// Register stores spec as an inactive version and reports whether this call
// created it. The same artifacts and period registered again, even
// concurrently, return the existing version in its current state. With dryRun
// nothing is written: the result is the existing version, or a zero version
// (ID.Valid false) when spec is not registered yet. Apply runs it after the
// checks.
func (s *Service) Register(ctx context.Context, spec VersionSpec, dryRun bool) (forecasts_sqlc.ForecastVersion, bool, error) {
	artifacts := forecasts_sqlc.GetForecastVersionByArtifactsParams{
		ModelVersion:   spec.ModelVersion,
		DatasetVersion: spec.DatasetVersion,
		HistoryEnd:     timestamptz(spec.HistoryEnd),
		ForecastFrom:   timestamptz(spec.ForecastFrom),
		ForecastTo:     timestamptz(spec.ForecastTo),
	}
	if dryRun {
		var version forecasts_sqlc.ForecastVersion
		err := s.readOnly(ctx, func(q queries) error {
			existing, err := q.versions.GetForecastVersionByArtifacts(ctx, artifacts)
			switch {
			case errors.Is(err, pgx.ErrNoRows):
				return nil
			case err != nil:
				return fmt.Errorf("find forecast version: %w", err)
			}
			version = existing
			return nil
		})
		return version, false, err
	}

	tx, err := s.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("begin version registration: %w", err)
	}
	// After Commit this is a no-op; otherwise it undoes the insert.
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	q := s.queries(tx).versions
	version, err := q.CreateForecastVersion(ctx, forecasts_sqlc.CreateForecastVersionParams{
		ID:             newUUID(),
		ModelVersion:   artifacts.ModelVersion,
		DatasetVersion: artifacts.DatasetVersion,
		HistoryEnd:     artifacts.HistoryEnd,
		ForecastFrom:   artifacts.ForecastFrom,
		ForecastTo:     artifacts.ForecastTo,
	})
	created := err == nil
	if errors.Is(err, pgx.ErrNoRows) {
		// The artifacts key conflicted. Under READ COMMITTED this new statement
		// sees the committed row, even one inserted concurrently.
		if version, err = q.GetForecastVersionByArtifacts(ctx, artifacts); err != nil {
			return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("find the registered forecast version: %w", err)
		}
	} else if err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("create forecast version: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("commit version registration: %w", err)
	}
	return version, created, nil
}

// Activate makes the version the default for new requests in one READ
// COMMITTED transaction under the activation lock and reports whether the
// active version changed. An already active version is returned without
// writes. A missing ID fails with ErrVersionNotFound and keeps the current
// active version.
func (s *Service) Activate(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, bool, error) {
	tx, err := s.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("begin version activation: %w", err)
	}
	// After Commit this is a no-op; otherwise it undoes every write.
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	q := s.queries(tx).versions
	if err := q.LockForecastVersionsForActivation(ctx); err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("lock forecast versions: %w", err)
	}
	version, err := q.GetForecastVersion(ctx, id)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("%w: %s", ErrVersionNotFound, id)
	case err != nil:
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("get forecast version: %w", err)
	case version.IsActive:
		return version, false, nil
	}
	// The partial unique index allows one active version: deactivate first.
	if err := q.DeactivateForecastVersions(ctx); err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("deactivate forecast versions: %w", err)
	}
	activated, err := q.ActivateForecastVersion(ctx, id)
	if err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("activate forecast version: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return forecasts_sqlc.ForecastVersion{}, false, fmt.Errorf("commit version activation: %w", err)
	}
	return activated, true, nil
}

// readOnly runs fn in a READ ONLY transaction, which is always rolled back.
func (s *Service) readOnly(ctx context.Context, fn func(queries) error) error {
	tx, err := s.db.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted, AccessMode: pgx.ReadOnly})
	if err != nil {
		return fmt.Errorf("begin read-only transaction: %w", err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	return fn(s.queries(tx))
}

func timestamptz(t time.Time) pgtype.Timestamptz {
	return pgtype.Timestamptz{Time: t, Valid: true}
}

// newUUID returns a random UUID (version 4); crypto/rand.Read never fails.
func newUUID() pgtype.UUID {
	id := pgtype.UUID{Valid: true}
	_, _ = rand.Read(id.Bytes[:])
	id.Bytes[6] = id.Bytes[6]&0x0f | 0x40 // version 4
	id.Bytes[8] = id.Bytes[8]&0x3f | 0x80 // RFC 9562 variant
	return id
}
