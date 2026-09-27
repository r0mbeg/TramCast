package forecasts_service

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

var (
	// ErrInvalidInterval rejects bounds that are not whole Moscow hours, an
	// empty interval or one outside the horizon of the version.
	ErrInvalidInterval = errors.New("invalid forecast interval")
	// ErrPredictionNotReady means the pair has no succeeded job; reading does
	// not create one.
	ErrPredictionNotReady = errors.New("prediction is not ready")
	// ErrVersionInactive refuses a new job of an inactive version: the worker
	// computes only the active one, so the job would fail for good and block
	// the pair of that version forever.
	ErrVersionInactive = errors.New("forecast version is not active")
)

// Point is one published hour of a slice.
type Point struct {
	HourStart time.Time // in core_domain.Moscow
	Boardings int64
}

// Slice is the published forecast of a route over [From, To): every hour in
// order, zeros included.
type Slice struct {
	Version  forecasts_sqlc.ForecastVersion
	RouteID  int64
	From, To time.Time // in core_domain.Moscow
	Points   []Point
}

// SliceQuery names a route and an interval of a forecast version.
type SliceQuery struct {
	// VersionID is not Valid when the request names no version: Query then
	// pins the active one.
	VersionID pgtype.UUID
	RouteID   int64
	From, To  time.Time
}

// QueryResult is the answer to Query: a ready slice or the job of the pair.
type QueryResult struct {
	// Slice is set when the job of the pair has succeeded.
	Slice *Slice
	// Job is the job of the pair. Without a Slice its status is queued,
	// running or failed; failed is never restarted.
	Job forecasts_sqlc.PredictionJob
}

// forecastJobs is the part of *Queue the reads use.
type forecastJobs interface {
	ActiveVersion(ctx context.Context) (forecasts_sqlc.ForecastVersion, error)
	Version(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error)
	Lookup(ctx context.Context, versionID pgtype.UUID, routeID int64) (forecasts_sqlc.PredictionJob, bool, error)
	Admit(ctx context.Context, versionID pgtype.UUID, routeID int64) (forecasts_sqlc.PredictionJob, bool, error)
	Job(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.PredictionJob, error)
}

// routeGetter and predictionReader are satisfied by the sqlc *Queries types.
type routeGetter interface {
	GetRouteByID(ctx context.Context, routeID int64) (routes_sqlc.Route, error)
}

type predictionReader interface {
	ListValidationPredictions(ctx context.Context, arg forecasts_sqlc.ListValidationPredictionsParams) ([]forecasts_sqlc.ValidationPrediction, error)
}

// Forecasts answers the forecast requests of users: it pins a version, checks
// the interval and the route, reads the slice of a succeeded job and otherwise
// admits or reports the job of the pair. Every statement reads committed data;
// published points are immutable, so a succeeded job has all of them.
type Forecasts struct {
	jobs        forecastJobs
	routes      routeGetter
	predictions predictionReader
}

func NewForecasts(jobs forecastJobs, routes routeGetter, predictions predictionReader) *Forecasts {
	return &Forecasts{jobs: jobs, routes: routes, predictions: predictions}
}

// ActiveVersion returns the active version or ErrNoActiveVersion.
func (f *Forecasts) ActiveVersion(ctx context.Context) (forecasts_sqlc.ForecastVersion, error) {
	return f.jobs.ActiveVersion(ctx)
}

// Job returns the job or ErrJobNotFound; reading it starts no work.
func (f *Forecasts) Job(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.PredictionJob, error) {
	return f.jobs.Job(ctx, id)
}

// Query returns the slice when the job of the pair has succeeded, and
// otherwise the job, admitting a new one when there is none. The version is
// the requested one or, without it, the active one read once here. It fails
// with ErrVersionNotFound, ErrNoActiveVersion, ErrInvalidInterval,
// ErrRouteNotFound, ErrForecastDisabled, ErrVersionInactive or ErrQueueFull.
func (f *Forecasts) Query(ctx context.Context, q SliceQuery) (QueryResult, error) {
	var (
		version forecasts_sqlc.ForecastVersion
		err     error
	)
	if q.VersionID.Valid {
		version, err = f.jobs.Version(ctx, q.VersionID)
	} else {
		version, err = f.jobs.ActiveVersion(ctx)
	}
	if err != nil {
		return QueryResult{}, err
	}
	if err := f.check(ctx, version, q); err != nil {
		return QueryResult{}, err
	}
	// Most requests find a job; the lookup avoids the admission table lock.
	job, found, err := f.jobs.Lookup(ctx, version.ID, q.RouteID)
	if err != nil {
		return QueryResult{}, err
	}
	if !found {
		// ponytail: a switch of the active version right after this read may
		// still admit a job of the old one, which then fails with
		// ml_version_mismatch; lock the versions in Admit if switches get frequent.
		if !version.IsActive {
			return QueryResult{}, fmt.Errorf("%w: %s", ErrVersionInactive, version.ID)
		}
		// A concurrent request may have admitted the pair meanwhile: Admit then
		// returns that job in whatever status it has.
		if job, _, err = f.jobs.Admit(ctx, version.ID, q.RouteID); err != nil {
			return QueryResult{}, err
		}
	}
	if job.Status != "succeeded" {
		return QueryResult{Job: job}, nil
	}
	slice, err := f.read(ctx, version, q)
	if err != nil {
		return QueryResult{}, err
	}
	return QueryResult{Slice: &slice, Job: job}, nil
}

// Slice reads the published slice of the version and route. Without a
// succeeded job it fails with ErrPredictionNotReady and creates nothing; the
// other errors are those of Query.
func (f *Forecasts) Slice(ctx context.Context, q SliceQuery) (Slice, error) {
	version, err := f.jobs.Version(ctx, q.VersionID)
	if err != nil {
		return Slice{}, err
	}
	if err := f.check(ctx, version, q); err != nil {
		return Slice{}, err
	}
	job, found, err := f.jobs.Lookup(ctx, version.ID, q.RouteID)
	switch {
	case err != nil:
		return Slice{}, err
	case !found:
		return Slice{}, fmt.Errorf("%w: route %d has no job in version %s", ErrPredictionNotReady, q.RouteID, version.ID)
	case job.Status != "succeeded":
		return Slice{}, fmt.Errorf("%w: job %s is %s", ErrPredictionNotReady, job.ID, job.Status)
	}
	return f.read(ctx, version, q)
}

// check applies the interval rules of the version, then the route rules.
func (f *Forecasts) check(ctx context.Context, version forecasts_sqlc.ForecastVersion, q SliceQuery) error {
	switch {
	case !core_domain.IsHourAligned(q.From) || !core_domain.IsHourAligned(q.To):
		return fmt.Errorf("%w: bounds must start whole Moscow hours", ErrInvalidInterval)
	case !q.From.Before(q.To):
		return fmt.Errorf("%w: from must precede to", ErrInvalidInterval)
	case q.From.Before(version.ForecastFrom.Time) || q.To.After(version.ForecastTo.Time):
		return fmt.Errorf("%w: [%s, %s) is outside the horizon of version %s", ErrInvalidInterval,
			q.From.Format(time.RFC3339), q.To.Format(time.RFC3339), version.ID)
	}
	route, err := f.routes.GetRouteByID(ctx, q.RouteID)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return fmt.Errorf("%w: %d", ErrRouteNotFound, q.RouteID)
	case err != nil:
		return fmt.Errorf("get route: %w", err)
	case !route.ForecastEnabled:
		return fmt.Errorf("%w: route %d", ErrForecastDisabled, route.RouteNumber)
	}
	return nil
}

// read returns the published hours of a succeeded job over the checked
// interval. A missing, extra or misplaced hour breaks the rule that a job
// succeeds with its full horizon, so it is an error, never a partial slice.
func (f *Forecasts) read(ctx context.Context, version forecasts_sqlc.ForecastVersion, q SliceQuery) (Slice, error) {
	from, to := q.From.In(core_domain.Moscow), q.To.In(core_domain.Moscow)
	rows, err := f.predictions.ListValidationPredictions(ctx, forecasts_sqlc.ListValidationPredictionsParams{
		ForecastVersionID: version.ID,
		RouteID:           q.RouteID,
		DateFrom:          localDate(from),
		HourFrom:          int16(from.Hour()),
		DateTo:            localDate(to),
		HourTo:            int16(to.Hour()),
	})
	if err != nil {
		return Slice{}, fmt.Errorf("list predictions: %w", err)
	}
	hours := int(to.Sub(from) / time.Hour)
	if len(rows) != hours {
		return Slice{}, fmt.Errorf("succeeded job of version %s and route %d has %d of %d hours in [%s, %s)",
			version.ID, q.RouteID, len(rows), hours, from.Format(time.RFC3339), to.Format(time.RFC3339))
	}
	points := make([]Point, hours)
	for i, row := range rows {
		// ponytail: Moscow keeps a fixed offset, so adding hours never skips
		// or repeats a local hour; a zone with DST would need the wall clock.
		hour := from.Add(time.Duration(i) * time.Hour)
		year, month, day := hour.Date()
		rowYear, rowMonth, rowDay := row.Date.Time.Date()
		if !row.Date.Valid || rowYear != year || rowMonth != month || rowDay != day || int(row.Hour) != hour.Hour() {
			return Slice{}, fmt.Errorf("succeeded job of version %s and route %d lacks hour %s",
				version.ID, q.RouteID, hour.Format(time.RFC3339))
		}
		points[i] = Point{HourStart: hour, Boardings: row.Boardings}
	}
	return Slice{Version: version, RouteID: q.RouteID, From: from, To: to, Points: points}, nil
}

// localDate is the calendar date of t in its own location as a pgtype.Date.
// pgx encodes the year, month and day of the time; UTC midnight also equals
// the values it decodes.
func localDate(t time.Time) pgtype.Date {
	year, month, day := t.Date()
	return pgtype.Date{Time: time.Date(year, month, day, 0, 0, 0, 0, time.UTC), Valid: true}
}

// ParseUUID accepts only the canonical 36-character form: pgtype also takes
// 32 hex digits and ignores the characters at the dash positions.
func ParseUUID(value string) (pgtype.UUID, error) {
	var id pgtype.UUID
	if err := id.Scan(value); err != nil || !strings.EqualFold(id.String(), value) {
		return pgtype.UUID{}, errors.New("want a UUID such as 0b9c6f1e-3a52-4d7e-9f10-2c4b8a6d5e31")
	}
	return id, nil
}
