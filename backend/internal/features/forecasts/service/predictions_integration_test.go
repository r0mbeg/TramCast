package forecasts_service_test

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

// forecasts activates the fixture version and returns the read service over
// the pool, wired as cmd/tramcast wires it.
func (f *dbFixture) forecasts(t *testing.T, queue *forecasts_service.Queue) *forecasts_service.Forecasts {
	t.Helper()
	version, _, err := forecasts_service.NewService(f.pool, nil).Activate(t.Context(), f.version.ID)
	if err != nil {
		t.Fatal(err)
	}
	f.version = version
	return forecasts_service.NewForecasts(queue, routes_sqlc.New(f.pool), forecasts_sqlc.New(f.pool))
}

func TestForecastsQueryAdmitsThenReadsThePublishedSlice(t *testing.T) {
	f := newDBFixture(t, 1)
	queue := f.queue(dbPolicy)
	forecasts := f.forecasts(t, queue)
	ctx := t.Context()
	full := forecasts_service.SliceQuery{RouteID: f.routes[1], From: horizonFrom, To: horizonTo}

	result, err := forecasts.Query(ctx, full)
	if err != nil || result.Slice != nil || result.Job.Status != "queued" {
		t.Fatalf("first query = %+v, %v; want a queued job", result, err)
	}
	pinned := full
	pinned.VersionID = f.version.ID
	if _, err := forecasts.Slice(ctx, pinned); !errors.Is(err, forecasts_service.ErrPredictionNotReady) {
		t.Fatalf("slice before publication: %v, want ErrPredictionNotReady", err)
	}

	claimed := claim(t, queue)
	if err := queue.Publish(ctx, claimed, horizon()); err != nil {
		t.Fatal(err)
	}
	job, err := forecasts.Job(ctx, result.Job.ID)
	if err != nil || job.Status != "succeeded" {
		t.Fatalf("job = %s, %v; want succeeded", job.Status, err)
	}

	result, err = forecasts.Query(ctx, full)
	if err != nil || result.Slice == nil || len(result.Slice.Points) != horizonHours {
		t.Fatalf("query after publication = %+v, %v; want %d hours", result.Job, err, horizonHours)
	}
	for i, point := range result.Slice.Points {
		if want := horizonFrom.Add(time.Duration(i) * time.Hour); !point.HourStart.Equal(want) || point.Boardings != int64(i) {
			t.Fatalf("point %d = %+v, want %s with %d boardings", i, point, want, i)
		}
	}

	// The last hours of the year end at the next date: the SQL row bound is
	// (2026-01-01, 0).
	tail := pinned
	tail.From = time.Date(2025, 12, 31, 20, 0, 0, 0, core_domain.Moscow)
	slice, err := forecasts.Slice(ctx, tail)
	if err != nil || len(slice.Points) != 4 || slice.Points[0].Boardings != horizonHours-4 || slice.Points[3].Boardings != horizonHours-1 {
		t.Fatalf("tail slice = %+v, %v; want the last 4 hours", slice.Points, err)
	}
	if n := f.count(t, "SELECT count(*) FROM prediction_jobs"); n != 1 {
		t.Fatalf("%d jobs stored, want 1", n)
	}

	// A lost hour of a succeeded job is an integrity error, never a partial slice.
	f.exec(t, `DELETE FROM validation_predictions WHERE forecast_version_id = $1 AND route_id = $2 AND date = '2025-12-31' AND hour = 21`,
		f.version.ID, f.routes[1])
	if _, err := forecasts.Slice(ctx, tail); err == nil || errors.Is(err, forecasts_service.ErrPredictionNotReady) {
		t.Fatalf("slice with a missing hour: %v, want an integrity error", err)
	}
}

func TestForecastsConcurrentQueriesShareOneJob(t *testing.T) {
	f := newDBFixture(t, 7)
	forecasts := f.forecasts(t, f.queue(dbPolicy))
	q := forecasts_service.SliceQuery{RouteID: f.routes[7], From: horizonFrom, To: horizonFrom.Add(24 * time.Hour)}

	ids := make([]pgtype.UUID, 12)
	errs := make([]error, len(ids))
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range ids {
		wg.Go(func() {
			<-start
			result, err := forecasts.Query(t.Context(), q)
			ids[i], errs[i] = result.Job.ID, err
		})
	}
	close(start)
	wg.Wait()
	for i := range ids {
		if errs[i] != nil || ids[i] != ids[0] {
			t.Fatalf("caller %d got job %s, %v; caller 0 %s", i, ids[i], errs[i], ids[0])
		}
	}
	if n := f.count(t, "SELECT count(*) FROM prediction_jobs"); n != 1 {
		t.Fatalf("%d jobs stored, want 1", n)
	}
}

func TestForecastsInactiveVersionGetsNoNewJob(t *testing.T) {
	f := newDBFixture(t, 1)
	forecasts := f.forecasts(t, f.queue(dbPolicy))
	previous, err := forecasts_sqlc.New(f.pool).CreateForecastVersion(t.Context(), forecasts_sqlc.CreateForecastVersionParams{
		ID: newID(), ModelVersion: "model-previous", DatasetVersion: "dataset-it",
		HistoryEnd:   pgtype.Timestamptz{Time: horizonFrom, Valid: true},
		ForecastFrom: pgtype.Timestamptz{Time: horizonFrom, Valid: true},
		ForecastTo:   pgtype.Timestamptz{Time: horizonTo, Valid: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	q := forecasts_service.SliceQuery{VersionID: previous.ID, RouteID: f.routes[1], From: horizonFrom, To: horizonTo}
	if _, err := forecasts.Query(t.Context(), q); !errors.Is(err, forecasts_service.ErrVersionInactive) {
		t.Fatalf("query of the inactive version: %v, want ErrVersionInactive", err)
	}
	if n := f.count(t, "SELECT count(*) FROM prediction_jobs"); n != 0 {
		t.Fatalf("%d jobs stored, want none", n)
	}
}
