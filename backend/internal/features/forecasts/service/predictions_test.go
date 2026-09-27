package forecasts_service

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
)

// ListValidationPredictions mirrors the SQL: rows of the version and route
// with (date, hour) in [from, to), ordered by date and hour.
func (s *fakeQueue) ListValidationPredictions(_ context.Context, arg forecasts_sqlc.ListValidationPredictionsParams) ([]forecasts_sqlc.ValidationPrediction, error) {
	if err := s.call("list predictions"); err != nil {
		return nil, err
	}
	s.listArgs = append(s.listArgs, arg)
	compare := func(date time.Time, hour int16, otherDate time.Time, otherHour int16) int {
		if c := date.Compare(otherDate); c != 0 {
			return c
		}
		return int(hour) - int(otherHour)
	}
	rows := []forecasts_sqlc.ValidationPrediction{}
	for _, p := range s.predictions {
		if p.ForecastVersionID == arg.ForecastVersionID && p.RouteID == arg.RouteID &&
			compare(p.Date.Time, p.Hour, arg.DateFrom.Time, arg.HourFrom) >= 0 &&
			compare(p.Date.Time, p.Hour, arg.DateTo.Time, arg.HourTo) < 0 {
			rows = append(rows, forecasts_sqlc.ValidationPrediction{
				ForecastVersionID: p.ForecastVersionID, RouteID: p.RouteID, Date: p.Date, Hour: p.Hour, Boardings: p.Boardings,
			})
		}
	}
	slices.SortStableFunc(rows, func(a, b forecasts_sqlc.ValidationPrediction) int {
		return compare(a.Date.Time, a.Hour, b.Date.Time, b.Hour)
	})
	return rows, nil
}

// publish adds job n of the version and route as succeeded, with the full
// horizon stored the way Publish stores it.
func (s *fakeQueue) publish(t *testing.T, n int, version forecasts_sqlc.ForecastVersion, routeID int64) forecasts_sqlc.PredictionJob {
	t.Helper()
	job := s.seed(n, routeID, "succeeded", 1)
	job.ForecastVersionID = version.ID
	s.jobs[len(s.jobs)-1] = job
	rows, err := predictionRows(s.claimed(job), horizonPoints(version))
	if err != nil {
		t.Fatal(err)
	}
	s.predictions = append(s.predictions, rows...)
	return job
}

func newTestForecasts(t *testing.T, store *fakeQueue, policy JobPolicy) *Forecasts {
	t.Helper()
	queue, _ := newTestQueue(t, store, policy)
	return NewForecasts(queue, store, store)
}

func moscowHour(year int, month time.Month, day, hour int) time.Time {
	return time.Date(year, month, day, hour, 0, 0, 0, core_domain.Moscow)
}

// fullHorizon is the horizon of the fixture version, bounds in UTC.
func fullHorizon(routeID int64) SliceQuery {
	return SliceQuery{RouteID: routeID, From: novemberStart.UTC(), To: januaryStart.UTC()}
}

// Reads of Query before it decides: the version, then the route, then the job.
var queryReads = []string{"begin read-only", "get active version", "rollback", "get route", "begin read-only", "find job", "rollback"}

func TestQueryReturnsTheSliceOfASucceededJob(t *testing.T) {
	store := queueFixture()
	job := store.publish(t, 1, store.versions[0], 101)
	forecasts := newTestForecasts(t, store, testPolicy)

	result, err := forecasts.Query(context.Background(), fullHorizon(101))
	if err != nil {
		t.Fatal(err)
	}
	if result.Slice == nil || result.Job != job {
		t.Fatalf("result = %+v, want the slice of job 1", result)
	}
	slice := result.Slice
	if slice.Version != store.versions[0] || slice.RouteID != 101 || len(slice.Points) != 1464 {
		t.Fatalf("slice of version %s route %d has %d points", slice.Version.ID, slice.RouteID, len(slice.Points))
	}
	if slice.From.Location() != core_domain.Moscow || !slice.From.Equal(novemberStart) || !slice.To.Equal(januaryStart) {
		t.Fatalf("slice bounds %s to %s, want the horizon in Moscow time", slice.From, slice.To)
	}
	for i, point := range slice.Points {
		if want := novemberStart.Add(time.Duration(i) * time.Hour); !point.HourStart.Equal(want) || point.Boardings != int64(i%7) ||
			point.HourStart.Location() != core_domain.Moscow {
			t.Fatalf("point %d = %+v, want %s with %d boardings", i, point, want, i%7)
		}
	}
	// A found job needs no admission and its lock.
	queueCalls(t, store, append(slices.Clone(queryReads), "list predictions")...)
}

func TestQueryReadsASubinterval(t *testing.T) {
	store := queueFixture()
	store.publish(t, 1, store.versions[0], 101)
	forecasts := newTestForecasts(t, store, testPolicy)

	// The last hours of the year end at the next date: the SQL bound is
	// 2026-01-01 hour 0.
	q := SliceQuery{RouteID: 101, From: moscowHour(2025, 12, 31, 20), To: moscowHour(2026, 1, 1, 0).UTC()}
	result, err := forecasts.Query(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	var hours []int
	for _, point := range result.Slice.Points {
		hours = append(hours, point.HourStart.Hour())
	}
	if !slices.Equal(hours, []int{20, 21, 22, 23}) {
		t.Fatalf("hours = %v, want 20 to 23", hours)
	}
	want := forecasts_sqlc.ListValidationPredictionsParams{
		ForecastVersionID: store.versions[0].ID, RouteID: 101,
		DateFrom: localDate(moscowHour(2025, 12, 31, 0)), HourFrom: 20,
		DateTo: localDate(moscowHour(2026, 1, 1, 0)), HourTo: 0,
	}
	if len(store.listArgs) != 1 || store.listArgs[0] != want {
		t.Fatalf("list params = %+v, want %+v", store.listArgs, want)
	}
}

func TestQueryAdmitsTheFirstJobOfThePair(t *testing.T) {
	store := queueFixture()
	forecasts := newTestForecasts(t, store, testPolicy)
	result, err := forecasts.Query(context.Background(), fullHorizon(100))
	if err != nil {
		t.Fatal(err)
	}
	if result.Slice != nil || result.Job.Status != "queued" || result.Job.RouteID != 100 || result.Job.ForecastVersionID != store.versions[0].ID {
		t.Fatalf("result = %+v, want a new queued job", result)
	}
	if len(store.jobs) != 1 || store.jobs[0] != result.Job {
		t.Fatalf("stored jobs = %+v", store.jobs)
	}
}

func TestQueryReportsAnUnfinishedOrFailedJob(t *testing.T) {
	for _, status := range []string{"queued", "running", "failed"} {
		t.Run(status, func(t *testing.T) {
			store := queueFixture()
			job := store.seed(1, 100, status, 1)
			forecasts := newTestForecasts(t, store, testPolicy)
			result, err := forecasts.Query(context.Background(), fullHorizon(100))
			if err != nil || result.Slice != nil || result.Job != job {
				t.Fatalf("result = %+v, error = %v; want job 1 as stored", result, err)
			}
			// Polling neither admits again nor restarts a failed job.
			queueCalls(t, store, queryReads...)
		})
	}
}

func TestQueryReturnsTheJobAConcurrentRequestAdmitted(t *testing.T) {
	for _, status := range []string{"queued", "succeeded"} {
		t.Run(status, func(t *testing.T) {
			store := queueFixture()
			queue, db := newTestQueue(t, store, testPolicy)
			var concurrent forecasts_sqlc.PredictionJob
			store.before = map[string]func(){"lock admission": func() {
				if status == "succeeded" {
					concurrent = store.publish(t, 9, store.versions[0], 100)
				} else {
					concurrent = store.seed(9, 100, status, 0)
				}
				// Another transaction committed it: the rollback of the
				// admission keeps it.
				db.tx.jobs, db.tx.predictions = slices.Clone(store.jobs), slices.Clone(store.predictions)
			}}
			forecasts := NewForecasts(queue, store, store)
			result, err := forecasts.Query(context.Background(), fullHorizon(100))
			if err != nil || result.Job != concurrent || (result.Slice != nil) != (status == "succeeded") {
				t.Fatalf("result = %+v, error = %v; want the concurrent %s job", result, err, status)
			}
			if len(store.jobs) != 1 {
				t.Fatalf("stored jobs = %d, want one per pair", len(store.jobs))
			}
		})
	}
}

func TestQueryPinsTheRequestedVersion(t *testing.T) {
	store := queueFixture()
	previous := storedVersion(2, false, contestSpec())
	previous.ModelVersion = "tabpfn030-previous"
	store.versions = append(store.versions, previous)
	store.publish(t, 1, previous, 101)
	forecasts := newTestForecasts(t, store, testPolicy)

	q := fullHorizon(101)
	q.VersionID = previous.ID
	result, err := forecasts.Query(context.Background(), q)
	if err != nil || result.Slice == nil || result.Slice.Version != previous {
		t.Fatalf("result = %+v, error = %v; want the slice of the inactive version", result, err)
	}

	// An inactive version gets no new job: the worker would fail it for good.
	q.RouteID = 100
	if _, err := forecasts.Query(context.Background(), q); !errors.Is(err, ErrVersionInactive) {
		t.Fatalf("error = %v, want ErrVersionInactive", err)
	}
	if len(store.jobs) != 1 {
		t.Fatalf("stored jobs = %d, want no new job", len(store.jobs))
	}
}

func TestQueryRejectsBadRequests(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*fakeQueue, *SliceQuery, *JobPolicy)
		want   error
	}{
		{"unknown version", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.VersionID = jobID(99) }, ErrVersionNotFound},
		{"no active version", func(s *fakeQueue, _ *SliceQuery, _ *JobPolicy) { s.versions[0].IsActive = false }, ErrNoActiveVersion},
		{"half hour", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.From = q.From.Add(30 * time.Minute) }, ErrInvalidInterval},
		{"fractional second", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.To = q.To.Add(-time.Nanosecond) }, ErrInvalidInterval},
		{"empty interval", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.To = q.From }, ErrInvalidInterval},
		{"reversed interval", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) {
			q.From, q.To = moscowHour(2025, 11, 2, 0), moscowHour(2025, 11, 1, 0)
		}, ErrInvalidInterval},
		{"before the horizon", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.From = q.From.Add(-time.Hour) }, ErrInvalidInterval},
		{"after the horizon", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.To = q.To.Add(time.Hour) }, ErrInvalidInterval},
		{"unknown route", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.RouteID = 999 }, ErrRouteNotFound},
		{"forecasts disabled", func(_ *fakeQueue, q *SliceQuery, _ *JobPolicy) { q.RouteID = 200 }, ErrForecastDisabled},
		{"queue full", func(s *fakeQueue, _ *SliceQuery, p *JobPolicy) {
			s.seed(1, 101, "queued", 0)
			p.QueueCapacity = 1
		}, ErrQueueFull},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			q, policy := fullHorizon(100), testPolicy
			tt.change(store, &q, &policy)
			jobs := len(store.jobs)
			forecasts := newTestForecasts(t, store, policy)
			if result, err := forecasts.Query(context.Background(), q); !errors.Is(err, tt.want) {
				t.Fatalf("result = %+v, error = %v; want %v", result, err, tt.want)
			}
			if len(store.jobs) != jobs {
				t.Fatalf("stored jobs = %d, want %d", len(store.jobs), jobs)
			}
		})
	}
}

func TestSliceNeedsASucceededJob(t *testing.T) {
	for _, status := range []string{"", "queued", "running", "failed"} {
		t.Run("job "+status, func(t *testing.T) {
			store := queueFixture()
			if status != "" {
				store.seed(1, 100, status, 1)
			}
			jobs := slices.Clone(store.jobs)
			forecasts := newTestForecasts(t, store, testPolicy)
			q := fullHorizon(100)
			q.VersionID = store.versions[0].ID
			if _, err := forecasts.Slice(context.Background(), q); !errors.Is(err, ErrPredictionNotReady) {
				t.Fatalf("error = %v, want ErrPredictionNotReady", err)
			}
			// Reading never admits.
			if !slices.Equal(store.jobs, jobs) {
				t.Fatalf("jobs = %+v, want them unchanged", store.jobs)
			}
			queueCalls(t, store, "begin read-only", "get version", "rollback", "get route", "begin read-only", "find job", "rollback")
		})
	}

	store := queueFixture()
	store.versions[0].IsActive = false
	store.publish(t, 1, store.versions[0], 100)
	forecasts := newTestForecasts(t, store, testPolicy)
	q := SliceQuery{VersionID: store.versions[0].ID, RouteID: 100, From: moscowHour(2025, 11, 3, 0), To: moscowHour(2025, 11, 4, 0)}
	slice, err := forecasts.Slice(context.Background(), q)
	if err != nil || len(slice.Points) != 24 || !slice.Points[0].HourStart.Equal(q.From) {
		t.Fatalf("slice = %+v, error = %v; want 24 hours of an inactive version", slice, err)
	}
}

func TestSliceRejectsBadRequests(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*SliceQuery)
		want   error
	}{
		{"unknown version", func(q *SliceQuery) { q.VersionID = jobID(99) }, ErrVersionNotFound},
		{"outside the horizon", func(q *SliceQuery) { q.To = q.To.Add(time.Hour) }, ErrInvalidInterval},
		{"unknown route", func(q *SliceQuery) { q.RouteID = 999 }, ErrRouteNotFound},
		{"forecasts disabled", func(q *SliceQuery) { q.RouteID = 200 }, ErrForecastDisabled},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			q := fullHorizon(100)
			q.VersionID = store.versions[0].ID
			tt.change(&q)
			if _, err := newTestForecasts(t, store, testPolicy).Slice(context.Background(), q); !errors.Is(err, tt.want) {
				t.Fatalf("error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestSliceOfASucceededJobMustBeComplete(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func([]forecasts_sqlc.CopyValidationPredictionsParams) []forecasts_sqlc.CopyValidationPredictionsParams
		want   string
	}{
		{"missing hour", func(rows []forecasts_sqlc.CopyValidationPredictionsParams) []forecasts_sqlc.CopyValidationPredictionsParams {
			return slices.Delete(rows, 30, 31)
		}, "has 23 of 24 hours"},
		{"hour stored twice", func(rows []forecasts_sqlc.CopyValidationPredictionsParams) []forecasts_sqlc.CopyValidationPredictionsParams {
			rows[30].Hour = rows[29].Hour
			return rows
		}, "lacks hour 2025-11-02T06:00:00+03:00"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := queueFixture()
			store.publish(t, 1, store.versions[0], 100)
			store.predictions = tt.change(store.predictions)
			forecasts := newTestForecasts(t, store, testPolicy)
			q := SliceQuery{VersionID: store.versions[0].ID, RouteID: 100, From: moscowHour(2025, 11, 2, 0), To: moscowHour(2025, 11, 3, 0)}
			_, err := forecasts.Slice(context.Background(), q)
			if err == nil || !strings.Contains(err.Error(), tt.want) || errors.Is(err, ErrPredictionNotReady) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
			if _, err := forecasts.Query(context.Background(), q); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("query error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestReadFailures(t *testing.T) {
	failure := errors.New("database unavailable")
	for _, name := range []string{"get active version", "get route", "find job", "list predictions"} {
		t.Run(name, func(t *testing.T) {
			store := queueFixture()
			store.publish(t, 1, store.versions[0], 100)
			store.fail = map[string]error{name: failure}
			if _, err := newTestForecasts(t, store, testPolicy).Query(context.Background(), fullHorizon(100)); !errors.Is(err, failure) {
				t.Fatalf("error = %v, want the failure", err)
			}
		})
	}
}

func TestJobReadsWithoutWriting(t *testing.T) {
	store := queueFixture()
	job := store.seed(1, 100, "running", 1)
	forecasts := newTestForecasts(t, store, testPolicy)
	if got, err := forecasts.Job(context.Background(), job.ID); err != nil || got != job {
		t.Fatalf("job = %+v, error = %v; want job 1", got, err)
	}
	if _, err := forecasts.Job(context.Background(), jobID(2)); !errors.Is(err, ErrJobNotFound) {
		t.Fatalf("error = %v, want ErrJobNotFound", err)
	}
	queueCalls(t, store, "begin read-only", "get job", "rollback", "begin read-only", "get job", "rollback")
}

func TestParseUUID(t *testing.T) {
	const canonical = "0b9c6f1e-3a52-4d7e-9f10-2c4b8a6d5e31"
	for _, value := range []string{canonical, strings.ToUpper(canonical)} {
		id, err := ParseUUID(value)
		if err != nil || id.String() != canonical {
			t.Fatalf("ParseUUID(%q) = %s, %v", value, id, err)
		}
	}
	for _, value := range []string{
		"", "0b9c6f1e3a524d7e9f102c4b8a6d5e31", "0b9c6f1e_3a52_4d7e_9f10_2c4b8a6d5e31",
		"0b9c6f1e-3a52-4d7e-9f10-2c4b8a6d5e3", "{0b9c6f1e-3a52-4d7e-9f10-2c4b8a6d5e31}", "0b9c6f1e-3a52-4d7e-9f10-2c4b8a6d5e3g",
	} {
		if id, err := ParseUUID(value); err == nil || id.Valid {
			t.Fatalf("ParseUUID(%q) = %s, want an error", value, id)
		}
	}
	var zero pgtype.UUID
	if id, _ := ParseUUID("nope"); id != zero {
		t.Fatalf("invalid input returned %s", id)
	}
}
