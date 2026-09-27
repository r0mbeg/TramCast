package forecasts_transport_http

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
)

const (
	versionID = "0b9c6f1e-3a52-4d7e-9f10-2c4b8a6d5e31"
	jobID     = "8d41e2b0-7c3a-4f65-b1d9-0e5a6c7f2b18"
)

var (
	november = time.Date(2025, 11, 1, 0, 0, 0, 0, core_domain.Moscow)
	january  = time.Date(2026, 1, 1, 0, 0, 0, 0, core_domain.Moscow)
)

func mustUUID(value string) pgtype.UUID {
	id, err := forecasts_service.ParseUUID(value)
	if err != nil {
		panic(err)
	}
	return id
}

// fakeService records the last query and returns the configured results.
type fakeService struct {
	version forecasts_sqlc.ForecastVersion
	result  forecasts_service.QueryResult
	slice   forecasts_service.Slice
	job     forecasts_sqlc.PredictionJob
	err     error

	calls   int
	lastQ   forecasts_service.SliceQuery
	lastJob pgtype.UUID
}

func (f *fakeService) ActiveVersion(context.Context) (forecasts_sqlc.ForecastVersion, error) {
	f.calls++
	return f.version, f.err
}

func (f *fakeService) Query(_ context.Context, q forecasts_service.SliceQuery) (forecasts_service.QueryResult, error) {
	f.calls++
	f.lastQ = q
	return f.result, f.err
}

func (f *fakeService) Slice(_ context.Context, q forecasts_service.SliceQuery) (forecasts_service.Slice, error) {
	f.calls++
	f.lastQ = q
	return f.slice, f.err
}

func (f *fakeService) Job(_ context.Context, id pgtype.UUID) (forecasts_sqlc.PredictionJob, error) {
	f.calls++
	f.lastJob = id
	return f.job, f.err
}

// serve runs one request through the real server with the handler registered.
// A POST body is sent as application/json unless header sets another type.
func serve(t *testing.T, service Service, method, target, body string, header ...string) (*httptest.ResponseRecorder, string) {
	t.Helper()
	var logs bytes.Buffer
	server := core_http_server.New(core_config.HTTPConfig{}, slog.New(slog.NewJSONHandler(&logs, nil)), func(context.Context) error { return nil })
	NewHandler(service).Register(server.Router().Group("/api"))
	request := httptest.NewRequest(method, target, strings.NewReader(body))
	if method == http.MethodPost {
		request.Header.Set("Content-Type", "application/json")
	}
	for i := 0; i+1 < len(header); i += 2 {
		request.Header.Set(header[i], header[i+1])
	}
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	return response, logs.String()
}

func expect(t *testing.T, response *httptest.ResponseRecorder, status int, body string) {
	t.Helper()
	if response.Code != status || response.Body.String() != body {
		t.Fatalf("response = %d %s\nwant %d %s", response.Code, response.Body.String(), status, body)
	}
}

func contestVersion() forecasts_sqlc.ForecastVersion {
	return forecasts_sqlc.ForecastVersion{
		ID: mustUUID(versionID), ModelVersion: "tabpfn030-0123456789abcdef", DatasetVersion: "prepared-844b17f7",
		// pgx returns timestamptz in UTC; the API shows Moscow time.
		HistoryEnd:   pgtype.Timestamptz{Time: november.UTC(), Valid: true},
		ForecastFrom: pgtype.Timestamptz{Time: november.UTC(), Valid: true},
		ForecastTo:   pgtype.Timestamptz{Time: january.UTC(), Valid: true},
		Timezone:     "Europe/Moscow", IsActive: true,
	}
}

// sundayNight spans the night from Sunday 2 to Monday 3 November 2025.
func sundayNight() forecasts_service.Slice {
	from := time.Date(2025, 11, 2, 23, 0, 0, 0, core_domain.Moscow)
	return forecasts_service.Slice{
		Version: contestVersion(), RouteID: 7, From: from, To: from.Add(2 * time.Hour),
		Points: []forecasts_service.Point{{HourStart: from, Boardings: 0}, {HourStart: from.Add(time.Hour), Boardings: 1234}},
	}
}

const sundayNightJSON = `{"forecast_version_id":"` + versionID + `","route_id":7,"timezone":"Europe/Moscow",` +
	`"from":"2025-11-02T23:00:00+03:00","to":"2025-11-03T01:00:00+03:00","points":[` +
	`{"date":"2025-11-02","weekday":7,"hour":23,"boardings":0},{"date":"2025-11-03","weekday":1,"hour":0,"boardings":1234}]}`

func job(status string) forecasts_sqlc.PredictionJob {
	return forecasts_sqlc.PredictionJob{
		ID: mustUUID(jobID), ForecastVersionID: mustUUID(versionID), RouteID: 7, Status: status, AttemptCount: 2,
		LastErrorCode: pgtype.Text{String: "ml_internal", Valid: true}, LastErrorMessage: pgtype.Text{String: "secret-detail", Valid: true},
	}
}

func TestActiveVersion(t *testing.T) {
	response, _ := serve(t, &fakeService{version: contestVersion()}, http.MethodGet, "/api/forecast-versions/active", "")
	expect(t, response, http.StatusOK, `{"id":"`+versionID+`","model_version":"tabpfn030-0123456789abcdef","dataset_version":"prepared-844b17f7",`+
		`"history_end":"2025-11-01T00:00:00+03:00","forecast_from":"2025-11-01T00:00:00+03:00","forecast_to":"2026-01-01T00:00:00+03:00","timezone":"Europe/Moscow"}`)

	response, _ = serve(t, &fakeService{err: fmt.Errorf("read: %w", forecasts_service.ErrNoActiveVersion)}, http.MethodGet, "/api/forecast-versions/active", "")
	expect(t, response, http.StatusServiceUnavailable, `{"error":"no_active_forecast_version"}`)
}

func TestQueryReturnsTheSlice(t *testing.T) {
	slice := sundayNight()
	service := &fakeService{result: forecasts_service.QueryResult{Slice: &slice, Job: job("succeeded")}}
	response, _ := serve(t, service, http.MethodPost, "/api/predictions/query",
		`{"route_id":7,"from":"2025-11-02T20:00:00Z","to":"2025-11-03T01:00:00+03:00"}`,
		"Content-Type", "application/json; charset=utf-8")
	expect(t, response, http.StatusOK, sundayNightJSON)

	want := forecasts_service.SliceQuery{RouteID: 7, From: slice.From, To: slice.To}
	if q := service.lastQ; q.VersionID.Valid || q.RouteID != want.RouteID || !q.From.Equal(want.From) || !q.To.Equal(want.To) {
		t.Fatalf("query = %+v, want %+v without a version", q, want)
	}
}

func TestQueryPassesTheVersion(t *testing.T) {
	for _, body := range []string{
		`{"route_id":7,"from":"2025-11-01T00:00:00+03:00","to":"2025-11-02T00:00:00+03:00","forecast_version_id":"` + versionID + `"}`,
		`{"route_id":7,"from":"2025-11-01T00:00:00+03:00","to":"2025-11-02T00:00:00+03:00","forecast_version_id":"` + strings.ToUpper(versionID) + `"}`,
	} {
		service := &fakeService{result: forecasts_service.QueryResult{Job: job("queued")}}
		serve(t, service, http.MethodPost, "/api/predictions/query", body)
		if service.lastQ.VersionID != mustUUID(versionID) {
			t.Fatalf("version = %s, want %s", service.lastQ.VersionID, versionID)
		}
	}
	// null means the active version, like an absent field.
	service := &fakeService{result: forecasts_service.QueryResult{Job: job("queued")}}
	serve(t, service, http.MethodPost, "/api/predictions/query",
		`{"route_id":7,"from":"2025-11-01T00:00:00+03:00","to":"2025-11-02T00:00:00+03:00","forecast_version_id":null}`)
	if service.calls != 1 || service.lastQ.VersionID.Valid {
		t.Fatalf("calls = %d, version = %s; want the active version", service.calls, service.lastQ.VersionID)
	}
}

func TestQueryAcceptsTheJob(t *testing.T) {
	body := `{"route_id":7,"from":"2025-11-01T00:00:00+03:00","to":"2026-01-01T00:00:00+03:00"}`
	for _, status := range []string{"queued", "running"} {
		response, logs := serve(t, &fakeService{result: forecasts_service.QueryResult{Job: job(status)}}, http.MethodPost, "/api/predictions/query", body)
		expect(t, response, http.StatusAccepted,
			`{"job_id":"`+jobID+`","forecast_version_id":"`+versionID+`","route_id":7,"status":"`+status+`","poll_interval_seconds":2}`)
		if strings.Contains(logs, "secret-detail") {
			t.Fatalf("job details leaked into the log: %s", logs)
		}
	}

	response, _ := serve(t, &fakeService{result: forecasts_service.QueryResult{Job: job("failed")}}, http.MethodPost, "/api/predictions/query", body)
	expect(t, response, http.StatusConflict, `{"error":"prediction_failed","job_id":"`+jobID+`"}`)
}

func TestQueryRejectsBadRequests(t *testing.T) {
	const from, to = `"from":"2025-11-01T00:00:00+03:00"`, `"to":"2025-11-02T00:00:00+03:00"`
	for _, tt := range []struct {
		name        string
		body        string
		contentType string
		status      int
		code        string
	}{
		{"no content type", `{"route_id":7,` + from + `,` + to + `}`, "-", http.StatusUnsupportedMediaType, "unsupported_media_type"},
		{"form content type", `route_id=7`, "application/x-www-form-urlencoded", http.StatusUnsupportedMediaType, "unsupported_media_type"},
		{"text content type", `{"route_id":7,` + from + `,` + to + `}`, "text/plain", http.StatusUnsupportedMediaType, "unsupported_media_type"},
		{"malformed JSON", `{"route_id":7,`, "", http.StatusBadRequest, "invalid_request"},
		{"empty body", ``, "", http.StatusBadRequest, "invalid_request"},
		{"unknown field", `{"route_id":7,"route":7,` + from + `,` + to + `}`, "", http.StatusBadRequest, "invalid_request"},
		{"two values", `{"route_id":7,` + from + `,` + to + `}{}`, "", http.StatusBadRequest, "invalid_request"},
		{"too large", `{"route_id":7,` + from + `,` + to + `,"forecast_version_id":"` + strings.Repeat(" ", maxQueryBody) + `"}`, "", http.StatusBadRequest, "invalid_request"},
		{"route as string", `{"route_id":"7",` + from + `,` + to + `}`, "", http.StatusBadRequest, "invalid_request"},
		{"fractional route", `{"route_id":7.5,` + from + `,` + to + `}`, "", http.StatusBadRequest, "invalid_request"},
		{"missing route", `{` + from + `,` + to + `}`, "", http.StatusBadRequest, "invalid_route_id"},
		{"negative route", `{"route_id":-7,` + from + `,` + to + `}`, "", http.StatusBadRequest, "invalid_route_id"},
		{"empty version", `{"route_id":7,` + from + `,` + to + `,"forecast_version_id":""}`, "", http.StatusBadRequest, "invalid_forecast_version_id"},
		{"short version", `{"route_id":7,` + from + `,` + to + `,"forecast_version_id":"0b9c6f1e3a524d7e9f102c4b8a6d5e31"}`, "", http.StatusBadRequest, "invalid_forecast_version_id"},
		{"missing from", `{"route_id":7,` + to + `}`, "", http.StatusBadRequest, "invalid_interval"},
		{"date only", `{"route_id":7,"from":"2025-11-01",` + to + `}`, "", http.StatusBadRequest, "invalid_interval"},
		{"no offset", `{"route_id":7,"from":"2025-11-01T00:00:00",` + to + `}`, "", http.StatusBadRequest, "invalid_interval"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			service := &fakeService{}
			var header []string
			switch tt.contentType {
			case "":
			case "-":
				header = []string{"Content-Type", ""}
			default:
				header = []string{"Content-Type", tt.contentType}
			}
			response, _ := serve(t, service, http.MethodPost, "/api/predictions/query", tt.body, header...)
			expect(t, response, tt.status, `{"error":"`+tt.code+`"}`)
			if service.calls != 0 {
				t.Fatal("the service must not be called for a bad request")
			}
		})
	}
}

func TestSlice(t *testing.T) {
	service := &fakeService{slice: sundayNight()}
	query := url.Values{
		"forecast_version_id": {versionID}, "route_id": {"7"},
		"from": {"2025-11-02T23:00:00+03:00"}, "to": {"2025-11-03T01:00:00+03:00"},
	}
	response, _ := serve(t, service, http.MethodGet, "/api/predictions?"+query.Encode(), "")
	expect(t, response, http.StatusOK, sundayNightJSON)
	if q := service.lastQ; q.VersionID != mustUUID(versionID) || q.RouteID != 7 || !q.From.Equal(sundayNight().From) || !q.To.Equal(sundayNight().To) {
		t.Fatalf("query = %+v", q)
	}
}

func TestSliceRejectsBadParameters(t *testing.T) {
	valid := "from=2025-11-02T23%3A00%3A00%2B03%3A00&to=2025-11-03T01%3A00%3A00Z"
	for _, tt := range []struct {
		name, query, code string
	}{
		{"no version", "route_id=7&" + valid, "invalid_forecast_version_id"},
		{"bad version", "forecast_version_id=abc&route_id=7&" + valid, "invalid_forecast_version_id"},
		{"no route", "forecast_version_id=" + versionID + "&" + valid, "invalid_route_id"},
		{"bad route", "forecast_version_id=" + versionID + "&route_id=abc&" + valid, "invalid_route_id"},
		{"zero route", "forecast_version_id=" + versionID + "&route_id=0&" + valid, "invalid_route_id"},
		{"no interval", "forecast_version_id=" + versionID + "&route_id=7", "invalid_interval"},
		// An unencoded + decodes as a space.
		{"unencoded plus", "forecast_version_id=" + versionID + "&route_id=7&from=2025-11-02T23:00:00+03:00&to=2025-11-03T01:00:00Z", "invalid_interval"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			service := &fakeService{}
			response, _ := serve(t, service, http.MethodGet, "/api/predictions?"+tt.query, "")
			expect(t, response, http.StatusBadRequest, `{"error":"`+tt.code+`"}`)
			if service.calls != 0 {
				t.Fatal("the service must not be called for bad parameters")
			}
		})
	}
}

func TestJob(t *testing.T) {
	service := &fakeService{job: job("running")}
	response, _ := serve(t, service, http.MethodGet, "/api/prediction-jobs/"+strings.ToUpper(jobID), "")
	expect(t, response, http.StatusOK, `{"job_id":"`+jobID+`","forecast_version_id":"`+versionID+`","route_id":7,"status":"running","poll_interval_seconds":2}`)
	if service.lastJob != mustUUID(jobID) {
		t.Fatalf("job ID = %s", service.lastJob)
	}
	for _, status := range []string{"succeeded", "failed"} {
		response, _ := serve(t, &fakeService{job: job(status)}, http.MethodGet, "/api/prediction-jobs/"+jobID, "")
		expect(t, response, http.StatusOK, `{"job_id":"`+jobID+`","forecast_version_id":"`+versionID+`","route_id":7,"status":"`+status+`","poll_interval_seconds":null}`)
	}

	service = &fakeService{}
	response, _ = serve(t, service, http.MethodGet, "/api/prediction-jobs/42", "")
	expect(t, response, http.StatusBadRequest, `{"error":"invalid_job_id"}`)
	if service.calls != 0 {
		t.Fatal("the service must not be called for a bad job ID")
	}
	response, _ = serve(t, &fakeService{err: forecasts_service.ErrJobNotFound}, http.MethodGet, "/api/prediction-jobs/"+jobID, "")
	expect(t, response, http.StatusNotFound, `{"error":"prediction_job_not_found"}`)
}

func TestServiceErrors(t *testing.T) {
	queryBody := `{"route_id":7,"from":"2025-11-01T00:00:00+03:00","to":"2025-11-02T00:00:00+03:00"}`
	sliceTarget := "/api/predictions?forecast_version_id=" + versionID + "&route_id=7&from=2025-11-01T00%3A00%3A00Z&to=2025-11-02T00%3A00%3A00Z"
	for _, tt := range []struct {
		err    error
		status int
		code   string
	}{
		{forecasts_service.ErrInvalidInterval, http.StatusBadRequest, "invalid_interval"},
		{forecasts_service.ErrVersionNotFound, http.StatusNotFound, "forecast_version_not_found"},
		{forecasts_service.ErrRouteNotFound, http.StatusNotFound, "route_not_found"},
		{forecasts_service.ErrPredictionNotReady, http.StatusConflict, "prediction_not_ready"},
		{forecasts_service.ErrVersionInactive, http.StatusConflict, "forecast_version_inactive"},
		{forecasts_service.ErrForecastDisabled, http.StatusUnprocessableEntity, "forecast_disabled"},
		{forecasts_service.ErrQueueFull, http.StatusServiceUnavailable, "queue_full"},
		{forecasts_service.ErrNoActiveVersion, http.StatusServiceUnavailable, "no_active_forecast_version"},
	} {
		t.Run(tt.code, func(t *testing.T) {
			err := fmt.Errorf("context: %w", tt.err)
			for _, response := range []*httptest.ResponseRecorder{
				first(serve(t, &fakeService{err: err}, http.MethodPost, "/api/predictions/query", queryBody)),
				first(serve(t, &fakeService{err: err}, http.MethodGet, sliceTarget, "")),
			} {
				expect(t, response, tt.status, `{"error":"`+tt.code+`"}`)
				if retry := response.Header().Get("Retry-After"); (retry != "") != (tt.code == "queue_full") || (retry != "" && retry != "30") {
					t.Fatalf("Retry-After = %q", retry)
				}
			}
		})
	}
}

func first(response *httptest.ResponseRecorder, _ string) *httptest.ResponseRecorder { return response }

func TestInternalErrorsAreNotExposed(t *testing.T) {
	queryBody := `{"route_id":7,"from":"2025-11-01T00:00:00+03:00","to":"2025-11-02T00:00:00+03:00"}`
	for _, request := range []struct{ method, target, body string }{
		{http.MethodGet, "/api/forecast-versions/active", ""},
		{http.MethodPost, "/api/predictions/query", queryBody},
		{http.MethodGet, "/api/predictions?forecast_version_id=" + versionID + "&route_id=7&from=2025-11-01T00%3A00%3A00Z&to=2025-11-02T00%3A00%3A00Z", ""},
		{http.MethodGet, "/api/prediction-jobs/" + jobID, ""},
	} {
		t.Run(request.target, func(t *testing.T) {
			response, logs := serve(t, &fakeService{err: errors.New("database-detail-secret")}, request.method, request.target, request.body)
			expect(t, response, http.StatusInternalServerError, `{"error":"internal_server_error"}`)
			if !strings.Contains(logs, "database-detail-secret") {
				t.Fatalf("internal error must be logged: %s", logs)
			}
		})
	}
}
