package stops_transport_http

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
)

type fakeService struct {
	stops []stops_sqlc.Stop
	err   error
}

func (f fakeService) ListStops(context.Context) ([]stops_sqlc.Stop, error) {
	return f.stops, f.err
}

func serve(t *testing.T, service Service) (*httptest.ResponseRecorder, string) {
	t.Helper()
	var logs bytes.Buffer
	server := core_http_server.New(core_config.HTTPConfig{}, slog.New(slog.NewJSONHandler(&logs, nil)), func(context.Context) error { return nil })
	NewHandler(service).Register(server.Router().Group("/api"))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/stops", nil))
	return response, logs.String()
}

func TestListStops(t *testing.T) {
	response, _ := serve(t, fakeService{stops: []stops_sqlc.Stop{
		{ID: 10, SourceStopID: "src-10", Name: "Depot", Latitude: 55.75, Longitude: 37.6},
	}})
	want := `{"stops":[{"id":10,"name":"Depot","latitude":55.75,"longitude":37.6}]}`
	if response.Code != http.StatusOK || response.Body.String() != want {
		t.Fatalf("response = %d %s\nwant 200 %s", response.Code, response.Body.String(), want)
	}
}

func TestListStopsEmpty(t *testing.T) {
	response, _ := serve(t, fakeService{stops: []stops_sqlc.Stop{}})
	if response.Code != http.StatusOK || response.Body.String() != `{"stops":[]}` {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestListStopsInternalErrorIsNotExposed(t *testing.T) {
	response, logs := serve(t, fakeService{err: errors.New("database-detail-secret")})
	if response.Code != http.StatusInternalServerError || response.Body.String() != `{"error":"internal_server_error"}` {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
	if !strings.Contains(logs, "database-detail-secret") {
		t.Fatalf("internal error must be logged: %s", logs)
	}
}
