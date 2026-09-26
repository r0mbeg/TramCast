package routes_transport_http

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgtype"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
	routes_service "github.com/r0mbeg/TramCast/backend/internal/features/routes/service"
)

type fakeService struct {
	routes      []routes_sqlc.Route
	patterns    []routes_service.Pattern
	err         error
	lastRouteID int64
}

func (f *fakeService) ListRoutes(context.Context) ([]routes_sqlc.Route, error) {
	return f.routes, f.err
}

func (f *fakeService) ListRouteStops(_ context.Context, routeID int64) ([]routes_service.Pattern, error) {
	f.lastRouteID = routeID
	return f.patterns, f.err
}

func serve(t *testing.T, service Service, target string) (*httptest.ResponseRecorder, string) {
	t.Helper()
	var logs bytes.Buffer
	server := core_http_server.New(core_config.HTTPConfig{}, slog.New(slog.NewJSONHandler(&logs, nil)), func(context.Context) error { return nil })
	NewHandler(service).Register(server.Router().Group("/api"))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, httptest.NewRequest(http.MethodGet, target, nil))
	return response, logs.String()
}

func TestListRoutes(t *testing.T) {
	service := &fakeService{routes: []routes_sqlc.Route{
		{ID: 7, RouteNumber: 1, Name: pgtype.Text{String: "Route one", Valid: true}, SourceRouteID: pgtype.Text{String: "4450", Valid: true}, ForecastEnabled: true},
		{ID: 1488, RouteNumber: 50},
	}}
	response, _ := serve(t, service, "/api/routes")
	want := `{"routes":[{"id":7,"route_number":1,"name":"Route one","forecast_enabled":true},{"id":1488,"route_number":50,"name":null,"forecast_enabled":false}]}`
	if response.Code != http.StatusOK || response.Body.String() != want {
		t.Fatalf("response = %d %s\nwant 200 %s", response.Code, response.Body.String(), want)
	}
}

func TestListRoutesEmpty(t *testing.T) {
	response, _ := serve(t, &fakeService{routes: []routes_sqlc.Route{}}, "/api/routes")
	if response.Code != http.StatusOK || response.Body.String() != `{"routes":[]}` {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestListRouteStops(t *testing.T) {
	service := &fakeService{patterns: []routes_service.Pattern{{
		Key:         "4450_0",
		DirectionID: 0,
		Stops: []routes_sqlc.ListRouteStopsRow{
			{RouteID: 7, PatternKey: "4450_0", StopSequence: 2, StopID: 10, SourceStopID: "src-10", Name: "Depot", Latitude: 55.75, Longitude: 37.6},
		},
	}}}
	response, _ := serve(t, service, "/api/routes/7/stops")
	want := `{"route_id":7,"patterns":[{"pattern_key":"4450_0","direction_id":0,"stops":[{"stop_sequence":2,"stop_id":10,"name":"Depot","latitude":55.75,"longitude":37.6}]}]}`
	if response.Code != http.StatusOK || response.Body.String() != want {
		t.Fatalf("response = %d %s\nwant 200 %s", response.Code, response.Body.String(), want)
	}
	if service.lastRouteID != 7 {
		t.Fatalf("route ID = %d, want 7", service.lastRouteID)
	}
}

func TestListRouteStopsWithoutGeography(t *testing.T) {
	response, _ := serve(t, &fakeService{patterns: []routes_service.Pattern{}}, "/api/routes/7/stops")
	if response.Code != http.StatusOK || response.Body.String() != `{"route_id":7,"patterns":[]}` {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestListRouteStopsInvalidRouteID(t *testing.T) {
	for _, routeID := range []string{"abc", "0", "-1", "1.5", "9223372036854775808"} {
		t.Run(routeID, func(t *testing.T) {
			service := &fakeService{}
			response, _ := serve(t, service, "/api/routes/"+routeID+"/stops")
			if response.Code != http.StatusBadRequest || response.Body.String() != `{"error":"invalid_route_id"}` {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			if service.lastRouteID != 0 {
				t.Fatal("service must not be called for an invalid route ID")
			}
		})
	}
}

func TestListRouteStopsUnknownRoute(t *testing.T) {
	response, _ := serve(t, &fakeService{err: routes_service.ErrRouteNotFound}, "/api/routes/404/stops")
	if response.Code != http.StatusNotFound || response.Body.String() != `{"error":"route_not_found"}` {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestInternalErrorsAreNotExposed(t *testing.T) {
	for _, target := range []string{"/api/routes", "/api/routes/7/stops"} {
		t.Run(target, func(t *testing.T) {
			response, logs := serve(t, &fakeService{err: errors.New("database-detail-secret")}, target)
			if response.Code != http.StatusInternalServerError || response.Body.String() != `{"error":"internal_server_error"}` {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			if !strings.Contains(logs, "database-detail-secret") {
				t.Fatalf("internal error must be logged: %s", logs)
			}
		})
	}
}
