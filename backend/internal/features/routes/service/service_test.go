package routes_service

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/jackc/pgx/v5"

	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

type fakeQueries struct {
	routes      []routes_sqlc.Route
	routeErr    error
	stops       []routes_sqlc.ListRouteStopsRow
	stopsErr    error
	listErr     error
	stopCalls   int
	geometry    []routes_sqlc.ListRouteGeometryRow
	geometryErr error
}

func (f *fakeQueries) ListRouteGeometry(context.Context) ([]routes_sqlc.ListRouteGeometryRow, error) {
	return f.geometry, f.geometryErr
}

func (f *fakeQueries) ListRoutes(context.Context) ([]routes_sqlc.Route, error) {
	return f.routes, f.listErr
}

func (f *fakeQueries) GetRouteByID(_ context.Context, routeID int64) (routes_sqlc.Route, error) {
	if f.routeErr != nil {
		return routes_sqlc.Route{}, f.routeErr
	}
	return routes_sqlc.Route{ID: routeID}, nil
}

func (f *fakeQueries) ListRouteStops(context.Context, int64) ([]routes_sqlc.ListRouteStopsRow, error) {
	f.stopCalls++
	return f.stops, f.stopsErr
}

func stopRow(pattern string, direction int16, sequence int32, stopID int64) routes_sqlc.ListRouteStopsRow {
	return routes_sqlc.ListRouteStopsRow{RouteID: 1, PatternKey: pattern, DirectionID: direction, StopSequence: sequence, StopID: stopID}
}

func TestListRouteStopsGroupsAdjacentVariants(t *testing.T) {
	queries := &fakeQueries{stops: []routes_sqlc.ListRouteStopsRow{
		stopRow("a", 0, 1, 10), stopRow("a", 0, 3, 11),
		stopRow("b", 0, 1, 12),
		stopRow("c", 1, 0, 11), stopRow("c", 1, 1, 10),
	}}
	patterns, err := NewService(queries).ListRouteStops(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	got := fmt.Sprint(summarize(patterns))
	want := "[a/0:[10 11] b/0:[12] c/1:[11 10]]"
	if got != want {
		t.Fatalf("patterns = %s, want %s", got, want)
	}
}

func TestListRouteStopsDoesNotMergeDirectionsOfOneKey(t *testing.T) {
	queries := &fakeQueries{stops: []routes_sqlc.ListRouteStopsRow{stopRow("a", 0, 1, 10), stopRow("a", 1, 1, 11)}}
	patterns, err := NewService(queries).ListRouteStops(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(summarize(patterns)); got != "[a/0:[10] a/1:[11]]" {
		t.Fatalf("patterns = %s", got)
	}
}

func TestListRouteStopsWithoutGeographyIsEmpty(t *testing.T) {
	patterns, err := NewService(&fakeQueries{stops: []routes_sqlc.ListRouteStopsRow{}}).ListRouteStops(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if patterns == nil || len(patterns) != 0 {
		t.Fatalf("patterns = %#v, want an empty non-nil slice", patterns)
	}
}

func TestListRouteStopsUnknownRoute(t *testing.T) {
	queries := &fakeQueries{routeErr: fmt.Errorf("scan: %w", pgx.ErrNoRows)}
	_, err := NewService(queries).ListRouteStops(context.Background(), 404)
	if !errors.Is(err, ErrRouteNotFound) {
		t.Fatalf("error = %v, want ErrRouteNotFound", err)
	}
	if queries.stopCalls != 0 {
		t.Fatal("stops must not be read for an unknown route")
	}
}

func TestListRouteStopsPropagatesDatabaseErrors(t *testing.T) {
	failure := errors.New("connection reset")
	for name, queries := range map[string]*fakeQueries{
		"route lookup": {routeErr: failure},
		"stops query":  {stopsErr: failure},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := NewService(queries).ListRouteStops(context.Background(), 1)
			if !errors.Is(err, failure) || errors.Is(err, ErrRouteNotFound) {
				t.Fatalf("error = %v, want wrapped database error", err)
			}
		})
	}
}

func TestListRoutesPropagatesDatabaseErrors(t *testing.T) {
	failure := errors.New("connection reset")
	_, err := NewService(&fakeQueries{listErr: failure}).ListRoutes(context.Background())
	if !errors.Is(err, failure) {
		t.Fatalf("error = %v, want wrapped database error", err)
	}
}

func summarize(patterns []Pattern) []string {
	result := make([]string, len(patterns))
	for i, pattern := range patterns {
		stopIDs := make([]int64, len(pattern.Stops))
		for j, stop := range pattern.Stops {
			stopIDs[j] = stop.StopID
		}
		result[i] = fmt.Sprintf("%s/%d:%v", pattern.Key, pattern.DirectionID, stopIDs)
	}
	return result
}

func geometryRow(routeID int64, number int16, pattern string, direction int16, sequence int32, lon, lat float64) routes_sqlc.ListRouteGeometryRow {
	return routes_sqlc.ListRouteGeometryRow{RouteID: routeID, RouteNumber: number, ForecastEnabled: number == 1,
		PatternKey: pattern, DirectionID: direction, StopSequence: sequence, Longitude: lon, Latitude: lat}
}

func TestListRouteGeometryBuildsLinesPerVariant(t *testing.T) {
	queries := &fakeQueries{geometry: []routes_sqlc.ListRouteGeometryRow{
		geometryRow(16, 1, "a", 0, 1, 37.59, 55.59), geometryRow(16, 1, "a", 0, 2, 37.58, 55.60),
		geometryRow(16, 1, "b", 1, 1, 37.58, 55.60), geometryRow(16, 1, "b", 1, 2, 37.59, 55.59),
		geometryRow(17, 2, "c", 0, 1, 37.70, 55.75), geometryRow(17, 2, "c", 0, 2, 37.71, 55.76), geometryRow(17, 2, "c", 0, 3, 37.72, 55.77),
	}}
	lines, err := NewService(queries).ListRouteGeometry(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 3 {
		t.Fatalf("lines = %+v, want 3", lines)
	}
	first := lines[0]
	if first.RouteID != 16 || first.RouteNumber != 1 || !first.ForecastEnabled || first.PatternKey != "a" || first.DirectionID != 0 ||
		first.Source != SourceWorkbook {
		t.Errorf("first line = %+v", first)
	}
	// GeoJSON order is [longitude, latitude].
	if fmt.Sprint(first.Coordinates) != "[[37.59 55.59] [37.58 55.6]]" {
		t.Errorf("coordinates = %v", first.Coordinates)
	}
	if lines[2].RouteNumber != 2 || lines[2].ForecastEnabled || len(lines[2].Coordinates) != 3 {
		t.Errorf("third line = %+v", lines[2])
	}
}

func TestListRouteGeometryMarksOSMVariants(t *testing.T) {
	queries := &fakeQueries{geometry: []routes_sqlc.ListRouteGeometryRow{
		geometryRow(16, 1, "2040920", 0, 1, 37.59, 55.59), geometryRow(16, 1, "2040920", 0, 2, 37.58, 55.60),
		geometryRow(20, 17, "osm:relation/540033", 0, 1, 37.61, 55.82), geometryRow(20, 17, "osm:relation/540033", 0, 2, 37.61, 55.83),
	}}
	lines, err := NewService(queries).ListRouteGeometry(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 2 || lines[0].Source != SourceWorkbook || lines[1].Source != SourceOSM {
		t.Fatalf("lines = %+v, want a workbook and an OSM variant", lines)
	}
}

func TestListRouteGeometrySkipsSingleStopVariants(t *testing.T) {
	queries := &fakeQueries{geometry: []routes_sqlc.ListRouteGeometryRow{
		geometryRow(16, 1, "a", 0, 1, 37.59, 55.59),
		geometryRow(16, 1, "b", 1, 1, 37.58, 55.60), geometryRow(16, 1, "b", 1, 2, 37.59, 55.59),
	}}
	lines, err := NewService(queries).ListRouteGeometry(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 1 || lines[0].PatternKey != "b" {
		t.Fatalf("lines = %+v, want only variant b", lines)
	}
}

func TestListRouteGeometryEmptyAndErrors(t *testing.T) {
	lines, err := NewService(&fakeQueries{geometry: []routes_sqlc.ListRouteGeometryRow{}}).ListRouteGeometry(context.Background())
	if err != nil || lines == nil || len(lines) != 0 {
		t.Fatalf("lines = %#v, err = %v; want an empty non-nil slice", lines, err)
	}
	failure := errors.New("connection reset")
	if _, err := NewService(&fakeQueries{geometryErr: failure}).ListRouteGeometry(context.Background()); !errors.Is(err, failure) {
		t.Fatalf("error = %v, want wrapped database error", err)
	}
}
