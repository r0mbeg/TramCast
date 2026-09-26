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
	routes    []routes_sqlc.Route
	routeErr  error
	stops     []routes_sqlc.ListRouteStopsRow
	stopsErr  error
	listErr   error
	stopCalls int
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
