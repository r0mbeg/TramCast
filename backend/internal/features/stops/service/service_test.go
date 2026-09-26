package stops_service

import (
	"context"
	"errors"
	"testing"

	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
)

type fakeQueries struct {
	stops []stops_sqlc.Stop
	err   error
}

func (f fakeQueries) ListStops(context.Context) ([]stops_sqlc.Stop, error) {
	return f.stops, f.err
}

func TestListStops(t *testing.T) {
	stops, err := NewService(fakeQueries{stops: []stops_sqlc.Stop{{ID: 10}, {ID: 11}}}).ListStops(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(stops) != 2 || stops[0].ID != 10 || stops[1].ID != 11 {
		t.Fatalf("stops = %+v", stops)
	}
}

func TestListStopsPropagatesDatabaseErrors(t *testing.T) {
	failure := errors.New("connection reset")
	_, err := NewService(fakeQueries{err: failure}).ListStops(context.Background())
	if !errors.Is(err, failure) {
		t.Fatalf("error = %v, want wrapped database error", err)
	}
}
