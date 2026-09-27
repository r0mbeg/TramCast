// Package routes_service contains read scenarios for the route catalog.
package routes_service

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

// ErrRouteNotFound means that no route has the requested internal ID.
var ErrRouteNotFound = errors.New("route not found")

// Sources of a movement variant, for provenance on the map.
const (
	SourceWorkbook = "workbook"
	SourceOSM      = "osm"
)

// Queries is the part of the generated sqlc API used by Service.
// *routes_sqlc.Queries satisfies it directly.
type Queries interface {
	ListRoutes(ctx context.Context) ([]routes_sqlc.Route, error)
	GetRouteByID(ctx context.Context, routeID int64) (routes_sqlc.Route, error)
	ListRouteStops(ctx context.Context, routeID int64) ([]routes_sqlc.ListRouteStopsRow, error)
	ListRouteGeometry(ctx context.Context) ([]routes_sqlc.ListRouteGeometryRow, error)
}

// Line is one movement variant as a polyline through its stops.
type Line struct {
	RouteID         int64
	RouteNumber     int16
	ForecastEnabled bool
	PatternKey      string
	DirectionID     int16
	// Source is SourceOSM for a variant from the OpenStreetMap snapshot and
	// SourceWorkbook for one from the reference workbook.
	Source string
	// Coordinates are [longitude, latitude] pairs in stop_sequence order.
	Coordinates [][2]float64
}

// Pattern is one movement variant with its stops in stop_sequence order.
type Pattern struct {
	Key         string
	DirectionID int16
	Stops       []routes_sqlc.ListRouteStopsRow
}

type Service struct {
	queries Queries
}

func NewService(queries Queries) *Service {
	return &Service{queries: queries}
}

func (s *Service) ListRoutes(ctx context.Context) ([]routes_sqlc.Route, error) {
	routes, err := s.queries.ListRoutes(ctx)
	if err != nil {
		return nil, fmt.Errorf("list routes: %w", err)
	}
	return routes, nil
}

// ListRouteStops returns the route's movement variants. A route without
// geography has no variants; an unknown route returns ErrRouteNotFound.
func (s *Service) ListRouteStops(ctx context.Context, routeID int64) ([]Pattern, error) {
	if _, err := s.queries.GetRouteByID(ctx, routeID); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrRouteNotFound
		}
		return nil, fmt.Errorf("get route %d: %w", routeID, err)
	}
	rows, err := s.queries.ListRouteStops(ctx, routeID)
	if err != nil {
		return nil, fmt.Errorf("list stops of route %d: %w", routeID, err)
	}
	return groupPatterns(rows), nil
}

// groupPatterns relies on ListRouteStops ordering by direction, pattern and
// stop_sequence, so the rows of one variant are adjacent.
func groupPatterns(rows []routes_sqlc.ListRouteStopsRow) []Pattern {
	patterns := []Pattern{}
	for _, row := range rows {
		last := len(patterns) - 1
		if last < 0 || patterns[last].Key != row.PatternKey || patterns[last].DirectionID != row.DirectionID {
			patterns = append(patterns, Pattern{Key: row.PatternKey, DirectionID: row.DirectionID})
			last++
		}
		patterns[last].Stops = append(patterns[last].Stops, row)
	}
	return patterns
}

// ListRouteGeometry returns every movement variant as a line through its stops,
// for the map. The line is a scheme between consecutive stops, not rail
// geometry; variants with fewer than two stops cannot form a line and are
// skipped.
func (s *Service) ListRouteGeometry(ctx context.Context) ([]Line, error) {
	rows, err := s.queries.ListRouteGeometry(ctx)
	if err != nil {
		return nil, fmt.Errorf("list route geometry: %w", err)
	}
	lines := []Line{}
	for i := 0; i < len(rows); {
		first := rows[i]
		line := Line{
			RouteID:         first.RouteID,
			RouteNumber:     first.RouteNumber,
			ForecastEnabled: first.ForecastEnabled,
			PatternKey:      first.PatternKey,
			DirectionID:     first.DirectionID,
			Source:          SourceWorkbook,
		}
		// The import keys OSM variants by their namespaced relation ID.
		if core_domain.IsOSM(first.PatternKey) {
			line.Source = SourceOSM
		}
		// ListRouteGeometry orders rows so that one variant is contiguous.
		for ; i < len(rows) && sameVariant(rows[i], first); i++ {
			line.Coordinates = append(line.Coordinates, [2]float64{rows[i].Longitude, rows[i].Latitude})
		}
		if len(line.Coordinates) >= 2 {
			lines = append(lines, line)
		}
	}
	return lines, nil
}

func sameVariant(a, b routes_sqlc.ListRouteGeometryRow) bool {
	return a.RouteID == b.RouteID && a.PatternKey == b.PatternKey && a.DirectionID == b.DirectionID
}
