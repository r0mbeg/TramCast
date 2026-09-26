// Package stops_service contains read scenarios for the stop catalog.
package stops_service

import (
	"context"
	"fmt"

	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
)

// Queries is the part of the generated sqlc API used by Service.
// *stops_sqlc.Queries satisfies it directly.
type Queries interface {
	ListStops(ctx context.Context) ([]stops_sqlc.Stop, error)
}

type Service struct {
	queries Queries
}

func NewService(queries Queries) *Service {
	return &Service{queries: queries}
}

func (s *Service) ListStops(ctx context.Context) ([]stops_sqlc.Stop, error) {
	stops, err := s.queries.ListStops(ctx)
	if err != nil {
		return nil, fmt.Errorf("list stops: %w", err)
	}
	return stops, nil
}
