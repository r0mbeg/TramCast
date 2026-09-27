package routes_transport_http

import (
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
	routes_service "github.com/r0mbeg/TramCast/backend/internal/features/routes/service"
)

type listRoutesResponse struct {
	Routes []routeDTO `json:"routes"`
}

type routeDTO struct {
	ID              int64   `json:"id"`
	RouteNumber     int16   `json:"route_number"`
	Name            *string `json:"name"`
	ForecastEnabled bool    `json:"forecast_enabled"`
}

type listRouteStopsResponse struct {
	RouteID  int64        `json:"route_id"`
	Patterns []patternDTO `json:"patterns"`
}

type patternDTO struct {
	PatternKey  string           `json:"pattern_key"`
	DirectionID int16            `json:"direction_id"`
	Stops       []patternStopDTO `json:"stops"`
}

type patternStopDTO struct {
	StopSequence int32   `json:"stop_sequence"`
	StopID       int64   `json:"stop_id"`
	Name         string  `json:"name"`
	Latitude     float64 `json:"latitude"`
	Longitude    float64 `json:"longitude"`
}

func newListRoutesResponse(catalog []routes_sqlc.Route) listRoutesResponse {
	response := listRoutesResponse{Routes: make([]routeDTO, len(catalog))}
	for i, route := range catalog {
		var name *string
		if route.Name.Valid {
			name = &route.Name.String
		}
		response.Routes[i] = routeDTO{
			ID:              route.ID,
			RouteNumber:     route.RouteNumber,
			Name:            name,
			ForecastEnabled: route.ForecastEnabled,
		}
	}
	return response
}

func newListRouteStopsResponse(routeID int64, patterns []routes_service.Pattern) listRouteStopsResponse {
	response := listRouteStopsResponse{RouteID: routeID, Patterns: make([]patternDTO, len(patterns))}
	for i, pattern := range patterns {
		stops := make([]patternStopDTO, len(pattern.Stops))
		for j, stop := range pattern.Stops {
			stops[j] = patternStopDTO{
				StopSequence: stop.StopSequence,
				StopID:       stop.StopID,
				Name:         stop.Name,
				Latitude:     stop.Latitude,
				Longitude:    stop.Longitude,
			}
		}
		response.Patterns[i] = patternDTO{PatternKey: pattern.Key, DirectionID: pattern.DirectionID, Stops: stops}
	}
	return response
}

// GeoJSON (RFC 7946): coordinates are [longitude, latitude].
type featureCollection struct {
	Type     string        `json:"type"`
	Features []lineFeature `json:"features"`
}

type lineFeature struct {
	Type       string         `json:"type"`
	Geometry   lineString     `json:"geometry"`
	Properties lineProperties `json:"properties"`
}

type lineString struct {
	Type        string       `json:"type"`
	Coordinates [][2]float64 `json:"coordinates"`
}

type lineProperties struct {
	RouteID         int64  `json:"route_id"`
	RouteNumber     int16  `json:"route_number"`
	PatternKey      string `json:"pattern_key"`
	DirectionID     int16  `json:"direction_id"`
	ForecastEnabled bool   `json:"forecast_enabled"`
	Source          string `json:"source"`
}

func newGeometryResponse(lines []routes_service.Line) featureCollection {
	response := featureCollection{Type: "FeatureCollection", Features: make([]lineFeature, len(lines))}
	for i, line := range lines {
		response.Features[i] = lineFeature{
			Type:     "Feature",
			Geometry: lineString{Type: "LineString", Coordinates: line.Coordinates},
			Properties: lineProperties{
				RouteID:         line.RouteID,
				RouteNumber:     line.RouteNumber,
				PatternKey:      line.PatternKey,
				DirectionID:     line.DirectionID,
				ForecastEnabled: line.ForecastEnabled,
				Source:          line.Source,
			},
		}
	}
	return response
}
