import { useQuery } from '@tanstack/react-query'
import { getJson } from './client'

// Shapes of the Go catalog API (see backend/README.md).

export interface Route {
  id: number
  route_number: number
  name: string | null
  forecast_enabled: boolean
}

export interface PatternStop {
  stop_sequence: number
  stop_id: number
  name: string
  latitude: number
  longitude: number
}

export interface RoutePattern {
  pattern_key: string
  direction_id: number
  stops: PatternStop[]
}

export interface RouteStops {
  route_id: number
  patterns: RoutePattern[]
}

export interface GeometryProperties {
  route_id: number
  route_number: number
  pattern_key: string
  direction_id: number
  forecast_enabled: boolean
  /** Where the scheme comes from: the organizers' workbook or the OpenStreetMap snapshot. */
  source: 'workbook' | 'osm'
}

export interface RouteGeometry {
  type: 'FeatureCollection'
  features: {
    type: 'Feature'
    geometry: { type: 'LineString'; coordinates: [number, number][] }
    properties: GeometryProperties
  }[]
}

/** Numbers of the routes drawn from OpenStreetMap, ascending and without repeats. */
export function osmRouteNumbers(geometry: RouteGeometry): number[] {
  const numbers = new Set<number>()
  for (const feature of geometry.features) {
    if (feature.properties.source === 'osm') numbers.add(feature.properties.route_number)
  }
  return [...numbers].sort((a, b) => a - b)
}

// The catalog is static for a session, so it is fetched once.
const catalogQuery = { staleTime: Infinity, gcTime: Infinity, retry: 1 } as const

export function useRoutes() {
  return useQuery({
    queryKey: ['routes'],
    queryFn: () => getJson<{ routes: Route[] }>('/api/routes').then((body) => body.routes),
    ...catalogQuery,
  })
}

export function useRouteGeometry() {
  return useQuery({
    queryKey: ['routes', 'geometry'],
    queryFn: () => getJson<RouteGeometry>('/api/routes/geometry'),
    ...catalogQuery,
  })
}

export function useRouteStops(routeId: number | null) {
  return useQuery({
    queryKey: ['routes', routeId, 'stops'],
    queryFn: () => getJson<RouteStops>(`/api/routes/${routeId}/stops`),
    enabled: routeId !== null,
    ...catalogQuery,
  })
}
