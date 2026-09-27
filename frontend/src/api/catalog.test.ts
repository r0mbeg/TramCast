import { describe, expect, it } from 'vitest'
import { osmRouteNumbers, type GeometryProperties, type RouteGeometry } from './catalog'

function geometry(lines: Pick<GeometryProperties, 'route_number' | 'direction_id' | 'source'>[]): RouteGeometry {
  return {
    type: 'FeatureCollection',
    features: lines.map((line) => ({
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: [
          [37.6, 55.7],
          [37.61, 55.71],
        ],
      },
      properties: { route_id: line.route_number + 1000, pattern_key: `${line.route_number}-${line.direction_id}`, forecast_enabled: true, ...line },
    })),
  }
}

describe('osmRouteNumbers', () => {
  it('lists each OpenStreetMap route once, in ascending order', () => {
    const lines = geometry([
      { route_number: 50, direction_id: 0, source: 'osm' },
      { route_number: 1, direction_id: 0, source: 'workbook' },
      { route_number: 17, direction_id: 0, source: 'osm' },
      { route_number: 50, direction_id: 1, source: 'osm' },
      { route_number: 1, direction_id: 1, source: 'workbook' },
      { route_number: 17, direction_id: 1, source: 'osm' },
    ])
    expect(osmRouteNumbers(lines)).toEqual([17, 50])
  })

  it('is empty without OpenStreetMap schemes', () => {
    expect(osmRouteNumbers(geometry([{ route_number: 7, direction_id: 0, source: 'workbook' }]))).toEqual([])
    expect(osmRouteNumbers(geometry([]))).toEqual([])
  })
})
