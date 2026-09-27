import type { ExpressionSpecification, LayerSpecification, StyleSpecification } from 'maplibre-gl'
import { hex } from '../lib/theme'
import base from './openfreemap-dark.json'

// OpenFreeMap Dark (no key, no limits), copied into the repository so upstream
// changes cannot surprise us, then recolored into the interface neutrals.
// Roads stay far quieter than route lines; labels are Russian.

const REMOVED = /^(aeroway|road_oneway|place_village|place_other|place_country|highway_path|road_pier|road_area_pier)/

const RU_NAME: ExpressionSpecification = ['coalesce', ['get', 'name:ru'], ['get', 'name']]

function recolor(layer: LayerSpecification): LayerSpecification {
  const id = layer.id
  const paint = { ...(layer.paint as Record<string, unknown> | undefined) }
  const layout = { ...(layer.layout as Record<string, unknown> | undefined) }
  let minzoom = layer.minzoom

  if (layer.type === 'background') paint['background-color'] = hex.land
  if (id === 'water') paint['fill-color'] = hex.water
  if (id === 'waterway') paint['line-color'] = hex.water
  if (id === 'landcover_wood' || id === 'landuse_park') paint['fill-color'] = hex.park
  if (id === 'landuse_residential') paint['fill-color'] = hex.park
  if (id === 'building') {
    paint['fill-color'] = hex.surface1
    paint['fill-outline-color'] = hex.surface2
    minzoom = 15
  }
  if (id === 'highway_minor') {
    paint['line-color'] = hex.roadMinor
    minzoom = 13
  }
  if (id.startsWith('highway_major') || id.startsWith('highway_motorway')) {
    paint['line-color'] = id.endsWith('casing') ? hex.surface2 : hex.roadMajor
  }
  if (id.startsWith('railway')) paint['line-color'] = id.endsWith('dashline') ? hex.land : hex.rail
  if (id.startsWith('boundary')) paint['line-color'] = hex.surface3

  if (layer.type === 'symbol' && 'text-field' in layout) {
    layout['text-field'] = id === 'highway_name_motorway' ? layout['text-field'] : RU_NAME
    paint['text-color'] = hex.mapLabel
    paint['text-halo-color'] = hex.land
    paint['text-halo-width'] = 1.5
    paint['text-halo-blur'] = 0
    if (id.startsWith('highway_name')) minzoom = 14
  }

  return { ...layer, paint, layout, ...(minzoom === undefined ? {} : { minzoom }) } as LayerSpecification
}

export function tramcastStyle(): StyleSpecification {
  const style = base as unknown as StyleSpecification
  const { ne2_shaded: _unused, ...sources } = style.sources
  return {
    ...style,
    sources,
    layers: style.layers.filter((layer) => !REMOVED.test(layer.id)).map(recolor),
  }
}

/** The font stack available from the OpenFreeMap glyph server. */
export const MAP_FONT_REGULAR = ['Noto Sans Regular']
export const MAP_FONT_BOLD = ['Noto Sans Bold']
