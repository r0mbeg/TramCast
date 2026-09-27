import '../../map/worker'
import { Maximize2, Minus, Plus } from 'lucide-react'
import { AttributionControl, LngLatBounds, Map as MapLibreMap, type ExpressionSpecification, type GeoJSONSource, type MapMouseEvent } from 'maplibre-gl'
import { useEffect, useMemo, useRef, useState } from 'react'
import { osmRouteNumbers, type Route, type RouteGeometry } from '../../api/catalog'
import { sum } from '../../forecast/aggregate'
import { MISSING_LABEL, jobLabel, routeStatus, type RouteStatus } from '../../forecast/status'
import type { ForecastState } from '../../forecast/useForecast'
import { formatMedium } from '../../lib/calendar'
import { formatNumber, hourRange, isServiceOff } from '../../lib/format'
import { routeHex } from '../../lib/routeColors'
import { hex } from '../../lib/theme'
import { MAP_FONT_BOLD, tramcastStyle } from '../../map/style'
import { actions, activeSlot, slotLetter, type AppState } from '../../state/appState'
import { Button, RouteBadge } from '../ui/controls'
import styles from './MapView.module.css'

type LineState = 'context' | 'ready' | 'pending'

interface MapViewProps {
  state: AppState
  routes: readonly Route[]
  geometry: RouteGeometry | undefined
  forecasts: Map<number, ForecastState>
}

interface HoverInfo {
  x: number
  y: number
  routeIds: number[]
}

interface Chooser {
  x: number
  y: number
  routeIds: number[]
  shift: boolean
}

const EMPTY: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }

// Credit for the schemes taken from the OpenStreetMap snapshot (ODbL). It sits on
// our own source, so it stays visible when the basemap fails to load.
const OSM_ATTRIBUTION = '<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© участники OpenStreetMap</a>'

/** Line width by zoom, plus extra pixels for casings and outlines. */
function width(extra = 0): ExpressionSpecification {
  return ['interpolate', ['linear'], ['zoom'], 10, 2 + extra, 12, 3 + extra, 14, 5 + extra, 16, 7 + extra]
}

// Both directions shift to their right, so they run side by side when zoomed in.
const OFFSET: ExpressionSpecification = ['interpolate', ['linear'], ['zoom'], 12, 0, 14, 2.5, 16, 4]

function reducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

function valueAt(status: RouteStatus, date: string, hour: number): number | null {
  if (status.kind !== 'ready') return null
  return status.series.get(date)?.[hour] ?? null
}

function statusText(status: RouteStatus, date: string, hour: number): string {
  switch (status.kind) {
    case 'loading':
      return 'загрузка…'
    case 'missing':
      return MISSING_LABEL[status.reason].toLowerCase()
    case 'running':
      return jobLabel(status.jobStatus).toLowerCase()
    case 'failed':
      return 'ошибка расчёта'
    case 'error':
      return 'прогноз не получен'
    case 'ready': {
      if (status.fallback) return '0 · нет истории'
      if (isServiceOff(hour)) return '0 · терминалы выключены'
      const value = valueAt(status, date, hour) ?? 0
      const total = sum(status.series.get(date) ?? [])
      return `${formatNumber(value)} посадок · сутки ${formatNumber(total)}`
    }
  }
}

function labelValue(status: RouteStatus, date: string, hour: number): string {
  switch (status.kind) {
    case 'ready':
      return formatNumber(isServiceOff(hour) ? 0 : (valueAt(status, date, hour) ?? 0))
    case 'missing':
      return '—'
    case 'running':
      return status.jobStatus === 'queued' ? 'очередь' : 'расчёт'
    case 'failed':
    case 'error':
      return 'ошибка'
    case 'loading':
      return '…'
  }
}

export function MapView({ state, routes, geometry, forecasts }: MapViewProps) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [ready, setReady] = useState(false)
  const [basemapFailed, setBasemapFailed] = useState(false)
  const [hover, setHover] = useState<HoverInfo | null>(null)
  const [chooser, setChooser] = useState<Chooser | null>(null)
  const fitted = useRef(false)
  // On low windows the legend starts collapsed so it does not cover the lines.
  const [legendOpen, setLegendOpen] = useState(() => window.innerHeight >= 860)

  const routesById = useMemo(() => new Map(routes.map((route) => [route.id, route])), [routes])
  const osmNumbers = useMemo(() => (geometry ? osmRouteNumbers(geometry) : []), [geometry])
  const selectedIds = useMemo(() => new Set(state.slots.map((slot) => slot.routeId).filter((id): id is number => id !== null)), [state.slots])
  const activeRouteId = activeSlot(state).routeId
  const letters = useMemo(() => {
    const result = new Map<number, string>()
    for (const slot of state.slots) {
      const letter = slotLetter(state, slot)
      if (slot.routeId !== null && letter) result.set(slot.routeId, letter)
    }
    return result
  }, [state])

  // Line features enriched with the current slice: identity color, status and label.
  const lineData = useMemo((): GeoJSON.FeatureCollection => {
    if (!geometry) return EMPTY
    return {
      type: 'FeatureCollection',
      features: geometry.features.map((feature) => {
        const { route_id: routeId, route_number: number, forecast_enabled: enabled } = feature.properties
        const status = routeStatus(forecasts.get(routeId))
        const lineState: LineState = !enabled ? 'context' : status.kind === 'ready' ? 'ready' : 'pending'
        const letter = letters.get(routeId)
        const short = letter ? `${number} ${letter}` : String(number)
        return {
          type: 'Feature',
          geometry: feature.geometry,
          properties: {
            ...feature.properties,
            color: routeHex(number),
            state: lineState,
            selected: selectedIds.has(routeId),
            label_short: short,
            label_long: enabled ? `${short} · ${labelValue(status, state.date, state.hour)}` : short,
          },
        }
      }),
    }
  }, [geometry, forecasts, letters, selectedIds, state.date, state.hour])

  const stopData = useMemo((): GeoJSON.FeatureCollection => {
    if (!geometry) return EMPTY
    return {
      type: 'FeatureCollection',
      features: geometry.features
        .filter((feature) => selectedIds.has(feature.properties.route_id))
        .flatMap((feature) =>
          feature.geometry.coordinates.map((coordinates) => ({
            type: 'Feature' as const,
            geometry: { type: 'Point' as const, coordinates },
            properties: { color: routeHex(feature.properties.route_number) },
          })),
        ),
    }
  }, [geometry, selectedIds])

  // Create the map once.
  useEffect(() => {
    if (!container.current) return
    const map = new MapLibreMap({
      container: container.current,
      style: tramcastStyle(),
      center: [37.6, 55.72],
      zoom: 10.3,
      minZoom: 9,
      maxZoom: 17,
      maxBounds: [
        [36.8, 55.3],
        [38.4, 56.2],
      ],
      dragRotate: false,
      pitchWithRotate: false,
      keyboard: false,
      attributionControl: false,
    })
    map.touchZoomRotate.disableRotation()
    map.addControl(new AttributionControl({ compact: false }), 'bottom-right')
    map.on('error', (event) => {
      const source = (event as unknown as { sourceId?: string }).sourceId
      if (source === 'openmaptiles' || String(event.error?.message ?? '').includes('openfreemap')) setBasemapFailed(true)
    })
    map.on('load', () => {
      map.addSource('tc-routes', { type: 'geojson', data: EMPTY, attribution: OSM_ATTRIBUTION })
      map.addSource('tc-stops', { type: 'geojson', data: EMPTY })
      const isState = (value: LineState): ExpressionSpecification => ['==', ['get', 'state'], value]
      map.addLayer({ id: 'tc-context', type: 'line', source: 'tc-routes', filter: isState('context'), layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': hex.contextLine, 'line-width': 1.5, 'line-offset': OFFSET } })
      map.addLayer({ id: 'tc-hover', type: 'line', source: 'tc-routes', filter: ['==', ['get', 'route_id'], -1], layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': hex.text1, 'line-width': width(4), 'line-offset': OFFSET } })
      map.addLayer({ id: 'tc-casing', type: 'line', source: 'tc-routes', filter: ['!=', ['get', 'state'], 'context'], layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': hex.land, 'line-width': width(2), 'line-offset': OFFSET } })
      map.addLayer({ id: 'tc-line', type: 'line', source: 'tc-routes', filter: isState('ready'), layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': ['get', 'color'], 'line-width': width(), 'line-offset': OFFSET } })
      map.addLayer({ id: 'tc-pending', type: 'line', source: 'tc-routes', filter: isState('pending'), layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': ['get', 'color'], 'line-width': width(), 'line-dasharray': [0.5, 2], 'line-offset': OFFSET } })
      // Selected routes: white outline, a land-colored gap, then the line on top.
      map.addLayer({ id: 'tc-selected-halo', type: 'line', source: 'tc-routes', filter: ['==', ['get', 'selected'], true], layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': hex.selection, 'line-width': width(6), 'line-offset': OFFSET } })
      map.addLayer({ id: 'tc-selected-gap', type: 'line', source: 'tc-routes', filter: ['==', ['get', 'selected'], true], layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': hex.land, 'line-width': width(2), 'line-offset': OFFSET } })
      map.addLayer({ id: 'tc-selected-line', type: 'line', source: 'tc-routes', filter: ['all', ['==', ['get', 'selected'], true], ['!=', ['get', 'state'], 'pending']], layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': ['get', 'color'], 'line-width': width(1), 'line-offset': OFFSET } })
      // A selected route without a result keeps its dotted line inside the outline.
      map.addLayer({ id: 'tc-selected-pending', type: 'line', source: 'tc-routes', filter: ['all', ['==', ['get', 'selected'], true], ['==', ['get', 'state'], 'pending']], layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': ['get', 'color'], 'line-width': width(1), 'line-dasharray': [0.5, 2], 'line-offset': OFFSET } })
      map.addLayer({ id: 'tc-stops', type: 'circle', source: 'tc-stops', minzoom: 12, paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 12, 2.5, 16, 5], 'circle-color': hex.land, 'circle-stroke-color': ['get', 'color'], 'circle-stroke-width': 1.5 } })
      map.addLayer({
        id: 'tc-labels',
        type: 'symbol',
        source: 'tc-routes',
        filter: ['==', ['get', 'direction_id'], 0],
        layout: {
          'symbol-placement': 'line',
          'symbol-spacing': 320,
          'text-field': ['step', ['zoom'], ['get', 'label_short'], 11, ['get', 'label_long']],
          'text-font': MAP_FONT_BOLD,
          'text-size': ['interpolate', ['linear'], ['zoom'], 10, 12, 14, 13],
          'text-letter-spacing': 0.02,
        },
        paint: { 'text-color': ['case', ['==', ['get', 'state'], 'context'], hex.text3, hex.text1], 'text-halo-color': hex.land, 'text-halo-width': 1.5 },
      })
      map.addLayer({ id: 'tc-hit', type: 'line', source: 'tc-routes', paint: { 'line-color': '#000', 'line-opacity': 0, 'line-width': 14 } })
      setReady(true)
    })
    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
      setReady(false)
    }
  }, [])

  // Feed the current slice into the sources.
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    ;(map.getSource('tc-routes') as GeoJSONSource).setData(lineData)
    ;(map.getSource('tc-stops') as GeoJSONSource).setData(stopData)
  }, [ready, lineData, stopData])

  // Fit the whole network once the geometry arrives.
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map || !geometry || fitted.current || geometry.features.length === 0) return
    fitted.current = true
    map.fitBounds(boundsOf(geometry, () => true), { padding: { top: 48, bottom: 64, left: 24, right: 24 }, duration: 0 })
  }, [ready, geometry])

  // Move the camera only when the selected route is mostly out of view.
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map || !geometry || activeRouteId === null) return
    const bounds = boundsOf(geometry, (id) => id === activeRouteId)
    if (bounds.isEmpty() || visibleShare(map, bounds) >= 0.7) return
    map.fitBounds(bounds, { padding: 48, maxZoom: 13.5, duration: reducedMotion() ? 0 : 200 })
  }, [ready, geometry, activeRouteId])

  // F and Shift+F from the keyboard shortcuts.
  useEffect(() => {
    const onFit = (event: Event) => {
      const map = mapRef.current
      if (!map || !geometry) return
      const target = (event as CustomEvent<'route' | 'network'>).detail
      const bounds = target === 'route' && activeRouteId !== null ? boundsOf(geometry, (id) => id === activeRouteId) : boundsOf(geometry, () => true)
      if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 48, maxZoom: 13.5, duration: reducedMotion() ? 0 : 200 })
    }
    window.addEventListener('tramcast:fit', onFit)
    return () => window.removeEventListener('tramcast:fit', onFit)
  }, [geometry, activeRouteId])

  // Hover and click on route lines.
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const routesAt = (event: MapMouseEvent) => {
      const { x, y } = event.point
      const features = map.queryRenderedFeatures(
        [
          [x - 6, y - 6],
          [x + 6, y + 6],
        ],
        { layers: ['tc-hit'] },
      )
      return [...new Set(features.map((feature) => Number(feature.properties?.route_id)))].filter((id) => routesById.has(id))
    }
    const onMove = (event: MapMouseEvent) => {
      const ids = routesAt(event)
      map.getCanvas().style.cursor = ids.length ? 'pointer' : ''
      map.setFilter('tc-hover', ['in', ['get', 'route_id'], ['literal', ids.length ? ids : [-1]]])
      setHover(ids.length ? { x: event.point.x, y: event.point.y, routeIds: ids } : null)
    }
    const onLeave = () => {
      map.getCanvas().style.cursor = ''
      map.setFilter('tc-hover', ['==', ['get', 'route_id'], -1])
      setHover(null)
    }
    const onClick = (event: MapMouseEvent) => {
      const ids = routesAt(event)
      const shift = event.originalEvent.shiftKey
      if (ids.length === 1) {
        setChooser(null)
        open(ids[0]!, shift)
      } else if (ids.length > 1) {
        setChooser({ x: event.point.x, y: event.point.y, routeIds: ids, shift })
      } else {
        setChooser(null)
      }
    }
    map.on('mousemove', onMove)
    map.on('mouseout', onLeave)
    map.on('click', onClick)
    return () => {
      map.off('mousemove', onMove)
      map.off('mouseout', onLeave)
      map.off('click', onClick)
    }
  }, [ready, routesById])

  const offMap = routes.filter((route) => route.forecast_enabled && geometry && !geometry.features.some((feature) => feature.properties.route_id === route.id))
  const onMapForecast = routes.filter((route) => route.forecast_enabled && !offMap.includes(route))
  const serviceOff = isServiceOff(state.hour)

  return (
    <section className={styles.wrap} aria-label="Карта маршрутов">
      <div ref={container} className={styles.map} />

      <div className={`${styles.panel} ${styles.caption}`}>
        {serviceOff ? (
          <>
            <strong>{hourRange(state.hour)}</strong> · терминалы выключены — прогноз 0 для всех маршрутов
          </>
        ) : (
          <>
            Посадки за <strong>{hourRange(state.hour)}</strong> · {formatMedium(state.date)}
          </>
        )}
      </div>

      {basemapFailed ? <div className={`${styles.panel} ${styles.basemapNote}`}>Подложка карты недоступна — показаны только схемы маршрутов</div> : null}

      <div className={`${styles.panel} ${styles.controls}`}>
        <Button variant="ghost" iconOnly aria-label="Приблизить" onClick={() => mapRef.current?.zoomIn({ duration: reducedMotion() ? 0 : 200 })}>
          <Plus size={16} />
        </Button>
        <Button variant="ghost" iconOnly aria-label="Отдалить" onClick={() => mapRef.current?.zoomOut({ duration: reducedMotion() ? 0 : 200 })}>
          <Minus size={16} />
        </Button>
        <Button variant="ghost" iconOnly aria-label="Вся сеть" title="Вся сеть (Shift+F)" onClick={() => window.dispatchEvent(new CustomEvent('tramcast:fit', { detail: 'network' }))}>
          <Maximize2 size={15} />
        </Button>
      </div>

      {offMap.length > 0 ? (
        <div className={`${styles.panel} ${styles.tray}`}>
          <span className={styles.trayTitle}>Нет на карте:</span>
          {offMap.map((route) => {
            const status = routeStatus(forecasts.get(route.id))
            return (
              <button
                key={route.id}
                type="button"
                className={styles.trayChip}
                aria-current={route.id === activeRouteId}
                title={`Маршрут ${route.route_number}: нет схемы маршрута · ${statusText(status, state.date, state.hour)}`}
                onClick={(event) => open(route.id, event.shiftKey)}
              >
                <RouteBadge routeNumber={route.route_number} letter={letters.get(route.id) ?? null} />
                {labelValue(status, state.date, state.hour)}
              </button>
            )
          })}
        </div>
      ) : null}

      <details className={`${styles.panel} ${styles.legend}`} open={legendOpen} onToggle={(event) => setLegendOpen(event.currentTarget.open)}>
        <summary className={styles.legendTitle}>Маршруты на карте</summary>
        <div className={styles.legendBody}>
          {onMapForecast.length > 0 ? (
            <div className={styles.legendRoutes}>
              {onMapForecast.map((route) => (
                <LegendRow key={route.id} color={routeHex(route.route_number)} label={`№ ${route.route_number}`} />
              ))}
            </div>
          ) : null}
          <LegendRow color={hex.contextLine} label="без прогноза (2, 3, 4, 6, 10)" />
          <span className={`${styles.swatch} ${styles.swatchDashed}`} aria-hidden />
          <span>нет результата прогноза</span>
          <span className={styles.legendNote}>Линия соединяет остановки по порядку — это схема, не трасса рельсов.</span>
          {osmNumbers.length > 0 ? <span className={styles.legendNote}>{`Схемы № ${osmNumbers.join(', ')} — по OpenStreetMap`}</span> : null}
        </div>
      </details>

      {hover && !chooser ? (
        <div className={styles.tooltip} style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {hover.routeIds.map((id) => {
            const route = routesById.get(id)
            if (!route) return null
            const status = routeStatus(forecasts.get(id))
            const meta = [route.name, osmNumbers.includes(route.route_number) ? 'схема: OpenStreetMap' : null].filter(Boolean).join(' · ')
            return (
              <div key={id} className={styles.tooltipRow}>
                <RouteBadge routeNumber={route.route_number} outline={!route.forecast_enabled} />
                <div>
                  <div>{route.forecast_enabled ? `${hourRange(state.hour)} — ${statusText(status, state.date, state.hour)}` : 'прогноз не строится'}</div>
                  {meta ? <div className={styles.tooltipMeta}>{meta}</div> : null}
                </div>
              </div>
            )
          })}
        </div>
      ) : null}

      {chooser ? (
        <div className={styles.chooser} style={{ left: chooser.x, top: chooser.y }} role="menu" aria-label="Маршруты в этом месте">
          <span className={styles.chooserTitle}>Здесь проходят:</span>
          {chooser.routeIds.map((id) => {
            const route = routesById.get(id)
            if (!route) return null
            return (
              <button
                key={id}
                type="button"
                role="menuitem"
                className={styles.chooserButton}
                onClick={() => {
                  setChooser(null)
                  open(id, chooser.shift)
                }}
              >
                <RouteBadge routeNumber={route.route_number} outline={!route.forecast_enabled} />
                <span>{route.name ?? `Маршрут ${route.route_number}`}</span>
              </button>
            )
          })}
        </div>
      ) : null}
    </section>
  )
}

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <>
      <span className={styles.swatch} style={{ background: color }} aria-hidden />
      <span>{label}</span>
    </>
  )
}

function open(routeId: number, shift: boolean) {
  if (shift && actions.addSlot(routeId)) return
  actions.selectRoute(routeId)
}

function boundsOf(geometry: RouteGeometry, match: (routeId: number) => boolean): LngLatBounds {
  const bounds = new LngLatBounds()
  for (const feature of geometry.features) {
    if (!match(feature.properties.route_id)) continue
    for (const coordinate of feature.geometry.coordinates) bounds.extend(coordinate)
  }
  return bounds
}

/** Share of the bounds' area currently in view, 0–1. */
function visibleShare(map: MapLibreMap, bounds: LngLatBounds): number {
  const view = map.getBounds()
  const west = Math.max(view.getWest(), bounds.getWest())
  const east = Math.min(view.getEast(), bounds.getEast())
  const south = Math.max(view.getSouth(), bounds.getSouth())
  const north = Math.min(view.getNorth(), bounds.getNorth())
  const area = (bounds.getEast() - bounds.getWest()) * (bounds.getNorth() - bounds.getSouth())
  if (area <= 0) return view.contains(bounds.getCenter()) ? 1 : 0
  return Math.max(0, east - west) * Math.max(0, north - south) / area
}
