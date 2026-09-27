import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp } from 'lucide-react'
import { useMemo } from 'react'
import styles from './App.module.css'
import { useRouteGeometry, useRoutes } from './api/catalog'
import { ContextBar } from './components/ContextBar/ContextBar'
import { Dock } from './components/Dock/Dock'
import { MapView } from './components/MapView/MapView'
import { RouteBoard } from './components/RouteBoard/RouteBoard'
import { Button } from './components/ui/controls'
import { routeStatus } from './forecast/status'
import { useForecastVersion, useForecasts } from './forecast/useForecast'
import { formatLong } from './lib/calendar'
import { formatNumber, hourRange } from './lib/format'
import { activeSlot, useAppState } from './state/appState'
import { BOARD_ID, DOCK_ID, layoutActions, useLayout } from './state/layout'
import { useShortcuts } from './state/useShortcuts'

const TAB_ICONS = {
  left: { expanded: ChevronLeft, collapsed: ChevronRight },
  bottom: { expanded: ChevronDown, collapsed: ChevronUp },
}

/** A tab on the edge a panel shares with the map; it hides the panel and brings it back. */
function PanelTab({ edge, panel, controls, expanded, onToggle }: { edge: 'left' | 'bottom'; panel: string; controls: string; expanded: boolean; onToggle: () => void }) {
  const Icon = TAB_ICONS[edge][expanded ? 'expanded' : 'collapsed']
  const label = `${expanded ? 'Скрыть' : 'Показать'} панель «${panel}»`
  return (
    <button
      type="button"
      className={`${styles.tab} ${edge === 'left' ? styles.tabLeft : styles.tabBottom}`}
      aria-expanded={expanded}
      aria-controls={controls}
      aria-label={label}
      title={label}
      onClick={onToggle}
    >
      <Icon size={14} aria-hidden />
    </button>
  )
}

export function App() {
  const state = useAppState()
  const layout = useLayout()
  const routesQuery = useRoutes()
  const geometryQuery = useRouteGeometry()
  const routes = useMemo(() => [...(routesQuery.data ?? [])].sort((a, b) => a.route_number - b.route_number), [routesQuery.data])
  const version = useForecastVersion()
  const forecasts = useForecasts(routes, version)
  const withGeometry = useMemo(() => new Set(geometryQuery.data?.features.map((feature) => feature.properties.route_id)), [geometryQuery.data])
  useShortcuts(routes)

  if (routesQuery.isError) {
    return (
      <main className={styles.fatal}>
        <h1>Не удалось загрузить маршруты</h1>
        <p>Справочник недоступен. Проверьте, что сервер TramCast запущен.</p>
        <Button onClick={() => void routesQuery.refetch()}>Повторить</Button>
      </main>
    )
  }

  // One short sentence for screen readers after every change of selection.
  const selected = routes.find((route) => route.id === activeSlot(state).routeId)
  const selectedStatus = selected ? routeStatus(forecasts.get(selected.id)) : null
  const selectedValue = selectedStatus?.kind === 'ready' ? selectedStatus.series.get(state.date)?.[state.hour] : undefined
  const announcement = selected
    ? `Маршрут ${selected.route_number}, ${formatLong(state.date)}, ${hourRange(state.hour)}${selectedValue === undefined ? '' : `, ${formatNumber(selectedValue)} посадок`}`
    : `Картина дня, ${formatLong(state.date)}, ${hourRange(state.hour)}`

  return (
    <div className={styles.shell} data-board={layout.boardCollapsed ? 'collapsed' : undefined}>
      <ContextBar state={state} version={version} />
      {layout.boardCollapsed ? null : <RouteBoard state={state} routes={routes} forecasts={forecasts} withGeometry={withGeometry} />}
      <main className={styles.workspace}>
        <div className={styles.stage}>
          <MapView state={state} routes={routes} geometry={geometryQuery.data} forecasts={forecasts} />
          <PanelTab
            edge="left"
            panel="Маршруты"
            controls={BOARD_ID}
            expanded={!layout.boardCollapsed}
            onToggle={() => layoutActions.setBoardCollapsed(!layout.boardCollapsed)}
          />
          <PanelTab
            edge="bottom"
            panel={state.slots.some((slot) => slot.routeId !== null) ? 'Прогноз маршрута' : 'Картина дня'}
            controls={DOCK_ID}
            expanded={!layout.dockCollapsed}
            onToggle={() => layoutActions.setDockCollapsed(!layout.dockCollapsed)}
          />
        </div>
        {layout.dockCollapsed ? null : <Dock state={state} routes={routes} forecasts={forecasts} />}
      </main>
      <div className="visually-hidden" aria-live="polite">
        {announcement}
      </div>
    </div>
  )
}
