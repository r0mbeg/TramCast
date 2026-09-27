import { useMemo } from 'react'
import styles from './App.module.css'
import { useRouteGeometry, useRoutes } from './api/catalog'
import { ContextBar } from './components/ContextBar/ContextBar'
import { Dock } from './components/Dock/Dock'
import { MapView } from './components/MapView/MapView'
import { RouteBoard } from './components/RouteBoard/RouteBoard'
import { Button } from './components/ui/controls'
import { routeStatus } from './forecast/status'
import { forecastVersion, useForecasts } from './forecast/useForecast'
import { formatLong } from './lib/calendar'
import { formatNumber, hourRange } from './lib/format'
import { activeSlot, useAppState } from './state/appState'
import { useShortcuts } from './state/useShortcuts'

export function App() {
  const state = useAppState()
  const routesQuery = useRoutes()
  const geometryQuery = useRouteGeometry()
  const routes = useMemo(() => [...(routesQuery.data ?? [])].sort((a, b) => a.route_number - b.route_number), [routesQuery.data])
  const forecasts = useForecasts(routes, state.scenario)
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
    ? `Маршрут ${selected.route_number}, ${formatLong(state.date)}, ${hourRange(state.hour)}${selectedValue === undefined ? '' : `, ${formatNumber(selectedValue)} посадок, демо-данные`}`
    : `Картина дня, ${formatLong(state.date)}, ${hourRange(state.hour)}`

  return (
    <div className={styles.shell}>
      <ContextBar state={state} version={forecastVersion()} />
      <RouteBoard state={state} routes={routes} forecasts={forecasts} withGeometry={withGeometry} />
      <main className={styles.workspace}>
        <MapView state={state} routes={routes} geometry={geometryQuery.data} forecasts={forecasts} />
        <Dock state={state} routes={routes} forecasts={forecasts} />
      </main>
      <div className="visually-hidden" aria-live="polite">
        {announcement}
      </div>
    </div>
  )
}
