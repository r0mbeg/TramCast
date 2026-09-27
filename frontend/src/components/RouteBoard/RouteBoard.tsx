import { AlertCircle, Clock3, MapPinOff } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { Route } from '../../api/catalog'
import { sum } from '../../forecast/aggregate'
import { routeStatus, useElapsed, type RouteStatus } from '../../forecast/status'
import type { ForecastState } from '../../forecast/useForecast'
import { formatMedium } from '../../lib/calendar'
import { formatElapsed, formatNumber, hourLabel, hourRange, isServiceOff } from '../../lib/format'
import { loadHex } from '../../lib/loadScale'
import { actions, activeSlot, slotLetter, type AppState } from '../../state/appState'
import { RouteBadge, Segmented } from '../ui/controls'
import styles from './RouteBoard.module.css'

type Sort = 'number' | 'total'

interface RouteBoardProps {
  state: AppState
  routes: readonly Route[]
  forecasts: Map<number, ForecastState>
  withGeometry: ReadonlySet<number>
}

function dayValues(status: RouteStatus, date: string): number[] | null {
  return status.kind === 'ready' ? (status.series.get(date) ?? null) : null
}

function Strip({ status, date, hour, onPickHour }: { status: RouteStatus; date: string; hour: number; onPickHour: (hour: number, event: { shiftKey: boolean }) => void }) {
  const elapsed = useElapsed(status.kind === 'running' ? status.startedAt : null)
  if (status.kind === 'loading') {
    return (
      <div className={styles.strip} aria-hidden>
        {Array.from({ length: 24 }, (_, i) => (
          <span key={i} className={`${styles.cell} ${styles.cellSkeleton}`} />
        ))}
      </div>
    )
  }
  if (status.kind === 'missing') return <div className={styles.stripMessage}>— Прогноз не рассчитан</div>
  if (status.kind === 'running') {
    return (
      <div className={styles.stripMessage}>
        <Clock3 size={12} aria-hidden /> Идёт расчёт · {formatElapsed(elapsed)}
      </div>
    )
  }
  if (status.kind === 'failed' || status.kind === 'error') {
    return (
      <div className={`${styles.stripMessage} ${styles.stripError}`}>
        <AlertCircle size={12} aria-hidden /> Ошибка расчёта
      </div>
    )
  }
  const values = dayValues(status, date) ?? []
  return (
    <div className={styles.strip}>
      {values.map((value, h) => {
        const off = isServiceOff(h)
        const color = off ? null : loadHex(value)
        return (
          <span
            key={h}
            role="button"
            tabIndex={-1}
            className={`${styles.cell} ${off ? styles.cellOff : ''} ${h === hour ? styles.cellSelected : ''}`}
            style={color ? { background: color } : undefined}
            title={`${hourRange(h)} — ${off ? 'терминалы выключены' : `${formatNumber(value)} посадок`}`}
            onClick={(event) => {
              event.stopPropagation()
              onPickHour(h, event)
            }}
          />
        )
      })}
    </div>
  )
}

function Subline({ route, status, hasGeometry }: { route: Route; status: RouteStatus; hasGeometry: boolean }) {
  if (status.kind === 'ready' && status.fallback) {
    return (
      <div className={styles.subline}>
        <span className={styles.flag}>Нет истории · 0</span>
        <span>{route.name}</span>
      </div>
    )
  }
  if (!hasGeometry) {
    return (
      <div className={styles.subline}>
        <MapPinOff size={12} aria-hidden />
        <span>Нет схемы маршрута</span>
      </div>
    )
  }
  return (
    <div className={styles.subline}>
      <span>{route.name}</span>
    </div>
  )
}

export function RouteBoard({ state, routes, forecasts, withGeometry }: RouteBoardProps) {
  const [sort, setSort] = useState<Sort>('number')
  const selected = activeSlot(state).routeId
  const forecastRoutes = routes.filter((route) => route.forecast_enabled)
  const schemaRoutes = routes.filter((route) => !route.forecast_enabled)

  const rows = useMemo(() => {
    const items = forecastRoutes.map((route) => {
      const status = routeStatus(forecasts.get(route.id))
      const values = dayValues(status, state.date)
      return { route, status, values, total: values ? sum(values) : null }
    })
    if (sort === 'total') items.sort((a, b) => (b.total ?? -1) - (a.total ?? -1))
    return items
  }, [forecastRoutes, forecasts, state.date, sort])

  const networkTotal = rows.reduce((total, row) => total + (row.total ?? 0), 0)
  const ready = rows.filter((row) => row.status.kind === 'ready' && !row.status.fallback).length
  const fallback = rows.filter((row) => row.status.kind === 'ready' && row.status.fallback).length
  const pending = rows.length - ready - fallback

  const letterOf = (routeId: number) => {
    const slot = state.slots.find((item) => item.routeId === routeId)
    return slot ? slotLetter(state, slot) : null
  }

  const open = (routeId: number, event: { shiftKey: boolean }, hour?: number) => {
    if (event.shiftKey && actions.addSlot(routeId)) {
      if (hour !== undefined) actions.setHour(hour)
      return
    }
    actions.selectRoute(routeId, hour)
  }

  return (
    <aside className={styles.board} aria-label="Маршруты">
      <div className={styles.header}>
        <h2 className={styles.title}>Маршруты</h2>
        <span className={styles.subtitle}>{formatMedium(state.date)}</span>
        <Segmented
          label="Порядок"
          value={sort}
          options={[
            { value: 'number', label: 'По номеру' },
            { value: 'total', label: 'По сумме' },
          ]}
          onChange={setSort}
        />
      </div>

      <div className={styles.ruler} aria-hidden>
        <div className={styles.rulerHours}>
          {Array.from({ length: 24 }, (_, h) => (
            <button key={h} type="button" tabIndex={-1} className={styles.rulerTick} aria-pressed={h === state.hour} onClick={() => actions.setHour(h)}>
              {h % 6 === 0 ? <span>{hourLabel(h)}</span> : null}
            </button>
          ))}
        </div>
        <span className={styles.rulerLabel}>час · сутки</span>
      </div>

      <div className={styles.list} role="list">
        {rows.map(({ route, status, values, total }) => {
          const value = values ? values[state.hour] : null
          return (
            <div
              key={route.id}
              role="listitem"
              tabIndex={0}
              className={styles.row}
              aria-current={route.id === selected}
              aria-label={`Маршрут ${route.route_number}`}
              onClick={(event) => open(route.id, event)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  open(route.id, event)
                }
              }}
            >
              <RouteBadge routeNumber={route.route_number} letter={letterOf(route.id)} />
              <Strip status={status} date={state.date} hour={state.hour} onPickHour={(hour, event) => open(route.id, event, hour)} />
              <span className={styles.value} title={hourRange(state.hour)}>
                {value === null || value === undefined ? '—' : isServiceOff(state.hour) ? '0' : formatNumber(value)}
              </span>
              <span className={styles.sum} title="Сумма за сутки">
                {total === null ? '—' : formatNumber(total)}
              </span>
              <Subline route={route} status={status} hasGeometry={withGeometry.has(route.id)} />
            </div>
          )
        })}
      </div>

      {schemaRoutes.length > 0 ? (
        <div className={styles.schema}>
          <span className={styles.schemaTitle}>Только схема · прогноз не строится</span>
          {schemaRoutes.map((route) => (
            <button
              key={route.id}
              type="button"
              className={styles.schemaButton}
              aria-current={route.id === selected}
              title={route.name ?? undefined}
              onClick={(event) => open(route.id, event)}
            >
              <RouteBadge routeNumber={route.route_number} outline letter={letterOf(route.id)} />
            </button>
          ))}
        </div>
      ) : null}

      <div className={styles.footer}>
        <span>
          Сеть за сутки <strong>{formatNumber(networkTotal)}</strong> · демо
        </span>
        <span>
          Готово {ready}
          {fallback ? ` · нет истории ${fallback}` : ''}
          {pending ? ` · без результата ${pending}` : ''}
        </span>
      </div>
    </aside>
  )
}
