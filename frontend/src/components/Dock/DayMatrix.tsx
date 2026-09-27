import { AlertCircle, Clock3 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Route } from '../../api/catalog'
import { peak, sum } from '../../forecast/aggregate'
import { routeStatus, type RouteStatus } from '../../forecast/status'
import type { ForecastState } from '../../forecast/useForecast'
import { formatLong } from '../../lib/calendar'
import { formatNumber, hourLabel, hourRange } from '../../lib/format'
import { LOAD_BUCKETS, LOAD_HEX, loadHex } from '../../lib/loadScale'
import { actions, type AppState } from '../../state/appState'
import { RouteBadge } from '../ui/controls'
import styles from './Dock.module.css'

// Hour columns: 00, the collapsed 01–04 block, then 05…23.
const COLUMNS: { label: string; hours: number[] }[] = [
  { label: '00', hours: [0] },
  { label: '01–04', hours: [1, 2, 3, 4] },
  ...Array.from({ length: 19 }, (_, i) => ({ label: i === 0 ? '05*' : hourLabel(i + 5), hours: [i + 5] })),
]

/** A number fits a cell from about this width; below it color and tooltip remain. */
const NUMBER_MIN_CELL = 36

function rowMessage(status: RouteStatus): { text: string; error?: boolean; running?: boolean } | null {
  switch (status.kind) {
    case 'loading':
      return { text: 'загрузка…' }
    case 'missing':
      return { text: '— Прогноз для маршрута ещё не рассчитан' }
    case 'running':
      return { text: 'Идёт расчёт — результат появится здесь', running: true }
    case 'failed':
    case 'error':
      return { text: 'Ошибка расчёта — откройте маршрут для подробностей', error: true }
    case 'ready':
      return null
  }
}

export function DayMatrix({ state, routes, forecasts }: { state: AppState; routes: readonly Route[]; forecasts: Map<number, ForecastState> }) {
  const container = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!container.current) return
    const observer = new ResizeObserver((entries) => setWidth(entries[0]?.contentRect.width ?? 0))
    observer.observe(container.current)
    return () => observer.disconnect()
  }, [])

  const forecastRoutes = routes.filter((route) => route.forecast_enabled)
  const rows = forecastRoutes.map((route) => {
    const status = routeStatus(forecasts.get(route.id))
    const values = status.kind === 'ready' ? (status.series.get(state.date) ?? null) : null
    return { route, status, values, total: values ? sum(values) : null }
  })
  const maxTotal = Math.max(1, ...rows.map((row) => row.total ?? 0))
  const network = Array.from({ length: 24 }, (_, hour) => rows.reduce((total, row) => total + (row.values?.[hour] ?? 0), 0))
  const networkPeak = peak(network)
  const cellWidth = (width - 72 - 28 - 80 - 22 * 2) / 20
  const showNumbers = cellWidth >= NUMBER_MIN_CELL

  return (
    <>
      <div className={styles.matrixHeader}>
        <h2 className={styles.matrixTitle}>Картина дня</h2>
        <span className={styles.matrixMeta}>{formatLong(state.date)} · демо-данные</span>
        <span className={styles.matrixMeta} style={{ marginLeft: 'auto' }}>
          Сеть за сутки {formatNumber(sum(network))} · пик сети {hourRange(networkPeak.index)}
        </span>
      </div>
      <div ref={container} className={styles.matrix}>
        <div className={styles.grid} role="grid" aria-label="Посадки по маршрутам и часам">
          <div role="row" style={{ display: 'contents' }}>
            <span role="columnheader" className={styles.gridHead}>
              Маршрут
            </span>
            {COLUMNS.map((column) => (
              <button
                key={column.label}
                type="button"
                role="columnheader"
                className={styles.gridHead}
                aria-selected={column.hours.includes(state.hour)}
                title={column.label === '01–04' ? 'Терминалы выключены 01:00–05:30' : column.label === '05*' ? 'Учитываются посадки с 05:30' : hourRange(column.hours[0]!)}
                onClick={() => actions.setHour(column.hours[0]!)}
              >
                {column.label}
              </button>
            ))}
            <span role="columnheader" className={styles.gridHead}>
              Сутки
            </span>
          </div>

          {rows.map(({ route, status, values, total }) => {
            const message = rowMessage(status)
            const rowPeak = values ? peak(values).index : -1
            return (
              <div key={route.id} role="row" style={{ display: 'contents' }}>
                <button type="button" role="rowheader" className={styles.rowHead} onClick={() => actions.selectRoute(route.id)} title={route.name ?? undefined}>
                  <RouteBadge routeNumber={route.route_number} />
                </button>
                {message ? (
                  <div role="gridcell" className={`${styles.gridMessage} ${message.error ? styles.error : ''}`} style={{ gridColumn: '2 / span 21' }}>
                    {message.running ? <Clock3 size={12} aria-hidden /> : null}
                    {message.error ? <AlertCircle size={12} aria-hidden /> : null}
                    {message.text}
                  </div>
                ) : (
                  COLUMNS.map((column) => {
                    if (column.label === '01–04') {
                      return <span key={column.label} role="gridcell" className={`${styles.gridCell} ${styles.gridCellOff}`} title="Терминалы выключены, прогноз 0" />
                    }
                    const hour = column.hours[0]!
                    const value = values?.[hour] ?? 0
                    const color = loadHex(value)
                    return (
                      <button
                        key={column.label}
                        type="button"
                        role="gridcell"
                        className={`${styles.gridCell} ${color ? '' : styles.gridCellZero} ${hour === state.hour ? styles.gridCellSelected : ''}`}
                        style={color ? { background: color, fontWeight: hour === rowPeak ? 600 : 500 } : undefined}
                        title={`№ ${route.route_number} · ${hourRange(hour)} — ${formatNumber(value)} посадок · демо`}
                        onClick={() => actions.selectRoute(route.id, hour)}
                      >
                        {showNumbers || value === 0 ? formatNumber(value) : ''}
                      </button>
                    )
                  })
                )}
                <div role="gridcell" className={styles.gridTotal}>
                  {total === null ? '—' : formatNumber(total)}
                  {total !== null ? <span className={styles.gridBar} style={{ width: `${(total / maxTotal) * 100}%` }} /> : null}
                </div>
              </div>
            )
          })}
        </div>

        <div className={styles.legend}>
          <span>Посадок за час:</span>
          {LOAD_BUCKETS.map((bucket, index) => (
            <span key={bucket.label} className={styles.legendItem}>
              <span className={styles.legendSwatch} style={{ background: LOAD_HEX[index] }} />
              {bucket.label}
            </span>
          ))}
          <span className={styles.legendItem}>
            <span className={styles.legendSwatch} style={{ background: 'var(--color-surface-3)' }} />0
          </span>
          <span className={styles.legendItem}>
            <span className={`${styles.legendSwatch} ${styles.gridCellOff}`} />
            01:00–05:30 терминалы выключены
          </span>
          <span>05* — посадки с 05:30</span>
          <span>Щелчок по ячейке открывает маршрут на этот час</span>
        </div>
      </div>
    </>
  )
}
