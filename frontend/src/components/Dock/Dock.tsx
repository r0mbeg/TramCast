import type { Route } from '../../api/catalog'
import { maxDaily, maxHourly, niceCeil } from '../../forecast/aggregate'
import { routeStatus } from '../../forecast/status'
import type { ForecastState } from '../../forecast/useForecast'
import { actions, slotDate, slotLetter, type AppState } from '../../state/appState'
import { DOCK_ID } from '../../state/layout'
import { DayMatrix } from './DayMatrix'
import styles from './Dock.module.css'
import { Slot, periodTotal, type SlotScale } from './Slot'

interface DockProps {
  state: AppState
  routes: readonly Route[]
  forecasts: Map<number, ForecastState>
}

export function Dock({ state, routes, forecasts }: DockProps) {
  const routesById = new Map(routes.map((route) => [route.id, route]))
  const open = state.slots
    .map((slot) => ({ slot, route: slot.routeId === null ? undefined : routesById.get(slot.routeId) }))
    .filter((item): item is { slot: (typeof state.slots)[number]; route: Route } => item.route !== undefined)

  if (open.length === 0) {
    return (
      <section id={DOCK_ID} className={styles.dock} aria-label="Картина дня">
        <DayMatrix state={state} routes={routes} forecasts={forecasts} />
      </section>
    )
  }

  const statuses = open.map(({ route }) => routeStatus(forecasts.get(route.id)))
  // Each route keeps one axis over the whole horizon so paging dates does not
  // rescale it; a comparison shares one axis so bar heights are comparable.
  const own = statuses.map((status): SlotScale => (status.kind === 'ready' ? { day: niceCeil(maxHourly(status.series)), period: niceCeil(maxDaily(status.series)) } : { day: 10, period: 10 }))
  const shared = state.commonScale && open.length > 1
  const common: SlotScale = { day: Math.max(...own.map((scale) => scale.day)), period: Math.max(...own.map((scale) => scale.period)) }

  const first = open[0]!
  const firstStatus = statuses[0]!
  const firstTotal = periodTotal(firstStatus, first.slot.period, slotDate(state, first.slot))
  const firstLetter = slotLetter(state, first.slot)
  const reference = open.length > 1 && firstTotal !== null && firstLetter ? { letter: firstLetter, period: first.slot.period, total: firstTotal } : null

  return (
    <section id={DOCK_ID} className={styles.dock} aria-label="Прогноз маршрута">
      {state.slots.length > 1 ? (
        <div className={styles.compareBar}>
          <span>Сравнение · дата и час из верхней строки, у слота с закреплённой датой — своя</span>
          <label className={styles.toggle}>
            <input type="checkbox" checked={state.commonScale} onChange={(event) => actions.setCommonScale(event.target.checked)} />
            Общая шкала
          </label>
        </div>
      ) : null}
      <div className={styles.slots}>
        {open.map(({ slot, route }, index) => (
          <Slot key={slot.id} state={state} slot={slot} route={route} status={statuses[index]!} scale={shared ? common : own[index]!} reference={reference} />
        ))}
      </div>
    </section>
  )
}
