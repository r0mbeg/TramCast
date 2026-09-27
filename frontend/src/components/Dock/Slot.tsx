import { AlertCircle, CalendarClock, ChevronLeft, ChevronRight, Clock3, Columns2, Copy, Info, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useRouteStops, type Route } from '../../api/catalog'
import { dayTotals, peak, sum } from '../../forecast/aggregate'
import { useElapsed, type RouteStatus } from '../../forecast/status'
import { HORIZON_END, HORIZON_START, formatMedium, formatMonth, formatRange, formatShort, monthDates, weekDates } from '../../lib/calendar'
import { formatElapsed, formatNumber, formatPercent, hourRange } from '../../lib/format'
import { actions, maxSlots, slotDate, slotLetter, type AppState, type Period, type Slot as SlotModel } from '../../state/appState'
import { EChart } from '../charts/EChart'
import { dayChartOption, periodChartOption } from '../charts/options'
import { Button, Note, RouteBadge, Segmented } from '../ui/controls'
import styles from './Dock.module.css'

export interface SlotScale {
  day: number
  period: number
}

interface SlotProps {
  state: AppState
  slot: SlotModel
  route: Route
  status: RouteStatus
  scale: SlotScale
  /** Total of the first slot's period, for "на N % больше, чем A". */
  reference: { letter: string; period: Period; total: number } | null
}

const PERIOD_OPTIONS: readonly { value: Period; label: string; hint: string }[] = [
  { value: 'day', label: 'День', hint: 'По часам (D)' },
  { value: 'week', label: 'Неделя', hint: 'По дням (W)' },
  { value: 'month', label: 'Месяц', hint: 'По дням (M)' },
]

export function periodDates(period: Period, date: string): string[] {
  if (period === 'week') return weekDates(date)
  if (period === 'month') return monthDates(date)
  return [date]
}

function periodLabel(period: Period, date: string): string {
  if (period === 'week') {
    const dates = weekDates(date)
    return formatRange(dates[0]!, dates[6]!)
  }
  if (period === 'month') return formatMonth(date)
  return formatMedium(date)
}

/** Sum of the slot's period from published hours; null without a ready forecast. */
export function periodTotal(status: RouteStatus, period: Period, date: string): number | null {
  if (status.kind !== 'ready') return null
  return dayTotals(status.series, periodDates(period, date)).reduce((total, day) => total + (day.total ?? 0), 0)
}

function PrototypeButton({ label }: { label: string }) {
  const [phase, setPhase] = useState<'idle' | 'busy' | 'done'>('idle')
  return (
    <>
      <Button
        variant="primary"
        busy={phase === 'busy'}
        disabled={phase === 'busy'}
        onClick={() => {
          setPhase('busy')
          window.setTimeout(() => setPhase('done'), 400)
        }}
      >
        {phase === 'busy' ? 'Запрос…' : label}
      </Button>
      {phase === 'done' ? <p className={styles.centerText}>Прототип: API прогнозов ещё не подключено.</p> : null}
    </>
  )
}

function SchemaOnly({ route }: { route: Route }) {
  const stops = useRouteStops(route.id)
  return (
    <div className={styles.body}>
      <Note icon={<Info size={14} />}>Маршрут не входит в конкурсный набор — прогноз не строится. Ниже порядок остановок из справочника.</Note>
      <div className={styles.stops} style={{ height: 'calc(100% - 3rem)', marginTop: '0.5rem' }}>
        {stops.isPending ? <p className={styles.centerText}>Загрузка остановок…</p> : null}
        {stops.isError ? <p className={`${styles.centerText} ${styles.error}`}>Схема не загрузилась.</p> : null}
        {stops.data?.patterns.map((pattern) => (
          <section key={pattern.pattern_key}>
            <h3>
              Направление {pattern.direction_id} · {pattern.stops.length} остановок
            </h3>
            <ol>
              {pattern.stops.map((stop) => (
                <li key={stop.stop_sequence}>{stop.name}</li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </div>
  )
}

function Running({ status }: { status: Extract<RouteStatus, { kind: 'running' }> }) {
  const elapsed = useElapsed(status.startedAt)
  return (
    <div className={styles.center}>
      <Clock3 size={20} aria-hidden />
      <p className={styles.centerTitle}>Расчёт выполняется · прошло {formatElapsed(elapsed)}</p>
      <p className={styles.centerText}>Задание {status.jobId.slice(0, 8)}… Можно продолжать работу — результат появится здесь.</p>
    </div>
  )
}

function Failed({ status }: { status: Extract<RouteStatus, { kind: 'failed' }> | { kind: 'error'; jobId?: undefined; code?: undefined } }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className={styles.center}>
      <AlertCircle size={20} className={styles.error} aria-hidden />
      <p className={`${styles.centerTitle} ${styles.error}`}>Расчёт не выполнен</p>
      <p className={styles.centerText}>
        {status.code ? `Код ${status.code}` : 'Не удалось получить прогноз'}
        {status.jobId ? ` · задание ${status.jobId.slice(0, 8)}…` : ''}. Автоматически не повторяется.
      </p>
      {status.jobId ? (
        <Button
          onClick={() => {
            void navigator.clipboard?.writeText(status.jobId!).then(() => setCopied(true))
          }}
        >
          <Copy size={14} aria-hidden /> {copied ? 'ID скопирован' : 'Скопировать ID задания'}
        </Button>
      ) : null}
    </div>
  )
}

export function Slot({ state, slot, route, status, scale, reference }: SlotProps) {
  const [width, setWidth] = useState(0)
  const date = slotDate(state, slot)
  const letter = slotLetter(state, slot)
  const active = slot.id === state.activeSlotId && state.slots.length > 1
  const canCompare = state.slots.length < maxSlots(window.innerWidth)
  const total = periodTotal(status, slot.period, date)

  const readout = useMemo(() => {
    if (status.kind !== 'ready') return null
    if (slot.period === 'day') {
      const values = status.series.get(date) ?? []
      const day = sum(values)
      const top = peak(values)
      const value = values[state.hour] ?? 0
      return { values, day, top, value }
    }
    return null
  }, [status, slot.period, date, state.hour])

  const days = useMemo(() => (status.kind === 'ready' && slot.period !== 'day' ? dayTotals(status.series, periodDates(slot.period, date)) : []), [status, slot.period, date])

  const option = useMemo(() => {
    if (status.kind !== 'ready') return null
    if (slot.period === 'day') {
      return dayChartOption({ values: status.series.get(date) ?? new Array(24).fill(0), selectedHour: state.hour, yMax: scale.day, width, demo: true })
    }
    return periodChartOption({
      dates: days.map((day) => day.date),
      totals: days.map((day) => day.total),
      selectedDate: date,
      yMax: scale.period,
      width,
      period: slot.period,
      demo: true,
    })
  }, [status, slot.period, date, state.hour, scale, width, days])

  const periodSummary = () => {
    const known = days.filter((day): day is { date: string; total: number } => day.total !== null)
    if (known.length === 0) return null
    const best = known.reduce((a, b) => (b.total > a.total ? b : a))
    return (
      <>
        <span>
          {periodLabel(slot.period, date)} · всего <strong>{formatNumber(sum(known.map((day) => day.total)))}</strong>
        </span>
        <span className={styles.readoutMeta}>
          максимум {formatShort(best.date)} — {formatNumber(best.total)}
        </span>
      </>
    )
  }

  const delta =
    reference && total !== null && reference.period === slot.period && reference.total > 0 && letter && letter !== reference.letter
      ? (total - reference.total) / reference.total
      : null

  return (
    <section className={`${styles.slot} ${active ? styles.slotActive : ''}`} aria-label={`Маршрут ${route.route_number}${letter ? `, слот ${letter}` : ''}`} onPointerDown={() => actions.setActiveSlot(slot.id)}>
      <div className={styles.slotHeader}>
        <div className={styles.slotTitle}>
          <RouteBadge routeNumber={route.route_number} outline={!route.forecast_enabled} letter={letter} />
          <span className={styles.slotName} title={route.name ?? undefined}>
            {route.name}
          </span>
        </div>
        {route.forecast_enabled ? (
          <>
            <Segmented label="Период" value={slot.period} options={PERIOD_OPTIONS} onChange={(period) => actions.setPeriod(slot.id, period)} />
            <div className={styles.periodStepper}>
              <Button variant="ghost" iconOnly aria-label="Назад на период" title="Назад ([)" disabled={periodDates(slot.period, date)[0]! <= HORIZON_START} onClick={() => actions.stepSlot(slot.id, -1)}>
                <ChevronLeft size={16} />
              </Button>
              <span className={`${styles.periodLabel} ${slot.date ? styles.pinned : ''}`}>{periodLabel(slot.period, date)}</span>
              <Button variant="ghost" iconOnly aria-label="Вперёд на период" title="Вперёд (])" disabled={periodDates(slot.period, date).at(-1)! >= HORIZON_END} onClick={() => actions.stepSlot(slot.id, 1)}>
                <ChevronRight size={16} />
              </Button>
            </div>
            <Button
              variant="ghost"
              iconOnly
              aria-pressed={slot.date !== null}
              aria-label={slot.date ? 'Своя дата закреплена — следовать общей дате' : 'Закрепить свою дату'}
              title={slot.date ? 'Своя дата: слот не следует за общей датой. Щелчок — следовать общей дате' : 'Закрепить свою дату для сравнения'}
              className={slot.date ? styles.pinned : undefined}
              onClick={() => actions.pinSlot(slot.id, slot.date === null)}
            >
              <CalendarClock size={16} />
            </Button>
          </>
        ) : null}
        <Button
          variant="ghost"
          iconOnly
          aria-label="Сравнить в новом слоте"
          title={canCompare ? 'Сравнить: открыть второй слот (Shift+клик по маршруту)' : `Не больше ${maxSlots(window.innerWidth)} слотов на этой ширине экрана`}
          disabled={!canCompare}
          onClick={() => actions.addSlot(null)}
        >
          <Columns2 size={16} />
        </Button>
        <Button variant="ghost" iconOnly aria-label={state.slots.length > 1 ? 'Закрыть слот' : 'Закрыть маршрут'} title="Закрыть (Esc)" onClick={() => actions.clearSlot(slot.id)}>
          <X size={16} />
        </Button>
      </div>

      {route.forecast_enabled && status.kind === 'ready' ? (
        <div className={styles.readout} aria-live="polite">
          {slot.period === 'day' && readout ? (
            <>
              <span>
                {formatMedium(date)} · {hourRange(state.hour)} — <strong>{formatNumber(readout.value)}</strong> посадок
                {readout.day > 0 ? ` · ${formatPercent((readout.value / readout.day) * 100)} суток` : ''}
              </span>
              <span className={styles.readoutMeta}>
                сутки {formatNumber(readout.day)} · пик {hourRange(readout.top.index)}
              </span>
            </>
          ) : (
            periodSummary()
          )}
          {delta !== null ? (
            <span className={styles.delta}>
              {delta === 0 ? `столько же, сколько ${reference!.letter}` : `на ${formatPercent(Math.abs(delta) * 100)} ${delta > 0 ? 'больше' : 'меньше'}, чем ${reference!.letter}`}
            </span>
          ) : null}
          <span className={styles.readoutMeta}>· демо-данные</span>
        </div>
      ) : null}

      {route.forecast_enabled && status.kind === 'ready' && status.fallback ? (
        <div className={styles.fallbackNote}>
          <Note icon={<Info size={14} />}>Нулевой прогноз (fallback): у маршрута № 5 нет истории валидаций.</Note>
        </div>
      ) : null}

      {!route.forecast_enabled ? (
        <SchemaOnly route={route} />
      ) : (
        <div className={styles.body}>
          {status.kind === 'loading' ? (
            <div className={styles.skeleton} aria-label="Загрузка прогноза">
              {Array.from({ length: 24 }, (_, i) => (
                <span key={i} style={{ height: `${20 + ((i * 37) % 60)}%` }} />
              ))}
            </div>
          ) : null}
          {status.kind === 'missing' ? (
            <div className={styles.center}>
              <p className={styles.centerTitle}>Прогноз для маршрута ещё не рассчитан</p>
              <p className={styles.centerText}>Расчёт займёт время; интерфейс останется рабочим, а результат появится здесь.</p>
              <PrototypeButton label="Рассчитать прогноз" />
            </div>
          ) : null}
          {status.kind === 'running' ? <Running status={status} /> : null}
          {status.kind === 'failed' ? <Failed status={status} /> : null}
          {status.kind === 'error' ? <Failed status={{ kind: 'error' }} /> : null}
          {status.kind === 'ready' && status.fallback && slot.period === 'day' ? (
            <div className={styles.center}>
              <p className={styles.centerTitle}>Нулевой прогноз во все часы</p>
            </div>
          ) : null}
          {status.kind === 'ready' && option && !(status.fallback && slot.period === 'day') ? (
            <EChart
              option={option}
              group={state.slots.length > 1 ? `compare-${slot.period}` : undefined}
              label={slot.period === 'day' ? 'Гистограмма посадок по часам' : 'Гистограмма посадок по дням'}
              onWidth={setWidth}
              onBarClick={(index) => {
                if (slot.period === 'day') actions.setHour(index)
                else {
                  const day = days[index]
                  if (day && day.total !== null) actions.openDay(slot.id, day.date)
                }
              }}
            />
          ) : null}
        </div>
      )}
    </section>
  )
}
