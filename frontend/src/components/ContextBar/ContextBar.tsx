import { ChevronLeft, ChevronRight, FlaskConical, Keyboard } from 'lucide-react'
import { useState } from 'react'
import type { ForecastVersion } from '../../forecast/types'
import {
  HORIZON_END,
  HORIZON_START,
  addDays,
  dayType,
  dayTypeLabel,
  formatLong,
  inHorizon,
  monthDates,
  weekday,
} from '../../lib/calendar'
import { hourRange, isServiceOff } from '../../lib/format'
import { actions, type AppState } from '../../state/appState'
import { Button, Kbd, Segmented, uiStyles } from '../ui/controls'
import { Popover } from '../ui/Popover'
import styles from './ContextBar.module.css'

const WEEKDAYS = ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс']

function MonthGrid({ month, title, selected, onPick }: { month: string; title: string; selected: string; onPick: (date: string) => void }) {
  const dates = monthDates(month)
  const offset = weekday(dates[0]!) - 1
  return (
    <div className={styles.month}>
      <div className={styles.monthTitle}>{title}</div>
      {WEEKDAYS.map((name) => (
        <div key={name} className={styles.weekday}>
          {name}
        </div>
      ))}
      {Array.from({ length: offset }, (_, i) => (
        <span key={`gap-${i}`} />
      ))}
      {dates.map((date) => (
        <button
          key={date}
          type="button"
          className={styles.day}
          data-type={dayType(date)}
          aria-current={date === selected ? 'date' : undefined}
          aria-label={`${formatLong(date)}, ${dayTypeLabel(dayType(date))}`}
          disabled={!inHorizon(date)}
          onClick={() => onPick(date)}
        >
          {Number(date.slice(8))}
        </button>
      ))}
    </div>
  )
}

/** A prototype action: the forecast API is not connected yet. */
function PrototypeAction({ label }: { label: string }) {
  const [phase, setPhase] = useState<'idle' | 'busy' | 'done'>('idle')
  return (
    <div>
      <Button
        busy={phase === 'busy'}
        disabled={phase === 'busy'}
        onClick={() => {
          setPhase('busy')
          window.setTimeout(() => setPhase('done'), 400)
        }}
      >
        {phase === 'busy' ? 'Запрос…' : label}
      </Button>
      {phase === 'done' ? <p className={styles.actionResult}>Прототип: API прогнозов ещё не подключено.</p> : null}
    </div>
  )
}

const SHORTCUTS: [string[], string][] = [
  [['↑', '↓'], 'Предыдущий или следующий маршрут'],
  [['0–9'], 'Маршрут по номеру'],
  [['←', '→'], 'Час назад или вперёд'],
  [['Shift', '←/→'], 'День назад или вперёд'],
  [['D', 'W', 'M'], 'День, неделя, месяц'],
  [['[', ']'], 'Шаг периода маршрута'],
  [['F'], 'Показать маршрут на карте, Shift+F — всю сеть'],
  [['Shift', 'клик'], 'Сравнить маршрут в новом слоте'],
  [['Esc'], 'Закрыть маршрут или слот'],
]

export function ContextBar({ state, version }: { state: AppState; version: ForecastVersion }) {
  const type = dayType(state.date)
  return (
    <header className={styles.bar}>
      <h1 className={styles.brand}>TramCast</h1>
      <span className={styles.divider} aria-hidden />

      <div className={styles.stepper}>
        <Button variant="ghost" iconOnly aria-label="Предыдущий день" disabled={state.date <= HORIZON_START} onClick={() => actions.stepDate(-1)}>
          <ChevronLeft size={18} />
        </Button>
        <Popover
          label="Выбор даты"
          trigger={(props) => (
            <button type="button" className={styles.stepperValue} {...props}>
              {formatLong(state.date)}
            </button>
          )}
        >
          {(close) => (
            <>
              <div className={styles.calendars}>
                {['2025-11-01', '2025-12-01'].map((month, index) => (
                  <MonthGrid
                    key={month}
                    month={month}
                    title={index === 0 ? 'Ноябрь 2025' : 'Декабрь 2025'}
                    selected={state.date}
                    onPick={(date) => {
                      actions.setDate(date)
                      close()
                    }}
                  />
                ))}
              </div>
              <p className={styles.legend}>Период прогноза 1 ноября — 31 декабря 2025. Подчёркнуты праздники; 1 ноября — рабочая суббота.</p>
            </>
          )}
        </Popover>
        <Button variant="ghost" iconOnly aria-label="Следующий день" disabled={state.date >= HORIZON_END || !inHorizon(addDays(state.date, 1))} onClick={() => actions.stepDate(1)}>
          <ChevronRight size={18} />
        </Button>
        <span className={styles.meta}>{dayTypeLabel(type)}</span>
      </div>

      <span className={styles.divider} aria-hidden />

      <div className={styles.stepper}>
        <Button variant="ghost" iconOnly aria-label="Предыдущий час" onClick={() => actions.stepHour(-1)}>
          <ChevronLeft size={18} />
        </Button>
        <span className={styles.stepperValue} aria-live="polite">
          {hourRange(state.hour)}
        </span>
        <Button variant="ghost" iconOnly aria-label="Следующий час" onClick={() => actions.stepHour(1)}>
          <ChevronRight size={18} />
        </Button>
        {isServiceOff(state.hour) ? <span className={`${styles.meta} ${styles.off}`}>терминалы выключены</span> : null}
      </div>

      <div className={styles.spacer} />

      <Popover
        label="Версия прогноза"
        align="end"
        trigger={(props) => (
          <button type="button" className={styles.versionButton} {...props}>
            Версия: демо · история до 01.11.2025
          </button>
        )}
      >
        {() => (
          <>
            <h2 className={uiStyles.popoverTitle}>Версия прогноза</h2>
            <dl className={styles.definition}>
              <dt>Модель</dt>
              <dd>{version.modelVersion}</dd>
              <dt>Данные</dt>
              <dd>{version.datasetVersion}</dd>
              <dt>История</dt>
              <dd>до 01.11.2025 00:00 МСК, не включая</dd>
              <dt>Период</dt>
              <dd>01.11–31.12.2025, Europe/Moscow</dd>
              <dt>Значения</dt>
              <dd>целые посадки за час, округление half-up</dd>
              <dt>№ 5</dt>
              <dd>нулевой fallback: нет истории валидаций</dd>
            </dl>
            <p className={styles.legend}>Конкурсная выгрузка: 10 маршрутов × 61 день × 24 часа = 14 640 строк.</p>
            <PrototypeAction label="Выгрузить конкурсный CSV" />
          </>
        )}
      </Popover>

      <Popover
        label="Демо-данные"
        align="end"
        trigger={(props) => (
          <button type="button" className={`${uiStyles.chip} ${uiStyles.chipDemo}`} {...props}>
            <FlaskConical size={14} aria-hidden />
            Демо-данные
          </button>
        )}
      >
        {() => (
          <>
            <h2 className={uiStyles.popoverTitle}>Демо-данные</h2>
            <p className={styles.popoverText}>
              Маршруты, остановки и схемы — настоящие: справочник организаторов и OpenStreetMap. Значения прогноза, версия и состояния расчёта — демонстрационные: API прогнозов ещё не подключено.
            </p>
            <Segmented
              label="Сценарий"
              value={state.scenario}
              options={[
                { value: 'ready', label: 'Всё готово' },
                { value: 'mixed', label: 'Смешанный', hint: '№ 1 — нет прогноза, № 11 — расчёт, № 12 — ошибка' },
              ]}
              onChange={actions.setScenario}
            />
            <p className={styles.legend}>«Смешанный»: у № 1 нет прогноза, у № 11 идёт расчёт, у № 12 ошибка.</p>
          </>
        )}
      </Popover>

      <Popover
        label="Клавиши"
        align="end"
        trigger={(props) => (
          <Button variant="ghost" iconOnly aria-label="Клавиши" {...props}>
            <Keyboard size={18} />
          </Button>
        )}
      >
        {() => (
          <>
            <h2 className={uiStyles.popoverTitle}>Клавиши</h2>
            <dl className={styles.shortcuts}>
              {SHORTCUTS.map(([keys, text]) => (
                <div key={text} style={{ display: 'contents' }}>
                  <dt>
                    {keys.map((key) => (
                      <Kbd key={key}>{key}</Kbd>
                    ))}
                  </dt>
                  <dd>{text}</dd>
                </div>
              ))}
            </dl>
          </>
        )}
      </Popover>
    </header>
  )
}
