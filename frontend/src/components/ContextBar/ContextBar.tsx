import { AlertCircle, ChevronLeft, ChevronRight, History, Keyboard } from 'lucide-react'
import type { ReactNode } from 'react'
import type { VersionState } from '../../forecast/useForecast'
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
import { Button, Kbd, uiStyles } from '../ui/controls'
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

/** 01.11.2025 from an RFC 3339 time in Moscow, as the API sends it. */
function dayOf(time: string): string {
  return `${time.slice(8, 10)}.${time.slice(5, 7)}.${time.slice(0, 4)}`
}

/** The last day of [from, to): the day before an end at midnight. */
function lastDay(to: string): string {
  const date = to.slice(0, 10)
  return dayOf(to.slice(11, 16) === '00:00' ? addDays(date, -1) : date)
}

function versionLabel(version: VersionState): ReactNode {
  switch (version.status) {
    case 'ready':
      return (
        <>
          Версия <span className={styles.versionModel}>{version.version.modelVersion}</span> · история до {dayOf(version.version.historyEnd)}
        </>
      )
    case 'loading':
      return 'Версия: загрузка…'
    case 'none':
      return 'Нет активной версии прогноза'
    case 'error':
      return 'Версия прогноза недоступна'
  }
}

function VersionDetails({ version }: { version: VersionState }) {
  if (version.status === 'none') {
    return (
      <p className={styles.popoverText}>
        Сервер не назначил активную версию прогноза, поэтому расчёт не запускается. Её регистрирует и активирует администратор: <code>make forecast-version-register ACTIVATE=1</code>. Страница проверяет снова каждые 30 с.
      </p>
    )
  }
  if (version.status === 'error') {
    return (
      <>
        <p className={styles.popoverText}>Не удалось получить версию прогноза: сервер недоступен или ответил ошибкой.</p>
        <Button onClick={version.retry}>Повторить</Button>
      </>
    )
  }
  if (version.status === 'loading') return <p className={styles.popoverText}>Загрузка…</p>
  const { id, modelVersion, datasetVersion, historyEnd, forecastFrom, forecastTo, timezone } = version.version
  return (
    <>
      <dl className={styles.definition}>
        <dt>Модель</dt>
        <dd>{modelVersion}</dd>
        <dt>Данные</dt>
        <dd>{datasetVersion}</dd>
        <dt>ID версии</dt>
        <dd>{id}</dd>
        <dt>История</dt>
        <dd>
          до {dayOf(historyEnd)} {historyEnd.slice(11, 16)} МСК, не включая
        </dd>
        <dt>Период</dt>
        <dd>
          {dayOf(forecastFrom)} — {lastDay(forecastTo)}, {timezone}
        </dd>
        <dt>Значения</dt>
        <dd>целые посадки за час, округление half-up</dd>
        <dt>№ 5</dt>
        <dd>нулевой fallback: нет истории валидаций</dd>
      </dl>
      <p className={styles.legend}>Конкурсная выгрузка: 10 маршрутов × 61 день × 24 часа = 14 640 строк.</p>
      <Button disabled>Выгрузить конкурсный CSV</Button>
      <p className={styles.actionResult}>Выгрузка из приложения — следующий этап.</p>
    </>
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

export function ContextBar({ state, version }: { state: AppState; version: VersionState }) {
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
          <button
            type="button"
            className={`${styles.versionButton} ${version.status === 'none' || version.status === 'error' ? styles.versionProblem : ''}`}
            title={version.status === 'ready' ? version.version.modelVersion : undefined}
            {...props}
          >
            {version.status === 'none' || version.status === 'error' ? <AlertCircle size={14} aria-hidden /> : null}
            {versionLabel(version)}
          </button>
        )}
      >
        {() => (
          <>
            <h2 className={uiStyles.popoverTitle}>Версия прогноза</h2>
            <VersionDetails version={version} />
          </>
        )}
      </Popover>

      {version.status === 'ready' && version.version.replay ? (
        <Popover
          label="Replay"
          align="end"
          trigger={(props) => (
            <button type="button" className={`${uiStyles.chip} ${uiStyles.chipDashed}`} {...props}>
              <History size={14} aria-hidden />
              Replay · сохранённый результат
            </button>
          )}
        >
          {() => (
            <>
              <h2 className={uiStyles.popoverTitle}>Replay</h2>
              <p className={styles.popoverText}>
                Эта версия отдаёт заранее сохранённый результат рецепта 030 без пересчёта моделью. Она нужна для разработки и проверки интеграции и не заменяет рабочую версию, которую считает модель на GPU.
              </p>
            </>
          )}
        </Popover>
      ) : null}

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
