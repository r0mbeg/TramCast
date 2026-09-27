// Dates are ISO strings "YYYY-MM-DD" in Europe/Moscow. Arithmetic runs on UTC
// midnights, which avoids daylight-saving and local time-zone effects.

export const HORIZON_START = '2025-11-01'
export const HORIZON_END = '2025-12-31'

export type DayType = 'working' | 'weekend' | 'holiday'

// Russian production calendar 2025: Saturday 1 November is a working day,
// 2–4 November and 31 December are days off.
const WORKING_WEEKENDS = new Set(['2025-11-01'])
const HOLIDAYS = new Set(['2025-11-03', '2025-11-04', '2025-12-31'])

const MONTHS_GENITIVE = [
  'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
  'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря',
]
const MONTHS_NOMINATIVE = [
  'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
  'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь',
]
const WEEKDAYS_SHORT = ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс']

function toUtc(date: string): Date {
  return new Date(`${date}T00:00:00Z`)
}

function fromUtc(value: Date): string {
  return value.toISOString().slice(0, 10)
}

export function isIsoDate(value: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && fromUtc(toUtc(value)) === value
}

export function addDays(date: string, days: number): string {
  const value = toUtc(date)
  value.setUTCDate(value.getUTCDate() + days)
  return fromUtc(value)
}

/** ISO weekday: 1 = Monday … 7 = Sunday. */
export function weekday(date: string): number {
  const day = toUtc(date).getUTCDay()
  return day === 0 ? 7 : day
}

export function inHorizon(date: string): boolean {
  return date >= HORIZON_START && date <= HORIZON_END
}

export function clampToHorizon(date: string): string {
  if (date < HORIZON_START) return HORIZON_START
  if (date > HORIZON_END) return HORIZON_END
  return date
}

export function dayType(date: string): DayType {
  if (HOLIDAYS.has(date)) return 'holiday'
  if (WORKING_WEEKENDS.has(date)) return 'working'
  return weekday(date) >= 6 ? 'weekend' : 'working'
}

export function dayTypeLabel(type: DayType): string {
  return { working: 'рабочий день', weekend: 'выходной', holiday: 'праздник' }[type]
}

/** Monday..Sunday of the ISO week containing date. */
export function weekDates(date: string): string[] {
  const monday = addDays(date, 1 - weekday(date))
  return Array.from({ length: 7 }, (_, i) => addDays(monday, i))
}

export function monthDates(date: string): string[] {
  const [year, month] = date.split('-').map(Number) as [number, number]
  const days = new Date(Date.UTC(year, month, 0)).getUTCDate()
  const prefix = date.slice(0, 8)
  return Array.from({ length: days }, (_, i) => `${prefix}${String(i + 1).padStart(2, '0')}`)
}

export function horizonDates(): string[] {
  const dates: string[] = []
  for (let date = HORIZON_START; date <= HORIZON_END; date = addDays(date, 1)) dates.push(date)
  return dates
}

function parts(date: string): { day: number; month: number; year: number } {
  const [year, month, day] = date.split('-').map(Number) as [number, number, number]
  return { day, month, year }
}

/** "ср, 5 ноября 2025" */
export function formatLong(date: string): string {
  const { day, month, year } = parts(date)
  return `${WEEKDAYS_SHORT[weekday(date) - 1]}, ${day} ${MONTHS_GENITIVE[month - 1]} ${year}`
}

/** "ср, 5 ноября" */
export function formatMedium(date: string): string {
  const { day, month } = parts(date)
  return `${WEEKDAYS_SHORT[weekday(date) - 1]}, ${day} ${MONTHS_GENITIVE[month - 1]}`
}

/** "пн 3.11" */
export function formatShort(date: string): string {
  const { day, month } = parts(date)
  return `${WEEKDAYS_SHORT[weekday(date) - 1]} ${day}.${String(month).padStart(2, '0')}`
}

/** "Ноябрь 2025" */
export function formatMonth(date: string): string {
  const { month, year } = parts(date)
  return `${MONTHS_NOMINATIVE[month - 1]} ${year}`
}

/** "3–9 ноября 2025" or "29 декабря 2025 – 4 января 2026" */
export function formatRange(first: string, last: string): string {
  const a = parts(first)
  const b = parts(last)
  if (a.year === b.year && a.month === b.month) return `${a.day}–${b.day} ${MONTHS_GENITIVE[b.month - 1]} ${b.year}`
  if (a.year === b.year) return `${a.day} ${MONTHS_GENITIVE[a.month - 1]} – ${b.day} ${MONTHS_GENITIVE[b.month - 1]} ${b.year}`
  return `${a.day} ${MONTHS_GENITIVE[a.month - 1]} ${a.year} – ${b.day} ${MONTHS_GENITIVE[b.month - 1]} ${b.year}`
}

export function weekdayShort(date: string): string {
  return WEEKDAYS_SHORT[weekday(date) - 1] ?? ''
}
