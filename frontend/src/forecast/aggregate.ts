import { inHorizon } from '../lib/calendar'
import type { DailySeries, ForecastSlice } from './types'

export function toDailySeries(slice: ForecastSlice): DailySeries {
  const series: DailySeries = new Map()
  for (const point of slice.points) {
    let day = series.get(point.date)
    if (!day) {
      day = new Array<number>(24).fill(0)
      series.set(point.date, day)
    }
    day[point.hour] = point.boardings
  }
  return series
}

/** Day totals are sums of published integer hours, never recomputed. */
export function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0)
}

export interface Peak {
  index: number
  value: number
}

export function peak(values: readonly number[]): Peak {
  let best: Peak = { index: 0, value: values[0] ?? 0 }
  values.forEach((value, index) => {
    if (value > best.value) best = { index, value }
  })
  return best
}

export interface DayTotal {
  date: string
  total: number | null
}

/** Totals for the given dates; dates outside the horizon have no value. */
export function dayTotals(series: DailySeries, dates: readonly string[]): DayTotal[] {
  return dates.map((date) => {
    const hours = inHorizon(date) ? series.get(date) : undefined
    return { date, total: hours ? sum(hours) : null }
  })
}

/** Largest hourly value of a route over the horizon, for a stable day axis. */
export function maxHourly(series: DailySeries): number {
  let max = 0
  for (const hours of series.values()) for (const value of hours) if (value > max) max = value
  return max
}

export function maxDaily(series: DailySeries): number {
  let max = 0
  for (const hours of series.values()) max = Math.max(max, sum(hours))
  return max
}

/** Rounds an axis maximum up to 1, 2 or 5 × 10ⁿ. */
export function niceCeil(value: number): number {
  if (value <= 0) return 10
  const power = 10 ** Math.floor(Math.log10(value))
  for (const step of [1, 2, 2.5, 5, 10]) if (value <= step * power) return step * power
  return 10 * power
}
