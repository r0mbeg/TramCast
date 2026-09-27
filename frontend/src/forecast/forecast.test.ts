import { describe, expect, it } from 'vitest'
import { horizonDates, weekday } from '../lib/calendar'
import { dayTotals, maxHourly, niceCeil, peak, sum, toDailySeries } from './aggregate'
import type { ForecastPoint } from './types'

/** A full horizon with a morning peak and zeros in hours 1–4. */
function fixturePoints(): ForecastPoint[] {
  return horizonDates().flatMap((date, day) =>
    Array.from({ length: 24 }, (_, hour) => ({
      date,
      weekday: weekday(date),
      hour,
      boardings: hour >= 1 && hour <= 4 ? 0 : 100 + day + (hour === 8 ? 900 : hour * 10),
    })),
  )
}

describe('aggregates', () => {
  const series = toDailySeries({ forecast_version_id: 'x', route_id: 1, timezone: 'Europe/Moscow', from: '', to: '', points: fixturePoints() })

  it('groups the points by date with 24 hours each', () => {
    expect(series.size).toBe(61)
    expect(series.get('2025-11-01')).toHaveLength(24)
    expect(series.get('2025-11-01')![8]).toBe(1000)
    expect(series.get('2025-12-31')![0]).toBe(160)
  })

  it('sums days from integer hours and leaves dates outside the horizon empty', () => {
    const week = dayTotals(series, ['2025-12-29', '2025-12-30', '2025-12-31', '2026-01-01'])
    expect(week[0]!.total).toBe(sum(series.get('2025-12-29')!))
    expect(week[3]!.total).toBeNull()
  })

  it('finds maxima and nice axis bounds', () => {
    expect(peak(series.get('2025-11-05')!).index).toBe(8)
    expect(maxHourly(series)).toBe(1060)
    expect([0, 7, 1234, 2600, 9001].map(niceCeil)).toEqual([10, 10, 2000, 5000, 10000])
  })
})
