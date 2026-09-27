import { describe, expect, it } from 'vitest'
import { dayTotals, maxHourly, niceCeil, peak, sum, toDailySeries } from './aggregate'
import { DEMO_VERSION, DemoForecastSource, demoPoints, roundHalfUp } from './demo'

describe('demo forecast', () => {
  it('is deterministic and covers the full grid', () => {
    const first = demoPoints(11)
    expect(first).toHaveLength(61 * 24)
    expect(demoPoints(11)).toEqual(first)
    expect(first.every((point) => Number.isInteger(point.boardings) && point.boardings >= 0)).toBe(true)
  })

  it('keeps hours 1–4 at zero and route 5 at the zero fallback', () => {
    expect(demoPoints(17).filter((point) => point.hour >= 1 && point.hour <= 4).every((point) => point.boardings === 0)).toBe(true)
    expect(demoPoints(5).every((point) => point.boardings === 0)).toBe(true)
  })

  it('makes working days busier than holidays', () => {
    const series = toDailySeries({ forecast_version_id: 'x', route_id: 1, timezone: 'Europe/Moscow', from: '', to: '', points: demoPoints(12) })
    expect(sum(series.get('2025-11-05')!)).toBeGreaterThan(sum(series.get('2025-11-04')!))
    expect(peak(series.get('2025-11-05')!).index).toBeGreaterThanOrEqual(7)
  })

  it('rounds half up', () => {
    expect([0.5, 1.5, 2.49, -3].map(roundHalfUp)).toEqual([1, 2, 2, 0])
  })

  it('answers in the shape of the future API, with scenario states', async () => {
    const ready = await new DemoForecastSource('ready').forecast({ id: 24, routeNumber: 11 })
    expect(ready.status).toBe('ready')
    if (ready.status === 'ready') {
      expect(ready.slice.forecast_version_id).toBe(DEMO_VERSION.id)
      expect(ready.slice.route_id).toBe(24)
      expect(ready.slice.points).toHaveLength(1464)
      expect(ready.fallback).toBe(false)
    }
    const mixed = new DemoForecastSource('mixed')
    expect((await mixed.forecast({ id: 16, routeNumber: 1 })).status).toBe('missing')
    expect((await mixed.forecast({ id: 24, routeNumber: 11 })).status).toBe('running')
    expect((await mixed.forecast({ id: 25, routeNumber: 12 })).status).toBe('failed')
    const fallback = await mixed.forecast({ id: 20, routeNumber: 5 })
    expect(fallback.status === 'ready' && fallback.fallback).toBe(true)
  })
})

describe('aggregates', () => {
  const series = toDailySeries({
    forecast_version_id: 'x',
    route_id: 1,
    timezone: 'Europe/Moscow',
    from: '',
    to: '',
    points: demoPoints(7),
  })

  it('sums days from integer hours and leaves dates outside the horizon empty', () => {
    const week = dayTotals(series, ['2025-12-29', '2025-12-30', '2025-12-31', '2026-01-01'])
    expect(week[0]!.total).toBe(sum(series.get('2025-12-29')!))
    expect(week[3]!.total).toBeNull()
  })

  it('finds maxima and nice axis bounds', () => {
    expect(maxHourly(series)).toBeGreaterThan(0)
    expect([0, 7, 1234, 2600, 9001].map(niceCeil)).toEqual([10, 10, 2000, 5000, 10000])
  })
})
