import { dayType, horizonDates, weekday } from '../lib/calendar'
import type { ForecastPoint, ForecastResult, ForecastSlice, ForecastSource, ForecastVersion } from './types'

export type DemoScenario = 'ready' | 'mixed'

// Rounded daily boardings of each contest route, derived from the organizers'
// labels (January–October 2025). Route 5 has no history: zero fallback.
const DAILY_SCALE: Record<number, number> = {
  1: 19000,
  5: 0,
  7: 24000,
  11: 31000,
  12: 35000,
  17: 51000,
  25: 6700,
  26: 17000,
  28: 9600,
  50: 24000,
}

// Relative weight of each hour. Hours 1–4 are closed; hour 5 counts only
// boardings from 05:30 and is about 7 % of hour 6.
const WORKDAY_PROFILE = [0.8, 0, 0, 0, 0, 0.35, 5, 7.6, 8.4, 6.8, 5.2, 4.8, 5, 5.2, 5.4, 5.9, 6.8, 8, 7.9, 6, 4.4, 3.3, 2.3, 1.5]
const WEEKEND_PROFILE = [1, 0, 0, 0, 0, 0.17, 2.4, 3.4, 4.4, 5.4, 6.4, 7, 7.4, 7.6, 7.6, 7.4, 7.2, 7, 6.6, 5.6, 4.4, 3.4, 2.4, 1.6]

function normalize(profile: number[]): number[] {
  const total = profile.reduce((sum, value) => sum + value, 0)
  return profile.map((value) => value / total)
}

const WORKDAY = normalize(WORKDAY_PROFILE)
const WEEKEND = normalize(WEEKEND_PROFILE)

export const DEMO_VERSION: ForecastVersion = {
  id: 'demo-2025-11',
  modelVersion: 'demo-profile-0 (не ML)',
  datasetVersion: 'профиль января–октября 2025, округлённые суточные масштабы',
  historyEnd: '2025-11-01T00:00:00+03:00',
  forecastFrom: '2025-11-01T00:00:00+03:00',
  forecastTo: '2026-01-01T00:00:00+03:00',
  timezone: 'Europe/Moscow',
  demo: true,
}

// Deterministic noise: the same route, date and hour always give the same value.
function hash(text: string): number {
  let value = 2166136261
  for (let i = 0; i < text.length; i++) {
    value ^= text.charCodeAt(i)
    value = Math.imul(value, 16777619)
  }
  return value >>> 0
}

function noise(seed: string): number {
  const value = hash(seed)
  return ((value % 10001) / 10000) * 0.16 - 0.08
}

function dayFactor(date: string): number {
  const type = dayType(date)
  if (type === 'holiday') return 0.6
  if (type === 'weekend') return weekday(date) === 6 ? 0.75 : 0.65
  // Slightly quieter working days before the New Year.
  return date >= '2025-12-26' ? 0.9 : 1
}

/** Half-up rounding for non-negative values, as the ML contract requires. */
export function roundHalfUp(value: number): number {
  return Math.floor(Math.max(0, value) + 0.5)
}

export function demoPoints(routeNumber: number): ForecastPoint[] {
  const scale = DAILY_SCALE[routeNumber] ?? 0
  const points: ForecastPoint[] = []
  for (const date of horizonDates()) {
    const profile = dayType(date) === 'working' ? WORKDAY : WEEKEND
    const daily = scale * dayFactor(date)
    for (let hour = 0; hour < 24; hour++) {
      const expected = daily * (profile[hour] ?? 0)
      const value = expected === 0 ? 0 : roundHalfUp(expected * (1 + noise(`${routeNumber}|${date}|${hour}`)))
      points.push({ date, weekday: weekday(date), hour, boardings: value })
    }
  }
  return points
}

const SCENARIO_STARTED_AT = Date.now() - 42_000

export class DemoForecastSource implements ForecastSource {
  constructor(private readonly scenario: DemoScenario) {}

  version(): ForecastVersion {
    return DEMO_VERSION
  }

  async forecast(route: { id: number; routeNumber: number }): Promise<ForecastResult> {
    // A short delay keeps loading states visible, as with the real API.
    await new Promise((resolve) => setTimeout(resolve, 120))
    if (this.scenario === 'mixed') {
      if (route.routeNumber === 1) return { status: 'missing' }
      if (route.routeNumber === 11) return { status: 'running', jobId: '3f2a9c1e-5b7d-4e21-9a0f-2c6d8e4b1a73', startedAt: SCENARIO_STARTED_AT }
      if (route.routeNumber === 12) return { status: 'failed', jobId: '8d41e2b0-7c3a-4f65-b1d9-0e5a6c7f2b18', code: 'model_unavailable' }
    }
    const slice: ForecastSlice = {
      forecast_version_id: DEMO_VERSION.id,
      route_id: route.id,
      timezone: 'Europe/Moscow',
      from: DEMO_VERSION.forecastFrom,
      to: DEMO_VERSION.forecastTo,
      points: demoPoints(route.routeNumber),
    }
    return { status: 'ready', slice, fallback: route.routeNumber === 5 }
  }
}
