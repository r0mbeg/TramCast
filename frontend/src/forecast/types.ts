// The future POST /api/predictions/query answers 200 with a full slice or 202
// with a job (AGENTS.md). The demo source returns exactly these shapes, so only
// the source changes when the API arrives.

export interface ForecastPoint {
  date: string
  weekday: number
  hour: number
  boardings: number
}

export interface ForecastSlice {
  forecast_version_id: string
  route_id: number
  timezone: 'Europe/Moscow'
  from: string
  to: string
  points: ForecastPoint[]
}

export interface ForecastVersion {
  id: string
  modelVersion: string
  datasetVersion: string
  historyEnd: string
  forecastFrom: string
  forecastTo: string
  timezone: 'Europe/Moscow'
  demo: boolean
}

export type ForecastResult =
  | { status: 'ready'; slice: ForecastSlice; fallback: boolean }
  | { status: 'missing' }
  | { status: 'running'; jobId: string; startedAt: number }
  | { status: 'failed'; jobId: string; code: string }

export interface ForecastSource {
  version(): ForecastVersion
  forecast(route: { id: number; routeNumber: number }): Promise<ForecastResult>
}

/** Hourly values by date: 24 integers per day of the horizon. */
export type DailySeries = Map<string, number[]>
