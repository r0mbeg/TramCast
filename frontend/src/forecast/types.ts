// A slice of POST /api/predictions/query and GET /api/predictions.

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

/** The version every forecast request of the session pins. */
export interface ForecastVersion {
  id: string
  modelVersion: string
  datasetVersion: string
  historyEnd: string
  forecastFrom: string
  forecastTo: string
  timezone: 'Europe/Moscow'
  /** A replay serves a saved result and is never a live model version. */
  replay: boolean
}

/** Why a route has no forecast and no job. */
export type MissingReason = 'no_version' | 'queue_full' | 'version_inactive'

export type ForecastResult =
  | { status: 'ready'; slice: ForecastSlice; fallback: boolean }
  | { status: 'missing'; reason: MissingReason; retryMs?: number }
  /** startedAt is when this page first saw the job, not when it started. */
  | { status: 'running'; jobId: string; jobStatus: 'queued' | 'running'; startedAt: number; pollMs: number }
  | { status: 'failed'; jobId: string }

/** Hourly values by date: 24 integers per day of the horizon. */
export type DailySeries = Map<string, number[]>
