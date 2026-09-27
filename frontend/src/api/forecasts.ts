import type { ForecastSlice } from '../forecast/types'
import { getJson, requestJson } from './client'

// Shapes of the Go forecast API (see backend/README.md, "HTTP API прогнозов").

export interface ForecastVersionDTO {
  id: string
  model_version: string
  dataset_version: string
  history_end: string
  forecast_from: string
  forecast_to: string
  timezone: 'Europe/Moscow'
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface PredictionJob {
  job_id: string
  forecast_version_id: string
  route_id: number
  status: JobStatus
  /** Null once the job has finished. */
  poll_interval_seconds: number | null
}

/** A route and an interval of a pinned version; bounds are RFC 3339. */
export interface SliceQuery {
  versionId: string
  routeId: number
  from: string
  to: string
}

export type QueryAnswer = { kind: 'slice'; slice: ForecastSlice } | { kind: 'job'; job: PredictionJob }

export function getActiveVersion(): Promise<ForecastVersionDTO> {
  return getJson<ForecastVersionDTO>('/api/forecast-versions/active')
}

/**
 * Returns the slice when the job of the pair has succeeded, otherwise its job,
 * which the server admits when there is none. A failed job, a full queue or an
 * inactive version throw ApiError.
 */
export async function queryForecast(q: SliceQuery): Promise<QueryAnswer> {
  const response = await requestJson<ForecastSlice | PredictionJob>('/api/predictions/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ route_id: q.routeId, from: q.from, to: q.to, forecast_version_id: q.versionId }),
  })
  return response.status === 202 ? { kind: 'job', job: response.body as PredictionJob } : { kind: 'slice', slice: response.body as ForecastSlice }
}

export function getJob(jobId: string): Promise<PredictionJob> {
  return getJson<PredictionJob>(`/api/prediction-jobs/${encodeURIComponent(jobId)}`)
}

/** Reads a published slice; it never admits a job. */
export function getSlice(q: SliceQuery): Promise<ForecastSlice> {
  // URLSearchParams encodes the "+" of the offset, which would read as a space.
  const params = new URLSearchParams({ forecast_version_id: q.versionId, route_id: String(q.routeId), from: q.from, to: q.to })
  return getJson<ForecastSlice>(`/api/predictions?${params}`)
}
