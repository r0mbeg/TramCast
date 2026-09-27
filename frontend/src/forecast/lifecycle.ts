import { ApiError } from '../api/client'
import { getJob, getSlice, queryForecast, type ForecastVersionDTO, type PredictionJob, type SliceQuery } from '../api/forecasts'
import type { ForecastResult, ForecastSlice, ForecastVersion } from './types'

// Route 5 has no history: by contract the model answers zeros for it, and the
// interface names the reason.
const FALLBACK_ROUTE = 5
const DEFAULT_POLL_MS = 2_000
const DEFAULT_RETRY_MS = 30_000

export function toVersion(dto: ForecastVersionDTO): ForecastVersion {
  return {
    id: dto.id,
    modelVersion: dto.model_version,
    datasetVersion: dto.dataset_version,
    historyEnd: dto.history_end,
    forecastFrom: dto.forecast_from,
    forecastTo: dto.forecast_to,
    timezone: dto.timezone,
    // ponytail: the API has no serving mode; replay bundles name themselves
    // "…-replay". Read a flag instead once the version response carries one.
    replay: dto.model_version.includes('replay'),
  }
}

/** The calls one step makes; tests replace them. */
export interface ForecastApi {
  query: (q: SliceQuery) => ReturnType<typeof queryForecast>
  job: (jobId: string) => Promise<PredictionJob>
  slice: (q: SliceQuery) => Promise<ForecastSlice>
}

const http: ForecastApi = { query: queryForecast, job: getJob, slice: getSlice }

function ready(slice: ForecastSlice, routeNumber: number): ForecastResult {
  return { status: 'ready', slice, fallback: routeNumber === FALLBACK_ROUTE }
}

async function fromJob(job: PredictionJob, q: SliceQuery, routeNumber: number, startedAt: number, api: ForecastApi): Promise<ForecastResult> {
  switch (job.status) {
    case 'succeeded':
      // The client reads the version it pinned, as the contract requires.
      return ready(await api.slice(q), routeNumber)
    case 'failed':
      return { status: 'failed', jobId: job.job_id }
    default:
      return { status: 'running', jobId: job.job_id, jobStatus: job.status, startedAt, pollMs: (job.poll_interval_seconds ?? DEFAULT_POLL_MS / 1000) * 1000 }
  }
}

/**
 * One step of the forecast of a route over the whole horizon of the pinned
 * version. Without a job in progress it queries, which admits a job when there
 * is none; with one it reads the job and, once it has succeeded, the slice.
 * Answers that are states, not failures, become results; anything else throws.
 */
export async function advance(
  previous: ForecastResult | undefined,
  version: ForecastVersion,
  route: { id: number; routeNumber: number },
  api: ForecastApi = http,
  now: () => number = Date.now,
): Promise<ForecastResult> {
  const q: SliceQuery = { versionId: version.id, routeId: route.id, from: version.forecastFrom, to: version.forecastTo }
  try {
    if (previous?.status === 'running') {
      return await fromJob(await api.job(previous.jobId), q, route.routeNumber, previous.startedAt, api)
    }
    const answer = await api.query(q)
    return answer.kind === 'slice' ? ready(answer.slice, route.routeNumber) : await fromJob(answer.job, q, route.routeNumber, now(), api)
  } catch (error) {
    if (error instanceof ApiError) {
      const jobId = error.body.job_id
      if (error.code === 'prediction_failed' && typeof jobId === 'string') return { status: 'failed', jobId }
      if (error.code === 'queue_full') return { status: 'missing', reason: 'queue_full', retryMs: (error.retryAfter ?? DEFAULT_RETRY_MS / 1000) * 1000 }
      if (error.code === 'forecast_version_inactive') return { status: 'missing', reason: 'version_inactive' }
    }
    throw error
  }
}

/** When to take the next step: while a job runs and while the queue is full. */
export function nextStepIn(result: ForecastResult | undefined): number | false {
  if (result?.status === 'running') return result.pollMs
  if (result?.status === 'missing' && result.reason === 'queue_full') return result.retryMs ?? DEFAULT_RETRY_MS
  return false
}
