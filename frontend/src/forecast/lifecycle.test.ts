import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import { getSlice, queryForecast, type PredictionJob } from '../api/forecasts'
import { advance, nextStepIn, toVersion, type ForecastApi } from './lifecycle'
import type { ForecastResult, ForecastSlice } from './types'

const version = toVersion({
  id: 'b095bd84-4060-4e25-a88c-6e697ab6eea4',
  model_version: 'tabpfn030-0123456789abcdef',
  dataset_version: 'prepared-844b17f7',
  history_end: '2025-11-01T00:00:00+03:00',
  forecast_from: '2025-11-01T00:00:00+03:00',
  forecast_to: '2026-01-01T00:00:00+03:00',
  timezone: 'Europe/Moscow',
})
const route = { id: 16, routeNumber: 1 }
const horizon = { versionId: version.id, routeId: 16, from: '2025-11-01T00:00:00+03:00', to: '2026-01-01T00:00:00+03:00' }

const slice: ForecastSlice = {
  forecast_version_id: version.id,
  route_id: 16,
  timezone: 'Europe/Moscow',
  from: horizon.from,
  to: horizon.to,
  points: [{ date: '2025-11-01', weekday: 6, hour: 0, boardings: 12 }],
}

function job(status: PredictionJob['status']): PredictionJob {
  return { job_id: 'job-1', forecast_version_id: version.id, route_id: 16, status, poll_interval_seconds: status === 'queued' || status === 'running' ? 3 : null }
}

/** An API whose calls fail unless a test sets them. */
function fakeApi(calls: Partial<ForecastApi>): ForecastApi {
  const unexpected = (name: string) => () => Promise.reject(new Error(`unexpected ${name}`))
  return { query: unexpected('query'), job: unexpected('job'), slice: unexpected('slice'), ...calls }
}

describe('forecast lifecycle', () => {
  it('reads a ready slice at once', async () => {
    const query = vi.fn(async () => ({ kind: 'slice' as const, slice }))
    expect(await advance(undefined, version, route, fakeApi({ query }))).toEqual({ status: 'ready', slice, fallback: false })
    // The whole horizon of the pinned version, whatever date is on screen.
    expect(query).toHaveBeenCalledWith(horizon)
  })

  it('names route 5 a zero fallback', async () => {
    const result = await advance(undefined, version, { id: 20, routeNumber: 5 }, fakeApi({ query: async () => ({ kind: 'slice', slice }) }))
    expect(result).toMatchObject({ status: 'ready', fallback: true })
  })

  it('polls an admitted job, keeps when it was first seen, then reads the slice', async () => {
    const queued = await advance(undefined, version, route, fakeApi({ query: async () => ({ kind: 'job', job: job('queued') }) }), () => 1000)
    expect(queued).toEqual({ status: 'running', jobId: 'job-1', jobStatus: 'queued', startedAt: 1000, pollMs: 3000 })
    expect(nextStepIn(queued)).toBe(3000)

    const running = await advance(queued, version, route, fakeApi({ job: async () => job('running') }), () => 9000)
    expect(running).toEqual({ status: 'running', jobId: 'job-1', jobStatus: 'running', startedAt: 1000, pollMs: 3000 })

    const read = vi.fn(async () => slice)
    const ready = await advance(running, version, route, fakeApi({ job: async () => job('succeeded'), slice: read }))
    expect(ready).toEqual({ status: 'ready', slice, fallback: false })
    expect(read).toHaveBeenCalledWith(horizon)
    expect(nextStepIn(ready)).toBe(false)
  })

  it('stops at a failed job, whether polled or queried', async () => {
    const running: ForecastResult = { status: 'running', jobId: 'job-1', jobStatus: 'running', startedAt: 0, pollMs: 2000 }
    expect(await advance(running, version, route, fakeApi({ job: async () => job('failed') }))).toEqual({ status: 'failed', jobId: 'job-1' })
    const failed = new ApiError(409, 'prediction_failed', { error: 'prediction_failed', job_id: 'job-7' })
    const result = await advance(undefined, version, route, fakeApi({ query: () => Promise.reject(failed) }))
    expect(result).toEqual({ status: 'failed', jobId: 'job-7' })
    expect(nextStepIn(result)).toBe(false)
  })

  it('waits for a full queue and reports an inactive version', async () => {
    const full = await advance(undefined, version, route, fakeApi({ query: () => Promise.reject(new ApiError(503, 'queue_full', {}, 45)) }))
    expect(full).toEqual({ status: 'missing', reason: 'queue_full', retryMs: 45_000 })
    expect(nextStepIn(full)).toBe(45_000)
    // After the wait the step queries again.
    expect(await advance(full, version, route, fakeApi({ query: async () => ({ kind: 'slice', slice }) }))).toMatchObject({ status: 'ready' })

    const inactive = await advance(undefined, version, route, fakeApi({ query: () => Promise.reject(new ApiError(409, 'forecast_version_inactive')) }))
    expect(inactive).toEqual({ status: 'missing', reason: 'version_inactive' })
    expect(nextStepIn(inactive)).toBe(false)
  })

  it('throws other failures for the query to retry', async () => {
    await expect(advance(undefined, version, route, fakeApi({ query: () => Promise.reject(new ApiError(500, 'internal_server_error')) }))).rejects.toThrow('internal_server_error')
    await expect(advance(undefined, version, route, fakeApi({ query: () => Promise.reject(new TypeError('Failed to fetch')) }))).rejects.toThrow('Failed to fetch')
  })

  it('marks a replay version', () => {
    expect(version.replay).toBe(false)
    expect(toVersion({ ...{ id: 'x', dataset_version: 'd', history_end: '', forecast_from: '', forecast_to: '', timezone: 'Europe/Moscow' }, model_version: 'tabpfn030-zhores8477154-replay' }).replay).toBe(true)
  })
})

describe('forecast API requests', () => {
  afterEach(() => vi.unstubAllGlobals())

  function stubFetch(status: number, body: unknown, headers: Record<string, string> = {}) {
    const fetch = vi.fn(async () => new Response(JSON.stringify(body), { status, headers }))
    vi.stubGlobal('fetch', fetch)
    return fetch
  }

  it('posts a JSON query and tells a slice from a job', async () => {
    const fetch = stubFetch(200, slice)
    expect(await queryForecast(horizon)).toEqual({ kind: 'slice', slice })
    const [path, init] = fetch.mock.calls[0] as unknown as [string, RequestInit]
    expect(path).toBe('/api/predictions/query')
    expect(init.method).toBe('POST')
    expect(init.headers).toMatchObject({ 'Content-Type': 'application/json', Accept: 'application/json' })
    expect(JSON.parse(init.body as string)).toEqual({ route_id: 16, from: horizon.from, to: horizon.to, forecast_version_id: version.id })

    stubFetch(202, job('queued'))
    expect(await queryForecast(horizon)).toEqual({ kind: 'job', job: job('queued') })
  })

  it('encodes the offset of a slice read', async () => {
    const fetch = stubFetch(200, slice)
    await getSlice(horizon)
    const [path] = fetch.mock.calls[0] as unknown as [string]
    expect(path).toContain('from=2025-11-01T00%3A00%3A00%2B03%3A00')
    expect(path).toContain(`forecast_version_id=${version.id}`)
  })

  it('turns an error answer into ApiError with its fields and Retry-After', async () => {
    stubFetch(503, { error: 'queue_full' }, { 'Retry-After': '30' })
    const error = await queryForecast(horizon).catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 503, code: 'queue_full', retryAfter: 30 })

    stubFetch(409, { error: 'prediction_failed', job_id: 'job-7' })
    await expect(queryForecast(horizon)).rejects.toMatchObject({ code: 'prediction_failed', body: { job_id: 'job-7' }, retryAfter: null })
  })
})
