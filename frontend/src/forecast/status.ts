import { useEffect, useState } from 'react'
import type { DailySeries } from './types'
import type { ForecastState } from './useForecast'

export type RouteStatus =
  | { kind: 'loading' }
  | { kind: 'error' }
  | { kind: 'missing' }
  | { kind: 'running'; jobId: string; startedAt: number }
  | { kind: 'failed'; jobId: string; code: string }
  | { kind: 'ready'; series: DailySeries; fallback: boolean }

/** One place that turns a query state into what every view shows. */
export function routeStatus(state: ForecastState | undefined): RouteStatus {
  if (!state || state.loading) return { kind: 'loading' }
  if (state.error) return { kind: 'error' }
  const { result, series } = state
  switch (result.status) {
    case 'missing':
      return { kind: 'missing' }
    case 'running':
      return { kind: 'running', jobId: result.jobId, startedAt: result.startedAt }
    case 'failed':
      return { kind: 'failed', jobId: result.jobId, code: result.code }
    case 'ready':
      return series ? { kind: 'ready', series, fallback: result.fallback } : { kind: 'error' }
  }
}

/** Seconds since a moment, updated every second while mounted. */
export function useElapsed(since: number | null): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (since === null) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [since])
  return since === null ? 0 : Math.max(0, Math.floor((now - since) / 1000))
}
