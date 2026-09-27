import { useEffect, useState } from 'react'
import type { DailySeries, MissingReason } from './types'
import type { ForecastState } from './useForecast'

export type RouteStatus =
  | { kind: 'loading' }
  | { kind: 'error'; retry: () => void }
  | { kind: 'missing'; reason: MissingReason; retryMs?: number }
  | { kind: 'running'; jobId: string; jobStatus: 'queued' | 'running'; startedAt: number }
  | { kind: 'failed'; jobId: string }
  | { kind: 'ready'; series: DailySeries; fallback: boolean }

/** One place that turns a query state into what every view shows. */
export function routeStatus(state: ForecastState | undefined): RouteStatus {
  if (!state || state.loading) return { kind: 'loading' }
  if (state.error) return { kind: 'error', retry: state.retry }
  const { result, series } = state
  switch (result.status) {
    case 'missing':
      return { kind: 'missing', reason: result.reason, retryMs: result.retryMs }
    case 'running':
      return { kind: 'running', jobId: result.jobId, jobStatus: result.jobStatus, startedAt: result.startedAt }
    case 'failed':
      return { kind: 'failed', jobId: result.jobId }
    case 'ready':
      return series ? { kind: 'ready', series, fallback: result.fallback } : { kind: 'error', retry: () => undefined }
  }
}

/** Short label of a route without a forecast; it fits the narrowest hour strip. */
export const MISSING_LABEL: Record<MissingReason, string> = {
  no_version: 'Нет активной версии прогноза',
  queue_full: 'Очередь заполнена · ждём',
  version_inactive: 'Версия прогноза неактивна',
}

/** A queued job has not reached the model yet; a running one is being computed. */
export function jobLabel(jobStatus: 'queued' | 'running'): string {
  return jobStatus === 'queued' ? 'В очереди' : 'Идёт расчёт'
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
