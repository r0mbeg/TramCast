import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import type { Route } from '../api/catalog'
import { ApiError } from '../api/client'
import { getActiveVersion } from '../api/forecasts'
import { toDailySeries } from './aggregate'
import { advance, nextStepIn, toVersion } from './lifecycle'
import type { DailySeries, ForecastResult, ForecastVersion } from './types'

export type VersionState =
  | { status: 'loading' }
  /** The server has no active version: 503 no_active_forecast_version. */
  | { status: 'none' }
  | { status: 'error'; retry: () => void }
  | { status: 'ready'; version: ForecastVersion }

function noActiveVersion(error: unknown): boolean {
  return error instanceof ApiError && error.code === 'no_active_forecast_version'
}

/** Client errors repeat the same answer; only server and network failures are retried. */
function retryable(count: number, error: unknown): boolean {
  return count < 2 && !(error instanceof ApiError && error.status < 500)
}

/**
 * The active version, read once a session: every request pins its ID, so a
 * switch on the server does not change what an open page shows.
 */
export function useForecastVersion(): VersionState {
  const query = useQuery({
    queryKey: ['forecast-version', 'active'],
    queryFn: async () => toVersion(await getActiveVersion()),
    staleTime: Infinity,
    gcTime: Infinity,
    retry: (count, error) => !noActiveVersion(error) && retryable(count, error),
    // An administrator may activate a version while the page is open.
    refetchInterval: (query) => (noActiveVersion(query.state.error) ? 30_000 : false),
  })
  if (query.data) return { status: 'ready', version: query.data }
  if (query.isPending) return { status: 'loading' }
  if (noActiveVersion(query.error)) return { status: 'none' }
  return { status: 'error', retry: () => void query.refetch() }
}

export interface RouteForecast {
  result: ForecastResult
  series: DailySeries | null
}

export type ForecastState = { loading: true } | { loading: false; error: true; retry: () => void } | ({ loading: false; error: false } & RouteForecast)

const NO_VERSION: ForecastState = { loading: false, error: false, result: { status: 'missing', reason: 'no_version' }, series: null }

/**
 * Forecast state per route ID for every forecast-enabled route. Each route
 * takes the whole horizon of the version in one query, so every switch of
 * date, hour or period is local filtering; while its job runs the query polls
 * the job and then reads the slice.
 */
export function useForecasts(routes: readonly Route[], versionState: VersionState): Map<number, ForecastState> {
  const client = useQueryClient()
  const version = versionState.status === 'ready' ? versionState.version : null
  const enabled = routes.filter((route) => route.forecast_enabled)
  return useQueries({
    queries: enabled.map((route) => ({
      queryKey: ['forecast', version?.id ?? null, route.id],
      queryFn: async ({ queryKey }: { queryKey: readonly unknown[] }): Promise<RouteForecast> => {
        const previous = client.getQueryData<RouteForecast>(queryKey)?.result
        const result = await advance(previous, version!, { id: route.id, routeNumber: route.route_number })
        return { result, series: result.status === 'ready' ? toDailySeries(result.slice) : null }
      },
      enabled: version !== null,
      staleTime: Infinity,
      gcTime: Infinity,
      retry: retryable,
      refetchInterval: (query: { state: { data?: RouteForecast } }) => nextStepIn(query.state.data?.result),
    })),
    combine: (results) =>
      new Map(
        results.map((query, index): [number, ForecastState] => {
          const id = enabled[index]!.id
          if (versionState.status === 'none') return [id, NO_VERSION]
          if (versionState.status === 'error') return [id, { loading: false, error: true, retry: versionState.retry }]
          if (query.isError) return [id, { loading: false, error: true, retry: () => void query.refetch() }]
          if (!query.data) return [id, { loading: true }]
          return [id, { loading: false, error: false, ...query.data }]
        }),
      ),
  })
}
