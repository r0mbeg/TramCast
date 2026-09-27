import { useQueries } from '@tanstack/react-query'
import type { Route } from '../api/catalog'
import { toDailySeries } from './aggregate'
import { DEMO_VERSION, DemoForecastSource, type DemoScenario } from './demo'
import type { DailySeries, ForecastResult, ForecastSource, ForecastVersion } from './types'

const sources: Record<DemoScenario, ForecastSource> = {
  ready: new DemoForecastSource('ready'),
  mixed: new DemoForecastSource('mixed'),
}

export function forecastVersion(): ForecastVersion {
  return DEMO_VERSION
}

export interface RouteForecast {
  result: ForecastResult
  series: DailySeries | null
}

export type ForecastState = { loading: true } | { loading: false; error: true } | ({ loading: false; error: false } & RouteForecast)

function forecastQuery(route: Route, scenario: DemoScenario) {
  return {
    // A forecast is cached for the whole horizon of its version, so every
    // switch of date, hour or period is local filtering without a request.
    queryKey: ['forecast', DEMO_VERSION.id, scenario, route.id],
    queryFn: async (): Promise<RouteForecast> => {
      const result = await sources[scenario].forecast({ id: route.id, routeNumber: route.route_number })
      return { result, series: result.status === 'ready' ? toDailySeries(result.slice) : null }
    },
    staleTime: Infinity,
    gcTime: Infinity,
  }
}

/** Forecast state per route ID for every forecast-enabled route. */
export function useForecasts(routes: readonly Route[], scenario: DemoScenario): Map<number, ForecastState> {
  const enabled = routes.filter((route) => route.forecast_enabled)
  return useQueries({
    queries: enabled.map((route) => forecastQuery(route, scenario)),
    combine: (results) =>
      new Map(
        results.map((query, index): [number, ForecastState] => {
          const id = enabled[index]!.id
          if (query.isPending) return [id, { loading: true }]
          if (query.isError || !query.data) return [id, { loading: false, error: true }]
          return [id, { loading: false, error: false, ...query.data }]
        }),
      ),
  })
}
