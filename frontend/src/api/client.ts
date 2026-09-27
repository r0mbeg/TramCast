/** An error answer of the Go API: {"error": "<code>"} plus its other fields. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    readonly body: Readonly<Record<string, unknown>> = {},
    /** Seconds from the Retry-After header, when the server sent one. */
    readonly retryAfter: number | null = null,
  ) {
    super(`API ${status}: ${code}`)
  }
}

export interface ApiResponse<T> {
  status: number
  body: T
}

/** Sends a request and decodes the JSON answer; any non-2xx answer throws ApiError. */
export async function requestJson<T>(path: string, init: RequestInit = {}): Promise<ApiResponse<T>> {
  const response = await fetch(path, { ...init, headers: { Accept: 'application/json', ...init.headers } })
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as Record<string, unknown>
    const retryAfter = Number(response.headers.get('Retry-After'))
    throw new ApiError(
      response.status,
      typeof body.error === 'string' ? body.error : 'unknown_error',
      body,
      Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : null,
    )
  }
  return { status: response.status, body: (await response.json()) as T }
}

export async function getJson<T>(path: string): Promise<T> {
  return (await requestJson<T>(path)).body
}
