import { oklch, toHex, type Oklch } from './color'

// "Smouldering ember": on dark surfaces more load is lighter and warmer.
// One unit only — boardings per hour — and the same thresholds for every date.
export const LOAD_THRESHOLDS = [300, 600, 1200, 2000] as const

export const LOAD_BUCKETS: readonly { color: Oklch; label: string }[] = [
  { color: oklch(0.6, 0.075, 50), label: '1–299' },
  { color: oklch(0.68, 0.1, 58), label: '300–599' },
  { color: oklch(0.76, 0.12, 68), label: '600–1 199' },
  { color: oklch(0.84, 0.13, 82), label: '1 200–1 999' },
  { color: oklch(0.92, 0.12, 95), label: '≥ 2 000' },
]

export const LOAD_HEX = LOAD_BUCKETS.map((bucket) => toHex(bucket.color))

/** 0 for a zero forecast, otherwise 1–5. */
export function loadBucket(boardings: number): number {
  if (boardings <= 0) return 0
  let bucket = 1
  for (const threshold of LOAD_THRESHOLDS) if (boardings >= threshold) bucket++
  return bucket
}

export function loadHex(boardings: number): string | null {
  const bucket = loadBucket(boardings)
  return bucket === 0 ? null : (LOAD_HEX[bucket - 1] ?? null)
}
