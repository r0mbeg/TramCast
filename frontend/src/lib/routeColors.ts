import { oklch, toCss, toHex, type Oklch } from './color'
import { hex } from './theme'

// Only the ten contest routes get an identity color, and all ten are drawn on
// the map. The workbook routes (1, 5, 7, 11, 12) take the most distant hues;
// the OpenStreetMap ones (17, 25, 26, 28, 50) fill the gaps, so neighbouring
// hues rely on the number label next to every line. Warm hues 50–95 belong to
// the load scale and are avoided.
const IDENTITY: Record<number, Oklch> = {
  1: oklch(0.72, 0.14, 250),
  5: oklch(0.72, 0.17, 340),
  7: oklch(0.8, 0.12, 195),
  11: oklch(0.7, 0.17, 295),
  12: oklch(0.76, 0.16, 150),
  17: oklch(0.8, 0.09, 225),
  25: oklch(0.74, 0.12, 5),
  26: oklch(0.78, 0.1, 270),
  28: oklch(0.82, 0.1, 170),
  50: oklch(0.72, 0.08, 315),
}

export function routeColor(routeNumber: number): Oklch | null {
  return IDENTITY[routeNumber] ?? null
}

/** Identity color for contest routes, neutral gray for the others. */
export function routeHex(routeNumber: number): string {
  const color = IDENTITY[routeNumber]
  return color ? toHex(color) : hex.contextLine
}

export function routeCss(routeNumber: number): string {
  const color = IDENTITY[routeNumber]
  return color ? toCss(color) : 'var(--color-context-line)'
}
