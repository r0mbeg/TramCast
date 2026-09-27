import { oklch, toCss, toHex, type Oklch } from './color'

// Cool neutrals (hue 255, chroma ≤ 0.012) for the chrome. Warm colors always
// mean load; moss (the brand seed, hue 140) only marks controls.
export const tokens = {
  app: oklch(0.165, 0.008, 255),
  land: oklch(0.18, 0.008, 255),
  surface1: oklch(0.195, 0.008, 255),
  surface2: oklch(0.235, 0.009, 255),
  surface3: oklch(0.27, 0.01, 255),
  border: oklch(0.3, 0.01, 255),
  borderControl: oklch(0.52, 0.01, 255),
  text1: oklch(0.94, 0.005, 255),
  text2: oklch(0.76, 0.01, 255),
  text3: oklch(0.66, 0.012, 255),
  disabled: oklch(0.48, 0.01, 255),
  // Selection inside data is white; selection of a control is moss.
  selection: oklch(0.97, 0.005, 255),
  accent: oklch(0.35, 0.11, 140),
  accentHover: oklch(0.4, 0.12, 140),
  accentActive: oklch(0.31, 0.1, 140),
  accentBorder: oklch(0.52, 0.12, 140),
  accentStrong: oklch(0.8, 0.14, 140),
  accentRow: oklch(0.26, 0.045, 140),
  error: oklch(0.7, 0.16, 25),
  // Period sums are another unit than hourly load, so they stay neutral.
  sum: oklch(0.72, 0.02, 255),
  zeroLine: oklch(0.62, 0.01, 255),
  contextLine: oklch(0.5, 0.01, 255),
  // Basemap.
  water: oklch(0.23, 0.02, 245),
  park: oklch(0.2, 0.008, 255),
  roadMajor: oklch(0.3, 0.006, 255),
  roadMinor: oklch(0.25, 0.006, 255),
  mapLabel: oklch(0.64, 0.01, 255),
  rail: oklch(0.34, 0.01, 255),
} satisfies Record<string, Oklch>

export type Token = keyof typeof tokens

export const hex = Object.fromEntries(
  Object.entries(tokens).map(([name, value]) => [name, toHex(value)]),
) as Record<Token, string>

function kebab(name: string): string {
  return name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)
}

/** Publishes tokens as CSS custom properties: --color-surface-1 etc. */
export function applyTokens(root: HTMLElement, extra: Record<string, Oklch> = {}): void {
  for (const [name, value] of Object.entries({ ...tokens, ...extra })) {
    root.style.setProperty(`--color-${kebab(name).replace(/(\d)/, '-$1')}`, toCss(value))
  }
}
