// Colors are defined once in OKLCH. MapLibre and ECharts need sRGB strings,
// so hex values are derived here instead of being duplicated by hand.

export interface Oklch {
  l: number
  c: number
  h: number
}

export function oklch(l: number, c: number, h: number): Oklch {
  return { l, c, h }
}

export function toCss({ l, c, h }: Oklch): string {
  return `oklch(${l} ${c} ${h})`
}

// OKLab → linear sRGB (Björn Ottosson), then the sRGB transfer function.
export function toHex({ l, c, h }: Oklch): string {
  const hue = (h * Math.PI) / 180
  const a = c * Math.cos(hue)
  const b = c * Math.sin(hue)
  const lp = l + 0.3963377774 * a + 0.2158037573 * b
  const mp = l - 0.1055613458 * a - 0.0638541728 * b
  const sp = l - 0.0894841775 * a - 1.291485548 * b
  const [lc, mc, sc] = [lp ** 3, mp ** 3, sp ** 3]
  const linear = [
    4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
    -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
    -0.0041960863 * lc - 0.7034186147 * mc + 1.707614701 * sc,
  ]
  return (
    '#' +
    linear
      .map((x) => {
        const v = x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055
        return Math.round(Math.min(1, Math.max(0, v)) * 255)
          .toString(16)
          .padStart(2, '0')
      })
      .join('')
  )
}
