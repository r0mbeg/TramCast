// ru-RU groups digits with a no-break space; a narrow no-break space (U+202F)
// keeps long numbers compact in dense cells.
const numberFormat = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 })
const percentFormat = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 })

export function formatNumber(value: number): string {
  return numberFormat.format(value).replace(/ /g, ' ')
}

export function formatPercent(value: number): string {
  return `${percentFormat.format(value)} %`
}

export function hourLabel(hour: number): string {
  return String(hour).padStart(2, '0')
}

/** "08:00–09:00" */
export function hourRange(hour: number): string {
  return `${hourLabel(hour)}:00–${hourLabel((hour + 1) % 24)}:00`
}

/** Hours 1–4 have no service: terminals are off from 01:00 to 05:30. */
export function isServiceOff(hour: number): boolean {
  return hour >= 1 && hour <= 4
}

/** Hour 5 counts only boardings from 05:30. */
export function isPartialHour(hour: number): boolean {
  return hour === 5
}

export function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${String(seconds % 60).padStart(2, '0')}`
}
