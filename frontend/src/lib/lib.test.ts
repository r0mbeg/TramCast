import { describe, expect, it } from 'vitest'
import { addDays, clampToHorizon, dayType, formatLong, formatRange, formatShort, horizonDates, monthDates, weekDates, weekday } from './calendar'
import { oklch, toHex } from './color'
import { formatNumber, hourRange, isServiceOff } from './format'
import { loadBucket, loadHex, LOAD_HEX } from './loadScale'

describe('calendar', () => {
  it('covers 61 days of November and December 2025', () => {
    const dates = horizonDates()
    expect(dates).toHaveLength(61)
    expect(dates[0]).toBe('2025-11-01')
    expect(dates.at(-1)).toBe('2025-12-31')
  })

  it('follows the 2025 production calendar', () => {
    expect(dayType('2025-11-01')).toBe('working') // working Saturday
    expect(dayType('2025-11-02')).toBe('weekend')
    expect(dayType('2025-11-03')).toBe('holiday')
    expect(dayType('2025-11-04')).toBe('holiday')
    expect(dayType('2025-11-05')).toBe('working')
    expect(dayType('2025-11-08')).toBe('weekend')
    expect(dayType('2025-12-31')).toBe('holiday')
  })

  it('computes ISO weeks and months', () => {
    expect(weekday('2025-11-03')).toBe(1)
    expect(weekday('2025-11-09')).toBe(7)
    expect(weekDates('2025-11-05')).toEqual(['2025-11-03', '2025-11-04', '2025-11-05', '2025-11-06', '2025-11-07', '2025-11-08', '2025-11-09'])
    expect(weekDates('2025-12-31').at(-1)).toBe('2026-01-04')
    expect(monthDates('2025-11-17')).toHaveLength(30)
    expect(monthDates('2025-12-01')).toHaveLength(31)
    expect(addDays('2025-12-31', 1)).toBe('2026-01-01')
    expect(clampToHorizon('2025-10-01')).toBe('2025-11-01')
  })

  it('formats Russian dates', () => {
    expect(formatLong('2025-11-05')).toBe('ср, 5 ноября 2025')
    expect(formatShort('2025-11-03')).toBe('пн 3.11')
    expect(formatRange('2025-11-03', '2025-11-09')).toBe('3–9 ноября 2025')
    expect(formatRange('2025-12-29', '2026-01-04')).toBe('29 декабря 2025 – 4 января 2026')
  })
})

describe('color', () => {
  it('converts the brief tokens from OKLCH to hex', () => {
    expect(toHex(oklch(0.6, 0.075, 50))).toBe('#a57357')
    expect(toHex(oklch(0.92, 0.12, 95))).toBe('#fee484')
    expect(toHex(oklch(0.18, 0.008, 255))).toBe('#0f1215')
  })
})

describe('load scale', () => {
  it('maps boardings per hour to five buckets with a separate zero', () => {
    expect(loadBucket(0)).toBe(0)
    expect(loadBucket(1)).toBe(1)
    expect(loadBucket(299)).toBe(1)
    expect(loadBucket(300)).toBe(2)
    expect(loadBucket(1199)).toBe(3)
    expect(loadBucket(1200)).toBe(4)
    expect(loadBucket(2000)).toBe(5)
    expect(loadHex(0)).toBeNull()
    expect(loadHex(5000)).toBe(LOAD_HEX[4])
  })
})

describe('format', () => {
  it('groups digits with a narrow no-break space', () => {
    expect(formatNumber(216540)).toBe('216 540')
    expect(hourRange(23)).toBe('23:00–00:00')
    expect([0, 1, 4, 5].map(isServiceOff)).toEqual([false, true, true, false])
  })
})
