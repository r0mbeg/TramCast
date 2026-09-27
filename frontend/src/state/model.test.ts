import { describe, expect, it } from 'vitest'
import { DEFAULT_DATE, maxSlots, parseState, serializeState, slotDate, stepByPeriod } from './model'

describe('URL state', () => {
  it('defaults to 5 November 08:00 with one empty slot', () => {
    const state = parseState('')
    expect(state.date).toBe(DEFAULT_DATE)
    expect(state.hour).toBe(8)
    expect(state.slots).toEqual([{ id: 1, routeId: null, period: 'day', date: null }])
    expect(serializeState(state)).toBe('d=2025-11-05&h=8')
  })

  it('round-trips comparison slots with a pinned date', () => {
    const search = 'd=2025-11-12&h=17&s=24%3Aday%2C24%3Aweek%402025-11-19&a=2&sc=mixed&cs=0'
    const state = parseState(search)
    expect(state.slots).toEqual([
      { id: 1, routeId: 24, period: 'day', date: null },
      { id: 2, routeId: 24, period: 'week', date: '2025-11-19' },
    ])
    expect(state.activeSlotId).toBe(2)
    expect(state.commonScale).toBe(false)
    expect(slotDate(state, state.slots[0]!)).toBe('2025-11-12')
    expect(slotDate(state, state.slots[1]!)).toBe('2025-11-19')
    expect(parseState(serializeState(state))).toEqual(state)
    // The demo scenario of old links is dropped.
    expect(serializeState(state)).not.toContain('sc=')
  })

  it('ignores invalid values and clamps dates to the horizon', () => {
    const state = parseState('d=2024-01-01&h=25&s=abc%3Ayear%402030-01-01&a=9')
    expect(state.date).toBe('2025-11-01')
    expect(state.hour).toBe(8)
    expect(state.slots).toEqual([{ id: 1, routeId: null, period: 'day', date: null }])
    expect(state.activeSlotId).toBe(1)
  })

  it('steps dates by period, keeping the day of month when possible', () => {
    expect(stepByPeriod('2025-11-05', 'day', 1)).toBe('2025-11-06')
    expect(stepByPeriod('2025-11-05', 'week', -1)).toBe('2025-10-29')
    expect(stepByPeriod('2025-11-30', 'month', 1)).toBe('2025-12-30')
    expect(stepByPeriod('2025-12-31', 'month', -1)).toBe('2025-11-30')
  })

  it('fits two comparison slots at 1440 px and three at 1920 px', () => {
    expect(maxSlots(1440)).toBe(2)
    expect(maxSlots(1920)).toBe(3)
  })
})
