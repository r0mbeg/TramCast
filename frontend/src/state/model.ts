import type { DemoScenario } from '../forecast/demo'
import { addDays, clampToHorizon, inHorizon, isIsoDate, monthDates } from '../lib/calendar'

export type Period = 'day' | 'week' | 'month'

/**
 * A slot is one forecast view of the dock. Its date is null while it follows
 * the global date and set once the dispatcher pins it for a comparison.
 */
export interface Slot {
  id: number
  routeId: number | null
  period: Period
  date: string | null
}

export interface AppState {
  date: string
  hour: number
  slots: Slot[]
  activeSlotId: number
  scenario: DemoScenario
  commonScale: boolean
}

// The first ordinary working day after the 2–4 November holidays.
export const DEFAULT_DATE = '2025-11-05'
export const DEFAULT_HOUR = 8
const PERIODS: readonly Period[] = ['day', 'week', 'month']

export function slotDate(state: AppState, slot: Slot): string {
  return slot.date ?? state.date
}

export function activeSlot(state: AppState): Slot {
  return state.slots.find((slot) => slot.id === state.activeSlotId) ?? state.slots[0]!
}

export function slotLetter(state: AppState, slot: Slot): string | null {
  return state.slots.length > 1 ? 'ABC'[state.slots.indexOf(slot)] ?? null : null
}

/** Two slots fit next to each other at 1440 px, three at 1920 px. */
export function maxSlots(width: number): number {
  return width >= 1680 ? 3 : 2
}

function defaultState(): AppState {
  return {
    date: DEFAULT_DATE,
    hour: DEFAULT_HOUR,
    slots: [{ id: 1, routeId: null, period: 'day', date: null }],
    activeSlotId: 1,
    scenario: 'ready',
    commonScale: true,
  }
}

// ?d=2025-11-05&h=8&s=16:day,24:week@2025-11-08&a=2&sc=mixed&cs=0
export function parseState(search: string): AppState {
  const params = new URLSearchParams(search)
  const state = defaultState()
  const date = params.get('d')
  if (date && isIsoDate(date)) state.date = clampToHorizon(date)
  const hour = Number(params.get('h'))
  if (params.has('h') && Number.isInteger(hour) && hour >= 0 && hour <= 23) state.hour = hour
  const slots = (params.get('s') ?? '')
    .split(',')
    .filter(Boolean)
    .slice(0, 3)
    .map((token, index): Slot => {
      const [body = '', pinned] = token.split('@')
      const [route = '', period = ''] = body.split(':')
      const routeId = Number(route)
      return {
        id: index + 1,
        routeId: route !== '' && Number.isInteger(routeId) && routeId > 0 ? routeId : null,
        period: PERIODS.includes(period as Period) ? (period as Period) : 'day',
        date: pinned && isIsoDate(pinned) && inHorizon(pinned) ? pinned : null,
      }
    })
  if (slots.length > 0) state.slots = slots
  const active = Number(params.get('a'))
  state.activeSlotId = state.slots.some((slot) => slot.id === active) ? active : state.slots[0]!.id
  if (params.get('sc') === 'mixed') state.scenario = 'mixed'
  if (params.get('cs') === '0') state.commonScale = false
  return state
}

export function serializeState(state: AppState): string {
  const params = new URLSearchParams()
  params.set('d', state.date)
  params.set('h', String(state.hour))
  const slots = state.slots.map((slot) => `${slot.routeId ?? ''}:${slot.period}${slot.date ? `@${slot.date}` : ''}`)
  if (slots.some((slot) => slot !== ':day')) params.set('s', slots.join(','))
  if (state.slots.length > 1) params.set('a', String(state.activeSlotId))
  if (state.scenario !== 'ready') params.set('sc', state.scenario)
  if (!state.commonScale) params.set('cs', '0')
  return params.toString()
}

function shiftMonth(date: string, direction: number): string {
  const [year, month, day] = date.split('-').map(Number) as [number, number, number]
  const target = new Date(Date.UTC(year, month - 1 + direction, 1)).toISOString().slice(0, 10)
  const days = monthDates(target).length
  return `${target.slice(0, 8)}${String(Math.min(day, days)).padStart(2, '0')}`
}

export function stepByPeriod(date: string, period: Period, direction: number): string {
  if (period === 'month') return shiftMonth(date, direction)
  return addDays(date, (period === 'week' ? 7 : 1) * direction)
}
