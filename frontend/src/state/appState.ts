import { useSyncExternalStore } from 'react'
import type { DemoScenario } from '../forecast/demo'
import { addDays, clampToHorizon, inHorizon } from '../lib/calendar'
import {
  activeSlot,
  maxSlots,
  parseState,
  serializeState,
  slotDate,
  stepByPeriod,
  type AppState,
  type Period,
  type Slot,
} from './model'

export * from './model'

const STORAGE_KEY = 'tramcast.view'

function readStorage(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

function writeStorage(search: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, search)
  } catch {
    // Storage may be unavailable; the URL still holds the view.
  }
}

let state: AppState = parseState(window.location.search || readStorage() || '')
const listeners = new Set<() => void>()

window.addEventListener('popstate', () => {
  state = parseState(window.location.search)
  listeners.forEach((listener) => listener())
})

/** Steps of date and hour replace the history entry; view changes push one. */
function commit(next: AppState, history: 'push' | 'replace'): void {
  state = next
  const search = serializeState(next)
  const url = `${window.location.pathname}?${search}`
  if (history === 'push') window.history.pushState(null, '', url)
  else window.history.replaceState(null, '', url)
  writeStorage(search)
  listeners.forEach((listener) => listener())
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function getState(): AppState {
  return state
}

export function useAppState(): AppState {
  return useSyncExternalStore(subscribe, getState)
}

function updateSlot(slotId: number, change: Partial<Slot>): Slot[] {
  return state.slots.map((slot) => (slot.id === slotId ? { ...slot, ...change } : slot))
}

export const actions = {
  setDate(date: string) {
    commit({ ...state, date: clampToHorizon(date) }, 'replace')
  },
  stepDate(days: number) {
    commit({ ...state, date: clampToHorizon(addDays(state.date, days)) }, 'replace')
  },
  setHour(hour: number) {
    commit({ ...state, hour: Math.min(23, Math.max(0, hour)) }, 'replace')
  },
  /** Moves the hour and crosses midnight within the forecast horizon. */
  stepHour(hours: number) {
    let hour = state.hour + hours
    let date = state.date
    if (hour > 23 && inHorizon(addDays(date, 1))) {
      hour -= 24
      date = addDays(date, 1)
    } else if (hour < 0 && inHorizon(addDays(date, -1))) {
      hour += 24
      date = addDays(date, -1)
    }
    commit({ ...state, date, hour: Math.min(23, Math.max(0, hour)) }, 'replace')
  },
  /** Opens a route in the active slot; the slot keeps its period and date. */
  selectRoute(routeId: number, hour?: number) {
    commit(
      { ...state, slots: updateSlot(state.activeSlotId, { routeId }), hour: hour ?? state.hour },
      'push',
    )
  },
  /** Closes the route of a slot; with several slots the slot itself closes. */
  clearSlot(slotId: number = state.activeSlotId) {
    if (state.slots.length > 1) {
      const slots = state.slots.filter((slot) => slot.id !== slotId)
      commit({ ...state, slots, activeSlotId: slots[0]!.id }, 'push')
    } else {
      commit({ ...state, slots: updateSlot(slotId, { routeId: null }) }, 'push')
    }
  },
  setActiveSlot(slotId: number) {
    if (slotId !== state.activeSlotId) commit({ ...state, activeSlotId: slotId }, 'replace')
  },
  setPeriod(slotId: number, period: Period) {
    commit({ ...state, slots: updateSlot(slotId, { period }) }, 'push')
  },
  /** Adds a comparison slot; returns false when the dock is full. */
  addSlot(routeId: number | null, width: number = window.innerWidth): boolean {
    if (state.slots.length >= maxSlots(width)) return false
    const source = activeSlot(state)
    const id = Math.max(...state.slots.map((slot) => slot.id)) + 1
    const slot: Slot = { id, routeId: routeId ?? source.routeId, period: source.period, date: null }
    commit({ ...state, slots: [...state.slots, slot], activeSlotId: id }, 'push')
    return true
  },
  /** Pins a slot to its own date, or releases it to follow the global date. */
  pinSlot(slotId: number, pinned: boolean) {
    const slot = state.slots.find((item) => item.id === slotId)
    if (!slot) return
    commit({ ...state, slots: updateSlot(slotId, { date: pinned ? slotDate(state, slot) : null }) }, 'push')
  },
  /** Steps a slot by its period: a pinned slot moves alone, a following slot moves the global date. */
  stepSlot(slotId: number, direction: number) {
    const slot = state.slots.find((item) => item.id === slotId)
    if (!slot) return
    const next = clampToHorizon(stepByPeriod(slotDate(state, slot), slot.period, direction))
    if (slot.date) commit({ ...state, slots: updateSlot(slotId, { date: next }) }, 'replace')
    else commit({ ...state, date: next }, 'replace')
  },
  /** Opens one day of a slot from its week or month view. */
  openDay(slotId: number, date: string) {
    const slot = state.slots.find((item) => item.id === slotId)
    if (!slot) return
    if (slot.date) commit({ ...state, slots: updateSlot(slotId, { period: 'day', date }) }, 'push')
    else commit({ ...state, date, slots: updateSlot(slotId, { period: 'day' }) }, 'push')
  },
  setScenario(scenario: DemoScenario) {
    commit({ ...state, scenario }, 'replace')
  },
  setCommonScale(commonScale: boolean) {
    commit({ ...state, commonScale }, 'replace')
  },
}
