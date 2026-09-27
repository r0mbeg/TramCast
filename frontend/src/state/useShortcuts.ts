import { useEffect, useRef } from 'react'
import type { Route } from '../api/catalog'
import { actions, activeSlot, getState, type Period } from './appState'

const PERIOD_KEYS: Record<string, Period> = { KeyD: 'day', KeyW: 'week', KeyM: 'month' }

function isTyping(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null
  return !!element && (element.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(element.tagName))
}

/**
 * Global shortcuts. Keys are matched by event.code, so they work in the
 * Russian layout too; they are off while typing in a field.
 */
export function useShortcuts(routes: readonly Route[]) {
  const buffer = useRef('')
  const timer = useRef<number | undefined>(undefined)
  const routesRef = useRef(routes)
  routesRef.current = routes

  useEffect(() => {
    const byNumber = (value: string) => routesRef.current.find((route) => String(route.route_number) === value)

    const commitNumber = () => {
      const route = byNumber(buffer.current)
      buffer.current = ''
      window.clearTimeout(timer.current)
      if (route) actions.selectRoute(route.id)
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || isTyping(event.target)) return
      const state = getState()
      const slot = activeSlot(state)
      const forecastRoutes = routesRef.current.filter((route) => route.forecast_enabled)

      const digit = /^(Digit|Numpad)(\d)$/.exec(event.code)?.[2]
      if (digit !== undefined && !event.shiftKey) {
        buffer.current += digit
        window.clearTimeout(timer.current)
        const candidates = routesRef.current.filter((route) => String(route.route_number).startsWith(buffer.current))
        if (candidates.length === 1 && String(candidates[0]!.route_number) === buffer.current) commitNumber()
        else if (candidates.length === 0) buffer.current = ''
        else timer.current = window.setTimeout(commitNumber, 600)
        return
      }
      if (event.code === 'Enter' && buffer.current) {
        event.preventDefault()
        commitNumber()
        return
      }

      switch (event.code) {
        case 'ArrowUp':
        case 'ArrowDown': {
          if (forecastRoutes.length === 0) return
          event.preventDefault()
          const index = forecastRoutes.findIndex((route) => route.id === slot.routeId)
          const step = event.code === 'ArrowDown' ? 1 : -1
          const next = index < 0 ? (step > 0 ? 0 : forecastRoutes.length - 1) : (index + step + forecastRoutes.length) % forecastRoutes.length
          actions.selectRoute(forecastRoutes[next]!.id)
          return
        }
        case 'ArrowLeft':
        case 'ArrowRight': {
          event.preventDefault()
          const step = event.code === 'ArrowRight' ? 1 : -1
          if (event.shiftKey) actions.stepDate(step)
          else actions.stepHour(step)
          return
        }
        case 'BracketLeft':
        case 'BracketRight':
          if (slot.routeId !== null) actions.stepSlot(slot.id, event.code === 'BracketRight' ? 1 : -1)
          return
        case 'Escape':
          if (slot.routeId !== null) actions.clearSlot(slot.id)
          return
        case 'KeyF':
          window.dispatchEvent(new CustomEvent('tramcast:fit', { detail: event.shiftKey ? 'network' : 'route' }))
          return
      }
      const period = PERIOD_KEYS[event.code]
      if (period && slot.routeId !== null && !event.shiftKey) actions.setPeriod(slot.id, period)
    }

    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.clearTimeout(timer.current)
    }
  }, [])
}
