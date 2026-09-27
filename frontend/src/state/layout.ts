import { useSyncExternalStore } from 'react'

/**
 * Which side panels are hidden. A preference of this browser, not part of the
 * view: it stays out of the URL, so a shared link opens with the panels as the
 * recipient left them.
 */
export interface Layout {
  boardCollapsed: boolean
  dockCollapsed: boolean
}

const STORAGE_KEY = 'tramcast.layout'

// Element IDs of the panels, for aria-controls of their tabs. A hidden panel
// is not rendered at all.
export const BOARD_ID = 'routes-panel'
export const DOCK_ID = 'dock-panel'

export function parseLayout(raw: string | null): Layout {
  try {
    const value: unknown = raw ? JSON.parse(raw) : null
    const stored = value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : {}
    return { boardCollapsed: stored.boardCollapsed === true, dockCollapsed: stored.dockCollapsed === true }
  } catch {
    return { boardCollapsed: false, dockCollapsed: false }
  }
}

function readStorage(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

let layout = parseLayout(readStorage())
const listeners = new Set<() => void>()

function commit(next: Layout): void {
  if (next.boardCollapsed === layout.boardCollapsed && next.dockCollapsed === layout.dockCollapsed) return
  layout = next
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    // Storage may be unavailable; the layout then lasts until reload.
  }
  listeners.forEach((listener) => listener())
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useLayout(): Layout {
  return useSyncExternalStore(subscribe, () => layout)
}

export const layoutActions = {
  setBoardCollapsed(boardCollapsed: boolean) {
    commit({ ...layout, boardCollapsed })
  },
  setDockCollapsed(dockCollapsed: boolean) {
    commit({ ...layout, dockCollapsed })
  },
}
