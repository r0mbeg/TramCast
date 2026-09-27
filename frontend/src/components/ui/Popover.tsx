import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { createPortal } from 'react-dom'
import styles from './ui.module.css'

interface TriggerProps {
  ref: RefObject<HTMLButtonElement | null>
  onClick: () => void
  'aria-expanded': boolean
  'aria-haspopup': 'dialog'
}

interface PopoverProps {
  label: string
  align?: 'start' | 'end'
  trigger: (props: TriggerProps) => ReactNode
  children: (close: () => void) => ReactNode
}

/**
 * Non-modal popover anchored to its trigger. It lives in a portal so panels
 * with overflow do not clip it; Esc and an outside click close it.
 */
export function Popover({ label, align = 'start', trigger, children }: PopoverProps) {
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)
  const close = useCallback(() => setOpen(false), [])

  useLayoutEffect(() => {
    if (!open || !triggerRef.current || !panelRef.current) return
    const anchor = triggerRef.current.getBoundingClientRect()
    const panel = panelRef.current.getBoundingClientRect()
    const left = align === 'end' ? anchor.right - panel.width : anchor.left
    setPosition({
      top: Math.min(anchor.bottom + 6, window.innerHeight - panel.height - 8),
      left: Math.max(8, Math.min(left, window.innerWidth - panel.width - 8)),
    })
  }, [open, align])

  useEffect(() => {
    if (!open) return
    const onPointer = (event: PointerEvent) => {
      const target = event.target as Node
      if (!panelRef.current?.contains(target) && !triggerRef.current?.contains(target)) setOpen(false)
    }
    // Capture phase: Esc closes the popover before the global shortcuts see it.
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        setOpen(false)
        triggerRef.current?.focus()
      }
    }
    const onResize = () => setOpen(false)
    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey, true)
    window.addEventListener('resize', onResize)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey, true)
      window.removeEventListener('resize', onResize)
    }
  }, [open])

  return (
    <>
      {trigger({
        ref: triggerRef,
        onClick: () => {
          setPosition(null)
          setOpen((value) => !value)
        },
        'aria-expanded': open,
        'aria-haspopup': 'dialog',
      })}
      {open
        ? createPortal(
            <div
              ref={panelRef}
              role="dialog"
              aria-label={label}
              className={styles.popover}
              style={position ?? { top: -9999, left: -9999 }}
            >
              {children(close)}
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
