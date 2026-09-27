import type { ButtonHTMLAttributes, ReactNode, Ref } from 'react'
import { routeCss } from '../../lib/routeColors'
import styles from './ui.module.css'

type ButtonVariant = 'secondary' | 'primary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  ref?: Ref<HTMLButtonElement>
  variant?: ButtonVariant
  iconOnly?: boolean
  busy?: boolean
}

export function Button({ variant = 'secondary', iconOnly, busy, className, children, ...rest }: ButtonProps) {
  const classes = [styles.button, variant !== 'secondary' && styles[variant], iconOnly && styles.icon, className]
    .filter(Boolean)
    .join(' ')
  return (
    <button type="button" className={classes} aria-busy={busy || undefined} {...rest}>
      {busy ? <span className={styles.spinner} aria-hidden /> : null}
      {children}
    </button>
  )
}

interface SegmentedProps<T extends string> {
  label: string
  value: T
  options: readonly { value: T; label: string; hint?: string }[]
  onChange: (value: T) => void
}

export function Segmented<T extends string>({ label, value, options, onChange }: SegmentedProps<T>) {
  return (
    <div className={styles.segmented} role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={styles.segment}
          aria-pressed={option.value === value}
          title={option.hint}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

interface RouteBadgeProps {
  routeNumber: number
  outline?: boolean
  letter?: string | null
}

/** Route number on its identity color; the number always carries the meaning. */
export function RouteBadge({ routeNumber, outline, letter }: RouteBadgeProps) {
  return (
    <span
      className={`${styles.badge} ${outline ? styles.badgeOutline : ''}`}
      style={outline ? undefined : { background: routeCss(routeNumber) }}
      aria-label={`Маршрут ${routeNumber}${letter ? `, слот ${letter}` : ''}`}
    >
      {routeNumber}
      {letter ? <span className={styles.letter}>{letter}</span> : null}
    </span>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className={styles.kbd}>{children}</kbd>
}

export function Note({ icon, children }: { icon?: ReactNode; children: ReactNode }) {
  return (
    <div className={styles.note}>
      {icon}
      <div>{children}</div>
    </div>
  )
}

export { styles as uiStyles }
