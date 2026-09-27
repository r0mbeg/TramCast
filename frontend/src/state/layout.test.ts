import { describe, expect, it } from 'vitest'
import { parseLayout } from './layout'

describe('panel layout', () => {
  it('shows both panels unless storage says otherwise', () => {
    const shown = { boardCollapsed: false, dockCollapsed: false }
    for (const raw of [null, '', 'not json', '42', 'null', '{"boardCollapsed":"yes"}']) {
      expect(parseLayout(raw)).toEqual(shown)
    }
    expect(parseLayout('{"boardCollapsed":true,"dockCollapsed":false,"extra":1}')).toEqual({ boardCollapsed: true, dockCollapsed: false })
  })
})
