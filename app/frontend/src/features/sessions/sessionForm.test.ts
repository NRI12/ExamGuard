import { describe, expect, it } from 'vitest'

import { defaultSessionName, nextSessionCode, scheduledEnd, toLocalDateTimeInput } from './sessionForm'

describe('session creation helpers', () => {
  it('increments the latest numeric session-code format', () => {
    expect(nextSessionCode([
      { session_code: 'CA-009' },
      { session_code: 'CA-008' },
      { session_code: 'OLD-20' },
    ], 9)).toBe('CA-010')
  })

  it('uses a stable default code when existing codes have no numeric suffix', () => {
    expect(nextSessionCode([{ session_code: 'PILOT' }], 4)).toBe('PT-005')
  })

  it('formats local input time and calculates the selected duration', () => {
    const date = new Date(2026, 8, 23, 14, 5)
    expect(toLocalDateTimeInput(date)).toBe('2026-09-23T14:05')

    const start = '2026-09-23T14:05'
    const expectedEnd = new Date(new Date(start).getTime() + 90 * 60_000).toISOString()
    expect(scheduledEnd(start, 90)).toBe(expectedEnd)
    expect(defaultSessionName(date)).toBe('Phiên thi 23/09/2026 - 14:05')
  })
})
