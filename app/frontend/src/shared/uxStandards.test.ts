import { describe, expect, it } from 'vitest'

import { formatDateTime, formatDurationMs, formatFps, formatResolution } from './formatters'
import { roleLabels, sessionStatusLabels } from './i18n/vi'
import { hasErrors, validateCandidate, validateRoom, validateSession, valuesChanged } from './validation'

describe('chuẩn hiển thị tiếng Việt', () => {
  it('maps every current role and session status', () => {
    expect(sessionStatusLabels.DRAFT).toBe('Chưa bắt đầu')
    expect(sessionStatusLabels.READY).toBe('Sẵn sàng')
    expect(sessionStatusLabels.COMPLETED).toBe('Đã kết thúc')
    expect(roleLabels.SUPERVISOR).toBe('Giám thị')
    expect(roleLabels.REVIEWER).toBe('Người xác minh')
    expect(roleLabels.ADMIN).toBe('Quản trị viên')
  })

  it('formats dates, durations, resolutions, and FPS consistently', () => {
    expect(formatDateTime('2026-09-21T14:32:00')).toContain('21/09/2026')
    expect(formatDateTime('2026-09-21T14:32:00')).toContain('14:32')
    expect(formatDurationMs(2_520_000)).toBe('42:00')
    expect(formatResolution(1920, 1080)).toBe('1920×1080')
    expect(formatFps(25)).toBe('25.0 FPS')
  })
})

describe('frontend validation', () => {
  it('validates required trimmed fields and schedule ordering', () => {
    expect(hasErrors(validateRoom({ code: ' ', name: '', description: null }))).toBe(true)
    expect(hasErrors(validateCandidate({ candidate_code: '', full_name: ' ', class_name: null, note: null }))).toBe(true)
    const sessionErrors = validateSession({
      session_code: 'CA-01', exam_name: 'Cơ sở dữ liệu', room_id: 'room',
      scheduled_start: '2026-09-21T15:00', scheduled_end: '2026-09-21T14:00',
    })
    expect(sessionErrors.scheduled_end).toBeTruthy()
  })

  it('detects dirty complex form values', () => {
    const original = [{ id: 'seat', code: 'A01' }]
    expect(valuesChanged(original, [{ id: 'seat', code: 'A01' }])).toBe(false)
    expect(valuesChanged(original, [{ id: 'seat', code: 'A02' }])).toBe(true)
  })
})
