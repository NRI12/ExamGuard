import { describe, expect, it } from 'vitest'

import { hasErrors, validateUserCreate } from '../../shared/validation'

describe('user creation validation', () => {
  it('accepts a normalized account with a valid password', () => {
    const errors = validateUserCreate({
      username: 'reviewer.01',
      full_name: 'Nguyễn Văn A',
      password: 'review-password-123',
      role: 'REVIEWER',
    })
    expect(hasErrors(errors)).toBe(false)
  })

  it('rejects unsupported usernames and weak passwords', () => {
    const errors = validateUserCreate({
      username: 'Invalid User',
      full_name: null,
      password: 'onlylettersxx',
      role: 'SUPERVISOR',
    })
    expect(errors.username).toBeTruthy()
    expect(errors.password).toBeTruthy()
  })
})
