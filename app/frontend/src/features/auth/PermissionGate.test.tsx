import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { AuthContext, type AuthContextValue } from './AuthProvider'
import { PermissionGate } from './PermissionGate'
import { permissions } from './permissions'
import type { UserRole } from './types'

function authValue(role: UserRole): AuthContextValue {
  return {
    user: { id: 'user-1', username: 'tester', full_name: 'Nguyễn Văn A', role, is_active: true },
    loading: false,
    login: async () => { throw new Error('not used') },
    logout: () => undefined,
  }
}

function renderGate(role: UserRole) {
  return renderToStaticMarkup(
    <AuthContext.Provider value={authValue(role)}>
      <PermissionGate permission={permissions.eventReview}><span>Review controls</span></PermissionGate>
    </AuthContext.Provider>,
  )
}

describe('PermissionGate', () => {
  it('renders children when permission is granted', () => {
    expect(renderGate('REVIEWER')).toContain('Review controls')
  })

  it('renders nothing when permission is missing', () => {
    expect(renderGate('SUPERVISOR')).toBe('')
  })
})
