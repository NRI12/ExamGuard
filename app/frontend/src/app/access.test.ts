import { describe, expect, it } from 'vitest'

import { permissions } from '../features/auth/permissions'
import { canAccessRoute, landingPathForRole, navigationForRole, requiredPermissionForPath } from './access'

describe('role-aware application access', () => {
  it('uses the expected landing page for each role', () => {
    expect(landingPathForRole('ADMIN')).toBe('/overview')
    expect(landingPathForRole('SUPERVISOR')).toBe('/overview')
    expect(landingPathForRole('REVIEWER')).toBe('/overview')
  })

  it('shows operational navigation to supervisors without settings', () => {
    const items = navigationForRole('SUPERVISOR')
    expect(items.map((item) => item.label)).toEqual(['Trang chủ', 'Giám sát', 'Sự kiện', 'Phiên thi'])
    expect(items.some((item) => item.to.startsWith('/settings'))).toBe(false)
  })

  it('shows only implemented read-only navigation to reviewers', () => {
    const items = navigationForRole('REVIEWER')
    expect(items.map((item) => item.label)).toEqual(['Trang chủ', 'Sự kiện', 'Phiên thi'])
    expect(items.some((item) => item.to === '/monitoring')).toBe(false)
    expect(items.some((item) => item.to.startsWith('/settings'))).toBe(false)
  })

  it('shows the focused operations and administration navigation to administrators', () => {
    const items = navigationForRole('ADMIN')
    expect(items.map((item) => item.label)).toEqual(['Trang chủ', 'Giám sát', 'Sự kiện', 'Phiên thi', 'Phòng thi', 'Camera', 'Người dùng', 'Hệ thống'])
    expect(items.filter((item) => item.section === 'QUẢN TRỊ')).toHaveLength(3)
  })

  it('protects direct routes using centralized route metadata', () => {
    expect(requiredPermissionForPath('/sessions/abc')).toBe(permissions.sessionRead)
    expect(canAccessRoute('SUPERVISOR', '/settings/rooms')).toBe(false)
    expect(canAccessRoute('REVIEWER', '/settings/rooms')).toBe(false)
    expect(canAccessRoute('ADMIN', '/settings/rooms')).toBe(true)
    expect(canAccessRoute('SUPERVISOR', '/settings/cameras')).toBe(false)
    expect(canAccessRoute('ADMIN', '/settings/cameras')).toBe(true)
    expect(canAccessRoute('SUPERVISOR', '/settings/users')).toBe(false)
    expect(canAccessRoute('ADMIN', '/settings/users')).toBe(true)
    expect(canAccessRoute('REVIEWER', '/settings/audit')).toBe(false)
    expect(canAccessRoute('ADMIN', '/settings/audit')).toBe(true)
  })
})
