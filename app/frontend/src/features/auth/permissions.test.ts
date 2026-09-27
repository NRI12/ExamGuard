import { describe, expect, it } from 'vitest'

import { allPermissions, hasPermission, permissions, rolePermissions } from './permissions'

describe('RBAC permission map', () => {
  it('separates supervisor operation from reviewer decisions', () => {
    expect(hasPermission('SUPERVISOR', permissions.sessionMonitor)).toBe(true)
    expect(hasPermission('SUPERVISOR', permissions.candidateManage)).toBe(true)
    expect(hasPermission('SUPERVISOR', permissions.eventReview)).toBe(false)
    expect(hasPermission('REVIEWER', permissions.eventReview)).toBe(true)
    expect(hasPermission('REVIEWER', permissions.sessionMonitor)).toBe(false)
    expect(hasPermission('REVIEWER', permissions.candidateManage)).toBe(false)
    expect(hasPermission('REVIEWER', permissions.diagnosticsRead)).toBe(false)
  })

  it('grants every declared permission to administrators', () => {
    expect(rolePermissions.ADMIN.size).toBe(allPermissions.length)
    for (const permission of allPermissions) {
      expect(hasPermission('ADMIN', permission)).toBe(true)
    }
  })

  it('denies permissions when no authenticated role is present', () => {
    expect(hasPermission(null, permissions.dashboardRead)).toBe(false)
  })
})
