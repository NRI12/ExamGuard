import { useCallback, useMemo } from 'react'

import { useAuth } from './AuthProvider'
import type { UserRole } from './types'

export const permissions = {
  dashboardRead: 'dashboard.read',
  roomRead: 'room.read',
  roomManage: 'room.manage',
  candidateRead: 'candidate.read',
  candidateManage: 'candidate.manage',
  sessionRead: 'session.read',
  sessionManage: 'session.manage',
  sessionMonitor: 'session.monitor',
  mediaRead: 'media.read',
  mediaUpload: 'media.upload',
  trackingRead: 'tracking.read',
  eventRead: 'event.read',
  eventCreateManual: 'event.create_manual',
  eventReview: 'event.review',
  evidenceRead: 'evidence.read',
  evidenceManage: 'evidence.manage',
  appealRead: 'appeal.read',
  appealManage: 'appeal.manage',
  reportRead: 'report.read',
  reportExport: 'report.export',
  userManage: 'user.manage',
  auditRead: 'audit.read',
  auditReadRelevant: 'audit.read_relevant',
  systemManage: 'system.manage',
  diagnosticsRead: 'diagnostics.read',
} as const

export type Permission = typeof permissions[keyof typeof permissions]

export const allPermissions = Object.freeze(Object.values(permissions)) as readonly Permission[]

export const rolePermissions: Readonly<Record<UserRole, ReadonlySet<Permission>>> = {
  SUPERVISOR: new Set([
    permissions.dashboardRead,
    permissions.roomRead,
    permissions.candidateRead,
    permissions.candidateManage,
    permissions.sessionRead,
    permissions.sessionManage,
    permissions.sessionMonitor,
    permissions.mediaRead,
    permissions.mediaUpload,
    permissions.trackingRead,
    permissions.eventRead,
    permissions.eventCreateManual,
    permissions.evidenceRead,
    permissions.reportRead,
  ]),
  REVIEWER: new Set([
    permissions.dashboardRead,
    permissions.roomRead,
    permissions.candidateRead,
    permissions.sessionRead,
    permissions.trackingRead,
    permissions.eventRead,
    permissions.eventReview,
    permissions.evidenceRead,
    permissions.evidenceManage,
    permissions.appealRead,
    permissions.appealManage,
    permissions.reportRead,
    permissions.auditReadRelevant,
  ]),
  ADMIN: new Set(allPermissions),
}

export function hasPermission(role: UserRole | null | undefined, permission: Permission): boolean {
  return role ? rolePermissions[role].has(permission) : false
}

export function usePermissions() {
  const { user } = useAuth()
  const granted = useMemo(
    () => user ? rolePermissions[user.role] : new Set<Permission>(),
    [user],
  )
  const can = useCallback((permission: Permission) => granted.has(permission), [granted])
  return { can, permissions: granted }
}
