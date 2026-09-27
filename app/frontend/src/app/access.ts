import { matchPath } from 'react-router-dom'

import { hasPermission, permissions, type Permission } from '../features/auth/permissions'
import type { UserRole } from '../features/auth/types'
import type { IconName } from '../shared/components/Icon'

export interface AppRouteAccess {
  id: 'overview' | 'monitoring' | 'events' | 'sessions' | 'session-detail' | 'candidates' | 'rooms' | 'cameras' | 'users' | 'audit'
  path: string
  permission: Permission
}

export interface NavigationItem {
  label: string
  to: string
  icon: IconName
  permission: Permission
  section?: string
  end?: boolean
}

export const appRouteAccess: readonly AppRouteAccess[] = [
  { id: 'overview', path: '/overview', permission: permissions.dashboardRead },
  { id: 'monitoring', path: '/monitoring', permission: permissions.trackingRead },
  { id: 'events', path: '/events', permission: permissions.eventRead },
  { id: 'sessions', path: '/sessions', permission: permissions.sessionRead },
  { id: 'session-detail', path: '/sessions/:id', permission: permissions.sessionRead },
  { id: 'candidates', path: '/candidates', permission: permissions.candidateRead },
  { id: 'rooms', path: '/settings/rooms', permission: permissions.roomManage },
  { id: 'cameras', path: '/settings/cameras', permission: permissions.systemManage },
  { id: 'users', path: '/settings/users', permission: permissions.userManage },
  { id: 'audit', path: '/settings/audit', permission: permissions.auditRead },
]

const navigationByRole: Readonly<Record<UserRole, readonly NavigationItem[]>> = {
  ADMIN: [
    { label: 'Trang chủ', to: '/overview', icon: 'chart', permission: permissions.dashboardRead },
    { label: 'Giám sát', to: '/monitoring', icon: 'camera', permission: permissions.trackingRead },
    { label: 'Sự kiện', to: '/events', icon: 'shield', permission: permissions.eventRead },
    { label: 'Phiên thi', to: '/sessions', icon: 'calendar', permission: permissions.sessionRead },
    { label: 'Phòng thi', to: '/settings/rooms', icon: 'door', permission: permissions.roomManage, section: 'QUẢN TRỊ' },
    { label: 'Camera', to: '/settings/cameras', icon: 'camera', permission: permissions.systemManage, section: 'QUẢN TRỊ' },
    { label: 'Người dùng', to: '/settings/users', icon: 'users', permission: permissions.userManage, section: 'QUẢN TRỊ' },
    { label: 'Hệ thống', to: '/settings/audit', icon: 'settings', permission: permissions.auditRead, section: 'HỆ THỐNG' },
  ],
  SUPERVISOR: [
    { label: 'Trang chủ', to: '/overview', icon: 'chart', permission: permissions.dashboardRead },
    { label: 'Giám sát', to: '/monitoring', icon: 'camera', permission: permissions.trackingRead },
    { label: 'Sự kiện', to: '/events', icon: 'shield', permission: permissions.eventRead },
    { label: 'Phiên thi', to: '/sessions', icon: 'calendar', permission: permissions.sessionRead },
  ],
  REVIEWER: [
    { label: 'Trang chủ', to: '/overview', icon: 'chart', permission: permissions.dashboardRead },
    { label: 'Sự kiện', to: '/events', icon: 'shield', permission: permissions.eventRead },
    { label: 'Phiên thi', to: '/sessions', icon: 'calendar', permission: permissions.sessionRead },
  ],
}

export function landingPathForRole(_role: UserRole): string {
  return '/overview'
}

export function navigationForRole(role: UserRole): readonly NavigationItem[] {
  return navigationByRole[role].filter((item) => hasPermission(role, item.permission))
}

export function requiredPermissionForPath(pathname: string): Permission | null {
  return appRouteAccess.find((route) => matchPath({ path: route.path, end: true }, pathname))?.permission ?? null
}

export function canAccessRoute(role: UserRole, pathname: string): boolean {
  const permission = requiredPermissionForPath(pathname)
  return permission !== null && hasPermission(role, permission)
}
