import type { ReactNode } from 'react'

import { usePermissions, type Permission } from './permissions'

export function PermissionGate({ permission, children }: { permission: Permission; children: ReactNode }) {
  const { can } = usePermissions()
  return can(permission) ? children : null
}
