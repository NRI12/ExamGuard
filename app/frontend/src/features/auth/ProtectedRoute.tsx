import type { ReactNode } from 'react'
import { Navigate, Outlet } from 'react-router-dom'

import { LoadingState } from '../../shared/components/LoadingState'
import { ForbiddenPage } from '../../shared/pages/ForbiddenPage'
import { useAuth } from './AuthProvider'
import { hasPermission, type Permission } from './permissions'

export function ProtectedRoute({ permission, children }: { permission?: Permission; children?: ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <LoadingState message="Đang xác thực phiên đăng nhập…" />
  if (!user) return <Navigate to="/login" replace />
  if (permission && !hasPermission(user.role, permission)) return <ForbiddenPage />
  return children ?? <Outlet />
}
