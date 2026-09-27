import { Navigate } from 'react-router-dom'

import { useAuth } from '../features/auth/AuthProvider'
import { landingPathForRole } from './access'

export function RoleLanding() {
  const { user } = useAuth()
  return user ? <Navigate to={landingPathForRole(user.role)} replace /> : null
}
