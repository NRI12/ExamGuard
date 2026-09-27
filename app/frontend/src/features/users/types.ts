import type { UserRole } from '../auth/types'

export interface ManagedUser {
  id: string
  username: string
  full_name: string | null
  role: UserRole
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface UserCreateInput {
  username: string
  full_name: string | null
  role: UserRole
  password: string
  is_active: boolean
}

export interface UserUpdateInput {
  full_name?: string | null
  role?: UserRole
  is_active?: boolean
}
