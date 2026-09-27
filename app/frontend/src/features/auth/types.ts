export type UserRole = 'SUPERVISOR' | 'REVIEWER' | 'ADMIN'

export interface AuthUser {
  id: string
  username: string
  full_name: string | null
  role: UserRole
  is_active: boolean
}

export interface LoginResponse {
  access_token: string
  token_type: 'bearer'
  user: AuthUser
}
