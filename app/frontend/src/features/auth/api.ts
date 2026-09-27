import { apiClient } from '../../shared/api/client'
import type { AuthUser, LoginResponse } from './types'

export async function loginRequest(username: string, password: string): Promise<LoginResponse> {
  const response = await apiClient.post<LoginResponse>('/auth/login', { username, password })
  return response.data
}

export async function getCurrentUser(): Promise<AuthUser> {
  const response = await apiClient.get<AuthUser>('/auth/me')
  return response.data
}

export async function logoutRequest(): Promise<void> {
  await apiClient.post('/auth/logout')
}
