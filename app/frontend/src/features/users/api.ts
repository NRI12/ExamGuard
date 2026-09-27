import { apiClient } from '../../shared/api/client'
import type { PageResponse } from '../../shared/api/types'
import type { UserRole } from '../auth/types'
import type { ManagedUser, UserCreateInput, UserUpdateInput } from './types'

export interface UserQuery {
  page?: number
  pageSize?: number
  query?: string
  role?: UserRole | ''
  active?: '' | 'true' | 'false'
}

export async function getUsers({ page = 1, pageSize = 20, query = '', role = '', active = '' }: UserQuery = {}): Promise<PageResponse<ManagedUser>> {
  return (await apiClient.get<PageResponse<ManagedUser>>('/users', {
    params: {
      page,
      page_size: pageSize,
      ...(query.trim() ? { q: query.trim() } : {}),
      ...(role ? { role } : {}),
      ...(active ? { is_active: active } : {}),
    },
  })).data
}

export async function createUser(payload: UserCreateInput): Promise<ManagedUser> {
  return (await apiClient.post<ManagedUser>('/users', payload)).data
}

export async function updateUser(id: string, payload: UserUpdateInput): Promise<ManagedUser> {
  return (await apiClient.patch<ManagedUser>(`/users/${id}`, payload)).data
}
