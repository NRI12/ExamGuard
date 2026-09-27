import { apiClient } from '../../shared/api/client'
import type { PageResponse } from '../../shared/api/types'
import type { Camera, CameraInput } from './types'

export async function getCameras({
  query = '', roomId, active, page = 1, pageSize = 20,
}: {
  query?: string
  roomId?: string
  active?: boolean
  page?: number
  pageSize?: number
} = {}): Promise<PageResponse<Camera>> {
  return (await apiClient.get<PageResponse<Camera>>('/cameras', {
    params: {
      page,
      page_size: pageSize,
      ...(query.trim() ? { q: query.trim() } : {}),
      ...(roomId ? { room_id: roomId } : {}),
      ...(active === undefined ? {} : { is_active: active }),
    },
  })).data
}

export async function createCamera(payload: CameraInput): Promise<Camera> {
  return (await apiClient.post<Camera>('/cameras', payload)).data
}

export async function updateCamera(id: string, payload: Partial<CameraInput>): Promise<Camera> {
  return (await apiClient.patch<Camera>(`/cameras/${id}`, payload)).data
}
