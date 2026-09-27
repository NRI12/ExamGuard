import { apiClient } from '../../shared/api/client'
import type { PageQuery, PageResponse } from '../../shared/api/types'
import type { Room, RoomInput, Seat } from './types'

export async function getRooms({ page = 1, pageSize = 20, query = '' }: PageQuery = {}): Promise<PageResponse<Room>> {
  return (await apiClient.get<PageResponse<Room>>('/rooms', {
    params: { page, page_size: pageSize, ...(query.trim() ? { q: query.trim() } : {}) },
  })).data
}

export async function createRoom(payload: RoomInput): Promise<Room> {
  return (await apiClient.post<Room>('/rooms', payload)).data
}

export async function updateRoom(id: string, payload: Partial<RoomInput>): Promise<Room> {
  return (await apiClient.patch<Room>(`/rooms/${id}`, payload)).data
}

export async function getSeats(roomId: string): Promise<Seat[]> {
  return (await apiClient.get<Seat[]>(`/rooms/${roomId}/seats`)).data
}

export async function saveSeats(roomId: string, seats: Seat[]): Promise<Seat[]> {
  return (await apiClient.put<Seat[]>(`/rooms/${roomId}/seats`, { seats })).data
}
