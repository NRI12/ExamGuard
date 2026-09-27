export interface Room {
  id: string
  code: string
  name: string
  description: string | null
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface Seat {
  id?: string
  room_id?: string
  code: string
  x: number
  y: number
  width: number
  height: number
  sort_order: number | null
  is_active: boolean
}

export type RoomInput = Pick<Room, 'code' | 'name' | 'description' | 'is_active'>
