import type { MediaAsset } from '../media/types'

export type CameraStatus = 'READY' | 'IN_USE' | 'ERROR' | 'DISABLED'

export interface Camera {
  id: string
  name: string
  room_id: string
  room: { id: string; code: string; name: string }
  source_media_asset_id: string
  source_media: MediaAsset | null
  description: string | null
  is_active: boolean
  status: CameraStatus
  created_at: string
  updated_at: string
}

export interface CameraInput {
  name: string
  room_id: string
  source_media_asset_id: string
  description: string | null
  is_active: boolean
}
