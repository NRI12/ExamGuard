export type SessionSourceType = 'CAMERA' | 'VIDEO_UPLOAD'

export type SessionStatus = 'DRAFT' | 'READY' | 'RUNNING' | 'PAUSED' | 'COMPLETED' | 'CANCELLED' | 'ERROR'

export interface SessionAssignment {
  id: string
  candidate: { id: string; candidate_code: string; full_name: string; class_name: string | null }
  seat: { id: string; code: string; is_active: boolean }
}

export interface ExamSession {
  id: string
  session_code: string
  exam_name: string
  room_id: string
  room: { id: string; code: string; name: string; is_active: boolean }
  source_type: SessionSourceType
  camera_id: string | null
  camera: { id: string; name: string; is_active: boolean } | null
  video_asset_id: string | null
  video: {
    id: string
    original_filename: string
    media_url: string
    codec: string
    width: number
    height: number
    fps: number
    duration_ms: number
    size_bytes: number
  } | null
  status: SessionStatus
  scheduled_start: string | null
  scheduled_end: string | null
  actual_start: string | null
  actual_end: string | null
  duration_minutes?: number
  runtime_profile: string | null
  created_by: string
  created_by_user: { id: string; username: string; full_name: string | null }
  candidate_count: number
  assignments: SessionAssignment[]
  readiness: {
    room_selected: boolean
    room_active: boolean
    seat_layout_available: boolean
    active_seats: number
    candidates_assigned: number
    video_configured: boolean
    monitoring_status: 'NOT_STARTED' | 'RUNNING' | 'PAUSED' | 'COMPLETED' | 'ERROR' | 'CANCELLED'
    can_mark_ready: boolean
  }
  created_at: string
  updated_at: string
}

export interface SessionInput {
  session_code: string
  exam_name: string
  room_id: string
  scheduled_start: string | null
  scheduled_end: string | null
  duration_minutes?: number
  runtime_profile: string | null
}
