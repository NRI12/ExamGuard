export type EventStatus = 'PENDING_REVIEW' | 'CONFIRMED' | 'DISMISSED' | 'NEEDS_REVIEW' | 'DISPUTED' | 'RESOLVED'
export type BehaviorType = 'SUSPICIOUS_LOOKING' | 'COMMUNICATING' | 'EXCHANGE_OBJECT' | 'USING_PHONE_CHEAT_SHEET' | 'OTHER'
export type ReviewDecision = 'CONFIRM' | 'DISMISS' | 'NEEDS_REVIEW'

export interface EventActor {
  session_candidate_id: string
  candidate_id: string
  candidate_code: string
  full_name: string
  seat_code: string | null
}

export interface EventReview {
  id: string
  decision: ReviewDecision
  note: string | null
  reviewer_id: string
  reviewer_name: string
  created_at: string
}

export interface ExamEvent {
  id: string
  event_code: string
  session_id: string
  session_code: string
  exam_name: string
  video_asset_id: string | null
  source: 'AI' | 'MANUAL'
  behavior_type: BehaviorType | null
  start_ms: number
  end_ms: number
  peak_ms: number | null
  ai_confidence: number | null
  status: EventStatus
  created_at: string
  updated_at: string
  actors: EventActor[]
}

export interface ExamEventDetail extends ExamEvent {
  reviews: EventReview[]
}

export interface EventCounts {
  total: number
  pending_review: number
  confirmed: number
  dismissed: number
  needs_review: number
}

export const behaviorLabels: Record<BehaviorType, string> = {
  SUSPICIOUS_LOOKING: 'Nhìn bài / quay ngang',
  COMMUNICATING: 'Trao đổi',
  EXCHANGE_OBJECT: 'Trao đổi vật',
  USING_PHONE_CHEAT_SHEET: 'Điện thoại / tài liệu',
  OTHER: 'Khác',
}

export const eventStatusLabels: Record<EventStatus, string> = {
  PENDING_REVIEW: 'Chờ xác minh',
  CONFIRMED: 'Đã xác nhận',
  DISMISSED: 'Đã bỏ qua',
  NEEDS_REVIEW: 'Cần xem lại',
  DISPUTED: 'Đang khiếu nại',
  RESOLVED: 'Đã giải quyết',
}

export const decisionLabels: Record<ReviewDecision, string> = {
  CONFIRM: 'Xác nhận vi phạm',
  DISMISS: 'Bỏ qua (báo nhầm)',
  NEEDS_REVIEW: 'Cần xem lại',
}
