import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import './events.css'

import { apiContentErrorMessage, apiErrorMessage } from '../../shared/api/errors'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { Pagination } from '../../shared/components/Pagination'
import { useToast } from '../../shared/components/ToastProvider'
import { formatDateTime, formatDurationMs } from '../../shared/formatters'
import { permissions, usePermissions } from '../auth/permissions'
import { eventMediaUrl, getEvent, getEventCounts, getEvents, reviewEvent } from './api'
import {
  behaviorLabels,
  decisionLabels,
  eventStatusLabels,
  type BehaviorType,
  type EventCounts,
  type EventStatus,
  type ExamEvent,
  type ExamEventDetail,
  type ReviewDecision,
} from './types'

const PAGE_SIZE = 20
const CONTEXT_MS = 3000 // video shown around the event

export function EventStatusPill({ status }: { status: EventStatus }) {
  return <span className={`event-status ${status.toLowerCase()}`}>{eventStatusLabels[status]}</span>
}

export function BehaviorPill({ behavior }: { behavior: BehaviorType | null }) {
  const value = behavior ?? 'OTHER'
  return <span className={`behavior-pill ${value.toLowerCase()}`}>{behaviorLabels[value]}</span>
}

export function actorLabel(event: ExamEvent): string {
  if (!event.actors.length) return 'Chưa gắn thí sinh'
  return event.actors.map((actor) => `${actor.candidate_code} — ${actor.full_name}${actor.seat_code ? ` (${actor.seat_code})` : ''}`).join(', ')
}

function EventClip({ event }: { event: ExamEvent }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const url = eventMediaUrl(event)
  const start = Math.max(0, event.start_ms - CONTEXT_MS) / 1000
  const end = (event.end_ms + CONTEXT_MS) / 1000
  const replay = () => { const video = videoRef.current; if (!video) return; video.currentTime = start; void video.play().catch(() => undefined) }
  if (!url) return <EmptyState title="Phiên thi không có video." description="Không thể xem lại đoạn này." />
  return <div className="event-clip">
    <video ref={videoRef} src={url} muted playsInline controls preload="metadata"
      onLoadedMetadata={replay}
      onTimeUpdate={(e) => { if (e.currentTarget.currentTime > end) e.currentTarget.pause() }} />
    <div className="event-clip-bar"><span>Đoạn {formatDurationMs(start * 1000)} – {formatDurationMs(end * 1000)} (sự kiện {formatDurationMs(event.start_ms)} – {formatDurationMs(event.end_ms)})</span><button type="button" className="secondary-button" onClick={replay}>Xem lại đoạn</button></div>
  </div>
}

function EventDetailPanel({ eventId, onReviewed }: { eventId: string; onReviewed(event: ExamEventDetail): void }) {
  const { can } = usePermissions()
  const toast = useToast()
  const canReview = can(permissions.eventReview)
  const [detail, setDetail] = useState<ExamEventDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState<ReviewDecision | null>(null)
  const [reviewError, setReviewError] = useState<string | null>(null)

  useEffect(() => {
    setDetail(null); setError(null); setNote('')
    getEvent(eventId).then(setDetail).catch((requestError) => setError(apiContentErrorMessage(requestError)))
  }, [eventId])

  const submit = async (decision: ReviewDecision) => {
    setSaving(decision); setReviewError(null)
    try {
      const updated = await reviewEvent(eventId, decision, note)
      setDetail(updated); setNote(''); onReviewed(updated)
      toast.success(`Đã lưu: ${decisionLabels[decision]}.`)
    } catch (requestError) { setReviewError(apiErrorMessage(requestError)) }
    finally { setSaving(null) }
  }

  if (error) return <ErrorState message={error} />
  if (!detail) return <LoadingState message="Đang tải sự kiện…" />
  return <div className="event-detail">
    <div className="event-detail-head"><div><p className="eyebrow">{detail.event_code}</p><h2><BehaviorPill behavior={detail.behavior_type} /></h2></div><EventStatusPill status={detail.status} /></div>
    <EventClip event={detail} />
    <dl className="event-facts">
      <div><dt>Thí sinh</dt><dd>{actorLabel(detail)}</dd></div>
      <div><dt>Phiên thi</dt><dd><Link to={`/sessions/${detail.session_id}`}>{detail.exam_name}</Link> <small>{detail.session_code}</small></dd></div>
      <div><dt>Thời điểm trong video</dt><dd>{formatDurationMs(detail.start_ms)} – {formatDurationMs(detail.end_ms)}{detail.peak_ms !== null && <small> · cao nhất {formatDurationMs(detail.peak_ms)}</small>}</dd></div>
      <div><dt>Độ tin cậy AI</dt><dd>{detail.ai_confidence === null ? '—' : `${Math.round(detail.ai_confidence * 100)} %`}<small> · {detail.source === 'AI' ? 'X3D-L tự động' : 'Thủ công'}</small></dd></div>
      <div><dt>Ghi nhận lúc</dt><dd>{formatDateTime(detail.created_at)}</dd></div>
    </dl>
    <p className="event-disclaimer">AI chỉ đánh dấu hành vi đáng ngờ. Kết luận vi phạm do người xác minh quyết định.</p>
    {canReview && <div className="event-review-form">
      <label htmlFor="event-note">Ghi chú xác minh</label>
      <textarea id="event-note" maxLength={2000} rows={3} value={note} onChange={(e) => setNote(e.target.value)} placeholder="Ví dụ: thí sinh cúi xuống dùng điện thoại dưới bàn." />
      <div className="event-review-actions">
        <button type="button" className="danger-button" disabled={saving !== null} onClick={() => void submit('CONFIRM')}>{saving === 'CONFIRM' ? 'Đang lưu…' : decisionLabels.CONFIRM}</button>
        <button type="button" className="secondary-button" disabled={saving !== null} onClick={() => void submit('DISMISS')}>{saving === 'DISMISS' ? 'Đang lưu…' : decisionLabels.DISMISS}</button>
        <button type="button" className="secondary-button" disabled={saving !== null} onClick={() => void submit('NEEDS_REVIEW')}>{saving === 'NEEDS_REVIEW' ? 'Đang lưu…' : decisionLabels.NEEDS_REVIEW}</button>
      </div>
      {reviewError && <p className="form-error" role="alert">{reviewError}</p>}
    </div>}
    <div className="event-reviews"><h3>Lịch sử xác minh</h3>
      {detail.reviews.length ? <ol>{detail.reviews.map((review) => <li key={review.id}><strong>{decisionLabels[review.decision]}</strong><span>{review.reviewer_name} · {formatDateTime(review.created_at)}</span>{review.note && <p>{review.note}</p>}</li>)}</ol> : <p className="muted-text">Chưa có ai xác minh sự kiện này.</p>}
    </div>
  </div>
}

export function EventsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const sessionId = searchParams.get('session') ?? ''
  const selectedId = searchParams.get('event') ?? ''
  const [status, setStatus] = useState<EventStatus | ''>((searchParams.get('status') as EventStatus | null) ?? '')
  const [behavior, setBehavior] = useState<BehaviorType | ''>('')
  const [page, setPage] = useState(1)
  const [events, setEvents] = useState<ExamEvent[]>([])
  const [total, setTotal] = useState(0)
  const [counts, setCounts] = useState<EventCounts | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const [result, summary] = await Promise.all([
        getEvents({ sessionId, status, behavior, page, pageSize: PAGE_SIZE }),
        getEventCounts(sessionId),
      ])
      setEvents(result.items); setTotal(result.total); setCounts(summary)
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [behavior, page, sessionId, status])

  useEffect(() => { void load() }, [load])
  useEffect(() => { // new AI events arrive while a session is monitored
    const timer = window.setInterval(() => { if (!document.hidden) void load() }, 15000)
    return () => window.clearInterval(timer)
  }, [load])

  const select = (eventId: string) => {
    const next = new URLSearchParams(searchParams)
    if (eventId) next.set('event', eventId); else next.delete('event')
    setSearchParams(next)
  }
  const reviewed = (updated: ExamEventDetail) => {
    setEvents((current) => current.map((item) => item.id === updated.id ? { ...item, status: updated.status } : item))
    void getEventCounts(sessionId).then(setCounts)
  }

  return <div className="page-stack events-page">
    <PageHeader eyebrow="SỰ KIỆN" title="Sự kiện bất thường" description="Hành vi đáng ngờ do AI (X3D-L) phát hiện khi giám sát. Xem lại đoạn video và xác minh từng sự kiện." />
    {sessionId && <div className="events-session-filter card">Đang lọc theo một phiên thi. <button type="button" className="link-button" onClick={() => { const next = new URLSearchParams(searchParams); next.delete('session'); setSearchParams(next); setPage(1) }}>Xem tất cả phiên</button></div>}
    {counts && <div className="event-count-tiles">
      {([['', 'Tất cả', counts.total], ['PENDING_REVIEW', 'Chờ xác minh', counts.pending_review], ['CONFIRMED', 'Đã xác nhận', counts.confirmed], ['DISMISSED', 'Đã bỏ qua', counts.dismissed], ['NEEDS_REVIEW', 'Cần xem lại', counts.needs_review]] as const).map(([value, label, count]) =>
        <button key={label} type="button" className={`event-count-tile card ${status === value ? 'active' : ''}`} onClick={() => { setStatus(value); setPage(1) }}><span>{label}</span><strong>{count}</strong></button>)}
    </div>}
    <div className="events-toolbar">
      <select aria-label="Lọc theo hành vi" value={behavior} onChange={(e) => { setBehavior(e.target.value as BehaviorType | ''); setPage(1) }}>
        <option value="">Mọi hành vi</option>
        {(Object.keys(behaviorLabels) as BehaviorType[]).map((value) => <option key={value} value={value}>{behaviorLabels[value]}</option>)}
      </select>
      <button type="button" className="secondary-button" onClick={() => void load()}>Làm mới</button>
    </div>
    {error && <ErrorState message={error} onRetry={() => void load()} />}
    <section className={`events-layout ${selectedId ? 'with-detail' : ''}`}>
      <div className="card events-list">
        {loading && !events.length ? <LoadingState message="Đang tải sự kiện…" />
          : !events.length ? <EmptyState title="Chưa có sự kiện." description="Sự kiện xuất hiện khi AI phát hiện hành vi đáng ngờ của một thí sinh đã được xếp chỗ trong lúc giám sát." />
          : <div className="table-wrap"><table className="events-table"><thead><tr><th>Hành vi</th><th>Thí sinh</th><th>Phiên thi</th><th>Trong video</th><th>Tin cậy</th><th>Trạng thái</th></tr></thead>
            <tbody>{events.map((event) => <tr key={event.id} className={event.id === selectedId ? 'selected' : ''} onClick={() => select(event.id)} tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') select(event.id) }}>
              <td><BehaviorPill behavior={event.behavior_type} /></td>
              <td>{actorLabel(event)}</td>
              <td>{event.exam_name}<small>{formatDateTime(event.created_at)}</small></td>
              <td>{formatDurationMs(event.start_ms)} – {formatDurationMs(event.end_ms)}</td>
              <td>{event.ai_confidence === null ? '—' : `${Math.round(event.ai_confidence * 100)} %`}</td>
              <td><EventStatusPill status={event.status} /></td>
            </tr>)}</tbody></table></div>}
        <Pagination page={page} pageSize={PAGE_SIZE} total={total} onChange={setPage} />
      </div>
      {selectedId && <aside className="card events-detail-card"><button type="button" className="icon-close" aria-label="Đóng" onClick={() => select('')}>×</button><EventDetailPanel eventId={selectedId} onReviewed={reviewed} /></aside>}
    </section>
  </div>
}
