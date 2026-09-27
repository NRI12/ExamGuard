import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import './monitoring.css'

import { apiContentErrorMessage, apiErrorMessage } from '../../shared/api/errors'
import { ConfirmDialog } from '../../shared/components/ConfirmDialog'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { StatusBadge } from '../../shared/components/StatusBadge'
import { formatDateTime, formatDurationMs, formatFps, formatResolution } from '../../shared/formatters'
import { runtimeStateLabels } from '../../shared/i18n/vi'
import { permissions, usePermissions } from '../auth/permissions'
import { getEvents } from '../events/api'
import { actorLabel, BehaviorPill, EventStatusPill } from '../events/EventsPage'
import type { ExamEvent } from '../events/types'
import { VideoMonitor } from '../media/VideoMonitor'
import { getRooms } from '../rooms/api'
import type { Room } from '../rooms/types'
import { getSessions } from '../sessions/api'
import type { ExamSession, SessionStatus } from '../sessions/types'
import {
  getMonitoringStatus,
  pauseMonitoring,
  resumeMonitoring,
  seekMonitoring,
  startMonitoring,
  stopMonitoring,
} from './api'
import { StartMonitoringModal } from './StartMonitoringModal'
import { cheatLabels, TrackingCanvas } from './TrackingCanvas'
import { isTrackingTimestampAligned, TrackingBuffer } from './trackingBuffer'
import type { CheatActorState, CheatDiagnostics, MonitoringMessage, MonitoringStatus, RuntimeDiagnostics, TrackingTrack } from './types'
import { useMonitoringSocket } from './useMonitoringSocket'

const inactiveStatus: MonitoringStatus = {
  session_id: '', state: 'INACTIVE', profile: null, error: null,
  subscriber_count: 0, queue_size: 0, dropped_analysis_frames: 0, diagnostics: null,
  runtime_instance_id: null, runtime_generation: null, worker_instance_id: null,
  tracker_instance_id: null, tracking_seq: 0,
}

function metric(value: number | null | undefined, suffix = ''): string {
  return value === null || value === undefined ? '—' : `${value.toFixed(1)}${suffix}`
}

export function MonitoringPage() {
  const { can } = usePermissions()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedSessionId = searchParams.get('session') ?? ''
  const canOperate = can(permissions.sessionMonitor)
  const debugOverlay = import.meta.env.DEV && can(permissions.diagnosticsRead)
  const [sessions, setSessions] = useState<ExamSession[]>([])
  const [sessionTotal, setSessionTotal] = useState(0)
  const [rooms, setRooms] = useState<Room[]>([])
  const [selectedId, setSelectedId] = useState(requestedSessionId)
  const [showStartModal, setShowStartModal] = useState(false)
  const [pendingAutoStart, setPendingAutoStart] = useState<string | null>(null)
  const [runtime, setRuntime] = useState<MonitoringStatus>(inactiveStatus)
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics | null>(null)
  const [activeTracks, setActiveTracks] = useState<TrackingTrack[]>([])
  const [synchronizing, setSynchronizing] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [overlayRevision, setOverlayRevision] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [errorDetail, setErrorDetail] = useState<string | null>(null)
  const [confirmStop, setConfirmStop] = useState(false)
  const [cheatStates, setCheatStates] = useState<ReadonlyMap<string, CheatActorState>>(new Map())
  const [cheatDiagnostics, setCheatDiagnostics] = useState<CheatDiagnostics | null>(null)
  const [sessionEvents, setSessionEvents] = useState<ExamEvent[]>([])
  const [sessionEventTotal, setSessionEventTotal] = useState(0)
  const canReadEvents = can(permissions.eventRead)
  const videoRef = useRef<HTMLVideoElement>(null)
  const trackingBuffer = useRef(new TrackingBuffer())
  const synchronizingRef = useRef(false)
  const suppressVideoEvents = useRef(false)
  const lastTrackUiUpdate = useRef(0)
  const runtimeStartedAt = useRef<number | null>(null)

  const selected = sessions.find((session) => session.id === selectedId) ?? null
  const runtimeActive = ['INITIALIZING', 'RUNNING', 'PAUSED'].includes(runtime.state)
  const candidateCodes = useMemo(() => new Map(selected?.assignments.map((item) => [item.id, item.candidate.candidate_code]) ?? []), [selected])

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const [sessionPage, roomPage] = await Promise.all([
        getSessions({ pageSize: 20 }),
        canOperate ? getRooms({ pageSize: 100 }) : Promise.resolve({ items: [], total: 0, page: 1, page_size: 100 }),
      ])
      setSessions(sessionPage.items); setSessionTotal(sessionPage.total)
      setRooms(roomPage.items.filter((room) => room.is_active))
      const requested = sessionPage.items.find((item) => item.id === requestedSessionId)
      const active = sessionPage.items.find((item) => item.status === 'RUNNING' || item.status === 'PAUSED')
      const target = requested ?? active
      if (target) {
        setSelectedId(target.id)
        runtimeStartedAt.current = target.actual_start ? new Date(target.actual_start).getTime() : null
      }
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [canOperate, requestedSessionId])

  useEffect(() => { void load() }, [load])
  useEffect(() => {
    trackingBuffer.current.reset(); setActiveTracks([]); setDiagnostics(null); setElapsedMs(0); setSynchronizing(false); synchronizingRef.current = false; setOverlayRevision((value) => value + 1)
    if (!selectedId) { setRuntime(inactiveStatus); setDiagnostics(null); return }
    getMonitoringStatus(selectedId).then((status) => { setRuntime(status); setDiagnostics(status.diagnostics) }).catch((requestError) => setError(apiContentErrorMessage(requestError)))
  }, [selectedId])
  useEffect(() => {
    if (!runtimeActive) return
    const timer = window.setInterval(() => {
      const startedAt = runtimeStartedAt.current
      setElapsedMs(startedAt ? Math.max(0, Date.now() - startedAt) : Math.round((videoRef.current?.currentTime ?? 0) * 1000))
    }, 500)
    return () => window.clearInterval(timer)
  }, [runtimeActive])

  const loadSessionEvents = useCallback(async () => {
    if (!selectedId || !canReadEvents) { setSessionEvents([]); setSessionEventTotal(0); return }
    try {
      const page = await getEvents({ sessionId: selectedId, pageSize: 6 })
      setSessionEvents(page.items); setSessionEventTotal(page.total)
    } catch { /* the live view keeps working without the event list */ }
  }, [canReadEvents, selectedId])
  useEffect(() => { void loadSessionEvents() }, [loadSessionEvents])

  const resetTracking = useCallback((sync = false) => {
    trackingBuffer.current.clearFrames(); setActiveTracks([]); setCheatStates(new Map())
    setSynchronizing(sync); synchronizingRef.current = sync
    setOverlayRevision((value) => value + 1)
  }, [])

  useEffect(() => {
    if (runtime.state !== 'RUNNING' || !selected?.video || !videoRef.current || pendingAutoStart) return
    const video = videoRef.current
    const latestTimestampMs = runtime.diagnostics?.latest_timestamp_ms
    suppressVideoEvents.current = true
    if (latestTimestampMs !== undefined && Number.isFinite(latestTimestampMs)) {
      video.currentTime = Math.max(0, latestTimestampMs / 1000)
      resetTracking(true)
    }
    video.muted = true
    void video.play().finally(() => {
      window.setTimeout(() => { suppressVideoEvents.current = false }, 0)
    })
  }, [pendingAutoStart, resetTracking, runtime.diagnostics?.latest_timestamp_ms, runtime.state, selected?.id, selected?.video])

  const handleSocketMessage = useCallback((message: MonitoringMessage) => {
    if (message.type === 'tracking') {
      if (synchronizingRef.current && !isTrackingTimestampAligned(message.timestamp_ms, (videoRef.current?.currentTime ?? 0) * 1000)) return
      const inserted = trackingBuffer.current.insert(message)
      if (!inserted.accepted) return
      if (inserted.reset) { setActiveTracks([]); setOverlayRevision((value) => value + 1) }
      synchronizingRef.current = false; setSynchronizing(false)
      const now = performance.now()
      if (now - lastTrackUiUpdate.current >= 250) {
        lastTrackUiUpdate.current = now; setActiveTracks(message.tracks)
      }
      return
    }
    if (message.type === 'cheat_prediction') {
      setCheatStates(new Map(message.actors.map((actor) => [actor.actor_id, actor])))
      setCheatDiagnostics(message.diagnostics)
      return
    }
    if (message.type === 'cheat_event') { void loadSessionEvents(); return }
    if (message.type === 'diagnostics') {
      const active = trackingBuffer.current.activateRuntime(message.runtime_instance_id, message.runtime_generation)
      if (!active.accepted) return
      if (active.reset) resetTracking()
      setDiagnostics(message)
      return
    }
    if (message.type !== 'state') return
    const active = trackingBuffer.current.activateRuntime(message.runtime_instance_id, message.runtime_generation)
    if (!active.accepted) return
    if (active.reset) resetTracking()
    setRuntime((current) => ({ ...current, state: message.state, error: message.error }))
    setSynchronizing(message.synchronizing); synchronizingRef.current = message.synchronizing
    if (message.state === 'ERROR') {
      resetTracking(); setError('Không thể khởi tạo hoặc duy trì AI.'); setErrorDetail(message.error)
    }
  }, [loadSessionEvents, resetTracking])

  useEffect(() => () => trackingBuffer.current.reset(), [])
  const activeAlerts = [...cheatStates.values()].filter((state) => state.alert)
  const candidateNames = new Map(selected?.assignments.map((item) => [item.id, `${item.candidate.candidate_code} — ${item.candidate.full_name}`]) ?? [])
  const socketConnected = useMonitoringSocket({ sessionId: selectedId, enabled: runtimeActive, onMessage: handleSocketMessage })

  const perform = async (operation: () => Promise<MonitoringStatus>) => {
    setError(null); setErrorDetail(null)
    try {
      const status = await operation(); setRuntime(status)
      const nextSessionStatus: SessionStatus | null = status.state === 'INACTIVE' ? null : status.state === 'INITIALIZING' ? 'RUNNING' : status.state
      if (nextSessionStatus) setSessions((current) => current.map((item) => item.id === selectedId ? { ...item, status: nextSessionStatus } : item))
      if (status.diagnostics) setDiagnostics(status.diagnostics)
      return status
    } catch (requestError) {
      setError('Không thể thực hiện thao tác giám sát.'); setErrorDetail(apiErrorMessage(requestError)); return null
    }
  }

  const start = useCallback(async () => {
    if (!selected || !videoRef.current) return
    const video = videoRef.current
    try { await video.play() } catch { setError('Trình duyệt không thể bắt đầu phát video.'); return }
    trackingBuffer.current.reset(); resetTracking(true)
    runtimeStartedAt.current = Date.now()
    const status = await perform(() => startMonitoring(selected.id, Math.round(video.currentTime * 1000)))
    if (!status) { suppressVideoEvents.current = true; video.pause(); resetTracking(); window.setTimeout(() => { suppressVideoEvents.current = false }, 0) }
  }, [selected, resetTracking])

  useEffect(() => {
    if (!pendingAutoStart || selected?.id !== pendingAutoStart || !selected.video || !videoRef.current) return
    setPendingAutoStart(null); void start()
  }, [pendingAutoStart, selected, start])

  const prepared = async (session: ExamSession) => {
    setSessions((current) => [session, ...current.filter((item) => item.id !== session.id)])
    setSessionTotal((current) => current + 1); setSelectedId(session.id); setSearchParams({ session: session.id }); setShowStartModal(false); setPendingAutoStart(session.id)
  }
  const onVideoPause = (video: HTMLVideoElement) => {
    if (canOperate && !video.ended && !suppressVideoEvents.current && runtime.state === 'RUNNING') void perform(() => pauseMonitoring(selectedId, Math.round(video.currentTime * 1000)))
  }
  const onVideoPlay = () => {
    if (canOperate && !suppressVideoEvents.current && runtime.state === 'PAUSED') void perform(() => resumeMonitoring(selectedId))
  }
  const seek = (video: HTMLVideoElement) => {
    if (!runtimeActive || !canOperate) return
    resetTracking(true); void perform(() => seekMonitoring(selectedId, Math.round(video.currentTime * 1000)))
  }
  const stop = async () => {
    suppressVideoEvents.current = true; videoRef.current?.pause()
    const status = await perform(() => stopMonitoring(selectedId)); trackingBuffer.current.reset(); resetTracking()
    if (status) {
      setSessions((current) => current.map((item) => item.id === selectedId ? { ...item, status: 'COMPLETED' } : item))
      navigate(`/sessions/${selectedId}`)
    }
    window.setTimeout(() => { suppressVideoEvents.current = false }, 0)
  }

  if (loading && sessions.length === 0) return <LoadingState message="Đang tải giám sát…" />

  if (!selected) return <div className="page-stack monitoring-page">
    <PageHeader eyebrow="GIÁM SÁT" title="Giám sát phòng thi" description="Tạo phiên và bắt đầu theo dõi chỉ trong một bước." actions={canOperate && <button className="primary-button start-monitoring-cta" type="button" onClick={() => setShowStartModal(true)}>Bắt đầu giám sát</button>} />
    {error && <ErrorState message={error} onRetry={() => void load()} />}
    <section className="monitoring-hero card"><div className="monitoring-hero-icon">▶</div><h2>Chưa có phiên đang giám sát</h2><p>Chọn video nguồn, hệ thống sẽ tự khởi tạo nhận diện người và theo dõi vị trí.</p>{canOperate && <button className="primary-button" type="button" onClick={() => setShowStartModal(true)}>Bắt đầu giám sát</button>}</section>
    <section className="card recent-sessions"><div className="section-heading"><div><h2>Phiên gần đây</h2><p>Mở lại thông tin hoặc tiếp tục một phiên đã sẵn sàng.</p></div><Link className="secondary-button link-button" to="/sessions">Xem tất cả</Link></div>
      {sessions.length ? <div className="recent-session-list">{sessions.slice(0, 6).map((session) => { const active = session.status === 'RUNNING' || session.status === 'PAUSED'; return <Link key={session.id} to={active ? `/monitoring?session=${session.id}` : `/sessions/${session.id}`}><span><strong>{session.exam_name}</strong><small>{session.room.code} · {session.camera?.name ?? (session.source_type === 'VIDEO_UPLOAD' ? 'Video tải lên' : 'Camera')} · {formatDateTime(session.scheduled_start)}</small></span><span><StatusBadge status={session.status} /><small>{active ? 'Mở giám sát' : 'Chi tiết'}</small></span></Link> })}</div> : <EmptyState title="Chưa có phiên thi." description="Bắt đầu phiên đầu tiên từ nút phía trên." />}
    </section>
    {showStartModal && <StartMonitoringModal rooms={rooms} recentSessions={sessions} sessionTotal={sessionTotal} onCancel={() => setShowStartModal(false)} onPrepared={prepared} />}
  </div>

  return <div className="page-stack monitoring-page live-monitoring-page">
    <div className="live-header"><button className="back-button" type="button" disabled={runtimeActive} onClick={() => { setSelectedId(''); setSearchParams({}) }}>← Trở về</button><div><p className="eyebrow">ĐANG GIÁM SÁT</p><h1>{selected.exam_name}</h1><p>{selected.room.code} — {selected.room.name}</p></div><div className="live-header-status"><StatusBadge status={selected.status} /><strong>{formatDurationMs(elapsedMs)}</strong></div></div>
    {error && <><ErrorState message={error} />{errorDetail && debugOverlay && <details className="error-detail"><summary>Chi tiết kỹ thuật</summary><code>{errorDetail}</code></details>}</>}
    {selected.video ? <>
      <section className="monitoring-live-grid">
        <div className="card monitoring-video-card"><div className="media-summary"><strong>{selected.source_type === 'CAMERA' ? selected.camera?.name : selected.video.original_filename}</strong><span>{formatResolution(selected.video.width, selected.video.height)} · {formatFps(selected.video.fps, 2)}</span></div>
          <VideoMonitor ref={videoRef} mediaUrl={selected.video.media_url} title={selected.camera?.name ?? selected.video.original_filename} loop={selected.source_type === 'CAMERA'} realtime={selected.source_type === 'CAMERA'} overlay={<><TrackingCanvas videoRef={videoRef} buffer={trackingBuffer.current} revision={overlayRevision} candidateCodes={candidateCodes} debug={debugOverlay} showConfidence={debugOverlay} cheatStates={cheatStates} />{(synchronizing || runtime.state === 'INITIALIZING') && <div className="sync-indicator">{runtime.state === 'INITIALIZING' ? 'Đang khởi tạo nhận diện người…' : 'Đang đồng bộ theo video…'}</div>}</>} onPause={onVideoPause} onPlay={onVideoPlay} onSeeking={selected.source_type === 'CAMERA' ? undefined : () => resetTracking(true)} onSeeked={selected.source_type === 'CAMERA' ? undefined : seek} onEnded={() => { if (selected.source_type === 'VIDEO_UPLOAD' && runtimeActive && canOperate) void stop() }} />
          <div className="monitoring-actions">{runtime.state === 'INACTIVE' && canOperate && <button className="primary-button" disabled={selected.status !== 'READY'} type="button" onClick={() => void start()}>Bắt đầu giám sát</button>}{runtime.state === 'INITIALIZING' && <button className="primary-button" disabled type="button">Đang khởi tạo…</button>}{runtime.state === 'RUNNING' && canOperate && <button className="secondary-button" type="button" onClick={() => videoRef.current?.pause()}>Tạm dừng</button>}{runtime.state === 'PAUSED' && canOperate && <button className="primary-button" type="button" onClick={() => void videoRef.current?.play()}>Tiếp tục</button>}{runtimeActive && canOperate && <button className="danger-button subtle" type="button" onClick={() => setConfirmStop(true)}>Kết thúc giám sát</button>}</div>
        </div>
        <aside className="card monitoring-side-panel"><p className="panel-label">TRẠNG THÁI HỆ THỐNG</p><div className={`runtime-state ${runtime.state.toLowerCase()}`}><span />{runtimeStateLabels[runtime.state]}</div><dl className="runtime-summary-list"><div><dt>Video</dt><dd>{runtimeActive ? 'Đang phát' : 'Sẵn sàng'}</dd></div><div><dt>Nhận diện người</dt><dd>{diagnostics ? 'Hoạt động' : runtime.state === 'INITIALIZING' ? 'Đang khởi tạo' : 'Chờ'}</dd></div><div><dt>Theo dõi</dt><dd>{socketConnected ? 'Đã kết nối' : 'Chưa kết nối'}</dd></div><div><dt>Người hiện tại</dt><dd>{activeTracks.length}</dd></div><div><dt>Active tracks</dt><dd>{diagnostics?.active_logical_actors ?? activeTracks.length}</dd></div><div><dt>Lost tracks</dt><dd>{diagnostics?.lost_logical_actors ?? 0}</dd></div><div><dt>Nguồn</dt><dd>{selected.source_type === 'CAMERA' ? selected.camera?.name : 'Video tải lên'}</dd></div><div><dt>Phân tích hành vi</dt><dd>{cheatDiagnostics ? `X3D-L · ${cheatDiagnostics.inference_ms_mean ?? '—'} ms` : runtimeActive ? 'Đang chờ 3,2 s đầu' : 'Chờ'}</dd></div></dl>
          <div className="ai-alerts"><p className="panel-label">CẢNH BÁO AI {activeAlerts.length > 0 && <span className="ai-alert-count">{activeAlerts.length}</span>}</p>
            {activeAlerts.length ? <ul>{activeAlerts.map((state) => <li key={state.actor_id} className={`ai-alert ${state.label}`}><strong>{cheatLabels[state.label]}</strong><span>{state.session_candidate_id ? candidateNames.get(state.session_candidate_id) ?? state.seat_code : `Người #${state.track_id} (chưa xếp chỗ)`}</span><small>Điểm bất thường {Math.round(state.cheat_score * 100)}%</small></li>)}</ul>
              : <p className="muted-text">{runtimeActive ? 'Không có hành vi đáng ngờ.' : 'Bắt đầu giám sát để phân tích.'}</p>}
          </div>
          {canReadEvents && <div className="ai-events"><p className="panel-label">SỰ KIỆN ĐÃ GHI NHẬN ({sessionEventTotal})</p>
            {sessionEvents.length ? <ul>{sessionEvents.map((event) => <li key={event.id}><Link to={`/events?session=${selected.id}&event=${event.id}`}><BehaviorPill behavior={event.behavior_type} /><span>{actorLabel(event)}</span><small>{formatDurationMs(event.start_ms)} – {formatDurationMs(event.end_ms)}</small><EventStatusPill status={event.status} /></Link></li>)}</ul>
              : <p className="muted-text">Chưa có sự kiện. Chỉ thí sinh đã được xếp chỗ mới được ghi nhận sự kiện.</p>}
            {sessionEventTotal > 0 && <Link className="secondary-button link-button" to={`/events?session=${selected.id}`}>Xác minh sự kiện</Link>}
          </div>}
        </aside>
      </section>
      <div className="monitoring-status-bar"><span className={socketConnected && runtime.state === 'RUNNING' ? 'online' : ''}>● {socketConnected ? 'Hệ thống trực tuyến' : 'Đang chờ kết nối'}</span><span>Video {formatFps(selected.video.fps)}</span><span>Phân tích {metric(diagnostics?.analysis_fps, ' FPS')}</span>{debugOverlay && <><span>Độ trễ {metric(diagnostics?.analysis_lag_ms, ' ms')}</span><span>Queue {diagnostics?.queue_size ?? '—'}</span></>}</div>
      {debugOverlay && <details className="card diagnostics-drawer"><summary>Chẩn đoán tracking</summary><div className="diagnostics-grid"><span>Raw tracks <strong>{diagnostics?.raw_track_count ?? '—'}</strong></span><span>Logical active / lost <strong>{diagnostics ? `${diagnostics.active_logical_actors} / ${diagnostics.lost_logical_actors}` : '—'}</strong></span><span>Recoveries <strong>{diagnostics?.recoveries_total ?? '—'}</strong></span><span>Dynamic pairs <strong>{diagnostics?.dynamic_pairs ?? '—'}</strong></span><span>Detector <strong>{metric(diagnostics?.detector_ms, ' ms')}</strong></span><span>Tracker <strong>{metric(diagnostics?.tracker_ms, ' ms')}</strong></span><span>GPU / VRAM <strong>{metric(diagnostics?.gpu_util_pct, '%')} / {metric(diagnostics?.vram_used_mb, ' MB')}</strong></span></div></details>}
    </> : <section className="card"><EmptyState title="Phiên thi không có video nguồn." description="Không thể mở nguồn video của phiên này." /></section>}
    <ConfirmDialog open={confirmStop} title="Kết thúc phiên giám sát?" description="Video realtime và quá trình theo dõi sẽ dừng." confirmLabel="Kết thúc" danger onCancel={() => setConfirmStop(false)} onConfirm={() => { setConfirmStop(false); void stop() }} />
  </div>
}
