import { useEffect, useState } from 'react'
import './sessionDetail.css'
import { Link, useParams } from 'react-router-dom'

import { apiContentErrorMessage } from '../../shared/api/errors'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { LoadingState } from '../../shared/components/LoadingState'
import { StatusBadge } from '../../shared/components/StatusBadge'
import { formatDateTime, formatDurationMs, formatFps, formatResolution } from '../../shared/formatters'
import { permissions, usePermissions } from '../auth/permissions'
import { getMonitoringStatus } from '../monitoring/api'
import type { MonitoringStatus } from '../monitoring/types'
import { getSession } from './api'
import type { ExamSession } from './types'

type Tab = 'OVERVIEW' | 'TRACKING' | 'SYSTEM'

function duration(start: string | null, end: string | null): string {
  if (!start || !end) return '—'
  return formatDurationMs(Math.max(0, new Date(end).getTime() - new Date(start).getTime()))
}

function sourceLabel(session: ExamSession): string {
  return session.source_type === 'CAMERA'
    ? session.camera?.name ?? 'Camera'
    : 'Video tải lên'
}

export function SessionDetailPage() {
  const { id = '' } = useParams()
  const { can } = usePermissions()
  const canSeeSystem = can(permissions.diagnosticsRead)
  const canMonitor = can(permissions.sessionMonitor)
  const [session, setSession] = useState<ExamSession | null>(null)
  const [runtime, setRuntime] = useState<MonitoringStatus | null>(null)
  const [tab, setTab] = useState<Tab>('OVERVIEW')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true); setError(null)
    try {
      const detail = await getSession(id)
      setSession(detail)
      try { setRuntime(await getMonitoringStatus(id)) } catch { setRuntime(null) }
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }

  useEffect(() => { void load() }, [id])

  if (loading) return <LoadingState message="Đang tải phiên thi…" />
  if (!session) return <div className="page-stack"><ErrorState message={error ?? 'Không tìm thấy phiên thi.'} onRetry={() => void load()} /></div>

  const running = session.status === 'RUNNING' || session.status === 'PAUSED'
  const diagnostics = runtime?.diagnostics

  return <div className="page-stack session-detail-page">
    <div className="breadcrumb"><Link to="/sessions">Phiên thi</Link><span>/</span><span>{session.session_code}</span></div>
    <div className="page-heading"><div><p className="eyebrow">CHI TIẾT PHIÊN THI</p><h1>{session.exam_name}</h1><p>{session.session_code} · {session.room.code} — {session.room.name}</p></div><div className="detail-heading-actions"><StatusBadge status={session.status} />{running && canMonitor && <Link className="primary-button link-button" to={`/monitoring?session=${session.id}`}>Mở giám sát</Link>}</div></div>
    {error && <ErrorState message={error} onRetry={() => void load()} />}
    <nav className="detail-tabs" aria-label="Thông tin phiên thi"><button className={tab === 'OVERVIEW' ? 'active' : ''} type="button" onClick={() => setTab('OVERVIEW')}>Tổng quan</button><button className={tab === 'TRACKING' ? 'active' : ''} type="button" onClick={() => setTab('TRACKING')}>Theo dõi</button>{canSeeSystem && <button className={tab === 'SYSTEM' ? 'active' : ''} type="button" onClick={() => setTab('SYSTEM')}>Hệ thống</button>}</nav>

    {tab === 'OVERVIEW' && <section className="card"><h2>Tổng quan phiên thi</h2><dl className="detail-list session-overview-list">
      <div><dt>Tên phiên</dt><dd>{session.exam_name}</dd></div>
      <div><dt>Mã phiên</dt><dd>{session.session_code}</dd></div>
      <div><dt>Trạng thái</dt><dd><StatusBadge status={session.status} /></dd></div>
      <div><dt>Phòng</dt><dd>{session.room.code} — {session.room.name}</dd></div>
      <div><dt>Nguồn</dt><dd>{sourceLabel(session)}</dd></div>
      <div><dt>Ngày thi</dt><dd>{formatDateTime(session.scheduled_start)}</dd></div>
      <div><dt>Bắt đầu dự kiến</dt><dd>{formatDateTime(session.scheduled_start)}</dd></div>
      <div><dt>Bắt đầu thực tế</dt><dd>{formatDateTime(session.actual_start)}</dd></div>
      <div><dt>Kết thúc thực tế</dt><dd>{formatDateTime(session.actual_end)}</dd></div>
      <div><dt>Thời lượng dự kiến</dt><dd>{duration(session.scheduled_start, session.scheduled_end)}</dd></div>
      <div><dt>Thời lượng thực tế</dt><dd>{duration(session.actual_start, session.actual_end)}</dd></div>
      <div><dt>Người tạo</dt><dd>{session.created_by_user.full_name ?? session.created_by_user.username}</dd></div>
    </dl></section>}

    {tab === 'TRACKING' && <section className="card"><h2>Kết quả theo dõi</h2>{diagnostics ? <dl className="detail-list session-overview-list">
      <div><dt>Người hiện tại / cuối phiên</dt><dd>{diagnostics.active_track_count}</dd></div>
      <div><dt>Active tracks</dt><dd>{diagnostics.active_logical_actors}</dd></div>
      <div><dt>Lost tracks</dt><dd>{diagnostics.lost_logical_actors}</dd></div>
      <div><dt>Tracking recoveries</dt><dd>{diagnostics.recoveries_total}</dd></div>
    </dl> : <EmptyState title="Chưa có số liệu theo dõi ổn định." description="Hệ thống chỉ hiển thị metric thật do runtime cung cấp." />}</section>}

    {tab === 'SYSTEM' && canSeeSystem && <section className="card"><h2>Chẩn đoán hệ thống</h2><dl className="detail-list session-overview-list">
      <div><dt>Nguồn video</dt><dd>{session.video?.original_filename ?? '—'}</dd></div>
      <div><dt>Độ phân giải</dt><dd>{session.video ? formatResolution(session.video.width, session.video.height) : '—'}</dd></div>
      <div><dt>Video FPS</dt><dd>{session.video ? formatFps(session.video.fps, 2) : '—'}</dd></div>
      <div><dt>Inference FPS</dt><dd>{diagnostics ? formatFps(diagnostics.analysis_fps, 1) : '—'}</dd></div>
      <div><dt>Detector</dt><dd>{diagnostics ? 'Healthy' : 'Chưa có dữ liệu'}</dd></div>
      <div><dt>Tracker</dt><dd>{diagnostics ? 'Healthy' : 'Chưa có dữ liệu'}</dd></div>
    </dl></section>}
  </div>
}
