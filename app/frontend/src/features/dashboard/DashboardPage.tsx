import { useEffect, useState } from 'react'
import './dashboard.css'
import { Link } from 'react-router-dom'

import { apiContentErrorMessage } from '../../shared/api/errors'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { StatusBadge } from '../../shared/components/StatusBadge'
import { formatDateTime } from '../../shared/formatters'
import { permissions, usePermissions } from '../auth/permissions'
import { getSessions } from '../sessions/api'
import type { ExamSession } from '../sessions/types'

function sourceLabel(session: ExamSession): string {
  return session.source_type === 'CAMERA'
    ? session.camera?.name ?? 'Camera'
    : 'Video tải lên'
}

export function DashboardPage() {
  const { can } = usePermissions()
  const canMonitor = can(permissions.sessionMonitor)
  const [sessions, setSessions] = useState<ExamSession[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true); setError(null)
    try { setSessions((await getSessions({ pageSize: 6 })).items) }
    catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }

  useEffect(() => { void load() }, [])
  if (loading) return <LoadingState message="Đang tải trang chủ…" />

  const active = sessions.find((session) => session.status === 'RUNNING' || session.status === 'PAUSED')
  return <div className="page-stack home-page">
    <PageHeader eyebrow="TRANG CHỦ" title="Hệ thống giám sát phòng thi" />
    {error && <ErrorState message={error} onRetry={() => void load()} />}
    {active ? <section className="card home-active-session"><div><p className="eyebrow">PHIÊN ĐANG GIÁM SÁT</p><h2>{active.exam_name}</h2><p>{active.room.code} · {sourceLabel(active)}</p></div><div><StatusBadge status={active.status} /><Link className="primary-button link-button" to={canMonitor ? `/monitoring?session=${active.id}` : `/sessions/${active.id}`}>{canMonitor ? 'Tiếp tục giám sát' : 'Xem chi tiết'}</Link></div></section> : <section className="card home-ready"><div><span className="status-dot" /><div><h2>Hệ thống sẵn sàng</h2><p>Bắt đầu một phiên để theo dõi camera hoặc video.</p></div></div>{canMonitor && <Link className="primary-button link-button" to="/monitoring">Bắt đầu giám sát</Link>}</section>}
    <section className="card"><div className="section-heading"><div><h2>Phiên gần đây</h2><p>Lịch sử phiên giám sát gần nhất.</p></div><Link className="secondary-button link-button" to="/sessions">Xem tất cả</Link></div>{sessions.length ? <div className="home-session-list">{sessions.map((session) => <Link key={session.id} to={canMonitor && (session.status === 'RUNNING' || session.status === 'PAUSED') ? `/monitoring?session=${session.id}` : `/sessions/${session.id}`}><span><strong>{session.exam_name}</strong><small>{formatDateTime(session.scheduled_start)} · {session.room.code} · {sourceLabel(session)}</small></span><StatusBadge status={session.status} /></Link>)}</div> : <EmptyState title="Chưa có phiên thi." />}</section>
  </div>
}
