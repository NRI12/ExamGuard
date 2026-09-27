import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { apiContentErrorMessage } from '../../shared/api/errors'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { Pagination } from '../../shared/components/Pagination'
import { StatusBadge } from '../../shared/components/StatusBadge'
import { formatDateTime, formatDurationMs } from '../../shared/formatters'
import { useDebouncedValue } from '../../shared/hooks/useDebouncedValue'
import { sessionStatusLabels } from '../../shared/i18n/vi'
import { permissions, usePermissions } from '../auth/permissions'
import { getSessions } from './api'
import type { ExamSession, SessionStatus } from './types'

const pageSize = 20

function sourceLabel(session: ExamSession): string {
  if (session.source_type === 'CAMERA') return session.camera?.name ?? 'Camera'
  return session.video ? 'Video tải lên' : 'Chưa cấu hình'
}

export function SessionsPage() {
  const { can } = usePermissions()
  const canMonitor = can(permissions.sessionMonitor)
  const [items, setItems] = useState<ExamSession[]>([])
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebouncedValue(query)
  const [statusFilter, setStatusFilter] = useState<SessionStatus | ''>('')
  const [dateFilter, setDateFilter] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (
    search: string,
    targetPage: number,
    selectedStatus: SessionStatus | '',
    selectedDate: string,
  ) => {
    setLoading(true); setError(null)
    try {
      const result = await getSessions({
        query: search,
        page: targetPage,
        pageSize,
        status: selectedStatus || undefined,
        date: selectedDate || undefined,
      })
      setItems(result.items); setTotal(result.total)
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    setPage(1); void load(debouncedQuery, 1, statusFilter, dateFilter)
  }, [debouncedQuery, dateFilter, load, statusFilter])

  if (loading && items.length === 0 && !query && !statusFilter && !dateFilter) {
    return <LoadingState message="Đang tải danh sách phiên thi…" />
  }

  return <div className="page-stack">
    <PageHeader eyebrow="PHIÊN THI" title="Phiên thi" actions={canMonitor && <Link className="primary-button link-button" to="/monitoring">Bắt đầu giám sát</Link>} />
    {error && <ErrorState message={error} onRetry={() => void load(debouncedQuery, page, statusFilter, dateFilter)} />}
    <form className="filter-row" role="search" onSubmit={(event) => { event.preventDefault(); setPage(1); void load(query, 1, statusFilter, dateFilter) }}>
      <label className="sr-only" htmlFor="session-search">Tìm kiếm phiên thi</label>
      <input id="session-search" placeholder="Tìm theo tên hoặc mã…" value={query} onChange={(event) => setQuery(event.target.value)} />
      <label className="sr-only" htmlFor="session-status-filter">Lọc theo trạng thái</label>
      <select id="session-status-filter" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as SessionStatus | '')}><option value="">Tất cả trạng thái</option>{Object.entries(sessionStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
      <label className="sr-only" htmlFor="session-date-filter">Lọc theo ngày</label>
      <input id="session-date-filter" type="date" value={dateFilter} onChange={(event) => setDateFilter(event.target.value)} />
    </form>
    <section className="card table-wrap">
      {loading && <p className="inline-loading">Đang tải…</p>}
      {items.length > 0 && <table><thead><tr><th>Tên phiên</th><th>Ngày / giờ</th><th>Phòng</th><th>Nguồn</th><th>Thời lượng</th><th>Trạng thái</th><th>Thao tác</th></tr></thead><tbody>
        {items.map((item) => {
          const running = item.status === 'RUNNING' || item.status === 'PAUSED'
          const duration = item.scheduled_start && item.scheduled_end
            ? new Date(item.scheduled_end).getTime() - new Date(item.scheduled_start).getTime()
            : null
          return <tr key={item.id}><td><strong>{item.exam_name}</strong><small className="table-subtitle">{item.session_code}</small></td><td>{formatDateTime(item.scheduled_start)}</td><td>{item.room.code}</td><td>{sourceLabel(item)}</td><td>{duration === null ? '—' : formatDurationMs(duration)}</td><td><StatusBadge status={item.status} /></td><td><Link className="secondary-button link-button" to={running && canMonitor ? `/monitoring?session=${item.id}` : `/sessions/${item.id}`}>{running && canMonitor ? 'Mở giám sát' : 'Chi tiết'}</Link></td></tr>
        })}
      </tbody></table>}
      {!loading && items.length === 0 && <EmptyState title={query || statusFilter || dateFilter ? 'Không tìm thấy kết quả phù hợp.' : 'Chưa có phiên thi.'} description={!query && !statusFilter && !dateFilter && canMonitor ? 'Bắt đầu giám sát để tạo phiên đầu tiên.' : undefined} />}
      <Pagination page={page} pageSize={pageSize} total={total} onChange={(next) => { setPage(next); void load(debouncedQuery, next, statusFilter, dateFilter) }} />
    </section>
  </div>
}
