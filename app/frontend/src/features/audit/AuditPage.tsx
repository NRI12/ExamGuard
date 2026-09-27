import { useCallback, useEffect, useState } from 'react'

import { apiContentErrorMessage } from '../../shared/api/errors'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { Pagination } from '../../shared/components/Pagination'
import { formatDateTime } from '../../shared/formatters'
import { roleLabels } from '../../shared/i18n/vi'
import type { UserRole } from '../auth/types'
import { getAuditLogs } from './api'
import type { AuditLogEntry } from './types'

const pageSize = 20
const actionLabels: Record<string, string> = {
  USER_CREATED: 'Tạo tài khoản',
  USER_UPDATED: 'Cập nhật tài khoản',
  USER_ROLE_CHANGED: 'Thay đổi vai trò',
  USER_DEACTIVATED: 'Vô hiệu hóa tài khoản',
  USER_REACTIVATED: 'Kích hoạt lại tài khoản',
  ROOM_CREATED: 'Tạo phòng thi',
  ROOM_UPDATED: 'Cập nhật phòng thi',
  SEAT_LAYOUT_UPDATED: 'Cập nhật bố trí chỗ ngồi',
  CANDIDATE_CREATED: 'Tạo thí sinh',
  CANDIDATE_UPDATED: 'Cập nhật thí sinh',
  SESSION_CREATED: 'Tạo phiên thi',
  SESSION_UPDATED: 'Cập nhật phiên thi',
  SESSION_CANCELLED: 'Hủy phiên thi',
  SESSION_CANDIDATES_UPDATED: 'Cập nhật phân công thí sinh',
  SESSION_STARTED: 'Bắt đầu giám sát',
  SESSION_PAUSED: 'Tạm dừng giám sát',
  SESSION_RESUMED: 'Tiếp tục giám sát',
  SESSION_STOPPED: 'Kết thúc giám sát',
  MEDIA_VIDEO_UPLOADED: 'Tải video lên',
}

const entityLabels: Record<string, string> = {
  USER: 'Tài khoản',
  ROOM: 'Phòng thi',
  CANDIDATE: 'Thí sinh',
  EXAM_SESSION: 'Phiên thi',
  MEDIA_ASSET: 'Video',
}

function localizedRole(value: unknown): string {
  const role = String(value) as UserRole
  return roleLabels[role] ?? String(value)
}

function auditDetail(entry: AuditLogEntry): string {
  if (!entry.metadata) return '—'
  const username = typeof entry.metadata.username === 'string' ? entry.metadata.username : null
  if (entry.action === 'USER_ROLE_CHANGED') {
    return `${username ?? 'Tài khoản'}: ${localizedRole(entry.metadata.previous_role)} → ${localizedRole(entry.metadata.new_role)}`
  }
  if (username) return `Tài khoản: ${username}`
  if (entry.metadata.room_code) return `Phòng: ${String(entry.metadata.room_code)}`
  if (entry.metadata.candidate_code) return `Thí sinh: ${String(entry.metadata.candidate_code)}`
  if (entry.metadata.session_code) return `Phiên thi: ${String(entry.metadata.session_code)}`
  if (entry.metadata.original_filename) return `Tệp: ${String(entry.metadata.original_filename)}`
  if (entry.metadata.timestamp_ms != null) return `Mốc video: ${String(entry.metadata.timestamp_ms)} ms`
  return 'Đã ghi nhận thay đổi.'
}

export function AuditPage() {
  const [items, setItems] = useState<AuditLogEntry[]>([])
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (targetPage: number) => {
    setLoading(true); setError(null)
    try {
      const result = await getAuditLogs(targetPage, pageSize)
      setItems(result.items); setTotal(result.total)
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { void load(1) }, [load])
  if (loading && items.length === 0) return <LoadingState message="Đang tải nhật ký hệ thống…" />

  return <div className="page-stack">
    <PageHeader eyebrow="CÀI ĐẶT" title="Nhật ký hệ thống" description="Lịch sử thao tác được lưu nối tiếp và không thể chỉnh sửa." />
    {error && <ErrorState message={error} onRetry={() => void load(page)} />}
    <section className="card table-wrap">
      {loading && <p className="inline-loading">Đang tải…</p>}
      {items.length > 0 && <table><thead><tr><th>Thời gian</th><th>Người thực hiện</th><th>Hành động</th><th>Đối tượng</th><th>Chi tiết</th></tr></thead><tbody>{items.map((entry) => <tr key={entry.id}><td>{formatDateTime(entry.created_at)}</td><td>{entry.actor_full_name ?? entry.actor_username ?? 'Hệ thống'}</td><td>{actionLabels[entry.action] ?? 'Thao tác hệ thống'}</td><td>{entityLabels[entry.entity_type] ?? 'Dữ liệu hệ thống'}{entry.entity_id ? ` · ${entry.entity_id.slice(0, 8)}` : ''}</td><td className="audit-detail">{auditDetail(entry)}</td></tr>)}</tbody></table>}
      {!loading && items.length === 0 && <EmptyState title="Chưa có thao tác nào được ghi nhận." />}
      <Pagination page={page} pageSize={pageSize} total={total} onChange={(next) => { setPage(next); void load(next) }} />
    </section>
  </div>
}
