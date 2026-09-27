import type { ReactNode } from 'react'

import type { ExamSession } from './types'

export function SessionReadinessPanel({ session, actions }: {
  session: ExamSession
  actions?: ReactNode
}) {
  const seatCount = session.readiness.active_seats
  return <section className="card preflight-panel">
    <div className="section-heading"><div><p className="panel-label">SẴN SÀNG GIÁM SÁT</p><h2>Kiểm tra trước khi bắt đầu</h2></div></div>
    <ul className="preflight-list">
      <li className={session.readiness.room_active ? 'ready' : 'warning'}><span>{session.readiness.room_active ? '✓' : '!'}</span><div><strong>Phòng thi</strong><small>{session.room.code} — {session.room.name}</small></div></li>
      <li><span>i</span><div><strong>Bố trí chỗ ngồi (tùy chọn)</strong><small>{seatCount ? `${seatCount} chỗ ngồi · dùng cho ánh xạ nâng cao` : 'Không cần cấu hình để bắt đầu'}</small></div></li>
      <li><span>i</span><div><strong>Ánh xạ thí sinh (tùy chọn)</strong><small>{session.candidate_count ? `${session.candidate_count} thí sinh đã xếp chỗ` : 'Stable Actor ID vẫn hoạt động khi chưa ánh xạ'}</small></div></li>
      <li className={session.video ? 'ready' : 'warning'}><span>{session.video ? '✓' : '!'}</span><div><strong>Video nguồn</strong><small>{session.video?.original_filename ?? 'Chưa có video'}</small></div></li>
      <li><span>i</span><div><strong>Hệ thống AI</strong><small>Được kiểm tra thực tế khi bắt đầu giám sát</small></div></li>
    </ul>
    {actions && <div className="button-row">{actions}</div>}
  </section>
}
