import { useCallback, useEffect, useState, type FormEvent } from 'react'

import { apiContentErrorMessage, apiErrorField, apiErrorMessage } from '../../shared/api/errors'
import { ConfirmDialog } from '../../shared/components/ConfirmDialog'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { FormField } from '../../shared/components/FormField'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { Pagination } from '../../shared/components/Pagination'
import { useToast } from '../../shared/components/ToastProvider'
import { useDebouncedValue } from '../../shared/hooks/useDebouncedValue'
import { hasErrors, normalizedOptional, validateRoom, type FieldErrors } from '../../shared/validation'
import { permissions, usePermissions } from '../auth/permissions'
import { getSessions } from '../sessions/api'
import { createRoom, getRooms, getSeats, updateRoom } from './api'
import { SeatLayoutEditor } from './SeatLayoutEditor'
import type { Room, RoomInput, Seat } from './types'

const blankRoom: RoomInput = { code: '', name: '', description: null, is_active: true }
const pageSize = 20

export function RoomsPage() {
  const { can } = usePermissions()
  const toast = useToast()
  const editable = can(permissions.roomManage)
  const [rooms, setRooms] = useState<Room[]>([])
  const [selected, setSelected] = useState<Room | null>(null)
  const [seats, setSeats] = useState<Seat[]>([])
  const [referenceMediaId, setReferenceMediaId] = useState<string | null>(null)
  const [referenceTimestampMs, setReferenceTimestampMs] = useState(5000)
  const [form, setForm] = useState<RoomInput>(blankRoom)
  const [editing, setEditing] = useState(false)
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebouncedValue(query)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [confirmDeactivate, setConfirmDeactivate] = useState(false)

  const load = useCallback(async (search: string, targetPage: number) => {
    setLoading(true); setError(null)
    try {
      const result = await getRooms({ query: search, page: targetPage, pageSize })
      setRooms(result.items); setTotal(result.total)
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { setPage(1); void load(debouncedQuery, 1) }, [debouncedQuery, load])

  const open = async (room: Room) => {
    setSelected(room); setForm({ code: room.code, name: room.name, description: room.description, is_active: room.is_active })
    setEditing(true); setError(null); setFieldErrors({}); setReferenceMediaId(null)
    try {
      const [roomSeats, sessions] = await Promise.all([
        getSeats(room.id),
        getSessions({ roomId: room.id, pageSize: 20 }),
      ])
      setSeats(roomSeats)
      const referenceSession = sessions.items.find((item) => item.video)
      setReferenceMediaId(referenceSession?.video_asset_id ?? null)
      setReferenceTimestampMs(referenceSession?.video ? Math.min(5000, referenceSession.video.duration_ms - 1) : 5000)
    }
    catch (requestError) { setError(apiContentErrorMessage(requestError)) }
  }

  const persist = async () => {
    setSaving(true); setError(null)
    const payload: RoomInput = {
      code: form.code.trim(), name: form.name.trim(),
      description: normalizedOptional(form.description), is_active: form.is_active,
    }
    try {
      const saved = selected ? await updateRoom(selected.id, payload) : await createRoom(payload)
      toast.success(selected ? 'Đã cập nhật phòng thi.' : 'Đã thêm phòng thi.')
      await load(debouncedQuery, page); await open(saved)
    } catch (requestError) {
      const field = apiErrorField(requestError)
      if (field) setFieldErrors((current) => ({ ...current, [field]: apiErrorMessage(requestError) }))
      else setError(apiErrorMessage(requestError))
    } finally { setSaving(false); setConfirmDeactivate(false) }
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const validation = validateRoom(form)
    setFieldErrors(validation)
    if (hasErrors(validation)) return
    if (selected?.is_active && !form.is_active) setConfirmDeactivate(true)
    else void persist()
  }

  if (loading && rooms.length === 0 && !query) return <LoadingState message="Đang tải danh sách phòng thi…" />

  return <div className="page-stack">
    <PageHeader eyebrow="CÀI ĐẶT" title="Phòng thi" actions={editable && <button className="primary-button" type="button" onClick={() => { setSelected(null); setForm(blankRoom); setSeats([]); setEditing(true); setFieldErrors({}) }}>Thêm mới</button>} />
    {error && <ErrorState message={error} onRetry={() => void load(debouncedQuery, page)} />}
    <form className="search-row" role="search" onSubmit={(event) => { event.preventDefault(); setPage(1); void load(query, 1) }}>
      <label className="sr-only" htmlFor="room-search">Tìm kiếm phòng thi</label>
      <input id="room-search" placeholder="Tìm theo mã hoặc tên phòng thi…" value={query} onChange={(event) => setQuery(event.target.value)} />
      <button className="secondary-button" type="submit">Tìm kiếm</button>
    </form>
    <div className="split-layout">
      <section className="card">
        <h2>Danh sách phòng thi</h2>
        {loading && <p className="inline-loading">Đang tải…</p>}
        <div className="list-stack">
          {rooms.map((room) => <button className={`list-item ${selected?.id === room.id ? 'selected' : ''}`} type="button" key={room.id} onClick={() => void open(room)}>
            <span><strong>{room.code}</strong><small>{room.name}</small></span>
            <span className={`status-pill ${room.is_active ? 'success' : 'muted'}`}>{room.is_active ? 'Đang hoạt động' : 'Đã vô hiệu hóa'}</span>
          </button>)}
          {!loading && rooms.length === 0 && <EmptyState title={query ? 'Không tìm thấy kết quả phù hợp.' : 'Chưa có phòng thi.'} />}
        </div>
        <Pagination page={page} pageSize={pageSize} total={total} onChange={(next) => { setPage(next); void load(debouncedQuery, next) }} />
      </section>
      <section className="card">
        <h2>{selected ? 'Thông tin phòng thi' : 'Thêm phòng thi'}</h2>
        {!editing && <EmptyState title="Chọn một phòng thi để xem thông tin." />}
        {editing && <form onSubmit={submit} noValidate>
          <FormField label="Mã phòng thi" htmlFor="room-code" required error={fieldErrors.code}>
            <input id="room-code" maxLength={100} value={form.code} disabled={!editable} onChange={(event) => setForm({ ...form, code: event.target.value })} />
          </FormField>
          <FormField label="Tên phòng thi" htmlFor="room-name" required error={fieldErrors.name}>
            <input id="room-name" maxLength={255} value={form.name} disabled={!editable} onChange={(event) => setForm({ ...form, name: event.target.value })} />
          </FormField>
          <FormField label="Mô tả" htmlFor="room-description" error={fieldErrors.description}>
            <textarea id="room-description" maxLength={5000} value={form.description ?? ''} disabled={!editable} onChange={(event) => setForm({ ...form, description: event.target.value || null })} />
          </FormField>
          <label className="check-row"><input type="checkbox" checked={form.is_active} disabled={!editable} onChange={(event) => setForm({ ...form, is_active: event.target.checked })} /> Phòng thi đang hoạt động</label>
          {editable && <button className="primary-button" type="submit" disabled={saving}>{saving ? 'Đang lưu…' : 'Lưu'}</button>}
        </form>}
      </section>
    </div>
    {selected && <SeatLayoutEditor key={selected.id} roomId={selected.id} initialSeats={seats} referenceMediaId={referenceMediaId} referenceTimestampMs={referenceTimestampMs} editable={Boolean(editable)} onSaved={setSeats} />}
    <ConfirmDialog open={confirmDeactivate} title="Vô hiệu hóa phòng thi?" description="Phòng thi sẽ không thể được chọn cho phiên thi mới. Dữ liệu lịch sử vẫn được giữ nguyên." confirmLabel="Vô hiệu hóa" danger onCancel={() => setConfirmDeactivate(false)} onConfirm={() => void persist()} />
  </div>
}
