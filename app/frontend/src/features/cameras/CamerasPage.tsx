import { useCallback, useEffect, useState, type FormEvent } from 'react'
import './cameras.css'

import { apiContentErrorMessage, apiErrorMessage } from '../../shared/api/errors'
import { ConfirmDialog } from '../../shared/components/ConfirmDialog'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { FormField } from '../../shared/components/FormField'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { Pagination } from '../../shared/components/Pagination'
import { useToast } from '../../shared/components/ToastProvider'
import { formatDurationMs, formatFps, formatResolution } from '../../shared/formatters'
import { useDebouncedValue } from '../../shared/hooks/useDebouncedValue'
import { VideoMonitor } from '../media/VideoMonitor'
import { VideoUpload } from '../media/VideoUpload'
import type { MediaAsset } from '../media/types'
import { getRooms } from '../rooms/api'
import type { Room } from '../rooms/types'
import { createCamera, getCameras, updateCamera } from './api'
import type { Camera, CameraInput, CameraStatus } from './types'

const blank: CameraInput = {
  name: '', room_id: '', source_media_asset_id: '', description: null, is_active: true,
}
const statusLabels: Record<CameraStatus, string> = {
  READY: 'Sẵn sàng', IN_USE: 'Đang sử dụng', ERROR: 'Có lỗi', DISABLED: 'Vô hiệu hóa',
}
const pageSize = 20

export function CamerasPage() {
  const toast = useToast()
  const [items, setItems] = useState<Camera[]>([])
  const [rooms, setRooms] = useState<Room[]>([])
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebouncedValue(query)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [editing, setEditing] = useState<Camera | null | undefined>(undefined)
  const [form, setForm] = useState<CameraInput>(blank)
  const [media, setMedia] = useState<MediaAsset | null>(null)
  const [preview, setPreview] = useState<Camera | null>(null)
  const [confirmToggle, setConfirmToggle] = useState<Camera | null>(null)

  const load = useCallback(async (search: string, targetPage: number) => {
    setLoading(true); setError(null)
    try {
      const [cameraPage, roomPage] = await Promise.all([
        getCameras({ query: search, page: targetPage, pageSize }),
        getRooms({ pageSize: 100 }),
      ])
      setItems(cameraPage.items); setTotal(cameraPage.total)
      setRooms(roomPage.items.filter((room) => room.is_active))
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { setPage(1); void load(debouncedQuery, 1) }, [debouncedQuery, load])

  const openCreate = () => {
    setEditing(null); setForm(blank); setMedia(null); setError(null)
  }
  const openEdit = (camera: Camera) => {
    setEditing(camera)
    setForm({
      name: camera.name,
      room_id: camera.room_id,
      source_media_asset_id: camera.source_media_asset_id,
      description: camera.description,
      is_active: camera.is_active,
    })
    setMedia(camera.source_media); setError(null)
  }
  const closeForm = () => { if (!saving) setEditing(undefined) }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const name = form.name.trim()
    if (name.length < 3 || name.length > 255 || !form.room_id || !form.source_media_asset_id) {
      setError('Vui lòng nhập đủ tên camera, phòng và video nguồn hợp lệ.')
      return
    }
    setSaving(true); setError(null)
    try {
      const payload = { ...form, name, description: form.description?.trim() || null }
      if (editing) await updateCamera(editing.id, payload)
      else await createCamera(payload)
      toast.success(editing ? 'Đã cập nhật camera.' : 'Đã thêm camera.')
      setEditing(undefined); await load(debouncedQuery, page)
    } catch (requestError) { setError(apiErrorMessage(requestError)) }
    finally { setSaving(false) }
  }

  const toggle = async () => {
    if (!confirmToggle) return
    setSaving(true); setError(null)
    try {
      await updateCamera(confirmToggle.id, { is_active: !confirmToggle.is_active })
      toast.success(confirmToggle.is_active ? 'Đã vô hiệu hóa camera.' : 'Đã kích hoạt camera.')
      setConfirmToggle(null); await load(debouncedQuery, page)
    } catch (requestError) { setError(apiErrorMessage(requestError)); setConfirmToggle(null) }
    finally { setSaving(false) }
  }

  if (loading && items.length === 0 && !query) return <LoadingState message="Đang tải danh sách camera…" />

  return <div className="page-stack cameras-page">
    <PageHeader eyebrow="QUẢN TRỊ" title="Camera" actions={<button className="primary-button" type="button" onClick={openCreate}>Thêm camera</button>} />
    {error && <ErrorState message={error} onRetry={() => void load(debouncedQuery, page)} />}
    <form className="search-row" role="search" onSubmit={(event) => { event.preventDefault(); setPage(1); void load(query, 1) }}>
      <label className="sr-only" htmlFor="camera-search">Tìm camera</label>
      <input id="camera-search" placeholder="Tìm camera hoặc phòng…" value={query} onChange={(event) => setQuery(event.target.value)} />
      <button className="secondary-button" type="submit">Tìm kiếm</button>
    </form>
    <section className="card table-wrap">
      {items.length > 0 ? <table><thead><tr><th>Tên camera</th><th>Phòng</th><th>Trạng thái</th><th>Thao tác</th></tr></thead><tbody>
        {items.map((camera) => <tr key={camera.id}><td><strong>{camera.name}</strong></td><td>{camera.room.code}</td><td><span className={`camera-status ${camera.status.toLowerCase()}`}><i />{statusLabels[camera.status]}</span></td><td><div className="table-actions"><button className="secondary-button" type="button" disabled={!camera.source_media} onClick={() => setPreview(camera)}>Xem trước</button><button className="secondary-button" type="button" onClick={() => openEdit(camera)}>Sửa</button><button className="danger-button subtle" type="button" disabled={camera.status === 'IN_USE'} onClick={() => setConfirmToggle(camera)}>{camera.is_active ? 'Vô hiệu hóa' : 'Kích hoạt'}</button></div></td></tr>)}
      </tbody></table> : !loading && <EmptyState title="Chưa có camera." description="Thêm camera và chọn một video làm nguồn mô phỏng." />}
      <Pagination page={page} pageSize={pageSize} total={total} onChange={(next) => { setPage(next); void load(debouncedQuery, next) }} />
    </section>

    {editing !== undefined && <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) closeForm() }}><section className="camera-dialog" role="dialog" aria-modal="true" aria-labelledby="camera-dialog-title"><header><div><p className="eyebrow">CAMERA MÔ PHỎNG</p><h2 id="camera-dialog-title">{editing ? 'Sửa camera' : 'Thêm camera'}</h2></div><button type="button" aria-label="Đóng" onClick={closeForm}>×</button></header><form onSubmit={submit} noValidate>
      <FormField label="Tên camera" htmlFor="camera-name" required><input id="camera-name" minLength={3} maxLength={255} value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></FormField>
      <FormField label="Phòng" htmlFor="camera-room" required><select id="camera-room" value={form.room_id} onChange={(event) => setForm({ ...form, room_id: event.target.value })}><option value="">Chọn phòng thi</option>{rooms.map((room) => <option key={room.id} value={room.id}>{room.code} — {room.name}</option>)}</select></FormField>
      <FormField label="Nguồn video" htmlFor="camera-video" required helper="Video được kiểm tra codec, độ phân giải, FPS và thời lượng ở backend."><VideoUpload disabled={saving} onUploaded={(uploaded) => { setMedia(uploaded); setForm({ ...form, source_media_asset_id: uploaded.id }) }} /></FormField>
      {media && <div className="source-success"><strong>✓ Video hợp lệ — {media.original_filename}</strong><span>{formatResolution(media.width, media.height)} · {formatFps(media.fps, 2)} · {formatDurationMs(media.duration_ms)}</span></div>}
      <FormField label="Mô tả" htmlFor="camera-description"><textarea id="camera-description" maxLength={5000} value={form.description ?? ''} onChange={(event) => setForm({ ...form, description: event.target.value })} /></FormField>
      <label className="camera-toggle"><input type="checkbox" checked={form.is_active} onChange={(event) => setForm({ ...form, is_active: event.target.checked })} /> Hoạt động</label>
      <footer><button className="secondary-button" type="button" onClick={closeForm}>Hủy</button><button className="primary-button" type="submit" disabled={saving}>{saving ? 'Đang lưu…' : 'Lưu camera'}</button></footer>
    </form></section></div>}

    {preview?.source_media && <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setPreview(null) }}><section className="camera-preview-dialog" role="dialog" aria-modal="true"><header><div><p className="eyebrow">XEM TRƯỚC</p><h2>{preview.name}</h2></div><button type="button" aria-label="Đóng" onClick={() => setPreview(null)}>×</button></header><VideoMonitor mediaUrl={preview.source_media.media_url} title={preview.name} loop /><div className="preview-meta"><span className="camera-status ready"><i />Sẵn sàng</span><span>{formatResolution(preview.source_media.width, preview.source_media.height)}</span><span>{formatFps(preview.source_media.fps, 2)}</span></div></section></div>}
    <ConfirmDialog open={Boolean(confirmToggle)} title={confirmToggle?.is_active ? 'Vô hiệu hóa camera?' : 'Kích hoạt camera?'} description={confirmToggle?.is_active ? 'Camera sẽ không còn xuất hiện trong lựa chọn bắt đầu giám sát.' : 'Camera sẽ có thể được chọn cho phiên giám sát mới.'} confirmLabel={confirmToggle?.is_active ? 'Vô hiệu hóa' : 'Kích hoạt'} danger={Boolean(confirmToggle?.is_active)} onCancel={() => setConfirmToggle(null)} onConfirm={() => void toggle()} />
  </div>
}
