import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'

import { apiErrorMessage } from '../../shared/api/errors'
import { ConfirmDialog } from '../../shared/components/ConfirmDialog'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { useToast } from '../../shared/components/ToastProvider'
import { useUnsavedChanges } from '../../shared/hooks/useUnsavedChanges'
import { valuesChanged } from '../../shared/validation'
import { getMediaFrame } from '../media/api'
import { saveSeats } from './api'
import type { Seat } from './types'

interface DragState {
  index: number
  mode: 'move' | 'resize'
  pointerX: number
  pointerY: number
  original: Seat
}

function copySeats(seats: Seat[]): Seat[] {
  return seats.map((seat) => ({ ...seat }))
}

export function SeatLayoutEditor({
  roomId,
  initialSeats,
  referenceMediaId,
  referenceTimestampMs,
  editable,
  onSaved,
}: {
  roomId: string
  initialSeats: Seat[]
  referenceMediaId: string | null
  referenceTimestampMs: number
  editable: boolean
  onSaved(seats: Seat[]): void
}) {
  const [seats, setSeats] = useState(() => copySeats(initialSeats))
  const [drag, setDrag] = useState<DragState | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [removeIndex, setRemoveIndex] = useState<number | null>(null)
  const [referenceUrl, setReferenceUrl] = useState<string | null>(null)
  const [referenceKind, setReferenceKind] = useState<'image' | 'video' | null>(null)
  const areaRef = useRef<HTMLDivElement>(null)
  const toast = useToast()
  const dirty = valuesChanged(seats, initialSeats)
  useUnsavedChanges(dirty)

  useEffect(() => setSeats(copySeats(initialSeats)), [initialSeats])

  useEffect(() => {
    let objectUrl: string | null = null
    let cancelled = false
    if (!referenceMediaId) {
      setReferenceUrl(null)
      setReferenceKind(null)
      return undefined
    }
    void getMediaFrame(referenceMediaId, referenceTimestampMs).then((blob) => {
      if (cancelled) return
      objectUrl = URL.createObjectURL(blob)
      setReferenceUrl(objectUrl)
      setReferenceKind('image')
    }).catch(() => {
      if (!cancelled) setReferenceUrl(null)
    })
    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [referenceMediaId, referenceTimestampMs])

  useEffect(() => () => {
    if (referenceUrl) URL.revokeObjectURL(referenceUrl)
  }, [referenceUrl])

  const chooseLocalReference = (file: File | undefined) => {
    if (!file) return
    if (referenceUrl) URL.revokeObjectURL(referenceUrl)
    setReferenceUrl(URL.createObjectURL(file))
    setReferenceKind('video')
  }

  useEffect(() => {
    if (!drag) return
    const move = (event: PointerEvent) => {
      const bounds = areaRef.current?.getBoundingClientRect()
      if (!bounds) return
      const dx = (event.clientX - drag.pointerX) / bounds.width
      const dy = (event.clientY - drag.pointerY) / bounds.height
      setSeats((current) => current.map((seat, index) => {
        if (index !== drag.index) return seat
        if (drag.mode === 'move') {
          return {
            ...seat,
            x: Math.max(0, Math.min(1 - seat.width, drag.original.x + dx)),
            y: Math.max(0, Math.min(1 - seat.height, drag.original.y + dy)),
          }
        }
        return {
          ...seat,
          width: Math.max(0.04, Math.min(1 - seat.x, drag.original.width + dx)),
          height: Math.max(0.04, Math.min(1 - seat.y, drag.original.height + dy)),
        }
      }))
    }
    const stop = () => setDrag(null)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', stop, { once: true })
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', stop)
    }
  }, [drag])

  const startDrag = (
    event: ReactPointerEvent,
    index: number,
    mode: DragState['mode'],
  ) => {
    if (!editable) return
    event.preventDefault()
    event.stopPropagation()
    setDrag({
      index,
      mode,
      pointerX: event.clientX,
      pointerY: event.clientY,
      original: { ...seats[index] },
    })
  }

  const addSeat = () => {
    const used = new Set(seats.map((seat) => seat.code))
    let number = seats.length + 1
    while (used.has(`A${String(number).padStart(2, '0')}`)) number += 1
    setSeats((current) => [...current, {
      code: `A${String(number).padStart(2, '0')}`,
      x: 0.05,
      y: 0.05,
      width: 0.14,
      height: 0.11,
      sort_order: current.length,
      is_active: true,
    }])
  }

  const submit = async () => {
    setError(null)
    const normalizedCodes = seats.map((seat) => seat.code.trim().toUpperCase())
    if (normalizedCodes.some((code) => !code)) {
      setError('Mã chỗ ngồi là bắt buộc.')
      return
    }
    if (new Set(normalizedCodes).size !== normalizedCodes.length) {
      setError('Mã chỗ ngồi không được trùng trong cùng phòng thi.')
      return
    }
    setSaving(true)
    try {
      const saved = await saveSeats(roomId, seats.map((seat, index) => ({
        ...seat,
        code: seat.code.trim().toUpperCase(),
        sort_order: index,
      })))
      setSeats(copySeats(saved))
      onSaved(saved)
      toast.success('Đã lưu bố trí chỗ ngồi.')
    } catch (requestError) {
      setError(apiErrorMessage(requestError))
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="card layout-editor">
      <div className="section-heading">
        <div>
          <h2>Bố trí chỗ ngồi</h2>
          <p>Tọa độ được lưu theo tỷ lệ chuẩn hóa, không phụ thuộc kích thước màn hình.</p>
        </div>
        {editable && <button className="secondary-button" type="button" onClick={addSeat} disabled={!referenceUrl}>Thêm chỗ ngồi</button>}
      </div>
      {error && <ErrorState message={error} />}
      <div className="calibration-toolbar">
        <span>{referenceUrl ? 'Đang hiệu chỉnh trên hình ảnh camera thực.' : 'Chọn video tham chiếu trước khi hiệu chỉnh vị trí ghế.'}</span>
        <label className="secondary-button calibration-file-button">
          Chọn video tham chiếu
          <input type="file" accept="video/mp4,video/*" onChange={(event) => chooseLocalReference(event.target.files?.[0])} />
        </label>
      </div>
      <div className="calibration-area" ref={areaRef} aria-label="Vùng hiệu chỉnh chỗ ngồi">
        {referenceUrl && referenceKind === 'image' && <img className="calibration-reference" src={referenceUrl} alt="Khung hình camera dùng để hiệu chỉnh ghế" />}
        {referenceUrl && referenceKind === 'video' && <video className="calibration-reference" src={referenceUrl} controls muted />}
        {!referenceUrl && <span className="calibration-placeholder">Chưa có khung hình camera tham chiếu</span>}
        <span className="calibration-label">Khung hình hiệu chỉnh</span>
        {seats.map((seat, index) => (
          <div
            className="seat-box"
            key={seat.id ?? `${seat.code}-${index}`}
            style={{
              left: `${seat.x * 100}%`,
              top: `${seat.y * 100}%`,
              width: `${seat.width * 100}%`,
              height: `${seat.height * 100}%`,
            }}
            onPointerDown={(event) => referenceUrl && startDrag(event, index, 'move')}
          >
            <strong>{seat.code}</strong>
            {editable && (
              <button
                aria-label={`Đổi kích thước chỗ ngồi ${seat.code}`}
                className="resize-handle"
                type="button"
                onPointerDown={(event) => referenceUrl && startDrag(event, index, 'resize')}
              />
            )}
          </div>
        ))}
      </div>
      {seats.length === 0 && <EmptyState title="Chưa có chỗ ngồi trong phòng thi." description="Thêm chỗ ngồi và đặt vị trí trên khung hình hiệu chỉnh." />}
      {editable && seats.map((seat, index) => (
        <div className="seat-row" key={seat.id ?? index}>
          <label htmlFor={`seat-${index}`}>Chỗ {index + 1}</label>
          <input
            id={`seat-${index}`}
            value={seat.code}
            onChange={(event) => setSeats((current) => current.map((item, itemIndex) => (
              itemIndex === index ? { ...item, code: event.target.value } : item
            )))}
          />
          <button className="danger-link" type="button" onClick={() => {
            if (seat.id) setRemoveIndex(index)
            else setSeats((current) => current.filter((_, itemIndex) => itemIndex !== index))
          }}>Bỏ</button>
        </div>
      ))}
      {editable && (
        <div className="button-row">
          <button className="primary-button" type="button" onClick={submit} disabled={saving}>
            {saving ? 'Đang lưu…' : 'Lưu sơ đồ'}
          </button>
          <button className="secondary-button" type="button" onClick={() => setSeats(copySeats(initialSeats))}>
            Hủy thay đổi
          </button>
          {dirty && <span className="unsaved-note">Có thay đổi chưa được lưu.</span>}
        </div>
      )}
      <ConfirmDialog
        open={removeIndex !== null}
        title="Bỏ chỗ ngồi khỏi bố trí?"
        description="Chỗ ngồi đã dùng trong lịch sử sẽ được vô hiệu hóa thay vì xóa dữ liệu liên quan. Thay đổi chỉ có hiệu lực sau khi lưu."
        confirmLabel="Bỏ chỗ ngồi"
        danger
        onCancel={() => setRemoveIndex(null)}
        onConfirm={() => {
          setSeats((current) => current.filter((_, index) => index !== removeIndex))
          setRemoveIndex(null)
        }}
      />
    </section>
  )
}
