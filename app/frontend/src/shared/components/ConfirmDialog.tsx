import { useEffect, useRef } from 'react'

export function ConfirmDialog({ open, title, description, confirmLabel, danger = false, onConfirm, onCancel }: {
  open: boolean
  title: string
  description: string
  confirmLabel: string
  danger?: boolean
  onConfirm(): void
  onCancel(): void
}) {
  const cancelRef = useRef<HTMLButtonElement>(null)
  useEffect(() => { if (open) cancelRef.current?.focus() }, [open])
  if (!open) return null
  return <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
    <section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" onMouseDown={(event) => event.stopPropagation()}>
      <h2 id="confirm-title">{title}</h2>
      <p>{description}</p>
      <div className="button-row">
        <button ref={cancelRef} type="button" className="secondary-button" onClick={onCancel}>Hủy</button>
        <button type="button" className={danger ? 'danger-button' : 'primary-button'} onClick={onConfirm}>{confirmLabel}</button>
      </div>
    </section>
  </div>
}
