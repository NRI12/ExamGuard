export function formatDateTime(value: string | null | undefined): string {
  if (!value) return 'Chưa đặt'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'Không hợp lệ'
  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(date)
}

export function formatDurationMs(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return '00:00'
  const totalSeconds = Math.floor(milliseconds / 1000)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  const base = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  return hours ? `${String(hours).padStart(2, '0')}:${base}` : base
}

export function formatResolution(width: number, height: number): string {
  return `${width}×${height}`
}

export function formatFps(value: number, digits = 1): string {
  return `${value.toFixed(digits)} FPS`
}
