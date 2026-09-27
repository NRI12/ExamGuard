import { useEffect, useRef, type RefObject } from 'react'

import { containedVideoRect, mapNormalizedBox } from './geometry'
import { TrackingBuffer } from './trackingBuffer'
import type { CheatActorState, CheatClass, TrackingTrack } from './types'

export const cheatLabels: Record<CheatClass, string> = {
  normal: 'Bình thường',
  looking: 'Nhìn bài',
  interaction: 'Trao đổi',
  phone_cheatsheet: 'Điện thoại / tài liệu',
  abnormal: 'Bất thường',
}
const cheatColors: Record<CheatClass, string> = {
  normal: '#22c55e',
  looking: '#f59e0b',
  interaction: '#8b5cf6',
  phone_cheatsheet: '#ef4444',
  abnormal: '#ef4444',
}
const CHEAT_STATE_MAX_AGE_MS = 2600 // a decision covers the last 3.2 s and is renewed every second

/** The actor's decision in force at this frame, or null (not classified yet / too old). */
export function cheatStateAt(
  states: ReadonlyMap<string, CheatActorState> | undefined,
  track: TrackingTrack,
  timestampMs: number,
): CheatActorState | null {
  const state = states?.get(track.actor_id)
  if (!state) return null
  const age = timestampMs - state.timestamp_ms
  return age >= -1200 && age <= CHEAT_STATE_MAX_AGE_MS ? state : null
}

interface TrackingCanvasProps {
  videoRef: RefObject<HTMLVideoElement | null>
  buffer: TrackingBuffer
  revision: number
  showConfidence?: boolean
  candidateCodes: ReadonlyMap<string, string>
  debug?: boolean
  cheatStates?: ReadonlyMap<string, CheatActorState>
}

export function trackingLabel(
  track: TrackingTrack,
  candidateCodes: ReadonlyMap<string, string>,
  debug = false,
): string[] {
  const labels: string[] = []
  const candidateCode = track.identity.session_candidate_id
    ? candidateCodes.get(track.identity.session_candidate_id)
    : undefined
  labels.push(track.identity.seat_code ?? 'Chưa xác định vị trí')
  if (debug) {
    labels.push(
      `${track.actor_id} • T${track.track_id}${candidateCode ? ` • ${candidateCode}` : ''}${track.recovered ? ' • recovered' : ''}${track.identity.score == null ? '' : ` • seat=${track.identity.score.toFixed(2)}`}`,
    )
  }
  return labels
}

interface ClearableCanvasContext {
  clearRect(x: number, y: number, width: number, height: number): void
  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void
}

export function clearCanvasBackingStore(
  context: ClearableCanvasContext,
  canvas: Pick<HTMLCanvasElement, 'width' | 'height'>,
): void {
  context.setTransform(1, 0, 0, 1, 0, 0)
  context.clearRect(0, 0, canvas.width, canvas.height)
}

interface VideoFrameScheduler {
  requestVideoFrameCallback(callback: () => void): number
  cancelVideoFrameCallback?(callbackId: number): void
}

export function startTrackingRenderLoop(
  video: VideoFrameScheduler,
  draw: () => void,
): () => void {
  let callbackId = 0
  let stopped = false
  const schedule = () => {
    if (stopped) return
    callbackId = video.requestVideoFrameCallback(() => {
      if (stopped) return
      draw()
      schedule()
    })
  }
  schedule()
  return () => {
    stopped = true
    if (callbackId) video.cancelVideoFrameCallback?.(callbackId)
  }
}

export function TrackingCanvas({ videoRef, buffer, revision, showConfidence = false, candidateCodes, debug = false, cheatStates }: TrackingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    let animationId = 0
    let stopped = false

    const draw = () => {
      const context = canvas.getContext('2d')
      if (!context) return
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      const ratio = window.devicePixelRatio || 1
      const targetWidth = Math.round(width * ratio)
      const targetHeight = Math.round(height * ratio)
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth
        canvas.height = targetHeight
      }
      clearCanvasBackingStore(context, canvas)
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      const frame = buffer.nearest(video.currentTime * 1000)
      if (!frame) return
      const content = containedVideoRect(width, height, video.videoWidth, video.videoHeight)
      context.lineWidth = 2
      context.strokeStyle = '#22d3ee'
      context.font = '600 13px Inter, sans-serif'
      for (const track of frame.tracks) {
        const [x1, y1, x2, y2] = mapNormalizedBox(track.bbox_norm, content)
        const cheat = cheatStateAt(cheatStates, track, frame.timestamp_ms)
        const color = cheat ? cheatColors[cheat.alert ? cheat.label : 'normal'] : '#22d3ee'
        context.lineWidth = cheat?.alert ? 3 : 2
        context.strokeStyle = color
        if (cheat?.alert) { context.fillStyle = `${color}26`; context.fillRect(x1, y1, x2 - x1, y2 - y1) }
        context.strokeRect(x1, y1, x2 - x1, y2 - y1)
        const labels = trackingLabel(track, candidateCodes, debug)
        if (cheat) labels[0] += cheat.alert ? ` • ${cheatLabels[cheat.label]} ${Math.round(cheat.cheat_score * 100)}%` : ` • ${cheatLabels.normal}`
        if (showConfidence && labels.length) labels[labels.length - 1] += ` • conf=${track.confidence.toFixed(2)}`
        labels.forEach((label, index) => {
          const labelWidth = context.measureText(label).width + 12
          const labelY = Math.max(0, y1 - (labels.length - index) * 22)
          context.fillStyle = index === 0 ? (cheat ? color : '#0891b2') : '#334155'
          context.fillRect(x1, labelY, labelWidth, 22)
          context.fillStyle = index === 0 && cheat ? '#0b0f14' : '#ecfeff'
          context.fillText(label, x1 + 6, labelY + 15)
        })
        if (cheat && x2 - x1 > 40) { // cheat score bar under the box
          context.fillStyle = 'rgba(0,0,0,.55)'; context.fillRect(x1, y2 + 3, x2 - x1, 5)
          context.fillStyle = color; context.fillRect(x1, y2 + 3, (x2 - x1) * cheat.cheat_score, 5)
        }
      }
    }

    const cancelVideoLoop = 'requestVideoFrameCallback' in video
      ? startTrackingRenderLoop(video, draw)
      : null
    const scheduleAnimation = () => {
      if (stopped || cancelVideoLoop) return
      animationId = window.requestAnimationFrame(() => { draw(); scheduleAnimation() })
    }
    const observer = new ResizeObserver(draw)
    if (canvas.parentElement) observer.observe(canvas.parentElement)
    document.addEventListener('fullscreenchange', draw)
    draw()
    scheduleAnimation()
    return () => {
      stopped = true
      observer.disconnect()
      document.removeEventListener('fullscreenchange', draw)
      if (animationId) window.cancelAnimationFrame(animationId)
      cancelVideoLoop?.()
      const context = canvas.getContext('2d')
      if (context) clearCanvasBackingStore(context, canvas)
    }
  }, [buffer, candidateCodes, cheatStates, debug, showConfidence, videoRef])

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (canvas && context) clearCanvasBackingStore(context, canvas)
  }, [revision])

  return <canvas ref={canvasRef} className="tracking-canvas" aria-hidden="true" />
}
