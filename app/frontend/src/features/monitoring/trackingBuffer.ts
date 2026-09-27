import type { TrackingFrame } from './types'

// Tracking results reach the browser after the video frame they describe: analysis time plus the network
// (tens of ms on a LAN, a few hundred through a remote GPU server). A result may therefore be late by up to
// LATE_TOLERANCE_MS, but never ahead of the player by more than AHEAD_TOLERANCE_MS (stale data after a seek).
export const AHEAD_TOLERANCE_MS = 250
export const LATE_TOLERANCE_MS = 1200

function withinTolerance(frameTimestampMs: number, videoTimestampMs: number, aheadMs: number, lateMs: number): boolean {
  const offset = frameTimestampMs - videoTimestampMs
  return offset <= aheadMs && offset >= -lateMs
}

export function isTrackingTimestampAligned(
  frameTimestampMs: number,
  videoTimestampMs: number,
  aheadMs = AHEAD_TOLERANCE_MS,
  lateMs = LATE_TOLERANCE_MS,
): boolean {
  return withinTolerance(frameTimestampMs, videoTimestampMs, aheadMs, lateMs)
}

export class TrackingBuffer {
  private frames: TrackingFrame[] = []
  private runtimeInstanceId: string | null = null
  private runtimeGeneration = -1
  private latestSequence = 0
  private retiredRuntimeInstances = new Set<string>()

  constructor(private readonly maxAgeMs = 3000, private readonly maxItems = 64) {}

  insert(frame: TrackingFrame): { accepted: boolean; reset: boolean } {
    const runtime = this.activateRuntime(frame.runtime_instance_id, frame.runtime_generation)
    if (!runtime.accepted || frame.tracking_seq <= this.latestSequence) {
      return { accepted: false, reset: runtime.reset }
    }
    this.latestSequence = frame.tracking_seq
    this.frames.push(frame)
    this.frames.sort((left, right) => left.timestamp_ms - right.timestamp_ms)
    const newest = this.frames.at(-1)?.timestamp_ms ?? frame.timestamp_ms
    this.frames = this.frames
      .filter((item) => item.timestamp_ms >= newest - this.maxAgeMs)
      .slice(-this.maxItems)
    return { accepted: true, reset: runtime.reset }
  }

  activateRuntime(
    runtimeInstanceId: string,
    runtimeGeneration: number,
  ): { accepted: boolean; reset: boolean } {
    if (this.runtimeInstanceId === runtimeInstanceId) {
      if (runtimeGeneration < this.runtimeGeneration) return { accepted: false, reset: false }
      if (runtimeGeneration === this.runtimeGeneration) return { accepted: true, reset: false }
      this.runtimeGeneration = runtimeGeneration
      this.latestSequence = 0
      this.frames = []
      return { accepted: true, reset: true }
    }
    if (this.retiredRuntimeInstances.has(runtimeInstanceId)) {
      return { accepted: false, reset: false }
    }
    if (this.runtimeInstanceId) this.retiredRuntimeInstances.add(this.runtimeInstanceId)
    this.runtimeInstanceId = runtimeInstanceId
    this.runtimeGeneration = runtimeGeneration
    this.latestSequence = 0
    this.frames = []
    return { accepted: true, reset: true }
  }

  nearest(timestampMs: number, aheadMs = AHEAD_TOLERANCE_MS, lateMs = LATE_TOLERANCE_MS): TrackingFrame | null {
    let nearest: TrackingFrame | null = null
    let distance = Number.POSITIVE_INFINITY
    for (const frame of this.frames) {
      if (!withinTolerance(frame.timestamp_ms, timestampMs, aheadMs, lateMs)) continue
      const currentDistance = Math.abs(frame.timestamp_ms - timestampMs)
      if (currentDistance < distance) {
        nearest = frame
        distance = currentDistance
      }
    }
    return nearest
  }

  clearFrames(): void {
    this.frames = []
  }

  reset(): void {
    this.frames = []
    this.runtimeInstanceId = null
    this.runtimeGeneration = -1
    this.latestSequence = 0
    this.retiredRuntimeInstances.clear()
  }

  get size(): number {
    return this.frames.length
  }

  get sequence(): number {
    return this.latestSequence
  }
}
