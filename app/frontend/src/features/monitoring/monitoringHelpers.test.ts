import { describe, expect, it } from 'vitest'

import { containedVideoRect, mapNormalizedBox } from './geometry'
import { clearCanvasBackingStore, startTrackingRenderLoop, trackingLabel } from './TrackingCanvas'
import { isTrackingTimestampAligned, TrackingBuffer } from './trackingBuffer'
import type { TrackingFrame } from './types'
import { monitoringSocketUrl, reconnectDelay, shouldReconnect } from './useMonitoringSocket'

function frame(
  timestamp_ms: number,
  tracking_seq = Math.floor(timestamp_ms / 100) + 1,
  runtime_instance_id = 'runtime-a',
  runtime_generation = 0,
): TrackingFrame {
  return {
    type: 'tracking', session_id: 'session', runtime_instance_id, runtime_generation,
    tracker_instance_id: 'tracker-a', tracking_seq, timestamp_ms, frame_id: timestamp_ms,
    source_width: 1920, source_height: 1080, tracks: [], seats: [],
  }
}

describe('tracking overlay helpers', () => {
  it('maps boxes through horizontal and vertical letterboxing', () => {
    const horizontal = containedVideoRect(1000, 700, 1920, 1080)
    expect(horizontal).toEqual({ x: 0, y: 68.75, width: 1000, height: 562.5 })
    expect(mapNormalizedBox([0.1, 0.2, 0.5, 0.8], horizontal)).toEqual([100, 181.25, 500, 518.75])
    const vertical = containedVideoRect(1200, 500, 1080, 1920)
    expect(vertical.x).toBeGreaterThan(400)
    expect(vertical.y).toBe(0)
  })

  it('keeps a bounded recent buffer, finds nearest, and clears on seek', () => {
    const buffer = new TrackingBuffer(1000, 3)
    for (const timestamp of [0, 500, 1000, 1500, 2000]) buffer.insert(frame(timestamp))
    expect(buffer.size).toBe(3)
    expect(buffer.nearest(1510)?.timestamp_ms).toBe(1500)
    expect(buffer.nearest(2600)?.timestamp_ms).toBe(2000) // late result from a remote server
    expect(buffer.nearest(3300)).toBeNull()
    expect(buffer.nearest(1600, 0)?.timestamp_ms).toBe(1500) // never a frame ahead of the player
    buffer.reset()
    expect(buffer.size).toBe(0)
    expect(isTrackingTimestampAligned(5100, 5000)).toBe(true)
    expect(isTrackingTimestampAligned(4600, 5000)).toBe(true)
    expect(isTrackingTimestampAligned(5400, 5000)).toBe(false)
    expect(isTrackingTimestampAligned(1000, 5000)).toBe(false)
  })

  it('rejects duplicate sequences and stale runtime generations', () => {
    const buffer = new TrackingBuffer()
    expect(buffer.insert(frame(100, 1)).accepted).toBe(true)
    expect(buffer.insert(frame(110, 1)).accepted).toBe(false)
    expect(buffer.insert(frame(5000, 1, 'runtime-a', 1))).toEqual({ accepted: true, reset: true })
    expect(buffer.insert(frame(120, 2, 'runtime-a', 0)).accepted).toBe(false)
    expect(buffer.insert(frame(200, 1, 'runtime-b', 0))).toEqual({ accepted: true, reset: true })
    expect(buffer.insert(frame(5100, 2, 'runtime-a', 1)).accepted).toBe(false)
    expect(buffer.nearest(200)?.runtime_instance_id).toBe('runtime-b')
  })

  it('clears the complete device-pixel backing store before drawing', () => {
    const calls: Array<[string, ...number[]]> = []
    const context = {
      setTransform: (...values: number[]) => calls.push(['transform', ...values]),
      clearRect: (...values: number[]) => calls.push(['clear', ...values]),
    }
    clearCanvasBackingStore(context, { width: 2000, height: 1200 })
    expect(calls).toEqual([
      ['transform', 1, 0, 0, 1, 0, 0],
      ['clear', 0, 0, 2000, 1200],
    ])
  })

  it('runs one video-frame callback chain and cancels it during cleanup', () => {
    const callbacks = new Map<number, () => void>()
    const cancelled: number[] = []
    let nextId = 0
    let draws = 0
    const video = {
      requestVideoFrameCallback(callback: () => void) {
        nextId += 1
        callbacks.set(nextId, callback)
        return nextId
      },
      cancelVideoFrameCallback(callbackId: number) { cancelled.push(callbackId) },
    }
    const cleanup = startTrackingRenderLoop(video, () => { draws += 1 })
    expect(callbacks.size).toBe(1)
    callbacks.get(1)?.()
    expect(draws).toBe(1)
    expect(callbacks.size).toBe(2)
    cleanup()
    callbacks.get(2)?.()
    expect(draws).toBe(1)
    expect(cancelled).toEqual([2])
  })

  it('builds secure websocket URLs and caps reconnect backoff', () => {
    const location = { protocol: 'https:', host: 'examguard.local' } as Location
    expect(monitoringSocketUrl('abc', location)).toBe('wss://examguard.local/ws/monitoring/abc')
    expect(reconnectDelay(0)).toBe(500)
    expect(reconnectDelay(10)).toBe(5000)
    expect(shouldReconnect(1006)).toBe(true)
    expect(shouldReconnect(4401)).toBe(false)
    expect(shouldReconnect(4409)).toBe(false)
  })

  it('shows only seat/location outside debug mode', () => {
    const track = {
      actor_id: 'A0007',
      actor_state: 'ACTIVE' as const,
      recovered: false,
      track_id: 17,
      bbox_norm: [0.1, 0.1, 0.3, 0.8] as [number, number, number, number],
      confidence: 0.9,
      identity: {
        state: 'ASSIGNED' as const,
        seat_id: 'seat-3',
        seat_code: 'B03',
        session_candidate_id: 'assignment-3',
        score: 0.82,
      },
    }
    const lookup = new Map([['assignment-3', 'SV103']])
    expect(trackingLabel(track, lookup)).toEqual(['B03'])
    expect(trackingLabel(track, lookup, true)).toEqual([
      'B03',
      'A0007 • T17 • SV103 • seat=0.82',
    ])
    expect(trackingLabel({
      ...track,
      identity: {
        state: 'TENTATIVE' as const,
        seat_id: 'seat-3',
        seat_code: 'B03',
        session_candidate_id: null,
        score: 0.5,
      },
    }, lookup)).toEqual(['B03'])
    expect(trackingLabel({
      ...track,
      identity: {
        state: 'UNASSIGNED' as const,
        seat_id: null,
        seat_code: null,
        session_candidate_id: null,
        score: null,
      },
    }, lookup)).toEqual(['Chưa xác định vị trí'])
  })
})
