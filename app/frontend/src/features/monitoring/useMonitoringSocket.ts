import { useEffect, useRef, useState } from 'react'

import type { MonitoringMessage } from './types'

export function monitoringSocketUrl(sessionId: string, location: Location = window.location): string {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${location.host}/ws/monitoring/${sessionId}`
}

export function reconnectDelay(attempt: number): number {
  return Math.min(500 * 2 ** Math.max(0, attempt), 5000)
}

export function shouldReconnect(code: number): boolean {
  return ![4401, 4403, 4404, 4409].includes(code)
}

interface SocketOptions {
  sessionId: string
  enabled: boolean
  onMessage(message: MonitoringMessage): void
}

export function useMonitoringSocket({ sessionId, enabled, onMessage }: SocketOptions) {
  const callbackRef = useRef(onMessage)
  const [connected, setConnected] = useState(false)
  callbackRef.current = onMessage

  useEffect(() => {
    if (!enabled || !sessionId) {
      setConnected(false)
      return
    }
    let socket: WebSocket | null = null
    let retryTimer = 0
    let attempt = 0
    let disposed = false
    const connect = () => {
      if (disposed) return
      const currentSocket = new WebSocket(monitoringSocketUrl(sessionId))
      socket = currentSocket
      currentSocket.onopen = () => {
        if (disposed) { currentSocket.close(); return }
        attempt = 0; setConnected(true)
      }
      currentSocket.onmessage = (event) => {
        if (disposed) return
        try {
          callbackRef.current(JSON.parse(event.data as string) as MonitoringMessage)
        } catch {
          // A later valid latest-state message supersedes malformed transport data.
        }
      }
      currentSocket.onerror = () => currentSocket.close()
      currentSocket.onclose = (event) => {
        if (disposed) return
        setConnected(false)
        if (shouldReconnect(event.code)) {
          retryTimer = window.setTimeout(connect, reconnectDelay(attempt))
          attempt += 1
        }
      }
    }
    connect()
    return () => {
      disposed = true
      window.clearTimeout(retryTimer)
      if (socket) {
        socket.onopen = null
        socket.onmessage = null
        socket.onerror = null
        socket.onclose = null
        socket.close()
      }
    }
  }, [enabled, sessionId])
  return connected
}
