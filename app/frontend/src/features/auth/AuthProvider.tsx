import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import { getCurrentUser, loginRequest, logoutRequest } from './api'
import { browserTokenStore } from './tokenStorage'
import type { AuthUser } from './types'

export interface AuthContextValue {
  user: AuthUser | null
  loading: boolean
  login(username: string, password: string): Promise<AuthUser>
  logout(): void
}

export const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    if (!browserTokenStore.get()) {
      setLoading(false)
      return () => {
        active = false
      }
    }

    getCurrentUser()
      .then((currentUser) => {
        if (active) setUser(currentUser)
      })
      .catch(() => browserTokenStore.clear())
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
    }
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      login: async (username, password) => {
        const result = await loginRequest(username, password)
        browserTokenStore.set(result.access_token)
        setUser(result.user)
        return result.user
      },
      logout: () => {
        void logoutRequest().catch(() => undefined)
        browserTokenStore.clear()
        setUser(null)
      },
    }),
    [loading, user],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside AuthProvider')
  return context
}
