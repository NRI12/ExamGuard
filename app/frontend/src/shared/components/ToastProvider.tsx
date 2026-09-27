import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

interface ToastItem { id: number; message: string }
interface ToastContextValue { success(message: string): void }

const ToastContext = createContext<ToastContextValue | null>(null)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const success = useCallback((message: string) => {
    const id = Date.now()
    setItems((current) => [...current, { id, message }])
    window.setTimeout(() => setItems((current) => current.filter((item) => item.id !== id)), 3500)
  }, [])
  const value = useMemo(() => ({ success }), [success])
  return <ToastContext.Provider value={value}>
    {children}
    <div className="toast-region" aria-live="polite" aria-atomic="true">
      {items.map((item) => <div className="toast" key={item.id}>{item.message}</div>)}
    </div>
  </ToastContext.Provider>
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext)
  if (!context) throw new Error('useToast must be used inside ToastProvider')
  return context
}
