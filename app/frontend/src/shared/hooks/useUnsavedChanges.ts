import { useEffect } from 'react'
import { unstable_usePrompt as usePrompt } from 'react-router-dom'

export function useUnsavedChanges(enabled: boolean): void {
  usePrompt({
    when: enabled,
    message: 'Bạn có thay đổi chưa được lưu. Bạn có muốn rời khỏi trang?',
  })
  useEffect(() => {
    if (!enabled) return
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = true
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [enabled])
}
