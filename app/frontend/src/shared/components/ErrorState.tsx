export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-state" role="alert">
      <span>{message}</span>
      {onRetry && <button type="button" className="error-retry" onClick={onRetry}>Thử lại</button>}
    </div>
  )
}
