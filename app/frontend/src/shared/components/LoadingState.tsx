export function LoadingState({ message }: { message: string }) {
  return (
    <main className="center-state" role="status">
      <span className="spinner" aria-hidden="true" />
      <p>{message}</p>
    </main>
  )
}
