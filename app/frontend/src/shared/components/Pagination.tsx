export function Pagination({ page, pageSize, total, onChange }: {
  page: number
  pageSize: number
  total: number
  onChange(page: number): void
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize))
  if (total <= pageSize) return null
  return <nav className="pagination" aria-label="Phân trang">
    <button type="button" className="secondary-button" disabled={page <= 1} onClick={() => onChange(page - 1)}>Trước</button>
    <span>Trang {page}/{pages}</span>
    <button type="button" className="secondary-button" disabled={page >= pages} onClick={() => onChange(page + 1)}>Sau</button>
  </nav>
}
