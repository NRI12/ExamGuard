import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'

export function EmptyFeaturePage({ title }: { title: string }) {
  return <div className="page-stack">
    <PageHeader eyebrow="CHƯA KHẢ DỤNG" title={title} />
    <section className="card empty-panel">
      <EmptyState title="Chức năng đang được hoàn thiện." description="Hiện chưa có dữ liệu để hiển thị." />
    </section>
  </div>
}
