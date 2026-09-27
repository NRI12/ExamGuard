import { useCallback, useEffect, useState, type FormEvent } from 'react'

import { apiContentErrorMessage, apiErrorField, apiErrorMessage } from '../../shared/api/errors'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { FormField } from '../../shared/components/FormField'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { Pagination } from '../../shared/components/Pagination'
import { useToast } from '../../shared/components/ToastProvider'
import { useDebouncedValue } from '../../shared/hooks/useDebouncedValue'
import { hasErrors, normalizedOptional, validateCandidate, type FieldErrors } from '../../shared/validation'
import { permissions, usePermissions } from '../auth/permissions'
import { createCandidate, getCandidates, updateCandidate } from './api'
import type { Candidate, CandidateInput } from './types'

const blank: CandidateInput = { candidate_code: '', full_name: '', class_name: null, note: null }
const pageSize = 20

export function CandidatesPage() {
  const { can } = usePermissions()
  const toast = useToast()
  const editable = can(permissions.candidateManage)
  const [items, setItems] = useState<Candidate[]>([])
  const [selected, setSelected] = useState<Candidate | null>(null)
  const [form, setForm] = useState<CandidateInput>(blank)
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebouncedValue(query)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})

  const load = useCallback(async (search: string, targetPage: number) => {
    setLoading(true); setError(null)
    try {
      const result = await getCandidates({ query: search, page: targetPage, pageSize })
      setItems(result.items); setTotal(result.total)
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { setPage(1); void load(debouncedQuery, 1) }, [debouncedQuery, load])

  const select = (candidate: Candidate) => {
    setSelected(candidate); setFieldErrors({})
    setForm({ candidate_code: candidate.candidate_code, full_name: candidate.full_name, class_name: candidate.class_name, note: candidate.note })
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const normalized: CandidateInput = {
      candidate_code: form.candidate_code.trim(), full_name: form.full_name.trim(),
      class_name: normalizedOptional(form.class_name), note: normalizedOptional(form.note),
    }
    const validation = validateCandidate(normalized)
    setFieldErrors(validation)
    if (hasErrors(validation)) return
    setSaving(true); setError(null)
    try {
      const saved = selected ? await updateCandidate(selected.id, normalized) : await createCandidate(normalized)
      toast.success(selected ? 'Đã cập nhật thí sinh.' : 'Đã thêm thí sinh.')
      select(saved); await load(debouncedQuery, page)
    } catch (requestError) {
      const field = apiErrorField(requestError)
      if (field) setFieldErrors((current) => ({ ...current, [field]: apiErrorMessage(requestError) }))
      else setError(apiErrorMessage(requestError))
    } finally { setSaving(false) }
  }

  if (loading && items.length === 0 && !query) return <LoadingState message="Đang tải danh sách thí sinh…" />

  return <div className="page-stack">
    <PageHeader eyebrow="NGHIỆP VỤ" title="Thí sinh" actions={editable && <button className="primary-button" type="button" onClick={() => { setSelected(null); setForm(blank); setFieldErrors({}) }}>Thêm mới</button>} />
    {error && <ErrorState message={error} onRetry={() => void load(debouncedQuery, page)} />}
    <form className="search-row" role="search" onSubmit={(event) => { event.preventDefault(); setPage(1); void load(query, 1) }}>
      <label className="sr-only" htmlFor="candidate-search">Tìm kiếm thí sinh</label>
      <input id="candidate-search" placeholder="Tìm theo mã, họ tên hoặc lớp…" value={query} onChange={(event) => setQuery(event.target.value)} />
      <button className="secondary-button" type="submit">Tìm kiếm</button>
    </form>
    <div className="split-layout">
      <section className="card">
        <h2>Danh sách thí sinh</h2>
        {loading && <p className="inline-loading">Đang tải…</p>}
        <div className="list-stack">
          {items.map((candidate) => <button className={`list-item ${selected?.id === candidate.id ? 'selected' : ''}`} type="button" key={candidate.id} onClick={() => select(candidate)}>
            <span><strong>{candidate.candidate_code}</strong><small>{candidate.full_name}</small></span><span>{candidate.class_name ?? '—'}</span>
          </button>)}
          {!loading && items.length === 0 && <EmptyState title={query ? 'Không tìm thấy kết quả phù hợp.' : 'Chưa có thí sinh.'} description={!query && editable ? 'Thêm thí sinh mới để bắt đầu.' : undefined} />}
        </div>
        <Pagination page={page} pageSize={pageSize} total={total} onChange={(next) => { setPage(next); void load(debouncedQuery, next) }} />
      </section>
      <section className="card">
        <h2>{selected ? 'Thông tin thí sinh' : 'Thêm thí sinh'}</h2>
        {!selected && !editable && <EmptyState title="Chọn một thí sinh để xem thông tin." />}
        {editable && <form onSubmit={submit} noValidate>
          <FormField label="Mã thí sinh" htmlFor="candidate-code" required error={fieldErrors.candidate_code}>
            <input id="candidate-code" maxLength={100} value={form.candidate_code} onChange={(event) => setForm({ ...form, candidate_code: event.target.value })} />
          </FormField>
          <FormField label="Họ và tên" htmlFor="candidate-name" required error={fieldErrors.full_name}>
            <input id="candidate-name" maxLength={255} value={form.full_name} onChange={(event) => setForm({ ...form, full_name: event.target.value })} />
          </FormField>
          <FormField label="Lớp" htmlFor="candidate-class" error={fieldErrors.class_name}>
            <input id="candidate-class" maxLength={255} value={form.class_name ?? ''} onChange={(event) => setForm({ ...form, class_name: event.target.value || null })} />
          </FormField>
          <FormField label="Ghi chú" htmlFor="candidate-note" error={fieldErrors.note}>
            <textarea id="candidate-note" maxLength={5000} value={form.note ?? ''} onChange={(event) => setForm({ ...form, note: event.target.value || null })} />
          </FormField>
          {editable && <button className="primary-button" type="submit" disabled={saving}>{saving ? 'Đang lưu…' : 'Lưu'}</button>}
        </form>}
        {selected && !editable && <dl className="detail-list">
          <div><dt>Mã thí sinh</dt><dd>{selected.candidate_code}</dd></div>
          <div><dt>Họ và tên</dt><dd>{selected.full_name}</dd></div>
          <div><dt>Lớp</dt><dd>{selected.class_name ?? '—'}</dd></div>
          <div><dt>Ghi chú</dt><dd>{selected.note ?? '—'}</dd></div>
        </dl>}
      </section>
    </div>
  </div>
}
