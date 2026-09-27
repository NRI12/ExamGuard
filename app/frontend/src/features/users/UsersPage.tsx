import { useCallback, useEffect, useState, type FormEvent } from 'react'

import { apiContentErrorMessage, apiErrorField, apiErrorMessage } from '../../shared/api/errors'
import { ConfirmDialog } from '../../shared/components/ConfirmDialog'
import { EmptyState } from '../../shared/components/EmptyState'
import { ErrorState } from '../../shared/components/ErrorState'
import { FormField } from '../../shared/components/FormField'
import { LoadingState } from '../../shared/components/LoadingState'
import { PageHeader } from '../../shared/components/PageHeader'
import { Pagination } from '../../shared/components/Pagination'
import { useToast } from '../../shared/components/ToastProvider'
import { roleLabels } from '../../shared/i18n/vi'
import { useDebouncedValue } from '../../shared/hooks/useDebouncedValue'
import { hasErrors, normalizedOptional, validateUserCreate, type FieldErrors } from '../../shared/validation'
import { useAuth } from '../auth/AuthProvider'
import type { UserRole } from '../auth/types'
import { createUser, getUsers, updateUser } from './api'
import type { ManagedUser, UserCreateInput, UserUpdateInput } from './types'

const roles: UserRole[] = ['ADMIN', 'SUPERVISOR', 'REVIEWER']
const pageSize = 20
const blankCreate: UserCreateInput = {
  username: '', full_name: null, role: 'SUPERVISOR', password: '', is_active: true,
}

export function UsersPage() {
  const { user: currentUser } = useAuth()
  const toast = useToast()
  const [items, setItems] = useState<ManagedUser[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebouncedValue(query)
  const [roleFilter, setRoleFilter] = useState<UserRole | ''>('')
  const [activeFilter, setActiveFilter] = useState<'' | 'true' | 'false'>('')
  const [mode, setMode] = useState<'create' | 'edit' | null>(null)
  const [selected, setSelected] = useState<ManagedUser | null>(null)
  const [createForm, setCreateForm] = useState<UserCreateInput>(blankCreate)
  const [editForm, setEditForm] = useState<{ full_name: string | null; role: UserRole }>({ full_name: null, role: 'SUPERVISOR' })
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmRole, setConfirmRole] = useState(false)
  const [deactivateTarget, setDeactivateTarget] = useState<ManagedUser | null>(null)

  const load = useCallback(async (search: string, targetPage: number, role: UserRole | '', active: '' | 'true' | 'false') => {
    setLoading(true); setError(null)
    try {
      const result = await getUsers({ query: search, page: targetPage, pageSize, role, active })
      setItems(result.items); setTotal(result.total)
    } catch (requestError) { setError(apiContentErrorMessage(requestError)) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    setPage(1)
    void load(debouncedQuery, 1, roleFilter, activeFilter)
  }, [activeFilter, debouncedQuery, load, roleFilter])

  const openCreate = () => {
    setMode('create'); setSelected(null); setCreateForm(blankCreate); setFieldErrors({}); setError(null)
  }

  const openEdit = (account: ManagedUser) => {
    setMode('edit'); setSelected(account)
    setEditForm({ full_name: account.full_name, role: account.role })
    setFieldErrors({}); setError(null)
  }

  const saveCreatedUser = async (event: FormEvent) => {
    event.preventDefault()
    const normalized = { ...createForm, username: createForm.username.trim().toLowerCase(), full_name: normalizedOptional(createForm.full_name) }
    const validation = validateUserCreate(normalized)
    setFieldErrors(validation)
    if (hasErrors(validation)) return
    setSaving(true); setError(null)
    try {
      await createUser(normalized)
      setMode(null); setCreateForm(blankCreate); toast.success('Đã tạo tài khoản.')
      await load(debouncedQuery, page, roleFilter, activeFilter)
    } catch (requestError) {
      const field = apiErrorField(requestError)
      if (field) setFieldErrors((current) => ({ ...current, [field]: apiErrorMessage(requestError) }))
      else setError(apiErrorMessage(requestError))
    } finally { setSaving(false) }
  }

  const persistEdit = async () => {
    if (!selected) return
    const fullName = normalizedOptional(editForm.full_name)
    if (fullName && fullName.length > 255) {
      setFieldErrors({ full_name: 'Họ và tên không được vượt quá 255 ký tự.' })
      return
    }
    setSaving(true); setError(null)
    const payload: UserUpdateInput = { full_name: fullName }
    if (selected.id !== currentUser?.id) payload.role = editForm.role
    try {
      await updateUser(selected.id, payload)
      setMode(null); setSelected(null); setConfirmRole(false); toast.success('Đã cập nhật tài khoản.')
      await load(debouncedQuery, page, roleFilter, activeFilter)
    } catch (requestError) { setError(apiErrorMessage(requestError)) }
    finally { setSaving(false) }
  }

  const submitEdit = (event: FormEvent) => {
    event.preventDefault()
    if (!selected) return
    if (selected.id !== currentUser?.id && selected.role !== editForm.role) {
      setConfirmRole(true)
      return
    }
    void persistEdit()
  }

  const setActive = async (account: ManagedUser, active: boolean) => {
    setSaving(true); setError(null)
    try {
      await updateUser(account.id, { is_active: active })
      setDeactivateTarget(null)
      toast.success(active ? 'Đã kích hoạt lại tài khoản.' : 'Đã vô hiệu hóa tài khoản.')
      await load(debouncedQuery, page, roleFilter, activeFilter)
    } catch (requestError) { setError(apiErrorMessage(requestError)) }
    finally { setSaving(false) }
  }

  if (loading && items.length === 0 && !query && !roleFilter && !activeFilter) return <LoadingState message="Đang tải danh sách người dùng…" />

  return <div className="page-stack">
    <PageHeader eyebrow="CÀI ĐẶT" title="Người dùng" description="Quản lý tài khoản và vai trò truy cập hệ thống." actions={<button className="primary-button" type="button" onClick={openCreate}>Thêm người dùng</button>} />
    {error && <ErrorState message={error} onRetry={() => void load(debouncedQuery, page, roleFilter, activeFilter)} />}
    {mode === 'create' && <section className="card user-form-panel"><div className="section-heading"><h2>Thêm người dùng</h2></div><form className="form-grid" onSubmit={saveCreatedUser} noValidate>
      <FormField label="Tên đăng nhập" htmlFor="user-username" required error={fieldErrors.username}><input id="user-username" autoComplete="off" maxLength={100} value={createForm.username} onChange={(event) => setCreateForm({ ...createForm, username: event.target.value })} /></FormField>
      <FormField label="Họ và tên" htmlFor="user-full-name" error={fieldErrors.full_name}><input id="user-full-name" maxLength={255} value={createForm.full_name ?? ''} onChange={(event) => setCreateForm({ ...createForm, full_name: event.target.value || null })} /></FormField>
      <FormField label="Vai trò" htmlFor="user-role" required><select id="user-role" value={createForm.role} onChange={(event) => setCreateForm({ ...createForm, role: event.target.value as UserRole })}>{roles.map((role) => <option key={role} value={role}>{roleLabels[role]}</option>)}</select></FormField>
      <FormField label="Mật khẩu ban đầu" htmlFor="user-password" required error={fieldErrors.password} helper="Tối thiểu 12 ký tự, gồm ít nhất một chữ cái và một chữ số."><input id="user-password" type="password" autoComplete="new-password" maxLength={128} value={createForm.password} onChange={(event) => setCreateForm({ ...createForm, password: event.target.value })} /></FormField>
      <label className="check-row"><input type="checkbox" checked={createForm.is_active} onChange={(event) => setCreateForm({ ...createForm, is_active: event.target.checked })} /> Tài khoản đang hoạt động</label>
      <div className="button-row"><button className="primary-button" type="submit" disabled={saving}>{saving ? 'Đang tạo…' : 'Tạo tài khoản'}</button><button className="secondary-button" type="button" onClick={() => setMode(null)}>Hủy</button></div>
    </form></section>}
    {mode === 'edit' && selected && <section className="card user-form-panel"><div className="section-heading"><div><h2>Chỉnh sửa tài khoản</h2><p>{selected.username}</p></div></div><form className="form-grid" onSubmit={submitEdit} noValidate>
      <FormField label="Họ và tên" htmlFor="edit-user-full-name" error={fieldErrors.full_name}><input id="edit-user-full-name" maxLength={255} value={editForm.full_name ?? ''} onChange={(event) => setEditForm({ ...editForm, full_name: event.target.value || null })} /></FormField>
      {selected.id === currentUser?.id ? <dl className="compact-detail"><div><dt>Vai trò</dt><dd>{roleLabels[selected.role]}</dd></div></dl> : <FormField label="Vai trò" htmlFor="edit-user-role" required><select id="edit-user-role" value={editForm.role} onChange={(event) => setEditForm({ ...editForm, role: event.target.value as UserRole })}>{roles.map((role) => <option key={role} value={role}>{roleLabels[role]}</option>)}</select></FormField>}
      <div className="button-row"><button className="primary-button" type="submit" disabled={saving}>{saving ? 'Đang lưu…' : 'Lưu thay đổi'}</button><button className="secondary-button" type="button" onClick={() => setMode(null)}>Hủy</button></div>
    </form></section>}
    <form className="filter-row user-filter-row" role="search" onSubmit={(event) => { event.preventDefault(); setPage(1); void load(query, 1, roleFilter, activeFilter) }}>
      <label className="sr-only" htmlFor="user-search">Tìm người dùng</label><input id="user-search" placeholder="Tìm theo tên đăng nhập hoặc họ tên…" value={query} onChange={(event) => setQuery(event.target.value)} />
      <label className="sr-only" htmlFor="user-role-filter">Lọc vai trò</label><select id="user-role-filter" value={roleFilter} onChange={(event) => setRoleFilter(event.target.value as UserRole | '')}><option value="">Tất cả vai trò</option>{roles.map((role) => <option key={role} value={role}>{roleLabels[role]}</option>)}</select>
      <label className="sr-only" htmlFor="user-status-filter">Lọc trạng thái</label><select id="user-status-filter" value={activeFilter} onChange={(event) => setActiveFilter(event.target.value as '' | 'true' | 'false')}><option value="">Tất cả trạng thái</option><option value="true">Đang hoạt động</option><option value="false">Đã vô hiệu hóa</option></select>
      <button className="secondary-button" type="submit">Tìm kiếm</button>
    </form>
    <section className="card table-wrap">
      {loading && <p className="inline-loading">Đang tải…</p>}
      {items.length > 0 && <table><thead><tr><th>Tên đăng nhập</th><th>Họ tên</th><th>Vai trò</th><th>Trạng thái</th><th>Thao tác</th></tr></thead><tbody>{items.map((account) => <tr key={account.id}><td><strong>{account.username}</strong>{account.id === currentUser?.id && <small className="current-user-label">Tài khoản hiện tại</small>}</td><td>{account.full_name ?? '—'}</td><td>{roleLabels[account.role]}</td><td><span className={`status-pill ${account.is_active ? 'success' : 'muted'}`}>{account.is_active ? 'Đang hoạt động' : 'Đã vô hiệu hóa'}</span></td><td><div className="table-actions"><button className="secondary-button" type="button" onClick={() => openEdit(account)}>Chỉnh sửa</button>{account.id !== currentUser?.id && (account.is_active ? <button className="danger-button subtle" type="button" onClick={() => setDeactivateTarget(account)}>Vô hiệu hóa</button> : <button className="secondary-button" type="button" disabled={saving} onClick={() => void setActive(account, true)}>Kích hoạt lại</button>)}</div></td></tr>)}</tbody></table>}
      {!loading && items.length === 0 && <EmptyState title={query || roleFilter || activeFilter ? 'Không tìm thấy tài khoản phù hợp.' : 'Chưa có người dùng nào ngoài tài khoản hiện tại.'} />}
      <Pagination page={page} pageSize={pageSize} total={total} onChange={(next) => { setPage(next); void load(debouncedQuery, next, roleFilter, activeFilter) }} />
    </section>
    <ConfirmDialog open={confirmRole && Boolean(selected)} title="Thay đổi vai trò?" description={`${selected?.full_name ?? selected?.username ?? ''}\n${selected ? roleLabels[selected.role] : ''} → ${roleLabels[editForm.role]}\n\nQuyền truy cập của tài khoản sẽ thay đổi ngay sau khi lưu.`} confirmLabel="Thay đổi" onCancel={() => setConfirmRole(false)} onConfirm={() => void persistEdit()} />
    <ConfirmDialog open={Boolean(deactivateTarget)} title="Vô hiệu hóa tài khoản?" description={`${deactivateTarget?.full_name ?? deactivateTarget?.username ?? ''}\n\nTài khoản sẽ không thể đăng nhập.`} confirmLabel="Vô hiệu hóa" danger onCancel={() => setDeactivateTarget(null)} onConfirm={() => { if (deactivateTarget) void setActive(deactivateTarget, false) }} />
  </div>
}
