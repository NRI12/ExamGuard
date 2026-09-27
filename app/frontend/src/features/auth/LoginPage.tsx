import axios from 'axios'
import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'

import { landingPathForRole } from '../../app/access'
import { ErrorState } from '../../shared/components/ErrorState'
import { FormField } from '../../shared/components/FormField'
import { Icon } from '../../shared/components/Icon'
import { useAuth } from './AuthProvider'

function loginErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error) && error.response?.status === 401) {
    return 'Tên đăng nhập hoặc mật khẩu không đúng.'
  }
  return 'Không thể kết nối tới ExamGuard. Vui lòng thử lại.'
}

export function LoginPage() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [fieldErrors, setFieldErrors] = useState<{ username?: string; password?: string }>({})
  const [submitting, setSubmitting] = useState(false)

  if (user) return <Navigate to={landingPathForRole(user.role)} replace />

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    const normalizedUsername = username.trim()
    const validation = {
      username: normalizedUsername ? undefined : 'Tên đăng nhập là bắt buộc.',
      password: password ? undefined : 'Mật khẩu là bắt buộc.',
    }
    setFieldErrors(validation)
    if (validation.username || validation.password) return
    setSubmitting(true)
    try {
      const loggedInUser = await login(normalizedUsername, password)
      navigate(landingPathForRole(loggedInUser.role), { replace: true })
    } catch (requestError) {
      setError(loginErrorMessage(requestError))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="login-screen">
      <section className="login-card" aria-labelledby="login-title">
        <div className="brand-badge"><Icon name="shield" /></div>
        <p className="eyebrow">EXAMGUARD</p>
        <h1 id="login-title">Đăng nhập hệ thống</h1>
        <p className="secondary-text">Nền tảng giám sát và rà soát phòng thi</p>
        {error && <ErrorState message={error} />}
        <form onSubmit={submit} noValidate>
          <FormField label="Tên đăng nhập" htmlFor="username" required error={fieldErrors.username}>
            <input id="username" autoComplete="username" maxLength={100} value={username} onChange={(event) => setUsername(event.target.value)} />
          </FormField>
          <FormField label="Mật khẩu" htmlFor="password" required error={fieldErrors.password}>
            <input id="password" type="password" autoComplete="current-password" maxLength={1024} value={password} onChange={(event) => setPassword(event.target.value)} />
          </FormField>
          <button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? 'Đang đăng nhập…' : 'Đăng nhập'}
          </button>
        </form>
      </section>
    </main>
  )
}
