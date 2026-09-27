import { useNavigate } from 'react-router-dom'

export function ForbiddenPage() {
  const navigate = useNavigate()
  return <div className="page-stack">
    <section className="card forbidden-panel" aria-labelledby="forbidden-title">
      <strong className="forbidden-code">403</strong>
      <h1 id="forbidden-title">Bạn không có quyền truy cập chức năng này.</h1>
      <p>Tài khoản hiện tại không được phép sử dụng chức năng này.</p>
      <button className="secondary-button" type="button" onClick={() => navigate(-1)}>Quay lại</button>
    </section>
  </div>
}
