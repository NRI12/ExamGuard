import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { AuthContext, type AuthContextValue } from './AuthProvider'
import { permissions } from './permissions'
import { ProtectedRoute } from './ProtectedRoute'

const supervisorAuth: AuthContextValue = {
  user: { id: 'user-1', username: 'supervisor', full_name: 'Nguyễn Văn A', role: 'SUPERVISOR', is_active: true },
  loading: false,
  login: async () => { throw new Error('not used') },
  logout: () => undefined,
}

describe('ProtectedRoute', () => {
  it('renders ForbiddenPage for an authenticated user missing the route permission', () => {
    const html = renderToStaticMarkup(
      <AuthContext.Provider value={supervisorAuth}>
        <MemoryRouter initialEntries={['/settings/rooms']}>
          <Routes>
            <Route path="/settings/rooms" element={<ProtectedRoute permission={permissions.roomManage}><span>Quản lý phòng</span></ProtectedRoute>} />
          </Routes>
        </MemoryRouter>
      </AuthContext.Provider>,
    )
    expect(html).toContain('403')
    expect(html).not.toContain('Quản lý phòng')
    expect(html).not.toContain('Đăng nhập hệ thống')
  })
})
