import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

import { navigationForRole } from '../../app/access'
import { useAuth } from '../../features/auth/AuthProvider'
import { Icon } from '../components/Icon'
import { roleLabels } from '../i18n/vi'

export function AppShell() {
  const { user, logout } = useAuth()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const displayName = user?.full_name ?? user?.username ?? ''
  const initials = displayName.trim().split(/\s+/).slice(-2).map((part) => part[0]?.toUpperCase()).join('') || 'EG'
  const navigation = user ? navigationForRole(user.role) : []

  const closeSidebar = () => setSidebarOpen(false)

  return (
    <div className={`app-shell ${sidebarOpen ? 'sidebar-open' : ''}`}>
      <aside className="sidebar" aria-label="Thanh điều hướng">
        <div className="brand-row">
          <span className="brand-badge small"><Icon name="shield" /></span>
          <div>
            <strong>GIÁM SÁT KỲ THI</strong>
          </div>
        </div>
        <div className="sidebar-user">
          <span className="user-avatar">{initials}</span>
          <span><strong>{displayName}</strong><small>{user ? roleLabels[user.role] : ''}</small></span>
        </div>
        <p className="nav-header">ĐIỀU HƯỚNG</p>
        <nav aria-label="Điều hướng chính">
          {navigation.map((item, index) => <span className="nav-entry" key={item.to}>
            {item.section && (index === 0 || navigation[index - 1]?.section !== item.section) && <span className="nav-section">{item.section}</span>}
            <NavLink to={item.to} end={item.end} onClick={closeSidebar} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}><Icon name={item.icon} /><span>{item.label}</span><Icon className="nav-chevron" name="chevron" /></NavLink>
          </span>)}
        </nav>
      </aside>
      <button className="sidebar-backdrop" type="button" aria-label="Đóng thanh điều hướng" onClick={closeSidebar} />
      <div className="workspace">
        <header className="topbar">
          <div className="topbar-start">
            <button className="menu-toggle" type="button" aria-label="Mở thanh điều hướng" onClick={() => setSidebarOpen((value) => !value)}><Icon name="bars" /></button>
            <div><strong>Hệ thống giám sát phòng thi</strong><small>Điều hành và theo dõi tập trung</small></div>
          </div>
          <details className="account-menu">
            <summary aria-label="Mở menu tài khoản"><span className="topbar-avatar">{initials}</span><span className="account-copy"><strong>{displayName}</strong><small>{user ? roleLabels[user.role] : ''}</small></span><Icon className="account-chevron" name="chevron" /></summary>
            <div className="account-popover">
              <p>Thông tin tài khoản</p>
              <strong>{displayName}</strong>
              <span>{user?.username}</span>
              <span>{user ? roleLabels[user.role] : ''}</span>
              <button type="button" onClick={logout}><Icon name="logout" />Đăng xuất</button>
            </div>
          </details>
        </header>
        <main className="content-wrapper"><Outlet /></main>
        <footer className="main-footer"><span><strong>GIÁM SÁT KỲ THI</strong> · Giám sát phòng thi</span><span>Phiên bản 1.0</span></footer>
      </div>
    </div>
  )
}
