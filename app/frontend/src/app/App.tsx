import type { ReactNode } from 'react'
import { Route, Routes } from 'react-router-dom'

import { LoginPage } from '../features/auth/LoginPage'
import { ProtectedRoute } from '../features/auth/ProtectedRoute'
import { AuditPage } from '../features/audit/AuditPage'
import { CandidatesPage } from '../features/candidates/CandidatesPage'
import { CamerasPage } from '../features/cameras/CamerasPage'
import { DashboardPage } from '../features/dashboard/DashboardPage'
import { EventsPage } from '../features/events/EventsPage'
import { MonitoringPage } from '../features/monitoring/MonitoringPage'
import { RoomsPage } from '../features/rooms/RoomsPage'
import { SessionDetailPage } from '../features/sessions/SessionDetailPage'
import { SessionsPage } from '../features/sessions/SessionsPage'
import { UsersPage } from '../features/users/UsersPage'
import { AppShell } from '../shared/layout/AppShell'
import { appRouteAccess } from './access'
import { RoleLanding } from './RoleLanding'

const routeElements = {
  overview: <DashboardPage />,
  monitoring: <MonitoringPage />,
  events: <EventsPage />,
  sessions: <SessionsPage />,
  'session-detail': <SessionDetailPage />,
  candidates: <CandidatesPage />,
  rooms: <RoomsPage />,
  cameras: <CamerasPage />,
  users: <UsersPage />,
  audit: <AuditPage />,
} satisfies Record<(typeof appRouteAccess)[number]['id'], ReactNode>

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<AppShell />}>
          <Route path="/" element={<RoleLanding />} />
          {appRouteAccess.map((route) => <Route
            key={route.id}
            path={route.path}
            element={<ProtectedRoute permission={route.permission}>{routeElements[route.id]}</ProtectedRoute>}
          />)}
          <Route path="*" element={<RoleLanding />} />
        </Route>
      </Route>
    </Routes>
  )
}
