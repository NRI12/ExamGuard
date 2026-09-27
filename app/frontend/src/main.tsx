import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import { App } from './app/App'
import { AuthProvider } from './features/auth/AuthProvider'
import { ToastProvider } from './shared/components/ToastProvider'
import './shared/styles/index.css'

const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('ExamGuard root element was not found.')
}

const router = createBrowserRouter([
  { path: '*', element: <AuthProvider><App /></AuthProvider> },
])

createRoot(rootElement).render(
  <StrictMode>
    <ToastProvider><RouterProvider router={router} /></ToastProvider>
  </StrictMode>,
)
