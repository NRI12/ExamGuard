import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, '.', 'VITE_')
  const apiTarget = environment.VITE_DEV_PROXY_TARGET || 'http://localhost:8000'
  const websocketTarget = apiTarget.replace(/^http/, 'ws')
  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
        '/ws': {
          target: websocketTarget,
          ws: true,
        },
      },
    },
  }
})
