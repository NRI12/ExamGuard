import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { ForbiddenPage } from './ForbiddenPage'

describe('ForbiddenPage', () => {
  it('shows a Vietnamese 403 state without redirecting to login', () => {
    const html = renderToStaticMarkup(<MemoryRouter><ForbiddenPage /></MemoryRouter>)
    expect(html).toContain('403')
    expect(html).toContain('Bạn không có quyền truy cập chức năng này.')
    expect(html).toContain('Tài khoản hiện tại không được phép sử dụng chức năng này.')
    expect(html).toContain('Quay lại')
  })
})
