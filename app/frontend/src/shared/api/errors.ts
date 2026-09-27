import axios from 'axios'

import { apiErrorLabels } from '../i18n/vi'

interface ApiErrorBody {
  error?: { code?: string; message?: string; field?: string }
}

export function apiErrorMessage(error: unknown): string {
  if (!axios.isAxiosError<ApiErrorBody>(error)) {
    return 'Đã xảy ra lỗi. Vui lòng thử lại.'
  }
  const code = error.response?.data?.error?.code
  if (code && apiErrorLabels[code]) return apiErrorLabels[code]
  if (!error.response) return 'Không thể kết nối tới máy chủ.'
  return 'Không thể thực hiện yêu cầu. Vui lòng thử lại.'
}

export function apiContentErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error) && error.response?.status === 403) {
    return 'Bạn không có quyền truy cập nội dung này.'
  }
  return apiErrorMessage(error)
}

export function apiErrorField(error: unknown): string | null {
  if (!axios.isAxiosError<ApiErrorBody>(error)) return null
  return error.response?.data?.error?.field ?? null
}
