export type FieldErrors<T extends string = string> = Partial<Record<T, string>>

function required(value: string, label: string, maxLength: number): string | undefined {
  const normalized = value.trim()
  if (!normalized) return `${label} là bắt buộc.`
  if (normalized.length > maxLength) return `${label} không được vượt quá ${maxLength} ký tự.`
  return undefined
}

export function validateRoom(values: { code: string; name: string; description: string | null }) {
  return {
    code: required(values.code, 'Mã phòng thi', 100),
    name: required(values.name, 'Tên phòng thi', 255),
    description: values.description && values.description.trim().length > 5000
      ? 'Mô tả không được vượt quá 5000 ký tự.' : undefined,
  }
}

export function validateCandidate(values: { candidate_code: string; full_name: string; class_name: string | null; note: string | null }) {
  return {
    candidate_code: required(values.candidate_code, 'Mã thí sinh', 100),
    full_name: required(values.full_name, 'Họ và tên', 255),
    class_name: values.class_name && values.class_name.trim().length > 255
      ? 'Tên lớp không được vượt quá 255 ký tự.' : undefined,
    note: values.note && values.note.trim().length > 5000
      ? 'Ghi chú không được vượt quá 5000 ký tự.' : undefined,
  }
}

export function validateSession(values: {
  session_code: string
  exam_name: string
  room_id: string
  scheduled_start: string | null
  scheduled_end: string | null
}) {
  const errors: FieldErrors = {
    session_code: required(values.session_code, 'Mã phiên thi', 100),
    exam_name: required(values.exam_name, 'Tên kỳ thi', 255),
    room_id: values.room_id ? undefined : 'Phòng thi là bắt buộc.',
  }
  if (values.scheduled_start && values.scheduled_end
    && new Date(values.scheduled_end) <= new Date(values.scheduled_start)) {
    errors.scheduled_end = 'Thời gian kết thúc phải sau thời gian bắt đầu.'
  }
  return errors
}

export function validateUserCreate(values: {
  username: string
  full_name: string | null
  password: string
  role: string
}) {
  const errors: FieldErrors = {
    username: required(values.username, 'Tên đăng nhập', 100),
    full_name: values.full_name && values.full_name.trim().length > 255
      ? 'Họ và tên không được vượt quá 255 ký tự.' : undefined,
    role: values.role ? undefined : 'Vai trò là bắt buộc.',
  }
  if (!errors.username && !/^[a-z0-9._-]+$/.test(values.username)) {
    errors.username = 'Tên đăng nhập chỉ gồm chữ thường, số, dấu chấm, gạch ngang hoặc gạch dưới.'
  }
  if (values.password.length < 12) {
    errors.password = 'Mật khẩu phải có ít nhất 12 ký tự.'
  } else if (values.password.length > 128) {
    errors.password = 'Mật khẩu không được vượt quá 128 ký tự.'
  } else if (!/[A-Za-z]/.test(values.password) || !/\d/.test(values.password)) {
    errors.password = 'Mật khẩu phải có ít nhất một chữ cái và một chữ số.'
  }
  return errors
}

export function hasErrors(errors: FieldErrors): boolean {
  return Object.values(errors).some(Boolean)
}

export function normalizedOptional(value: string | null): string | null {
  const normalized = value?.trim() ?? ''
  return normalized || null
}

export function valuesChanged(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) !== JSON.stringify(right)
}
