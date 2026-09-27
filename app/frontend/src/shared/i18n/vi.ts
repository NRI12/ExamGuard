import type { UserRole } from '../../features/auth/types'
import type { RuntimeState } from '../../features/monitoring/types'
import type { SessionStatus } from '../../features/sessions/types'

export const sessionStatusLabels: Record<SessionStatus, string> = {
  DRAFT: 'Chưa bắt đầu',
  READY: 'Sẵn sàng',
  RUNNING: 'Đang giám sát',
  PAUSED: 'Tạm dừng',
  COMPLETED: 'Đã kết thúc',
  CANCELLED: 'Đã hủy',
  ERROR: 'Có lỗi',
}

export const runtimeStateLabels: Record<RuntimeState, string> = {
  INACTIVE: 'Chưa chạy',
  INITIALIZING: 'Đang khởi tạo AI…',
  RUNNING: 'Đang giám sát',
  PAUSED: 'Tạm dừng',
  COMPLETED: 'Đã kết thúc',
  ERROR: 'Có lỗi',
}

export const roleLabels: Record<UserRole, string> = {
  SUPERVISOR: 'Giám thị',
  REVIEWER: 'Người xác minh',
  ADMIN: 'Quản trị viên',
}

export const apiErrorLabels: Record<string, string> = {
  AUTHENTICATION_REQUIRED: 'Phiên đăng nhập không hợp lệ. Vui lòng đăng nhập lại.',
  USER_INACTIVE: 'Tài khoản đã bị vô hiệu hóa.',
  ROOM_CODE_EXISTS: 'Mã phòng thi đã tồn tại.',
  ROOM_NOT_FOUND: 'Không tìm thấy phòng thi.',
  CANDIDATE_CODE_EXISTS: 'Mã thí sinh đã tồn tại.',
  CANDIDATE_NOT_FOUND: 'Không tìm thấy thí sinh.',
  SESSION_CODE_EXISTS: 'Mã phiên thi đã tồn tại.',
  SESSION_NOT_FOUND: 'Không tìm thấy phiên thi.',
  SESSION_HAS_ASSIGNMENTS: 'Hãy bỏ phân công thí sinh trước khi đổi phòng thi.',
  INVALID_SCHEDULE: 'Thời gian kết thúc phải sau thời gian bắt đầu.',
  SEAT_CODE_DUPLICATE: 'Mã chỗ ngồi bị trùng trong phòng thi.',
  SEAT_ID_DUPLICATE: 'Một chỗ ngồi không thể xuất hiện nhiều lần.',
  SEAT_OUTSIDE_FRAME: 'Vị trí chỗ ngồi phải nằm hoàn toàn trong khung hình.',
  SEAT_NOT_IN_ROOM: 'Chỗ ngồi không thuộc phòng thi này.',
  SEAT_NOT_IN_SESSION_ROOM: 'Chỗ ngồi không thuộc phòng của phiên thi.',
  CANDIDATE_ALREADY_ASSIGNED: 'Một thí sinh không thể được xếp vào nhiều chỗ ngồi.',
  SEAT_ALREADY_ASSIGNED: 'Một chỗ ngồi không thể được xếp cho nhiều thí sinh.',
  ASSIGNMENT_CONFLICT: 'Không thể lưu phân công thí sinh. Vui lòng kiểm tra lại.',
  ROOM_INACTIVE: 'Phòng thi đã bị vô hiệu hóa.',
  SEAT_INACTIVE: 'Chỗ ngồi đã bị vô hiệu hóa.',
  SESSION_NOT_READY: 'Phiên thi chưa đủ điều kiện để chuyển sang sẵn sàng.',
  INVALID_SESSION_STATE: 'Không thể thực hiện thao tác ở trạng thái hiện tại.',
  FORBIDDEN: 'Bạn không có quyền thực hiện thao tác này.',
  USERNAME_EXISTS: 'Tên đăng nhập đã tồn tại.',
  USER_NOT_FOUND: 'Không tìm thấy tài khoản.',
  USER_SELF_ROLE_CHANGE: 'Bạn không thể thay đổi vai trò của chính mình.',
  USER_SELF_DEACTIVATION: 'Bạn không thể vô hiệu hóa tài khoản đang đăng nhập.',
  LAST_ACTIVE_ADMIN: 'Hệ thống phải còn ít nhất một quản trị viên đang hoạt động.',
  INVALID_VIDEO_EXTENSION: 'Vui lòng sử dụng tệp MP4 hợp lệ.',
  INVALID_VIDEO: 'Không thể đọc video này. Vui lòng sử dụng tệp MP4 hợp lệ.',
  INVALID_VIDEO_CONTAINER: 'Tệp tải lên không phải MP4 hợp lệ.',
  UNSUPPORTED_VIDEO_CODEC: 'Video MP4 phải sử dụng codec H.264.',
  VIDEO_EMPTY: 'Tệp video đang trống.',
  VIDEO_TOO_LARGE: 'Video vượt quá dung lượng tải lên cho phép.',
  MEDIA_SAVE_FAILED: 'Không thể lưu video. Vui lòng thử lại.',
  CAMERA_NOT_FOUND: 'Không tìm thấy camera.',
  CAMERA_REQUIRED: 'Vui lòng chọn camera.',
  CAMERA_DISABLED: 'Camera đã bị vô hiệu hóa.',
  CAMERA_NOT_IN_ROOM: 'Camera không thuộc phòng thi đã chọn.',
  CAMERA_IN_USE: 'Camera đang được sử dụng và chưa thể thay đổi.',
  CAMERA_SAVE_FAILED: 'Không thể lưu camera. Vui lòng thử lại.',
  CAMERA_SOURCE_MISSING: 'Không thể đọc video nguồn của camera.',
  INVALID_SESSION_SOURCE: 'Nguồn giám sát của phiên không hợp lệ.',
  MEDIA_FILE_MISSING: 'Tệp video không còn khả dụng trên máy chủ.',
  VALIDATION_ERROR: 'Thông tin chưa hợp lệ. Vui lòng kiểm tra lại.',
}
