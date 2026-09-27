# Triển khai ExamGuard trên máy chủ có GPU (RTX 3060)

Gói này chứa toàn bộ hệ thống đang chạy: mã nguồn (backend FastAPI + frontend React đã build), mô hình AI
(YOLO11n, X3D-L v1), cơ sở dữ liệu (phòng, thí sinh, phiên thi, sự kiện, lịch sử xác minh) và video S07/S08.
Không cần Docker.

## Yêu cầu

| | Windows 10/11, Windows Server 2019+ | Linux (Ubuntu 22.04/24.04, Debian 12) |
|---|---|---|
| GPU | NVIDIA RTX 3060 (hoặc mạnh hơn), driver NVIDIA **≥ 560** | như Windows |
| RAM / ổ đĩa | 16 GB / 15 GB trống | như Windows |
| Mạng khi cài | tải khoảng 3,5 GB (PyTorch CUDA, Python, PostgreSQL, ffmpeg) | như Windows |
| Quyền | người dùng thường (quản trị chỉ cần khi mở tường lửa) | sudo |

Kiểm tra driver: chạy `nvidia-smi`, cột "Driver Version" phải từ 560 trở lên.

## Windows

1. Giải nén gói vào một thư mục **không có dấu cách và dấu tiếng Việt**, ví dụ `C:\ExamGuard`.
2. Nháy đúp `deploy\windows\install.cmd` (khoảng 10 đến 20 phút lần đầu). Script tự tải vào
   `deploy\windows\tools`: uv + Python 3.11, PostgreSQL 16 bản portable (cổng 5433, không đụng tới
   PostgreSQL có sẵn), ffmpeg; tạo môi trường Python với PyTorch CUDA 12.6; tạo `.env` với mật khẩu ngẫu nhiên;
   khôi phục cơ sở dữ liệu từ `data\examguard.dump`.
3. Nháy đúp `deploy\windows\start.cmd`. Lần đầu mất khoảng 1 đến 3 phút để nạp mô hình lên GPU.
   Mở `http://localhost:8000`; máy khác trong mạng LAN dùng địa chỉ IP mà script in ra.
4. Dừng: `deploy\windows\stop.cmd`.

Máy khác trong LAN không truy cập được: chạy PowerShell **với quyền quản trị**
`powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -Firewall` (mở cổng 8000).

## Linux

```bash
tar -xzf ExamGuard.tar.gz && cd ExamGuard       # hoặc giải nén file .zip
sudo bash deploy/linux/install.sh --service      # cài đặt + dịch vụ systemd "examguard" tự chạy khi khởi động máy
# hoặc: sudo bash deploy/linux/install.sh  rồi  bash deploy/linux/start.sh   (dừng: bash deploy/linux/stop.sh)
```
Mở `http://<ip-máy-chủ>:8000`. Nhật ký: `journalctl -u examguard -f` (dịch vụ) hoặc `logs/backend.log`.

## Đăng nhập

Cơ sở dữ liệu khôi phục giữ nguyên tài khoản cũ: **admin / DlnHYYS5TGmk8OAa1**. Hãy đổi mật khẩu hoặc tạo
tài khoản mới trong *Người dùng* sau khi đăng nhập. (`ADMIN_PASSWORD` trong `.env` chỉ dùng khi cơ sở dữ liệu
chưa có tài khoản quản trị nào.)

## Dữ liệu có sẵn

- Phòng `DOAN1-S07`, `DOAN1-S08`: mỗi phòng 6 chỗ ngồi đã hiệu chỉnh theo camera, 1 camera (video 1080p).
- Phiên `DOAN1-S08`: **Sẵn sàng**, bấm *Giám sát* để chạy AI.
- Phiên `DOAN1-S07`: đã kết thúc, giữ 7 sự kiện AI mẫu (1 sự kiện đã xác nhận) trong trang *Sự kiện*.
- Tạo phiên mới: *Giám sát → Bắt đầu giám sát*, chọn phòng; camera của phòng được chọn tự động.

## HTTPS

Mặc định chạy HTTP trong mạng LAN (`COOKIE_SECURE=false` trong `.env`). Để truy cập qua Internet có HTTPS
mà không cần tên miền: `deploy\windows\tunnel.cmd` hoặc `bash deploy/linux/tunnel.sh` in ra một địa chỉ
`https://...trycloudflare.com` (đổi mỗi lần chạy). Khi dùng HTTPS, đặt `COOKIE_SECURE=true` trong `.env`
rồi khởi động lại.

## Cấu hình (`.env`)

| Biến | Ý nghĩa |
|---|---|
| `PORT` | cổng web (mặc định 8000) |
| `APP_PROFILE` | `rtx3060`: YOLO11n 18 FPS + X3D-L; `gtx1650` cho card yếu hơn (không có X3D) |
| `COOKIE_SECURE` | `false` cho HTTP trong LAN, `true` khi có HTTPS |
| `UPLOAD_ROOT`, `MODEL_ROOT` | thư mục video tải lên và mô hình |
| `DATABASE_URL`, mật khẩu | sinh tự động khi cài, không cần sửa |

Ngưỡng cảnh báo AI (mặc định 0,85 / 0,70, làm mượt 3 cửa sổ, tối thiểu 2 cửa sổ) nằm ở mục
`cheating_classifier.rule` của `backend/configs/runtime/rtx3060.yaml`.

## Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `CUDA không khả dụng` khi bắt đầu giám sát | cập nhật driver NVIDIA ≥ 560, kiểm tra `nvidia-smi` |
| Video không phát, lỗi 401 | đang dùng HTTP mà `COOKIE_SECURE=true`: đặt `false` hoặc dùng HTTPS |
| Không có khung người trên video | xem `logs/backend.log` (Windows: `logs\backend.err.log`) |
| Log báo "eager mode" (thường gặp trên Windows) | không có Triton nên bỏ qua torch.compile: X3D mất khoảng 305 ms thay vì 245 ms mỗi giây video, vẫn theo kịp thời gian thực |
| Cổng 8000 đã bị dùng | đổi `PORT` trong `.env` |

Chi tiết kỹ thuật (luồng AI, API sự kiện, số liệu đo trên RTX 3060): `docs/X3D_INTEGRATION.md`.
