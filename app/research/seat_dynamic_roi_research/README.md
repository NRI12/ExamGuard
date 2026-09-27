# Seat-Anchored Dynamic ROI Dataset Builder

Pipeline nghiên cứu để rebuild dataset từ **full-frame source video** theo đúng deployment policy:

`Full video + source events + Seat Map -> YOLO person detection -> SeatID association -> stabilized Dynamic Local ROI + Interaction ROI -> cropped MP4 clips + deployment_roi_manifest.jsonl`

## Nguyên tắc khoa học

- Event annotation chỉ quyết định **label / target participant / thời gian**, không quyết định crop sát hành vi.
- ROI train được sinh bằng cùng policy có thể chạy ở deployment.
- Giữ `source_event_id`, `session_id`, `participant_id`, `global_subject_id`, `split`, `fold` để kiểm soát leakage.
- `normal` dùng cùng ROI policy với các class bất thường.
- Các normal khó như `pick_up_dropped_object`, `bend_to_adjust_shoe`, ... nên được annotate bằng `subtype` để phân tích hard negative.
- Mỗi sample luôn tạo Local ROI và Interaction ROI với **tất cả neighbor được Seat Map khai báo**; không dùng ground-truth class để chọn neighbor/view.

## Input 1: source_events.csv

Các cột bắt buộc:

```csv
source_event_id,session_id,video_path,participant_id,global_subject_id,behavior,start_sec,end_sec,split,fold
```

Các cột tùy chọn được giữ lại trong manifest:

```text
subtype
quality
related_participant_id
note
```

Canonical class:

```text
normal
suspicious_looking
communicating
exchange_object
using_phone/cheat_sheet
```

`video_path` có thể là absolute path hoặc relative theo `--project-root`.

## Input 2: seat_map.csv

```csv
session_id,camera_id,seat_id,participant_id,global_subject_id,x1,y1,x2,y2,neighbors
```

- `x1,y1,x2,y2` normalized trong `[0,1]`.
- ROI là **Seat Activity Anchor ROI**, không chỉ rectangle của ghế: nên chứa đầu, thân trên, tay, bàn, bài thi và vùng dưới bàn hợp lý.
- `neighbors` dùng `;`, ví dụ `seat_02;seat_05`.
- Mỗi participant trong mỗi session phải ánh xạ đúng một seat.
- Nếu camera/layout thay đổi giữa session, tạo seat map riêng theo session.

## Cài dependency nghiên cứu

Project hiện pin Torch/OpenCV/NumPy/PyYAML. Không cài lại Torch thủ công.

Thêm Ultralytics vào research environment với exact pin:

```bash
pip install ultralytics==8.4.147
```

Trong project dùng `uv`, nên thêm exact pin vào research dependency/extra rồi lock lại thay vì dùng floating dependency.

Code mặc định dùng `yolo11n.pt` cho person detection COCO. Lần đầu Ultralytics có thể tải weight; hãy copy weight vào một vị trí cố định và ghi SHA256 cho experiment sau khi chốt detector.

## 1. Validate input

```bash
python -m seat_dynamic_roi.cli_validate \
  --events /path/source_events.csv \
  --seats /path/seat_map.csv \
  --project-root /media/congthieu/UBUNTU_DATA4/DOAN_TN
```

## 2. Preview Seat Map trước khi chạy toàn bộ

```bash
python -m seat_dynamic_roi.cli_preview \
  --events /path/source_events.csv \
  --seats /path/seat_map.csv \
  --out /path/seat_previews \
  --project-root /media/congthieu/UBUNTU_DATA4/DOAN_TN
```

Mỗi session lấy frame giữa một source event và vẽ Seat Activity Anchor ROI + SeatID.

## 3. Dry-run build manifest + ROI metadata, chưa ghi clip

Sửa `config.yaml`:

```yaml
output:
  write_clips: false
```

sau đó:

```bash
python -m seat_dynamic_roi.cli_build \
  --events /path/source_events.csv \
  --seats /path/seat_map.csv \
  --config config.yaml \
  --out /path/seat_roi_dataset \
  --project-root /media/congthieu/UBUNTU_DATA4/DOAN_TN
```

Kiểm tra:

```text
seat_roi_dataset/
  deployment_roi_manifest.jsonl
  dataset_summary.json
  detection_cache.sqlite3
  previews/
```

## 4. Sinh clip thật

Đổi:

```yaml
output:
  write_clips: true
```

và chạy lại. Output:

```text
seat_roi_dataset/
├── deployment_roi_manifest.jsonl
├── dataset_summary.json
├── detection_cache.sqlite3
├── clips/
│   ├── local/<class_slug>/<sample_id>.mp4
│   └── interaction/<class_slug>/<sample_id>__<neighbor_seat>.mp4
└── previews/
```

## Window policy

Mặc định:

- 4 giây/window
- stride 2 giây
- tối đa 4 window/source event để giảm redundancy
- T=8 frame được dùng **chỉ để detect và stabilize ROI**
- sau khi có stable ROI, crop toàn bộ frame của window thành MP4; loader TSM/X3D sau đó vẫn thực hiện sampling T=8 theo training config.

Nếu event ngắn hơn 4 giây, window được center quanh event và clamp vào duration video.

## Dynamic Local ROI

YOLO chỉ detect `person` (COCO class 0). Detection được gán **one-to-one** vào SeatID theo từng frame (một person box không được dùng cho hai seat) bằng:

- IoU với Seat Activity Anchor ROI;
- khoảng cách center;
- bonus nếu center nằm trong anchor.

Trong một 4-second window, box không crop riêng từng frame. Box được robust-aggregate từ T=8 sampled detections (quantile 10/90%) rồi expand thành Activity ROI. Cách này giảm detector jitter gây motion giả cho TSM/X3D.

Nếu số frame detect target dưới `min_detection_frames`, code fallback sang Seat Activity Anchor ROI và ghi:

```json
"local_roi_source": "SEAT_FALLBACK"
```

## Interaction ROI

Với mọi neighbor khai báo trong Seat Map:

```text
interaction_roi = union(target_activity_roi, neighbor_activity_roi) + margin
```

Neighbor cũng dùng dynamic detection nếu đủ, nếu không fallback seat anchor.

**Không** chỉ tạo Pair ROI cho `communicating`/`exchange_object`, vì dùng class ground truth để quyết định input view sẽ tạo train/deployment mismatch.

## Normal hard negatives

Code không tự đoán semantic `pick_up_dropped_object`. Hãy annotate các đoạn đó trong source events:

```text
behavior = normal
subtype = pick_up_dropped_object
```

Tương tự:

```text
bend_to_adjust_shoe
pick_up_pen
pick_up_paper
reach_for_allowed_item
```

Sau đó có thể oversample các subtype normal khó ở training, nhưng không đổi label.

## Quan trọng

Không random split lại dataset sau khi generate. `split` và `fold` được copy từ source-event manifest vào mọi window con. Tất cả windows của một `source_event_id` vì thế giữ cùng partition.
