from seat_dynamic_roi.detector import Detection
from seat_dynamic_roi.geometry import Box
from seat_dynamic_roi.dataset import _global_assign_frame


def test_detection_not_reused_across_two_seats():
    cfg = {
        "iou_weight": 0.55,
        "center_weight": 0.35,
        "inside_bonus_weight": 0.10,
        "min_score": 0.01,
    }
    anchors = {
        "seat_01": Box(0, 0, 100, 200),
        "seat_02": Box(90, 0, 190, 200),
    }
    detections = [
        Detection(Box(70, 10, 130, 190), 0.9),
    ]
    assigned = _global_assign_frame(detections, anchors, cfg)
    used = [x for x in assigned.values() if x is not None]
    assert len(used) == 1
