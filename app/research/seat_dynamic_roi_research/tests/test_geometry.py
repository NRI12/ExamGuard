from seat_dynamic_roi.geometry import Box, association_score, expand_box, iou, union_boxes


def test_iou_identity():
    a = Box(0, 0, 10, 10)
    assert abs(iou(a, a) - 1.0) < 1e-9


def test_union():
    a = Box(0, 0, 10, 10)
    b = Box(5, 4, 20, 30)
    assert union_boxes([a, b]) == Box(0, 0, 20, 30)


def test_expand_clips_image():
    a = Box(10, 10, 20, 20)
    b = expand_box(a, 25, 25, 2, 2, 2, 2)
    assert b.x1 == 0
    assert b.y1 == 0
    assert b.x2 == 25
    assert b.y2 == 25


def test_association_prefers_near_box():
    anchor = Box(100, 100, 200, 300)
    near = Box(110, 110, 195, 295)
    far = Box(500, 500, 600, 700)
    s_near = association_score(near, anchor, 0.55, 0.35, 0.10)
    s_far = association_score(far, anchor, 0.55, 0.35, 0.10)
    assert s_near > s_far
