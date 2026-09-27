from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.ai.domain import Detection
from app.monitoring.config import DetectorConfig, DuplicateSuppressionConfig


class DetectorInitializationError(RuntimeError):
    pass


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _is_nested_duplicate(
    left: Detection,
    right: Detection,
    config: DuplicateSuppressionConfig,
) -> bool:
    left_area = _area(left.bbox_xyxy)
    right_area = _area(right.bbox_xyxy)
    if left_area <= 0 or right_area <= 0:
        return False
    smaller_area = min(left_area, right_area)
    larger_area = max(left_area, right_area)
    if smaller_area / larger_area > config.max_area_ratio:
        return False
    containment = _intersection_area(left.bbox_xyxy, right.bbox_xyxy) / smaller_area
    if containment < config.containment_threshold:
        return False

    larger = left.bbox_xyxy if left_area >= right_area else right.bbox_xyxy
    left_center = (
        (left.bbox_xyxy[0] + left.bbox_xyxy[2]) / 2,
        (left.bbox_xyxy[1] + left.bbox_xyxy[3]) / 2,
    )
    right_center = (
        (right.bbox_xyxy[0] + right.bbox_xyxy[2]) / 2,
        (right.bbox_xyxy[1] + right.bbox_xyxy[3]) / 2,
    )
    larger_diagonal = ((larger[2] - larger[0]) ** 2 + (larger[3] - larger[1]) ** 2) ** 0.5
    center_distance = (
        (left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2
    ) ** 0.5
    return center_distance / larger_diagonal <= config.max_center_distance_ratio


def suppress_nested_person_detections(
    detections: list[Detection],
    config: DuplicateSuppressionConfig,
) -> list[Detection]:
    """Remove conservative partial/full-body duplicates left by ordinary IoU NMS.

    Ordinary NMS cannot remove a torso box nested in a full-person box when their
    union is much larger than the torso. This pass only considers strongly
    contained, center-aligned boxes with a material area difference. Adjacent
    boxes that merely overlap are deliberately retained.
    """
    if not config.enabled or len(detections) < 2:
        return detections

    retained = [True] * len(detections)
    pairs: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(detections):
        for right_index in range(left_index + 1, len(detections)):
            right = detections[right_index]
            if not _is_nested_duplicate(left, right, config):
                continue
            smaller_area = min(_area(left.bbox_xyxy), _area(right.bbox_xyxy))
            containment = _intersection_area(left.bbox_xyxy, right.bbox_xyxy) / smaller_area
            pairs.append((containment, left_index, right_index))

    for _, left_index, right_index in sorted(pairs, reverse=True):
        if not retained[left_index] or not retained[right_index]:
            continue
        left = detections[left_index]
        right = detections[right_index]
        left_area = _area(left.bbox_xyxy)
        right_area = _area(right.bbox_xyxy)
        larger_index, smaller_index = (
            (left_index, right_index) if left_area >= right_area else (right_index, left_index)
        )
        larger = detections[larger_index]
        smaller = detections[smaller_index]
        if larger.confidence < config.preferred_detection_confidence <= smaller.confidence:
            retained[larger_index] = False
        elif larger.confidence >= (smaller.confidence * config.larger_box_min_confidence_ratio):
            retained[smaller_index] = False
        else:
            retained[larger_index] = False

    return [detection for index, detection in enumerate(detections) if retained[index]]


class PersonDetector:
    def __init__(
        self,
        config: DetectorConfig,
        *,
        model: Any | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.last_raw_detection_count = 0
        self.last_suppressed_detection_count = 0
        self.last_raw_detections: tuple[Detection, ...] = ()
        if model is not None:
            self._model = model
            return
        self.validate_environment(config)
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        try:
            self._model = model_factory(str(config.model))
        except Exception as error:
            raise DetectorInitializationError(f"Không thể tải YOLO11n: {error}") from error

    @staticmethod
    def validate_environment(config: DetectorConfig) -> None:
        if not Path(config.model).is_file():
            raise DetectorInitializationError(f"Không tìm thấy model YOLO11n: {config.model}")
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise DetectorInitializationError(
                f"CUDA không khả dụng nhưng profile yêu cầu {config.device}"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.last_raw_detection_count = 0
        self.last_suppressed_detection_count = 0
        self.last_raw_detections = ()
        results = self._model.predict(
            source=frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            classes=self.config.classes,
            max_det=self.config.max_det,
            device=self.config.device,
            quantize=16 if self.config.half else None,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        detections = [
            Detection(
                bbox_xyxy=tuple(float(value) for value in box),  # type: ignore[arg-type]
                confidence=float(confidence),
                class_id=int(class_id),
            )
            for box, confidence, class_id in zip(xyxy, confidences, classes, strict=True)
            if int(class_id) == 0
        ]
        self.last_raw_detection_count = len(detections)
        self.last_raw_detections = tuple(detections)
        filtered = suppress_nested_person_detections(
            detections,
            self.config.duplicate_suppression,
        )
        self.last_suppressed_detection_count = len(detections) - len(filtered)
        return filtered
