from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ai.detector.yolo import PersonDetector
from app.monitoring.config import DetectorConfig


@pytest.mark.gpu
def test_yolo11n_loads_and_runs_on_cuda_when_explicitly_enabled() -> None:
    if os.getenv("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to run optional CUDA integration")
    model = Path(os.getenv("YOLO_MODEL", "/models/detection/yolo11n.pt"))
    if not torch.cuda.is_available() or not model.is_file():
        pytest.skip("CUDA or YOLO11n artifact is unavailable")
    detector = PersonDetector(
        DetectorConfig(
            model=model,
            imgsz=640,
            conf=0.10,
            iou=0.70,
            classes=[0],
            max_det=32,
            half=True,
            device="cuda:0",
        )
    )
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert all(item.class_id == 0 for item in detections)
