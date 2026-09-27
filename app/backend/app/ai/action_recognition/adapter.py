from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.ai.action_recognition.model import KineticsTSMClassifier
from app.ai.action_recognition.preprocessing import preprocess_rgb_clips_profiled
from app.ai.action_recognition.types import ActionClip, ActionPrediction
from app.monitoring.config import ActionRecognitionConfig

R3_CLASS_NAMES = (
    "normal",
    "suspicious_looking",
    "communicating",
    "exchange_object",
    "using_phone/cheat_sheet",
)


@dataclass(frozen=True, slots=True)
class ActionModelResult:
    predictions: tuple[ActionPrediction, ...]
    preprocessing_ms: float
    inference_ms: float
    tensor_assembly_ms: float = 0.0
    resize_ms: float = 0.0
    normalization_ms: float = 0.0
    h2d_ms: float = 0.0
    forward_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0


def _safe_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility for older torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError("R3 checkpoint must be a mapping")
    return value


class ActionModelAdapter:
    """One lazily loaded, thread-safe R3 model shared by monitoring sessions."""

    def __init__(self, config: ActionRecognitionConfig, device: str) -> None:
        self.config = config
        self.device_name = device
        self._model: KineticsTSMClassifier | None = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()
        self.load_count = 0

    @staticmethod
    def validate_artifact(config: ActionRecognitionConfig) -> None:
        if not config.model.is_file():
            raise RuntimeError(f"R3 checkpoint does not exist: {config.model}")
        hasher = hashlib.sha256()
        with config.model.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != config.checkpoint_sha256:
            raise RuntimeError(f"R3 checkpoint SHA-256 mismatch: {digest}")

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        self.validate_artifact(self.config)
        checkpoint = _safe_load(self.config.model)
        expected = {
            "architecture": "TSM-ResNet50",
            "num_segments": self.config.num_segments,
            "input_size": self.config.input_size,
            "num_classes": len(R3_CLASS_NAMES),
            "class_names": list(R3_CLASS_NAMES),
            "view": "compact_actor_context_roi",
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"R3 checkpoint contract mismatch: {mismatches}")
        model_config = checkpoint.get("config", {}).get("model", {})
        if not isinstance(model_config, dict):
            raise RuntimeError("R3 checkpoint is missing model configuration")
        if (
            model_config.get("backbone") != "resnet50"
            or model_config.get("fold_div") != 8
            or model_config.get("shift_place") != "blockres"
        ):
            raise RuntimeError(f"Unsupported R3 architecture config: {model_config}")
        model = KineticsTSMClassifier(
            num_classes=len(R3_CLASS_NAMES),
            num_segments=self.config.num_segments,
            fold_div=8,
            dropout=float(model_config.get("dropout", 0.6)),
        )
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("R3 checkpoint is missing model_state_dict")
        model.load_state_dict(state, strict=True)
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required by action runtime profile: {self.device_name}")
        device = torch.device(self.device_name)
        model.to(device)
        if self.config.precision == "fp16" and device.type == "cuda":
            model.half()
        model.eval()
        self._model, self._device = model, device
        self.load_count += 1

    @property
    def device(self) -> str:
        return self.device_name

    def ensure_loaded(self) -> None:
        with self._lock:
            self._load_locked()

    def predict(self, clips: tuple[ActionClip, ...]) -> ActionModelResult:
        if not clips:
            return ActionModelResult((), 0.0, 0.0)
        with self._lock:
            self._load_locked()
            assert self._model is not None and self._device is not None
            preprocessing_ms = 0.0
            inference_ms = 0.0
            tensor_assembly_ms = 0.0
            resize_ms = 0.0
            normalization_ms = 0.0
            h2d_ms = 0.0
            forward_ms = 0.0
            postprocess_ms = 0.0
            total_ms = 0.0
            rows: list[list[float]] = []
            # The scheduler has already enforced the compute budget. Keeping
            # batching independent from the registry's first session config
            # lets one shared model safely serve runtimes with different budgets.
            batch_size = len(clips)
            for start in range(0, len(clips), batch_size):
                chunk = clips[start : start + batch_size]
                total_started = time.perf_counter()
                tensor, profile = preprocess_rgb_clips_profiled(
                    [clip.frames for clip in chunk],
                    input_size=self.config.input_size,
                    num_segments=self.config.num_segments,
                )
                tensor_assembly_ms += profile.tensor_assembly_ms
                resize_ms += profile.resize_ms
                normalization_ms += profile.normalization_ms
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                transfer_started = time.perf_counter()
                tensor = tensor.to(self._device)
                if self.config.precision == "fp16" and self._device.type == "cuda":
                    tensor = tensor.half()
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                chunk_h2d_ms = (time.perf_counter() - transfer_started) * 1000
                h2d_ms += chunk_h2d_ms
                chunk_preprocessing_ms = (
                    profile.tensor_assembly_ms
                    + profile.resize_ms
                    + profile.normalization_ms
                    + chunk_h2d_ms
                )
                preprocessing_ms += chunk_preprocessing_ms
                forward_started = time.perf_counter()
                with torch.inference_mode():
                    logits = self._model(tensor)
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                chunk_forward_ms = (time.perf_counter() - forward_started) * 1000
                forward_ms += chunk_forward_ms
                postprocess_started = time.perf_counter()
                output = torch.softmax(logits, dim=1).float().cpu()
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                chunk_postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
                postprocess_ms += chunk_postprocess_ms
                inference_ms += chunk_forward_ms + chunk_postprocess_ms
                total_ms += (time.perf_counter() - total_started) * 1000
                if not torch.isfinite(output).all():
                    raise RuntimeError(
                        "R3 produced non-finite probabilities at "
                        f"precision={self.config.precision} "
                        f"batch_size={len(chunk)}"
                    )
                rows.extend(output.tolist())
        predictions = []
        for clip, row in zip(clips, rows, strict=True):
            class_index = max(range(len(row)), key=row.__getitem__)
            predictions.append(
                ActionPrediction(
                    proposal_id=clip.proposal.proposal_id,
                    proposal_type=clip.proposal.proposal_type,
                    session_candidate_ids=clip.proposal.session_candidate_ids,
                    seat_codes=clip.proposal.seat_codes,
                    actor_ids=clip.proposal.actor_ids,
                    timestamp_ms=clip.end_timestamp_ms,
                    probabilities=tuple(float(value) for value in row),
                    predicted_class=R3_CLASS_NAMES[class_index],
                    confidence=float(row[class_index]),
                )
            )
        return ActionModelResult(
            predictions=tuple(predictions),
            preprocessing_ms=preprocessing_ms,
            inference_ms=inference_ms,
            tensor_assembly_ms=tensor_assembly_ms,
            resize_ms=resize_ms,
            normalization_ms=normalization_ms,
            h2d_ms=h2d_ms,
            forward_ms=forward_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
        )


class ActionModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[tuple[str, str, str], ActionModelAdapter] = {}

    def get(self, config: ActionRecognitionConfig, device: str) -> ActionModelAdapter:
        key = (str(config.model.resolve()), device, config.precision)
        with self._lock:
            return self._models.setdefault(key, ActionModelAdapter(config, device))
