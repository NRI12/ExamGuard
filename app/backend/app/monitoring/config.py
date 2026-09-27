from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator

from app.core.config import Settings


class AnalysisConfig(BaseModel):
    target_fps: float = Field(gt=0)
    minimum_fps: float = Field(gt=0)
    queue_size: Literal[1]
    drop_stale_frames: Literal[True]


class DuplicateSuppressionConfig(BaseModel):
    enabled: bool = True
    containment_threshold: float = Field(default=0.90, gt=0, le=1)
    max_area_ratio: float = Field(default=0.70, gt=0, lt=1)
    max_center_distance_ratio: float = Field(default=0.35, gt=0, le=1)
    preferred_detection_confidence: float = Field(default=0.25, gt=0, le=1)
    larger_box_min_confidence_ratio: float = Field(default=0.65, gt=0, le=1)


class DetectorConfig(BaseModel):
    model: Path
    imgsz: int = Field(gt=0)
    conf: float = Field(ge=0, le=1)
    iou: float = Field(ge=0, le=1)
    classes: list[int]
    max_det: int = Field(gt=0)
    half: bool
    device: str = "cuda:0"
    duplicate_suppression: DuplicateSuppressionConfig = Field(
        default_factory=DuplicateSuppressionConfig
    )

    @model_validator(mode="after")
    def person_only(self) -> DetectorConfig:
        if self.classes != [0]:
            raise ValueError("Phase 4 detector must use COCO person class only")
        return self


class ByteTrackConfig(BaseModel):
    tracker_type: Literal["bytetrack"]
    track_high_thresh: float = Field(ge=0, le=1)
    track_low_thresh: float = Field(ge=0, le=1)
    new_track_thresh: float = Field(ge=0, le=1)
    track_buffer: int = Field(gt=0)
    match_thresh: float = Field(ge=0, le=1)
    fuse_score: bool


class DiagnosticsConfig(BaseModel):
    publish_hz: float = Field(gt=0, le=10)


class SeatAssignmentConfig(BaseModel):
    enabled: bool = True
    overlap_weight: float = Field(default=0.70, ge=0)
    distance_weight: float = Field(default=0.30, ge=0)
    min_score: float = Field(default=0.35, ge=0, le=1)
    seat_expand_ratio: float = Field(default=0.08, ge=0)
    confirm_ms: int = Field(default=600, ge=0)
    release_ms: int = Field(default=1500, ge=0)
    switch_margin: float = Field(default=0.15, ge=0)
    switch_confirm_ms: int = Field(default=800, ge=0)

    @model_validator(mode="after")
    def positive_weight_sum(self) -> SeatAssignmentConfig:
        if self.overlap_weight + self.distance_weight <= 0:
            raise ValueError("seat-assignment weights must have a positive sum")
        return self


class IdentityConfig(BaseModel):
    mode: Literal["logical_track", "seat"] = "logical_track"


class LogicalRecoveryConfig(BaseModel):
    max_lost_ms: int = Field(default=5500, gt=0)
    max_position_distance: float = Field(default=0.18, gt=0, le=1)
    min_scale_similarity: float = Field(default=0.40, gt=0, le=1)
    min_score: float = Field(default=0.55, ge=0, le=1)
    ambiguity_margin: float = Field(default=0.08, ge=0, le=1)
    position_weight: float = Field(default=0.55, ge=0)
    motion_weight: float = Field(default=0.25, ge=0)
    scale_weight: float = Field(default=0.20, ge=0)

    @model_validator(mode="after")
    def normalized_weights(self) -> LogicalRecoveryConfig:
        total = self.position_weight + self.motion_weight + self.scale_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("logical recovery weights must sum to 1.0")
        return self


class LogicalTrackingConfig(BaseModel):
    center_history_size: int = Field(default=6, ge=2, le=32)
    raw_track_history_size: int = Field(default=8, ge=1, le=64)
    recovery: LogicalRecoveryConfig = Field(default_factory=LogicalRecoveryConfig)


class ReIdConfig(BaseModel):
    enabled: bool = False
    mode: Literal["on_demand"] = "on_demand"
    encoder: Literal["hsv_histogram_v1"] = "hsv_histogram_v1"
    max_batch_size: int = Field(default=16, gt=0, le=64)
    min_similarity: float = Field(default=0.72, ge=-1, le=1)
    appearance_weight: float = Field(default=0.45, ge=0, le=1)
    min_combined_score: float = Field(default=0.68, ge=0, le=1)
    ambiguity_margin: float = Field(default=0.06, ge=0, le=1)
    prototype_alpha: float = Field(default=0.20, gt=0, le=1)
    prototype_seed_delay_ms: int = Field(default=800, ge=0)
    min_prototype_confidence: float = Field(default=0.50, ge=0, le=1)
    max_request_age_ms: int = Field(default=250, gt=0)


class DynamicNeighborsConfig(BaseModel):
    enabled: bool = True
    max_neighbors_per_actor: int = Field(default=2, ge=0, le=8)
    max_horizontal_gap_ratio: float = Field(default=2.5, ge=0)
    max_vertical_gap_ratio: float = Field(default=0.75, ge=0)
    min_scale_similarity: float = Field(default=0.40, gt=0, le=1)


class ActionRoiGeometryConfig(BaseModel):
    expand_x: float = Field(ge=0)
    expand_top: float = Field(ge=0)
    expand_bottom: float = Field(ge=0)
    min_crop_width_px: int = Field(gt=0)
    min_crop_height_px: int = Field(gt=0)


class ActionAdjacencyConfig(BaseModel):
    row_tolerance_ratio: float = Field(default=0.75, ge=0)
    max_horizontal_gap_ratio: float = Field(default=2.5, ge=0)


class ActionSchedulingConfig(BaseModel):
    prediction_stride_ms: int = Field(default=1500, gt=0)
    max_batch_size: int = Field(default=2, gt=0, le=8)
    max_queue_size: Literal[1] = 1
    max_prediction_age_ms: int = Field(default=2000, gt=0)
    min_inference_interval_ms: int = Field(default=200, ge=0)


class ActionRecognitionConfig(BaseModel):
    enabled: bool = False
    model: Path = Path("action/r3/model.pth")
    checkpoint_sha256: str = "c5ef406cfe404d2883575d8dcf1306fd7bebc6d6b1ec723b5093f2a3e09b357c"
    architecture: Literal["TSM-ResNet50"] = "TSM-ResNet50"
    num_segments: Literal[8] = 8
    input_size: Literal[224] = 224
    clip_span_ms: Literal[4000] = 4000
    capture_interval_ms: int = Field(default=200, gt=0)
    sample_tolerance_ms: int = Field(default=180, ge=0)
    max_gap_ms: int = Field(default=750, gt=0)
    precision: Literal["fp16", "fp32"] = "fp32"
    scheduling: ActionSchedulingConfig = Field(default_factory=ActionSchedulingConfig)
    single_roi: ActionRoiGeometryConfig = Field(
        default_factory=lambda: ActionRoiGeometryConfig(
            expand_x=0.04,
            expand_top=0.03,
            expand_bottom=0.08,
            min_crop_width_px=128,
            min_crop_height_px=128,
        )
    )
    pair_roi: ActionRoiGeometryConfig = Field(
        default_factory=lambda: ActionRoiGeometryConfig(
            expand_x=0.025,
            expand_top=0.02,
            expand_bottom=0.06,
            min_crop_width_px=128,
            min_crop_height_px=128,
        )
    )
    adjacency: ActionAdjacencyConfig = Field(default_factory=ActionAdjacencyConfig)


class EventSmoothingConfig(BaseModel):
    type: Literal["ema"] = "ema"
    alpha: float = Field(gt=0, le=1)


class EventBehaviorConfig(BaseModel):
    proposal_types: list[Literal["SINGLE", "PAIR"]] = Field(min_length=1)
    smoothing: EventSmoothingConfig
    start_threshold: float = Field(gt=0, le=1)
    keep_threshold: float = Field(ge=0, lt=1)
    min_active_ms: int = Field(gt=0)
    end_grace_ms: int = Field(ge=0)
    merge_gap_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_hysteresis(self) -> EventBehaviorConfig:
        if self.start_threshold <= self.keep_threshold:
            raise ValueError("event start_threshold must be greater than keep_threshold")
        if len(set(self.proposal_types)) != len(self.proposal_types):
            raise ValueError("event proposal_types must not contain duplicates")
        return self


class EventDetectionConfig(BaseModel):
    enabled: bool = False
    max_discontinuity_ms: int = Field(default=5000, gt=0)
    dedup_temporal_iou: float = Field(default=0.30, ge=0, le=1)
    dedup_max_gap_ms: int = Field(default=2000, ge=0)
    suspicious_looking: EventBehaviorConfig | None = None
    communicating: EventBehaviorConfig | None = None
    exchange_object: EventBehaviorConfig | None = None
    using_phone_cheat_sheet: EventBehaviorConfig | None = None

    @model_validator(mode="after")
    def require_calibrated_behavior_config(self) -> EventDetectionConfig:
        behaviors = {
            "suspicious_looking": self.suspicious_looking,
            "communicating": self.communicating,
            "exchange_object": self.exchange_object,
            "using_phone_cheat_sheet": self.using_phone_cheat_sheet,
        }
        if self.enabled and any(value is None for value in behaviors.values()):
            missing = sorted(name for name, value in behaviors.items() if value is None)
            raise ValueError(
                "event detection cannot be enabled without calibrated config for: "
                + ", ".join(missing)
            )
        expected = {
            "suspicious_looking": {"SINGLE"},
            "using_phone_cheat_sheet": {"SINGLE"},
            "communicating": {"PAIR"},
            "exchange_object": {"PAIR"},
        }
        for name, value in behaviors.items():
            if value is not None and set(value.proposal_types) != expected[name]:
                raise ValueError(f"invalid proposal routing for {name}")
        return self

    def behavior(self, name: str) -> EventBehaviorConfig | None:
        if name not in {
            "suspicious_looking",
            "communicating",
            "exchange_object",
            "using_phone_cheat_sheet",
        }:
            return None
        return getattr(self, name)


class CheatBoxMapConfig(BaseModel):
    """YOLO person box -> annotated 'upper body + desk' crop box, as fractions of the box width."""

    dx1: float = -0.101
    dx2: float = 0.094
    dy1: float = -0.131
    h: float = Field(default=1.021, gt=0)


class CheatAlertRuleConfig(BaseModel):
    """Decision rule validated on DOAN1 S07/S08 (43/52 events at 15.6 false alarms/person-hour)."""

    smooth_windows: int = Field(default=3, ge=1, le=10)
    start_threshold: float = Field(default=0.85, gt=0, le=1)
    keep_threshold: float = Field(default=0.70, ge=0, lt=1)
    min_windows: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_hysteresis(self) -> CheatAlertRuleConfig:
        if self.start_threshold <= self.keep_threshold:
            raise ValueError("cheat alert start_threshold must be greater than keep_threshold")
        return self


class CheatingClassifierConfig(BaseModel):
    """Per-actor X3D-L classifier (cheating_detection_v1): 16 frames 0.2 s apart, 4 classes."""

    enabled: bool = False
    model: Path = Path("cheating/x3d_l_v1_best.pt")
    sample_interval_ms: int = Field(default=200, gt=0)
    infer_interval_ms: int = Field(default=1000, gt=0)
    min_coverage: float = Field(default=0.75, gt=0, le=1)
    clip_size: int = Field(default=356, gt=0)
    crop_pad: float = Field(default=0.15, ge=0)
    box_map: CheatBoxMapConfig = Field(default_factory=CheatBoxMapConfig)
    batch_size: int = Field(default=8, ge=1, le=32)
    compile: bool = True
    max_actor_gap_ms: int = Field(default=5000, gt=0)
    rule: CheatAlertRuleConfig = Field(default_factory=CheatAlertRuleConfig)


class TrackingDebugConfig(BaseModel):
    enabled: bool = False
    log_every_n_frames: int = Field(default=10, gt=0)
    suspicious_iou_threshold: float = Field(default=0.5, gt=0, le=1)


class TrackingUiConfig(BaseModel):
    tracking_interpolation: Literal[False] = False
    tracking_smoothing: Literal[False] = False


class RuntimeProfile(BaseModel):
    profile: Literal["gtx1650", "rtx3060"]
    device: str
    precision: Literal["fp16"]
    analysis: AnalysisConfig
    detector: DetectorConfig
    tracker: ByteTrackConfig
    diagnostics: DiagnosticsConfig
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    logical_tracking: LogicalTrackingConfig = Field(default_factory=LogicalTrackingConfig)
    reid: ReIdConfig = Field(default_factory=ReIdConfig)
    dynamic_neighbors: DynamicNeighborsConfig = Field(default_factory=DynamicNeighborsConfig)
    seat_assignment: SeatAssignmentConfig = Field(default_factory=SeatAssignmentConfig)
    action_recognition: ActionRecognitionConfig = Field(default_factory=ActionRecognitionConfig)
    event_detection: EventDetectionConfig = Field(default_factory=EventDetectionConfig)
    cheating_classifier: CheatingClassifierConfig = Field(default_factory=CheatingClassifierConfig)
    tracking_debug: TrackingDebugConfig = Field(default_factory=TrackingDebugConfig)
    ui: TrackingUiConfig = Field(default_factory=TrackingUiConfig)

    @model_validator(mode="after")
    def one_event_source(self) -> RuntimeProfile:
        if self.cheating_classifier.enabled and self.event_detection.enabled:
            raise ValueError("enable either the X3D cheating classifier or the R3 event FSM")
        return self

    @model_validator(mode="after")
    def event_detection_requires_action_runtime(self) -> RuntimeProfile:
        if self.event_detection.enabled and not self.action_recognition.enabled:
            raise ValueError("event detection requires action recognition to be enabled")
        if self.event_detection.enabled and self.identity.mode != "seat":
            raise ValueError(
                "event detection persistence currently requires seat identity actor bindings"
            )
        return self


def _yaml_mapping(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Không thể đọc cấu hình {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Cấu hình {path} phải là YAML mapping")
    return payload


def load_runtime_profile(settings: Settings, profile_name: str | None = None) -> RuntimeProfile:
    return load_runtime_profile_from_paths(
        config_root=settings.config_root,
        model_root=settings.model_root,
        profile_name=profile_name or settings.app_profile,
    )


def load_runtime_profile_from_paths(
    *,
    config_root: Path,
    model_root: Path,
    profile_name: str,
) -> RuntimeProfile:
    name = profile_name
    if name not in {"gtx1650", "rtx3060"}:
        raise ValueError(f"Runtime profile không được hỗ trợ: {name}")
    validated_name = cast(Literal["gtx1650", "rtx3060"], name)
    payload = _yaml_mapping(config_root / "runtime" / f"{name}.yaml")
    runtime = payload.get("runtime")
    detector = payload.get("detector")
    tracker = payload.get("tracker")
    if (
        not isinstance(runtime, dict)
        or not isinstance(detector, dict)
        or not isinstance(tracker, dict)
    ):
        raise ValueError("Runtime, detector và tracker configuration là bắt buộc")

    model = Path(str(detector.get("model", "")))
    if model.is_absolute() and model.parts[:2] == ("/", "models"):
        model = model_root.joinpath(*model.parts[2:])
    detector_payload = {**detector, "model": model, "device": runtime.get("device")}

    raw_action_payload = payload.get("action_recognition") or {}
    if not isinstance(raw_action_payload, dict):
        raise ValueError("Action recognition configuration must be a mapping")
    action_payload = dict(raw_action_payload)
    action_model = Path(str(action_payload.get("model", "action/r3/model.pth")))
    if action_model.is_absolute() and action_model.parts[:2] == ("/", "models"):
        action_model = model_root.joinpath(*action_model.parts[2:])
    elif not action_model.is_absolute():
        action_model = model_root / action_model
    action_payload["model"] = action_model

    raw_cheat_payload = payload.get("cheating_classifier") or {}
    if not isinstance(raw_cheat_payload, dict):
        raise ValueError("Cheating classifier configuration must be a mapping")
    cheat_payload = dict(raw_cheat_payload)
    cheat_model = Path(str(cheat_payload.get("model", "cheating/x3d_l_v1_best.pt")))
    if cheat_model.is_absolute() and cheat_model.parts[:2] == ("/", "models"):
        cheat_model = model_root.joinpath(*cheat_model.parts[2:])
    elif not cheat_model.is_absolute():
        cheat_model = model_root / cheat_model
    cheat_payload["model"] = cheat_model

    tracker_path = Path(str(tracker.get("config", "")))
    if tracker_path.is_absolute() and tracker_path.parts[:3] == ("/", "app", "configs"):
        tracker_path = config_root.joinpath(*tracker_path.parts[3:])

    if runtime.get("precision") != "fp16":
        raise ValueError("Phase 4 runtime precision phải là fp16")
    return RuntimeProfile(
        profile=validated_name,
        device=str(runtime.get("device")),
        precision="fp16",
        analysis=AnalysisConfig.model_validate(payload.get("analysis")),
        detector=DetectorConfig.model_validate(detector_payload),
        tracker=ByteTrackConfig.model_validate(_yaml_mapping(tracker_path)),
        diagnostics=DiagnosticsConfig.model_validate(payload.get("diagnostics")),
        identity=IdentityConfig.model_validate(payload.get("identity") or {}),
        logical_tracking=LogicalTrackingConfig.model_validate(
            payload.get("logical_tracking") or {}
        ),
        reid=ReIdConfig.model_validate(payload.get("reid") or {}),
        dynamic_neighbors=DynamicNeighborsConfig.model_validate(
            payload.get("dynamic_neighbors") or {}
        ),
        seat_assignment=SeatAssignmentConfig.model_validate(payload.get("seat_assignment") or {}),
        action_recognition=ActionRecognitionConfig.model_validate(action_payload),
        event_detection=EventDetectionConfig.model_validate(payload.get("event_detection") or {}),
        cheating_classifier=CheatingClassifierConfig.model_validate(cheat_payload),
        tracking_debug=TrackingDebugConfig.model_validate(payload.get("tracking_debug") or {}),
        ui=TrackingUiConfig.model_validate(payload.get("ui") or {}),
    )
