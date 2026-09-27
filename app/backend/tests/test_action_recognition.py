from __future__ import annotations

import ast
import importlib.util
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from app.ai.action_recognition.adapter import (
    R3_CLASS_NAMES,
    ActionModelAdapter,
    ActionModelRegistry,
    ActionModelResult,
)
from app.ai.action_recognition.buffer import TimestampedRoiBuffer
from app.ai.action_recognition.model import KineticsTSMClassifier, TemporalShift
from app.ai.action_recognition.preprocessing import preprocess_rgb_clips, sample_segment_indices
from app.ai.action_recognition.proposals import adjacent_seat_pairs, build_action_proposals
from app.ai.action_recognition.roi import training_crop_box, union_bbox
from app.ai.action_recognition.runtime import ActionRecognitionRuntime
from app.ai.action_recognition.scheduler import ActionScheduler
from app.ai.action_recognition.types import (
    ActionClip,
    ActionPrediction,
    ActionProposal,
    ProposalType,
)
from app.ai.domain import Track
from app.ai.seat_identity.types import (
    AssignmentState,
    SeatDefinition,
    SeatIdentityContext,
    TrackIdentity,
)
from app.monitoring.config import ActionRecognitionConfig


def action_config(**overrides: Any) -> ActionRecognitionConfig:
    payload: dict[str, Any] = {
        "enabled": True,
        "model": Path("unused.pth"),
        "precision": "fp32",
        "capture_interval_ms": 200,
        "sample_tolerance_ms": 180,
        "scheduling": {
            "prediction_stride_ms": 2000,
            "max_batch_size": 2,
            "max_queue_size": 1,
            "max_prediction_age_ms": 2000,
            "min_inference_interval_ms": 0,
        },
    }
    payload.update(overrides)
    return ActionRecognitionConfig.model_validate(payload)


def assigned_track(
    track_id: int,
    seat_id: uuid.UUID,
    candidate_id: uuid.UUID,
    bbox: tuple[float, float, float, float],
) -> Track:
    return Track(
        track_id=track_id,
        bbox_norm=bbox,
        confidence=0.9,
        identity=TrackIdentity(
            state=AssignmentState.ASSIGNED,
            seat_id=seat_id,
            seat_code="A01",
            session_candidate_id=candidate_id,
            score=0.9,
        ),
    )


def test_r3_crop_geometry_matches_training_manifest_example() -> None:
    bbox = (314.75 / 1920, 156.7 / 1080, 568.29 / 1920, 439.26 / 1080)
    assert training_crop_box(
        bbox,
        1920,
        1080,
        expand_x=0.04,
        expand_top=0.03,
        expand_bottom=0.08,
        min_width_px=128,
        min_height_px=128,
    ) == (304, 148, 274, 314)


@pytest.mark.parametrize(
    "bbox",
    [
        (0.2, 0.2, 0.4, 0.8),
        (0.0, 0.0, 0.12, 0.45),
        (0.8, 0.65, 1.0, 1.0),
        (-0.1, 0.2, 0.05, 0.9),
        (0.45, 0.45, 0.46, 0.47),
    ],
)
def test_single_roi_clamps_edges_and_preserves_minimum_crop(
    bbox: tuple[float, float, float, float],
) -> None:
    x, y, width, height = training_crop_box(
        bbox,
        1920,
        1080,
        expand_x=0.04,
        expand_top=0.03,
        expand_bottom=0.08,
        min_width_px=128,
        min_height_px=128,
    )
    assert 0 <= x < x + width <= 1920
    assert 0 <= y < y + height <= 1080
    assert width >= 128 and height >= 128
    assert x % 2 == y % 2 == width % 2 == height % 2 == 0


def test_pair_roi_union_contains_both_actors_and_interaction_space() -> None:
    left = (0.05, 0.05, 0.30, 0.85)
    right = (0.28, 0.10, 0.60, 0.95)
    pair = union_bbox((left, right))
    assert pair == (0.05, 0.05, 0.60, 0.95)
    x, y, width, height = training_crop_box(
        pair,
        1920,
        1080,
        expand_x=0.025,
        expand_top=0.02,
        expand_bottom=0.06,
        min_width_px=128,
        min_height_px=128,
    )
    assert x <= left[0] * 1920 and x + width >= right[2] * 1920
    assert y <= left[1] * 1080 and y + height >= right[3] * 1080


def test_adjacency_only_connects_consecutive_seats_in_same_row() -> None:
    seats = tuple(
        SeatDefinition(uuid.uuid4(), code, bbox)
        for code, bbox in (
            ("A01", (0.05, 0.10, 0.15, 0.30)),
            ("A02", (0.20, 0.10, 0.30, 0.30)),
            ("A03", (0.35, 0.10, 0.45, 0.30)),
            ("B01", (0.05, 0.55, 0.15, 0.75)),
            ("B02", (0.20, 0.55, 0.30, 0.75)),
            ("B04", (0.80, 0.55, 0.90, 0.75)),
        )
    )
    pairs = adjacent_seat_pairs(seats, row_tolerance_ratio=0.75, max_gap_ratio=2.5)
    assert pairs == (
        (seats[0].id, seats[1].id),
        (seats[1].id, seats[2].id),
        (seats[3].id, seats[4].id),
    )


def test_explicit_2d_neighbor_graph_overrides_perspective_row_inference() -> None:
    seats = tuple(
        SeatDefinition(uuid.uuid4(), code, bbox)
        for code, bbox in (
            ("A1", (0.15, 0.17, 0.29, 0.50)),
            ("A2", (0.36, 0.16, 0.58, 0.42)),
            ("A3", (0.62, 0.11, 0.80, 0.39)),
            ("B1", (0.00, 0.21, 0.31, 0.68)),
            ("B2", (0.40, 0.23, 0.68, 0.58)),
            ("B3", (0.70, 0.16, 0.98, 0.51)),
        )
    )
    explicit = (
        (seats[0].id, seats[1].id),
        (seats[1].id, seats[2].id),
        (seats[3].id, seats[4].id),
        (seats[4].id, seats[5].id),
        (seats[0].id, seats[3].id),
        (seats[1].id, seats[4].id),
        (seats[2].id, seats[5].id),
    )
    context = SeatIdentityContext(uuid.uuid4(), seats, (), explicit)
    runtime = ActionRecognitionRuntime(
        session_id=context.session_id,
        runtime_instance_id=uuid.uuid4(),
        config=action_config(),
        identity_context=context,
        model=FakeActionModel(),  # type: ignore[arg-type]
        publish=lambda _: None,
    )
    try:
        assert runtime._adjacent_pairs == explicit
    finally:
        assert runtime.close()


def test_proposal_ids_are_candidate_stable_and_unassigned_tracks_are_excluded() -> None:
    left_seat, right_seat = uuid.uuid4(), uuid.uuid4()
    left_candidate, right_candidate = uuid.uuid4(), uuid.uuid4()
    adjacent = ((left_seat, right_seat),)
    first = build_action_proposals(
        (
            assigned_track(17, left_seat, left_candidate, (0.1, 0.1, 0.3, 0.8)),
            assigned_track(21, right_seat, right_candidate, (0.3, 0.1, 0.5, 0.8)),
            Track(
                track_id=99,
                bbox_norm=(0.7, 0.1, 0.9, 0.8),
                confidence=0.9,
                identity=TrackIdentity(state=AssignmentState.UNASSIGNED),
            ),
        ),
        adjacent,
        1000,
    )
    fragmented = build_action_proposals(
        (
            assigned_track(24, left_seat, left_candidate, (0.11, 0.1, 0.31, 0.8)),
            assigned_track(25, right_seat, right_candidate, (0.31, 0.1, 0.51, 0.8)),
        ),
        adjacent,
        1200,
    )
    assert {proposal.proposal_id for proposal in first} == {
        proposal.proposal_id for proposal in fragmented
    }
    assert len(first) == 3
    assert all(99 not in proposal.current_track_ids for proposal in first)
    pair = next(proposal for proposal in first if proposal.proposal_type is ProposalType.PAIR)
    assert pair.session_candidate_ids == tuple(sorted((left_candidate, right_candidate), key=str))
    expected_pair_id = f"pair:{pair.session_candidate_ids[0]}:{pair.session_candidate_ids[1]}"
    assert pair.proposal_id == expected_pair_id


def test_timestamp_buffer_samples_eight_segments_and_resets_on_gap() -> None:
    proposal = ActionProposal(
        proposal_id="single:candidate",
        proposal_type=ProposalType.SINGLE,
        session_candidate_ids=(uuid.uuid4(),),
        seat_ids=(uuid.uuid4(),),
        seat_codes=("A01",),
        current_track_ids=(17,),
        bbox_norm=(0.1, 0.1, 0.5, 0.9),
        timestamp_ms=0,
    )
    buffer = TimestampedRoiBuffer(
        clip_span_ms=4000,
        num_segments=8,
        capture_interval_ms=200,
        sample_tolerance_ms=180,
        max_gap_ms=750,
    )
    for timestamp in range(0, 4001, 200):
        frame = np.full((2, 2, 3), timestamp // 200, dtype=np.uint8)
        assert buffer.append(proposal, timestamp, frame)
    clip = buffer.clip_if_ready(proposal, 4000)
    assert clip is not None
    assert [int(frame[0, 0, 0]) for frame in clip.frames] == [1, 4, 6, 9, 11, 14, 16, 19]
    assert buffer.clip_if_ready(proposal, 5000) is None
    buffer.append(proposal, 5001, np.zeros((2, 2, 3), dtype=np.uint8))
    assert buffer.stored_frame_count == 1


def test_temporal_buffer_is_bounded_handles_pause_duplicates_seek_and_expiry() -> None:
    proposal = ActionProposal(
        proposal_id="single:buffer",
        proposal_type=ProposalType.SINGLE,
        session_candidate_ids=(uuid.uuid4(),),
        seat_ids=(uuid.uuid4(),),
        seat_codes=("A01",),
        current_track_ids=(1,),
        bbox_norm=(0.1, 0.1, 0.5, 0.9),
        timestamp_ms=0,
    )
    buffer = TimestampedRoiBuffer(
        clip_span_ms=4000,
        num_segments=8,
        capture_interval_ms=200,
        sample_tolerance_ms=180,
        max_gap_ms=750,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert buffer.append(proposal, 0, frame)
    assert not buffer.should_capture(proposal.proposal_id, 100)
    assert not buffer.append(proposal, 0, frame)
    assert buffer.stored_frame_count == 1
    # Wall-clock pause cannot advance a timestamp-based clip.
    time.sleep(0.01)
    assert buffer.clip_if_ready(proposal, 0) is None
    for timestamp in range(200, 6000, 200):
        buffer.append(proposal, timestamp, frame)
    assert buffer.stored_frame_count <= buffer.max_frames
    buffer.prune(set(), 11000)
    assert buffer.proposal_count == 0
    buffer.append(proposal, 12000, frame)
    buffer.append(proposal, 100, frame)
    assert buffer.stored_frame_count == 1
    buffer.clear()
    assert buffer.stored_frame_count == 0


def test_eval_sampling_matches_frozen_r3_reference() -> None:
    reference_path = (
        Path(__file__).parents[2]
        / "ai_reference/r3_tsm_r50_k400_diff_final/dataset_tsm_reference.py"
    )
    if not reference_path.exists():
        reference_path = Path("/models/action/r3/dataset_tsm_reference.py")
    if not reference_path.exists():
        pytest.skip("frozen R3 reference package is not mounted")
    source = ast.parse(reference_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_segment_indices"
    )
    namespace: dict[str, Any] = {"np": np}
    compiled = compile(ast.Module(body=[function], type_ignores=[]), str(reference_path), "exec")
    exec(compiled, namespace)
    reference_sampling = namespace["sample_segment_indices"]
    for frame_count in (1, 5, 8, 17, 100):
        expected = reference_sampling(frame_count, 8, False)
        assert np.array_equal(sample_segment_indices(frame_count, 8), expected)


def test_preprocessing_matches_frozen_r3_equations() -> None:
    generator = np.random.default_rng(42)
    frames = tuple(generator.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(8))
    actual = preprocess_rgb_clips([frames])
    reference = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    reference = torch.nn.functional.interpolate(
        reference, size=(224, 224), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.equal(actual[0], (reference - mean) / std)


def test_temporal_shift_and_r3_model_input_contract() -> None:
    source = torch.arange(2 * 3 * 4 * 1 * 1).reshape(6, 4, 1, 1).float()
    shifted = TemporalShift.shift(source, num_segments=3, fold_div=2)
    assert shifted.shape == source.shape
    model = KineticsTSMClassifier(num_classes=5)
    model.eval()
    with pytest.raises(ValueError, match="Expected T=8"):
        model(torch.zeros(1, 7, 3, 32, 32))


def test_runtime_model_inference_matches_frozen_r3_reference() -> None:
    reference_path = (
        Path(__file__).parents[2] / "ai_reference/r3_tsm_r50_k400_diff_final/tsm_kinetics_model.py"
    )
    if not reference_path.exists():
        reference_path = Path("/models/action/r3/tsm_kinetics_model.py")
    if not reference_path.exists():
        pytest.skip("frozen R3 reference package is not mounted")
    spec = importlib.util.spec_from_file_location("frozen_r3_model", reference_path)
    assert spec is not None and spec.loader is not None
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    torch.manual_seed(7)
    runtime_model = KineticsTSMClassifier(num_classes=5).eval()
    reference_model = reference_module.KineticsTSMClassifier(
        num_classes=5,
        num_segments=8,
        fold_div=8,
        dropout=0.6,
        imagenet_init=False,
    ).eval()
    reference_model.load_state_dict(runtime_model.state_dict(), strict=True)
    sample = torch.randn(1, 8, 3, 32, 32)
    with torch.inference_mode():
        expected = reference_model(sample)
        actual = runtime_model(sample)
    assert torch.equal(actual, expected)


class FakeActionModel:
    @property
    def device(self) -> str:
        return "cpu"

    def predict(self, clips: tuple[ActionClip, ...]) -> ActionModelResult:
        predictions = tuple(
            ActionPrediction(
                proposal_id=clip.proposal.proposal_id,
                proposal_type=clip.proposal.proposal_type,
                session_candidate_ids=clip.proposal.session_candidate_ids,
                seat_codes=clip.proposal.seat_codes,
                timestamp_ms=clip.end_timestamp_ms,
                probabilities=(0.8, 0.05, 0.05, 0.05, 0.05),
                predicted_class="normal",
                confidence=0.8,
            )
            for clip in clips
        )
        return ActionModelResult(predictions, 1.0, 3.0)


def test_action_runtime_emits_raw_prediction_and_seek_clears_buffers() -> None:
    seat_id, candidate_id, session_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    context = SeatIdentityContext(
        session_id,
        (SeatDefinition(seat_id, "A01", (0.1, 0.1, 0.4, 0.9)),),
        (),
    )
    messages: list[dict[str, Any]] = []
    runtime = ActionRecognitionRuntime(
        session_id=session_id,
        runtime_instance_id=uuid.uuid4(),
        config=action_config(),
        identity_context=context,
        model=FakeActionModel(),  # type: ignore[arg-type]
        publish=messages.append,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    track = assigned_track(17, seat_id, candidate_id, (0.1, 0.1, 0.4, 0.9))
    for timestamp in range(0, 4201, 200):
        runtime.update(frame, (track,), timestamp, 0)
        time.sleep(0.005)
    deadline = time.monotonic() + 2
    while not messages and time.monotonic() < deadline:
        time.sleep(0.01)
    assert messages and messages[-1]["type"] == "action_prediction"
    assert messages[-1]["predictions"][0]["proposal_id"] == f"single:{candidate_id}"
    runtime.reset(1)
    assert runtime.diagnostics().buffered_roi_frames == 0
    assert runtime.diagnostics().action_predictions_total == 0
    runtime.update(frame, (track,), 100, 0)
    time.sleep(0.05)
    assert runtime.diagnostics().buffered_roi_frames == 0
    assert runtime.close()


def scheduler_clip(
    proposal_id: str,
    proposal_type: ProposalType,
    timestamp_ms: int,
) -> ActionClip:
    candidate_ids = (
        (uuid.uuid4(),) if proposal_type is ProposalType.SINGLE else (uuid.uuid4(), uuid.uuid4())
    )
    proposal = ActionProposal(
        proposal_id=proposal_id,
        proposal_type=proposal_type,
        session_candidate_ids=candidate_ids,
        seat_ids=tuple(uuid.uuid4() for _ in candidate_ids),
        seat_codes=tuple(f"A{index + 1:02d}" for index in range(len(candidate_ids))),
        current_track_ids=tuple(range(1, len(candidate_ids) + 1)),
        bbox_norm=(0.1, 0.1, 0.5, 0.9),
        timestamp_ms=timestamp_ms,
    )
    frames = tuple(np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(8))
    return ActionClip(proposal, timestamp_ms - 4000, timestamp_ms, frames)


def scheduler_prediction(clip: ActionClip) -> ActionPrediction:
    return ActionPrediction(
        proposal_id=clip.proposal.proposal_id,
        proposal_type=clip.proposal.proposal_type,
        session_candidate_ids=clip.proposal.session_candidate_ids,
        seat_codes=clip.proposal.seat_codes,
        timestamp_ms=clip.end_timestamp_ms,
        probabilities=(0.8, 0.05, 0.05, 0.05, 0.05),
        predicted_class="normal",
        confidence=0.8,
    )


def test_action_scheduler_enforces_timestamp_stride_and_latest_only() -> None:
    scheduler = ActionScheduler(
        prediction_stride_ms=1000,
        max_batch_size=1,
        max_prediction_age_ms=2000,
        min_inference_interval_ms=0,
    )
    first = scheduler_clip("single:a", ProposalType.SINGLE, 4000)
    scheduler.observe((first,), 4000)
    assert scheduler.next_batch(4000) == (first,)
    scheduler.complete((scheduler_prediction(first),))

    intermediate = scheduler_clip("single:a", ProposalType.SINGLE, 4500)
    latest = scheduler_clip("single:a", ProposalType.SINGLE, 4800)
    scheduler.observe((intermediate,), 4500)
    scheduler.observe((latest,), 4800)
    assert scheduler.next_batch(4800) == ()
    assert scheduler.snapshot().replaced_ready_requests == 1
    assert scheduler.next_batch(5000) == (latest,)


def test_action_scheduler_is_oldest_first_bounded_and_covers_single_and_pair() -> None:
    scheduler = ActionScheduler(
        prediction_stride_ms=1000,
        max_batch_size=2,
        max_prediction_age_ms=5000,
        min_inference_interval_ms=0,
    )
    old_single = scheduler_clip("single:old", ProposalType.SINGLE, 4000)
    pair = scheduler_clip("pair:adjacent", ProposalType.PAIR, 4200)
    scheduler.observe((old_single, pair), 4200)
    initial = scheduler.next_batch(4200)
    assert {clip.proposal.proposal_id for clip in initial} == {
        "single:old",
        "pair:adjacent",
    }
    scheduler.complete(tuple(scheduler_prediction(clip) for clip in initial))

    never_seen = scheduler_clip("single:new", ProposalType.SINGLE, 6000)
    old_again = scheduler_clip("single:old", ProposalType.SINGLE, 6000)
    pair_again = scheduler_clip("pair:adjacent", ProposalType.PAIR, 6000)
    scheduler.observe((old_again, pair_again, never_seen), 6000)
    selected = scheduler.next_batch(6000)
    assert len(selected) == 2
    assert selected[0].proposal.proposal_id == "single:new"
    assert selected[1].proposal.proposal_id == "single:old"
    scheduler.complete(tuple(scheduler_prediction(clip) for clip in selected))
    remaining = scheduler.next_batch(6000)
    assert remaining == (pair_again,)
    scheduler.complete((scheduler_prediction(pair_again),))
    snapshot = scheduler.snapshot()
    assert snapshot.single_predictions == 3
    assert snapshot.pair_predictions == 2
    assert snapshot.single_interval_ms_max == 2000
    assert snapshot.pair_interval_ms_max == 1800


def test_action_scheduler_drops_stale_requests_and_reset_clears_state() -> None:
    scheduler = ActionScheduler(
        prediction_stride_ms=1000,
        max_batch_size=2,
        max_prediction_age_ms=500,
        min_inference_interval_ms=0,
    )
    stale = scheduler_clip("single:stale", ProposalType.SINGLE, 1000)
    scheduler.observe((stale,), 1600)
    assert scheduler.next_batch(1600) == ()
    assert scheduler.snapshot().stale_drops == 1
    scheduler.reset()
    assert scheduler.snapshot().ready_proposals == 0
    assert scheduler.snapshot().stale_drops == 0


def test_action_config_rejects_contract_drift() -> None:
    config = action_config()
    assert config.single_roi.model_dump() == {
        "expand_x": 0.04,
        "expand_top": 0.03,
        "expand_bottom": 0.08,
        "min_crop_width_px": 128,
        "min_crop_height_px": 128,
    }
    assert config.pair_roi.model_dump() == {
        "expand_x": 0.025,
        "expand_top": 0.02,
        "expand_bottom": 0.06,
        "min_crop_width_px": 128,
        "min_crop_height_px": 128,
    }
    with pytest.raises(ValidationError):
        action_config(num_segments=16)
    with pytest.raises(ValidationError):
        action_config(clip_span_ms=2000)
    with pytest.raises(ValidationError):
        action_config(scheduling={"max_queue_size": 2})


def test_action_model_registry_reuses_one_model_per_artifact_device_precision() -> None:
    registry = ActionModelRegistry()
    config = action_config(model=Path("same-model.pth"), precision="fp32")
    assert registry.get(config, "cpu") is registry.get(config, "cpu")
    assert registry.get(config, "cpu") is not registry.get(
        config.model_copy(update={"precision": "fp16"}),
        "cpu",
    )


def test_real_r3_checkpoint_cpu_smoke() -> None:
    path = Path(__file__).parents[2] / "ai_reference/r3_tsm_r50_k400_diff_final/model.pth"
    if not path.exists():
        path = Path("/models/action/r3/model.pth")
    if not path.exists():
        pytest.skip("R3 checkpoint is not mounted")
    config = action_config(model=path)
    adapter = ActionModelAdapter(config, "cpu")
    proposal = ActionProposal(
        proposal_id="single:smoke",
        proposal_type=ProposalType.SINGLE,
        session_candidate_ids=(uuid.uuid4(),),
        seat_ids=(uuid.uuid4(),),
        seat_codes=("A01",),
        current_track_ids=(17,),
        bbox_norm=(0.1, 0.1, 0.9, 0.9),
        timestamp_ms=4000,
    )
    frames = tuple(np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8))
    result = adapter.predict((ActionClip(proposal, 0, 4000, frames),))
    predictions, inference_ms = result.predictions, result.inference_ms
    assert len(predictions) == 1
    assert predictions[0].predicted_class in R3_CLASS_NAMES
    assert sum(predictions[0].probabilities) == pytest.approx(1.0, abs=1e-5)
    assert inference_ms > 0
    assert adapter.load_count == 1


def test_real_r3_clip_matches_reference_preprocessing_and_inference() -> None:
    clip_path = Path(os.environ.get("R3_REFERENCE_CLIP", ""))
    if not clip_path.is_file():
        pytest.skip("R3_REFERENCE_CLIP is not a mounted real R3 ROI clip")
    package = Path("/models/action/r3")
    if not package.exists():
        package = Path(__file__).parents[2] / "ai_reference/r3_tsm_r50_k400_diff_final"
    dataset_source = ast.parse((package / "dataset_tsm_reference.py").read_text())
    functions = [
        node
        for node in dataset_source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"decode_video_cv2", "sample_segment_indices"}
    ]
    import cv2

    namespace: dict[str, Any] = {"cv2": cv2, "np": np, "Path": Path}
    compiled = compile(ast.Module(body=functions, type_ignores=[]), "reference_dataset", "exec")
    exec(compiled, namespace)
    all_frames = namespace["decode_video_cv2"](clip_path)
    reference_indices = namespace["sample_segment_indices"](len(all_frames), 8, False)
    runtime_indices = sample_segment_indices(len(all_frames), 8)
    assert np.array_equal(runtime_indices, reference_indices)
    frames = tuple(all_frames[reference_indices])
    reference_input = torch.from_numpy(all_frames[reference_indices]).permute(0, 3, 1, 2).float()
    reference_input = reference_input / 255.0
    reference_input = torch.nn.functional.interpolate(
        reference_input, size=(224, 224), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    reference_input = ((reference_input - mean) / std).unsqueeze(0)
    runtime_input = preprocess_rgb_clips([frames])
    preprocessing_max_abs = float((runtime_input - reference_input).abs().max())
    assert preprocessing_max_abs == 0.0

    reference_path = package / "tsm_kinetics_model.py"
    spec = importlib.util.spec_from_file_location("real_r3_reference_model", reference_path)
    assert spec is not None and spec.loader is not None
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    reference_model = reference_module.KineticsTSMClassifier(
        num_classes=5, num_segments=8, fold_div=8, dropout=0.6, imagenet_init=False
    ).eval()
    checkpoint_path = package / "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reference_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.inference_mode():
        reference_logits = reference_model(reference_input)
        reference_probabilities = torch.softmax(reference_logits, dim=1)[0]
    adapter = ActionModelAdapter(action_config(model=checkpoint_path), "cpu")
    proposal = ActionProposal(
        proposal_id="single:reference",
        proposal_type=ProposalType.SINGLE,
        session_candidate_ids=(uuid.uuid4(),),
        seat_ids=(uuid.uuid4(),),
        seat_codes=("A01",),
        current_track_ids=(17,),
        bbox_norm=(0.1, 0.1, 0.9, 0.9),
        timestamp_ms=4000,
    )
    result = adapter.predict((ActionClip(proposal, 0, 4000, frames),))
    runtime_probabilities = torch.tensor(result.predictions[0].probabilities)
    probability_max_abs = float((runtime_probabilities - reference_probabilities).abs().max())
    assert probability_max_abs < 1e-6
    assert (
        result.predictions[0].predicted_class
        == R3_CLASS_NAMES[int(reference_probabilities.argmax())]
    )
    print(
        json.dumps(
            {
                "clip": str(clip_path),
                "decoded_frames": len(all_frames),
                "sample_indices": reference_indices.tolist(),
                "preprocessing_max_abs": preprocessing_max_abs,
                "probability_max_abs": probability_max_abs,
                "top_class": result.predictions[0].predicted_class,
            }
        )
    )
