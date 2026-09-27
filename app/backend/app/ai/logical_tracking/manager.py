from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from math import sqrt

import numpy as np

from app.ai.logical_tracking.appearance import AppearanceEncoder, cosine_similarity, l2_normalize
from app.ai.logical_tracking.types import (
    ActorMemory,
    ActorState,
    LogicalActor,
    LogicalTrackingDiagnostics,
    LogicalTrackingSnapshot,
    LogicalTrackObservation,
    NormalizedRect,
)
from app.monitoring.config import LogicalTrackingConfig, ReIdConfig


@dataclass(frozen=True, slots=True)
class _RecoveryCandidate:
    track_id: int
    actor_id: str
    position_score: float
    motion_score: float
    scale_score: float
    motion_total: float


def _center(bbox: NormalizedRect) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _area(bbox: NormalizedRect) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _percentile(samples: deque[float], fraction: float) -> float | None:
    if not samples:
        return None
    values = sorted(samples)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


class LogicalTrackManager:
    """Map ephemeral ByteTrack IDs to conservative runtime actor identities."""

    def __init__(
        self,
        config: LogicalTrackingConfig,
        reid_config: ReIdConfig,
        appearance_encoder: AppearanceEncoder | None = None,
    ) -> None:
        self.config = config
        self.reid_config = reid_config
        self._appearance_encoder = appearance_encoder
        self._actors: dict[str, ActorMemory] = {}
        self._raw_to_actor: dict[int, str] = {}
        self._next_actor_number = 1
        self._last_timestamp_ms: int | None = None
        self._recoveries_total = 0
        self._motion_recoveries = 0
        self._reid_recoveries = 0
        self._ambiguous_recoveries = 0
        self._reid_requests_total = 0
        self._reid_batches_total = 0
        self._reid_dropped_stale = 0
        self._reid_latency_samples: deque[float] = deque(maxlen=256)

    def reset(self) -> None:
        self._actors.clear()
        self._raw_to_actor.clear()
        self._next_actor_number = 1
        self._last_timestamp_ms = None
        self._recoveries_total = 0
        self._motion_recoveries = 0
        self._reid_recoveries = 0
        self._ambiguous_recoveries = 0
        self._reid_requests_total = 0
        self._reid_batches_total = 0
        self._reid_dropped_stale = 0
        self._reid_latency_samples.clear()

    def update(
        self,
        observations: tuple[LogicalTrackObservation, ...],
        timestamp_ms: int,
        frame_bgr: np.ndarray | None = None,
    ) -> LogicalTrackingSnapshot:
        started = time.perf_counter()
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("Logical tracking timestamp must increase within one generation")
        self._last_timestamp_ms = timestamp_ms
        by_track = {item.track_id: item for item in observations}
        self._mark_missing_lost(set(by_track), timestamp_ms)
        self._expire_old(timestamp_ms)

        known = [
            item
            for item in observations
            if item.track_id in self._raw_to_actor
            and self._actors[self._raw_to_actor[item.track_id]].current_track_id
            in {None, item.track_id}
        ]
        known_track_ids = {item.track_id for item in known}
        new = [item for item in observations if item.track_id not in known_track_ids]
        for observation in known:
            self._update_actor(self._actors[self._raw_to_actor[observation.track_id]], observation)

        candidates = self._recovery_candidates(new, timestamp_ms)
        assigned_tracks: set[int] = set()
        assigned_actors: set[str] = set()
        ambiguous_tracks: list[int] = []
        by_new_track: dict[int, list[_RecoveryCandidate]] = {}
        for candidate in candidates:
            by_new_track.setdefault(candidate.track_id, []).append(candidate)
        unambiguous: list[_RecoveryCandidate] = []
        for track_id, rows in by_new_track.items():
            ranked = sorted(rows, key=lambda item: (-item.motion_total, item.actor_id))
            best = ranked[0]
            second_score = ranked[1].motion_total if len(ranked) > 1 else 0.0
            if (
                len(ranked) > 1
                and best.motion_total - second_score < self.config.recovery.ambiguity_margin
            ):
                ambiguous_tracks.append(track_id)
                self._ambiguous_recoveries += 1
            else:
                unambiguous.append(best)
        for candidate in sorted(
            unambiguous,
            key=lambda item: (-item.motion_total, item.track_id, item.actor_id),
        ):
            if candidate.track_id in assigned_tracks or candidate.actor_id in assigned_actors:
                continue
            observation = by_track[candidate.track_id]
            self._recover(candidate.actor_id, observation)
            assigned_tracks.add(candidate.track_id)
            assigned_actors.add(candidate.actor_id)
            self._recoveries_total += 1
            self._motion_recoveries += 1

        reid_matches = self._appearance_recovery(
            ambiguous_tracks,
            by_new_track,
            by_track,
            assigned_actors,
            frame_bgr,
            timestamp_ms,
        )
        for track_id, actor_id in reid_matches:
            self._recover(actor_id, by_track[track_id])
            assigned_tracks.add(track_id)
            assigned_actors.add(actor_id)
            self._recoveries_total += 1
            self._reid_recoveries += 1

        for observation in new:
            if observation.track_id not in assigned_tracks:
                self._create_actor(observation, timestamp_ms)

        self._seed_prototypes(by_track, frame_bgr, timestamp_ms)
        actors_by_track = {
            track_id: self._snapshot(self._actors[actor_id])
            for track_id, actor_id in self._raw_to_actor.items()
            if track_id in by_track
        }
        logical_tracking_ms = (time.perf_counter() - started) * 1000
        return LogicalTrackingSnapshot(
            actors_by_track=actors_by_track,
            active_actors=tuple(
                self._snapshot(actor)
                for actor in sorted(self._actors.values(), key=lambda item: item.actor_id)
                if actor.state is ActorState.ACTIVE
            ),
            lost_actors=tuple(
                self._snapshot(actor)
                for actor in sorted(self._actors.values(), key=lambda item: item.actor_id)
                if actor.state is ActorState.LOST
            ),
            diagnostics=LogicalTrackingDiagnostics(
                active_logical_actors=sum(
                    actor.state is ActorState.ACTIVE for actor in self._actors.values()
                ),
                lost_logical_actors=sum(
                    actor.state is ActorState.LOST for actor in self._actors.values()
                ),
                raw_track_count=len(observations),
                recoveries_total=self._recoveries_total,
                motion_recoveries=self._motion_recoveries,
                reid_recoveries=self._reid_recoveries,
                ambiguous_recoveries=self._ambiguous_recoveries,
                reid_requests_total=self._reid_requests_total,
                reid_batches_total=self._reid_batches_total,
                reid_dropped_stale=self._reid_dropped_stale,
                logical_tracking_ms=logical_tracking_ms,
                reid_latency_ms_mean=(
                    sum(self._reid_latency_samples) / len(self._reid_latency_samples)
                    if self._reid_latency_samples
                    else None
                ),
                reid_latency_ms_p95=_percentile(self._reid_latency_samples, 0.95),
            ),
        )

    def _mark_missing_lost(self, active_raw_ids: set[int], timestamp_ms: int) -> None:
        del timestamp_ms
        for actor in self._actors.values():
            if actor.state is not ActorState.ACTIVE or actor.current_track_id in active_raw_ids:
                continue
            actor.state = ActorState.LOST
            actor.current_track_id = None
            actor.recovered_on_last_update = False

    def _expire_old(self, timestamp_ms: int) -> None:
        expired: list[str] = []
        for actor in self._actors.values():
            if (
                actor.state is ActorState.LOST
                and timestamp_ms - actor.last_seen_ms > self.config.recovery.max_lost_ms
            ):
                actor.state = ActorState.EXPIRED
                expired.append(actor.actor_id)
        for actor_id in expired:
            actor = self._actors.pop(actor_id)
            for raw_id in actor.raw_track_ids:
                if self._raw_to_actor.get(raw_id) == actor_id:
                    del self._raw_to_actor[raw_id]

    def _recovery_candidates(
        self,
        observations: list[LogicalTrackObservation],
        timestamp_ms: int,
    ) -> list[_RecoveryCandidate]:
        candidates: list[_RecoveryCandidate] = []
        recovery = self.config.recovery
        for observation in observations:
            center = _center(observation.bbox_norm)
            area = _area(observation.bbox_norm)
            for actor in self._actors.values():
                if actor.state is not ActorState.LOST:
                    continue
                gap_ms = timestamp_ms - actor.last_seen_ms
                if gap_ms < 0 or gap_ms > recovery.max_lost_ms:
                    continue
                velocity = self._velocity(actor)
                previous = _center(actor.bbox_norm)
                predicted = (
                    previous[0] + velocity[0] * gap_ms,
                    previous[1] + velocity[1] * gap_ms,
                )
                distance = sqrt((center[0] - predicted[0]) ** 2 + (center[1] - predicted[1]) ** 2)
                if distance > recovery.max_position_distance:
                    continue
                previous_area = _area(actor.bbox_norm)
                if area <= 0 or previous_area <= 0:
                    continue
                scale = min(area, previous_area) / max(area, previous_area)
                if scale < recovery.min_scale_similarity:
                    continue
                position_score = 1.0 - distance / recovery.max_position_distance
                displacement = (center[0] - previous[0], center[1] - previous[1])
                speed = sqrt(velocity[0] ** 2 + velocity[1] ** 2)
                displacement_size = sqrt(displacement[0] ** 2 + displacement[1] ** 2)
                if speed < 1e-8 or displacement_size < 1e-8:
                    motion_score = position_score
                else:
                    direction = (velocity[0] * displacement[0] + velocity[1] * displacement[1]) / (
                        speed * displacement_size
                    )
                    motion_score = max(0.0, (direction + 1.0) / 2.0)
                total = (
                    recovery.position_weight * position_score
                    + recovery.motion_weight * motion_score
                    + recovery.scale_weight * scale
                )
                if total >= recovery.min_score:
                    candidates.append(
                        _RecoveryCandidate(
                            observation.track_id,
                            actor.actor_id,
                            position_score,
                            motion_score,
                            scale,
                            total,
                        )
                    )
        return candidates

    def _appearance_recovery(
        self,
        ambiguous_tracks: list[int],
        candidates: dict[int, list[_RecoveryCandidate]],
        observations: dict[int, LogicalTrackObservation],
        assigned_actors: set[str],
        frame_bgr: np.ndarray | None,
        timestamp_ms: int,
    ) -> list[tuple[int, str]]:
        if (
            not self.reid_config.enabled
            or self._appearance_encoder is None
            or frame_bgr is None
            or not ambiguous_tracks
        ):
            return []
        eligible = [
            track_id
            for track_id in sorted(ambiguous_tracks)
            if any(
                self._actors[row.actor_id].appearance_prototype is not None
                and row.actor_id not in assigned_actors
                for row in candidates.get(track_id, ())
            )
        ]
        batch = eligible[: self.reid_config.max_batch_size]
        if len(eligible) > len(batch):
            self._reid_dropped_stale += len(eligible) - len(batch)
        if not batch:
            return []
        started = time.perf_counter()
        embeddings = self._appearance_encoder.encode(
            frame_bgr,
            [observations[track_id].bbox_norm for track_id in batch],
        )
        self._reid_latency_samples.append((time.perf_counter() - started) * 1000)
        self._reid_batches_total += 1
        self._reid_requests_total += len(batch)
        scored: list[tuple[float, int, str]] = []
        for track_id, embedding in zip(batch, embeddings, strict=True):
            normalized = l2_normalize(embedding) if embedding is not None else None
            if normalized is None:
                continue
            rows: list[tuple[float, str]] = []
            for candidate in candidates.get(track_id, ()):
                actor = self._actors[candidate.actor_id]
                prototype = actor.appearance_prototype
                if prototype is None or actor.actor_id in assigned_actors:
                    continue
                similarity = cosine_similarity(normalized, prototype)
                if similarity < self.reid_config.min_similarity:
                    continue
                appearance_score = (similarity + 1.0) / 2.0
                combined = (
                    self.reid_config.appearance_weight * appearance_score
                    + (1.0 - self.reid_config.appearance_weight) * candidate.motion_total
                )
                if combined >= self.reid_config.min_combined_score:
                    rows.append((combined, actor.actor_id))
            rows.sort(key=lambda item: (-item[0], item[1]))
            if not rows:
                continue
            if len(rows) > 1 and rows[0][0] - rows[1][0] < self.reid_config.ambiguity_margin:
                continue
            scored.append((rows[0][0], track_id, rows[0][1]))
        matches: list[tuple[int, str]] = []
        used_tracks: set[int] = set()
        used_actors = set(assigned_actors)
        for _, track_id, actor_id in sorted(scored, key=lambda item: (-item[0], item[1], item[2])):
            if track_id in used_tracks or actor_id in used_actors:
                continue
            matches.append((track_id, actor_id))
            used_tracks.add(track_id)
            used_actors.add(actor_id)
        return matches

    def _seed_prototypes(
        self,
        observations: dict[int, LogicalTrackObservation],
        frame_bgr: np.ndarray | None,
        timestamp_ms: int,
    ) -> None:
        if not self.reid_config.enabled or self._appearance_encoder is None or frame_bgr is None:
            return
        actor_rows = [
            (actor, observations[actor.current_track_id])
            for actor in sorted(self._actors.values(), key=lambda item: item.actor_id)
            if actor.state is ActorState.ACTIVE
            and actor.current_track_id in observations
            and actor.appearance_prototype is None
            and not actor.recovered_on_last_update
            and timestamp_ms - actor.first_seen_ms >= self.reid_config.prototype_seed_delay_ms
            and observations[actor.current_track_id].confidence
            >= self.reid_config.min_prototype_confidence
        ][: self.reid_config.max_batch_size]
        if not actor_rows:
            return
        started = time.perf_counter()
        embeddings = self._appearance_encoder.encode(
            frame_bgr,
            [observation.bbox_norm for _, observation in actor_rows],
        )
        self._reid_latency_samples.append((time.perf_counter() - started) * 1000)
        self._reid_batches_total += 1
        self._reid_requests_total += len(actor_rows)
        for (actor, _), embedding in zip(actor_rows, embeddings, strict=True):
            normalized = l2_normalize(embedding) if embedding is not None else None
            if normalized is not None:
                actor.appearance_prototype = normalized
                actor.prototype_seeded = True

    def _create_actor(self, observation: LogicalTrackObservation, timestamp_ms: int) -> None:
        actor_id = f"A{self._next_actor_number:04d}"
        self._next_actor_number += 1
        center = _center(observation.bbox_norm)
        self._actors[actor_id] = ActorMemory(
            actor_id=actor_id,
            current_track_id=observation.track_id,
            state=ActorState.ACTIVE,
            first_seen_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
            bbox_norm=observation.bbox_norm,
            centers=[(timestamp_ms, *center)],
            raw_track_ids=[observation.track_id],
            confidence_sum=observation.confidence,
            confidence_count=1,
        )
        self._raw_to_actor[observation.track_id] = actor_id

    def _recover(self, actor_id: str, observation: LogicalTrackObservation) -> None:
        actor = self._actors[actor_id]
        actor.current_track_id = observation.track_id
        actor.state = ActorState.ACTIVE
        actor.recovered_on_last_update = True
        if not actor.raw_track_ids or actor.raw_track_ids[-1] != observation.track_id:
            actor.raw_track_ids.append(observation.track_id)
            evicted = actor.raw_track_ids[: -self.config.raw_track_history_size]
            del actor.raw_track_ids[: -self.config.raw_track_history_size]
            for raw_id in evicted:
                if self._raw_to_actor.get(raw_id) == actor_id:
                    del self._raw_to_actor[raw_id]
        self._raw_to_actor[observation.track_id] = actor_id
        self._update_actor(actor, observation, recovered=True)

    def _update_actor(
        self,
        actor: ActorMemory,
        observation: LogicalTrackObservation,
        *,
        recovered: bool = False,
    ) -> None:
        actor.current_track_id = observation.track_id
        actor.state = ActorState.ACTIVE
        actor.last_seen_ms = self._last_timestamp_ms or actor.last_seen_ms
        actor.bbox_norm = observation.bbox_norm
        actor.centers.append((actor.last_seen_ms, *_center(observation.bbox_norm)))
        del actor.centers[: -self.config.center_history_size]
        actor.confidence_sum += observation.confidence
        actor.confidence_count += 1
        actor.recovered_on_last_update = recovered

    @staticmethod
    def _velocity(actor: ActorMemory) -> tuple[float, float]:
        if len(actor.centers) < 2:
            return (0.0, 0.0)
        start, end = actor.centers[0], actor.centers[-1]
        elapsed = end[0] - start[0]
        if elapsed <= 0:
            return (0.0, 0.0)
        return ((end[1] - start[1]) / elapsed, (end[2] - start[2]) / elapsed)

    def _snapshot(self, actor: ActorMemory) -> LogicalActor:
        return LogicalActor(
            actor_id=actor.actor_id,
            current_track_id=actor.current_track_id,
            state=actor.state,
            first_seen_ms=actor.first_seen_ms,
            last_seen_ms=actor.last_seen_ms,
            bbox_norm=actor.bbox_norm,
            velocity=self._velocity(actor),
            confidence=actor.confidence,
            recovered=actor.recovered_on_last_update,
        )
