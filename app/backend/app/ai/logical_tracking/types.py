from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

NormalizedRect = tuple[float, float, float, float]


class ActorState(StrEnum):
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class LogicalTrackObservation:
    track_id: int
    bbox_norm: NormalizedRect
    confidence: float


@dataclass(frozen=True, slots=True)
class LogicalActor:
    actor_id: str
    current_track_id: int | None
    state: ActorState
    first_seen_ms: int
    last_seen_ms: int
    bbox_norm: NormalizedRect
    velocity: tuple[float, float]
    confidence: float
    recovered: bool = False


@dataclass(frozen=True, slots=True)
class LogicalTrackingDiagnostics:
    active_logical_actors: int
    lost_logical_actors: int
    raw_track_count: int
    recoveries_total: int
    motion_recoveries: int
    reid_recoveries: int
    ambiguous_recoveries: int
    reid_requests_total: int
    reid_batches_total: int
    reid_dropped_stale: int
    logical_tracking_ms: float
    reid_latency_ms_mean: float | None
    reid_latency_ms_p95: float | None


@dataclass(frozen=True, slots=True)
class LogicalTrackingSnapshot:
    actors_by_track: dict[int, LogicalActor]
    active_actors: tuple[LogicalActor, ...]
    lost_actors: tuple[LogicalActor, ...]
    diagnostics: LogicalTrackingDiagnostics


@dataclass(slots=True)
class ActorMemory:
    actor_id: str
    current_track_id: int | None
    state: ActorState
    first_seen_ms: int
    last_seen_ms: int
    bbox_norm: NormalizedRect
    centers: list[tuple[int, float, float]]
    raw_track_ids: list[int]
    confidence_sum: float
    confidence_count: int
    appearance_prototype: np.ndarray | None = None
    recovered_on_last_update: bool = False
    prototype_seeded: bool = False

    @property
    def confidence(self) -> float:
        return self.confidence_sum / max(1, self.confidence_count)
