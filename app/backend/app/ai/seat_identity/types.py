from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum

NormalizedRect = tuple[float, float, float, float]


class AssignmentState(StrEnum):
    UNASSIGNED = "UNASSIGNED"
    TENTATIVE = "TENTATIVE"
    ASSIGNED = "ASSIGNED"


class SeatOccupancyState(StrEnum):
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"
    GRACE = "GRACE"


@dataclass(frozen=True, slots=True)
class SeatDefinition:
    id: uuid.UUID
    code: str
    bbox_norm: NormalizedRect


@dataclass(frozen=True, slots=True)
class SeatCandidateBinding:
    seat_id: uuid.UUID
    session_candidate_id: uuid.UUID
    candidate_id: uuid.UUID
    candidate_code: str
    candidate_name: str


@dataclass(frozen=True, slots=True)
class SeatIdentityContext:
    session_id: uuid.UUID
    seats: tuple[SeatDefinition, ...]
    bindings: tuple[SeatCandidateBinding, ...]
    adjacent_seat_pairs: tuple[tuple[uuid.UUID, uuid.UUID], ...] = ()

    @classmethod
    def empty(cls, session_id: uuid.UUID) -> SeatIdentityContext:
        return cls(session_id=session_id, seats=(), bindings=())


@dataclass(frozen=True, slots=True)
class SeatMatchCandidate:
    track_id: int
    seat_id: uuid.UUID
    overlap: float
    distance: float
    score: float


@dataclass(frozen=True, slots=True)
class TrackIdentity:
    state: AssignmentState
    seat_id: uuid.UUID | None = None
    seat_code: str | None = None
    session_candidate_id: uuid.UUID | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class SeatRuntimeSnapshot:
    seat_id: uuid.UUID
    seat_code: str
    session_candidate_id: uuid.UUID | None
    state: SeatOccupancyState
    track_id: int | None


@dataclass(frozen=True, slots=True)
class SeatAssignmentSnapshot:
    identities: dict[int, TrackIdentity]
    seats: tuple[SeatRuntimeSnapshot, ...]
    assigned_tracks: int
    tentative_tracks: int
    unassigned_tracks: int
    occupied_seats: int
    grace_seats: int
    empty_seats: int
    seat_switches: int
    identity_recoveries: int
