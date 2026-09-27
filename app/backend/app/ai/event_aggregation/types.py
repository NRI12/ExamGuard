from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from enum import StrEnum


class EventBehavior(StrEnum):
    SUSPICIOUS_LOOKING = "suspicious_looking"
    COMMUNICATING = "communicating"
    EXCHANGE_OBJECT = "exchange_object"
    USING_PHONE_CHEAT_SHEET = "using_phone_cheat_sheet"


class EventFsmState(StrEnum):
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True, slots=True)
class AggregatedEvent:
    session_id: uuid.UUID
    behavior: EventBehavior
    session_candidate_ids: tuple[uuid.UUID, ...]
    proposal_ids: tuple[str, ...]
    start_ms: int
    end_ms: int
    peak_ms: int
    peak_probability: float
    active_probability_sum: float
    active_prediction_count: int

    @property
    def ai_confidence(self) -> float:
        return self.active_probability_sum / self.active_prediction_count

    @property
    def fingerprint(self) -> str:
        actors = ",".join(str(value) for value in self.session_candidate_ids)
        payload = (
            f"{self.session_id}|{self.behavior.value}|{actors}|"
            f"{self.start_ms}|{self.end_ms}|{self.peak_ms}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EventAggregationDiagnostics:
    candidate_fsms: int
    active_fsms: int
    cooldown_fsms: int
    events_created_total: int
    events_suppressed_total: int
    events_deduplicated_total: int
    per_behavior_event_count: dict[str, int]

