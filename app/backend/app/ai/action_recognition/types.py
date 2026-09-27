from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ProposalType(StrEnum):
    SINGLE = "SINGLE"
    PAIR = "PAIR"


@dataclass(frozen=True, slots=True)
class ActionProposal:
    proposal_id: str
    proposal_type: ProposalType
    session_candidate_ids: tuple[uuid.UUID, ...]
    seat_ids: tuple[uuid.UUID, ...]
    seat_codes: tuple[str, ...]
    current_track_ids: tuple[int, ...]
    bbox_norm: tuple[float, float, float, float]
    timestamp_ms: int
    actor_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BufferedRoiFrame:
    timestamp_ms: int
    rgb: Any


@dataclass(frozen=True, slots=True)
class ActionClip:
    proposal: ActionProposal
    start_timestamp_ms: int
    end_timestamp_ms: int
    frames: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    proposal_id: str
    proposal_type: ProposalType
    session_candidate_ids: tuple[uuid.UUID, ...]
    seat_codes: tuple[str, ...]
    timestamp_ms: int
    probabilities: tuple[float, ...]
    predicted_class: str
    confidence: float
    actor_ids: tuple[str, ...] = ()
    model_name: str = "r3_tsm_r50_k400_diff_final"

    def as_dict(self, class_names: tuple[str, ...]) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.value,
            "actor_ids": list(self.actor_ids),
            "session_candidate_ids": [str(value) for value in self.session_candidate_ids],
            "seat_codes": list(self.seat_codes),
            "timestamp_ms": self.timestamp_ms,
            "class_probabilities": {
                name: probability
                for name, probability in zip(class_names, self.probabilities, strict=True)
            },
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "model_name": self.model_name,
        }
