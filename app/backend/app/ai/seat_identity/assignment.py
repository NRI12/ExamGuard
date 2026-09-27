from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.ai.domain import Track
from app.ai.seat_identity.matcher import build_candidates, greedy_one_to_one, score_track_seat
from app.ai.seat_identity.types import (
    AssignmentState,
    SeatAssignmentSnapshot,
    SeatCandidateBinding,
    SeatIdentityContext,
    SeatMatchCandidate,
    SeatOccupancyState,
    SeatRuntimeSnapshot,
    TrackIdentity,
)
from app.monitoring.config import SeatAssignmentConfig


@dataclass(slots=True)
class _TrackSeatState:
    state: AssignmentState = AssignmentState.UNASSIGNED
    seat_id: uuid.UUID | None = None
    score: float | None = None
    tentative_since_ms: int | None = None
    invalid_since_ms: int | None = None
    switch_seat_id: uuid.UUID | None = None
    switch_since_ms: int | None = None


class SeatAssignmentEngine:
    """Stateful runtime Track→Seat assignment; no state is persisted to PostgreSQL."""

    def __init__(self, context: SeatIdentityContext, config: SeatAssignmentConfig) -> None:
        self.context = context
        self.config = config
        self._seats = {seat.id: seat for seat in context.seats}
        self._bindings: dict[uuid.UUID, SeatCandidateBinding] = {
            binding.seat_id: binding for binding in context.bindings
        }
        self._tracks: dict[int, _TrackSeatState] = {}
        self._last_timestamp_ms: int | None = None
        self._seat_switches = 0
        self._identity_recoveries = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._last_timestamp_ms = None
        self._seat_switches = 0
        self._identity_recoveries = 0

    def update(self, tracks: tuple[Track, ...], timestamp_ms: int) -> SeatAssignmentSnapshot:
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("Seat assignment timestamp must increase within one generation")
        self._last_timestamp_ms = timestamp_ms
        if not self.config.enabled or not self._seats:
            return self._empty_snapshot(tracks)

        active = {track.track_id: track for track in tracks}
        scores = {
            (track.track_id, seat.id): score_track_seat(
                track.track_id,
                track.bbox_norm,
                seat,
                self.config,
            )
            for track in tracks
            for seat in self.context.seats
        }
        self._advance_missing_tracks(set(active), timestamp_ms)
        self._release_invalid_assignments(active, scores, timestamp_ms)

        for track_id in active:
            self._tracks.setdefault(track_id, _TrackSeatState())

        active_assigned = {
            track_id: state
            for track_id, state in self._tracks.items()
            if track_id in active
            and state.state == AssignmentState.ASSIGNED
            and state.seat_id is not None
        }
        reserved_seats = {
            state.seat_id for state in active_assigned.values() if state.seat_id is not None
        }
        proposal_candidates = self._proposal_candidates(active, scores, reserved_seats)
        selected = greedy_one_to_one(proposal_candidates)

        for track_id, state in list(active_assigned.items()):
            proposal = selected.get(track_id)
            if proposal is not None:
                self._advance_switch(track_id, state, proposal, timestamp_ms)
            else:
                state.switch_seat_id = None
                state.switch_since_ms = None

        for track_id in active:
            state = self._tracks[track_id]
            if state.state != AssignmentState.ASSIGNED:
                self._advance_tentative(track_id, state, selected.get(track_id), timestamp_ms)

        identities = {
            track_id: self._identity_for(state)
            for track_id, state in self._tracks.items()
            if track_id in active
        }
        seats = self._seat_snapshots(set(active))
        return self._snapshot(identities, seats)

    def _advance_missing_tracks(self, active_ids: set[int], timestamp_ms: int) -> None:
        for track_id, state in list(self._tracks.items()):
            if track_id in active_ids:
                continue
            if state.state != AssignmentState.ASSIGNED:
                del self._tracks[track_id]
                continue
            if state.invalid_since_ms is None:
                state.invalid_since_ms = timestamp_ms
            if timestamp_ms - state.invalid_since_ms >= self.config.release_ms:
                del self._tracks[track_id]

    def _release_invalid_assignments(
        self,
        active: dict[int, Track],
        scores: dict[tuple[int, uuid.UUID], SeatMatchCandidate],
        timestamp_ms: int,
    ) -> None:
        for track_id, state in self._tracks.items():
            if (
                track_id not in active
                or state.state != AssignmentState.ASSIGNED
                or state.seat_id is None
            ):
                continue
            current = scores[(track_id, state.seat_id)]
            state.score = current.score
            if current.score >= self.config.min_score:
                state.invalid_since_ms = None
                continue
            if state.invalid_since_ms is None:
                state.invalid_since_ms = timestamp_ms
            if timestamp_ms - state.invalid_since_ms >= self.config.release_ms:
                self._clear_state(state)

    def _proposal_candidates(
        self,
        active: dict[int, Track],
        scores: dict[tuple[int, uuid.UUID], SeatMatchCandidate],
        reserved_seats: set[uuid.UUID],
    ) -> list[SeatMatchCandidate]:
        candidates: list[SeatMatchCandidate] = []
        for track_id, track in active.items():
            state = self._tracks[track_id]
            if state.state == AssignmentState.ASSIGNED and state.seat_id is not None:
                current_score = scores[(track_id, state.seat_id)].score
                for seat_id in self._seats:
                    if seat_id == state.seat_id or seat_id in reserved_seats:
                        continue
                    candidate = scores[(track_id, seat_id)]
                    if (
                        candidate.score >= self.config.min_score
                        and candidate.score > current_score + self.config.switch_margin
                    ):
                        candidates.append(candidate)
                continue
            free_seats = tuple(seat for seat in self.context.seats if seat.id not in reserved_seats)
            candidates.extend(
                build_candidates({track_id: track.bbox_norm}, free_seats, self.config)
            )
        return candidates

    def _advance_switch(
        self,
        track_id: int,
        state: _TrackSeatState,
        proposal: SeatMatchCandidate,
        timestamp_ms: int,
    ) -> None:
        if state.switch_seat_id != proposal.seat_id:
            state.switch_seat_id = proposal.seat_id
            state.switch_since_ms = timestamp_ms
            return
        if state.switch_since_ms is None:
            state.switch_since_ms = timestamp_ms
            return
        if timestamp_ms - state.switch_since_ms < self.config.switch_confirm_ms:
            return
        state.seat_id = proposal.seat_id
        state.score = proposal.score
        state.invalid_since_ms = None
        state.switch_seat_id = None
        state.switch_since_ms = None
        self._seat_switches += 1

    def _advance_tentative(
        self,
        track_id: int,
        state: _TrackSeatState,
        proposal: SeatMatchCandidate | None,
        timestamp_ms: int,
    ) -> None:
        if proposal is None:
            self._clear_state(state)
            return
        if state.state != AssignmentState.TENTATIVE or state.seat_id != proposal.seat_id:
            state.state = AssignmentState.TENTATIVE
            state.seat_id = proposal.seat_id
            state.score = proposal.score
            state.tentative_since_ms = timestamp_ms
            state.invalid_since_ms = None
            return
        state.score = proposal.score
        if state.tentative_since_ms is None:
            state.tentative_since_ms = timestamp_ms
            return
        if timestamp_ms - state.tentative_since_ms < self.config.confirm_ms:
            return

        recovered = [
            previous_id
            for previous_id, previous in self._tracks.items()
            if previous_id != track_id
            and previous.state == AssignmentState.ASSIGNED
            and previous.seat_id == proposal.seat_id
        ]
        for previous_id in recovered:
            del self._tracks[previous_id]
        if recovered:
            self._identity_recoveries += 1
        state.state = AssignmentState.ASSIGNED
        state.tentative_since_ms = None
        state.invalid_since_ms = None

    @staticmethod
    def _clear_state(state: _TrackSeatState) -> None:
        state.state = AssignmentState.UNASSIGNED
        state.seat_id = None
        state.score = None
        state.tentative_since_ms = None
        state.invalid_since_ms = None
        state.switch_seat_id = None
        state.switch_since_ms = None

    def _identity_for(self, state: _TrackSeatState) -> TrackIdentity:
        if state.state == AssignmentState.UNASSIGNED or state.seat_id is None:
            return TrackIdentity(state=AssignmentState.UNASSIGNED)
        seat = self._seats[state.seat_id]
        binding = self._bindings.get(state.seat_id)
        return TrackIdentity(
            state=state.state,
            seat_id=seat.id,
            seat_code=seat.code,
            session_candidate_id=(
                binding.session_candidate_id
                if binding is not None and state.state == AssignmentState.ASSIGNED
                else None
            ),
            score=state.score,
        )

    def _seat_snapshots(self, active_ids: set[int]) -> tuple[SeatRuntimeSnapshot, ...]:
        snapshots: list[SeatRuntimeSnapshot] = []
        for seat in self.context.seats:
            assigned = [
                (track_id, state)
                for track_id, state in self._tracks.items()
                if state.state == AssignmentState.ASSIGNED and state.seat_id == seat.id
            ]
            active_assignment = next(
                ((track_id, state) for track_id, state in assigned if track_id in active_ids),
                None,
            )
            if active_assignment is not None:
                track_id, state = active_assignment
                occupancy = (
                    SeatOccupancyState.OCCUPIED
                    if state.invalid_since_ms is None
                    else SeatOccupancyState.GRACE
                )
            elif assigned:
                track_id, _ = assigned[0]
                occupancy = SeatOccupancyState.GRACE
            else:
                track_id = None
                occupancy = SeatOccupancyState.EMPTY
            binding = self._bindings.get(seat.id)
            snapshots.append(
                SeatRuntimeSnapshot(
                    seat_id=seat.id,
                    seat_code=seat.code,
                    session_candidate_id=(binding.session_candidate_id if binding else None),
                    state=occupancy,
                    track_id=track_id,
                )
            )
        return tuple(snapshots)

    def _empty_snapshot(self, tracks: tuple[Track, ...]) -> SeatAssignmentSnapshot:
        identities = {
            track.track_id: TrackIdentity(state=AssignmentState.UNASSIGNED) for track in tracks
        }
        seats = tuple(
            SeatRuntimeSnapshot(
                seat_id=seat.id,
                seat_code=seat.code,
                session_candidate_id=(
                    self._bindings[seat.id].session_candidate_id
                    if seat.id in self._bindings
                    else None
                ),
                state=SeatOccupancyState.EMPTY,
                track_id=None,
            )
            for seat in self.context.seats
        )
        return self._snapshot(identities, seats)

    def _snapshot(
        self,
        identities: dict[int, TrackIdentity],
        seats: tuple[SeatRuntimeSnapshot, ...],
    ) -> SeatAssignmentSnapshot:
        return SeatAssignmentSnapshot(
            identities=identities,
            seats=seats,
            assigned_tracks=sum(
                identity.state == AssignmentState.ASSIGNED for identity in identities.values()
            ),
            tentative_tracks=sum(
                identity.state == AssignmentState.TENTATIVE for identity in identities.values()
            ),
            unassigned_tracks=sum(
                identity.state == AssignmentState.UNASSIGNED for identity in identities.values()
            ),
            occupied_seats=sum(seat.state == SeatOccupancyState.OCCUPIED for seat in seats),
            grace_seats=sum(seat.state == SeatOccupancyState.GRACE for seat in seats),
            empty_seats=sum(seat.state == SeatOccupancyState.EMPTY for seat in seats),
            seat_switches=self._seat_switches,
            identity_recoveries=self._identity_recoveries,
        )
