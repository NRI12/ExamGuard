import uuid

from fastapi import APIRouter, Query, status

from app.features.auth.dependencies import CandidateManager, CandidateReader, DatabaseSession
from app.features.candidates.schemas import CandidateCreate, CandidateResponse, CandidateUpdate
from app.features.candidates.service import (
    candidate_or_error,
    create_candidate,
    list_candidates,
    update_candidate,
)
from app.shared.pagination import Page

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.get("", response_model=Page[CandidateResponse])
def get_candidates(
    _: CandidateReader,
    db: DatabaseSession,
    q: str | None = Query(default=None, max_length=255),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[CandidateResponse]:
    candidates, total = list_candidates(db, q, page, page_size)
    return Page(
        items=[CandidateResponse.model_validate(item) for item in candidates],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("", response_model=CandidateResponse, status_code=status.HTTP_201_CREATED)
def post_candidate(
    payload: CandidateCreate,
    actor: CandidateManager,
    db: DatabaseSession,
) -> CandidateResponse:
    return CandidateResponse.model_validate(create_candidate(db, payload, actor))


@router.get("/{candidate_id}", response_model=CandidateResponse)
def get_candidate(
    candidate_id: uuid.UUID,
    _: CandidateReader,
    db: DatabaseSession,
) -> CandidateResponse:
    return CandidateResponse.model_validate(candidate_or_error(db, candidate_id))


@router.patch("/{candidate_id}", response_model=CandidateResponse)
def patch_candidate(
    candidate_id: uuid.UUID,
    payload: CandidateUpdate,
    actor: CandidateManager,
    db: DatabaseSession,
) -> CandidateResponse:
    return CandidateResponse.model_validate(update_candidate(db, candidate_id, payload, actor))
