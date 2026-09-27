"""initial_schema

Revision ID: 20260920_0001
Revises:
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260920_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamp_columns(*, updated: bool = False) -> list[sa.Column[object]]:
    columns: list[sa.Column[object]] = [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        )
    ]
    if updated:
        columns.append(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            )
        )
    return columns


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        *timestamp_columns(updated=True),
        sa.CheckConstraint(
            "role IN ('SUPERVISOR', 'REVIEWER', 'ADMIN')",
            name="user_role",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )

    op.create_table(
        "rooms",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("code", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        *timestamp_columns(updated=True),
        sa.PrimaryKeyConstraint("id", name="pk_rooms"),
        sa.UniqueConstraint("code", name="uq_rooms_code"),
    )

    op.create_table(
        "candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("candidate_code", sa.String(length=100), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("class_name", sa.String(length=255), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        *timestamp_columns(updated=True),
        sa.PrimaryKeyConstraint("id", name="pk_candidates"),
        sa.UniqueConstraint("candidate_code", name="uq_candidates_candidate_code"),
    )

    op.create_table(
        "media_assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("codec", sa.String(length=100), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        *timestamp_columns(),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name="fk_media_assets_created_by_users",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_media_assets"),
    )

    op.create_table(
        "seats",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("room_id", sa.Uuid(), nullable=False),
        sa.Column("code", sa.String(length=100), nullable=False),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("width", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        *timestamp_columns(updated=True),
        sa.CheckConstraint("x >= 0 AND x <= 1", name="x_range"),
        sa.CheckConstraint("y >= 0 AND y <= 1", name="y_range"),
        sa.CheckConstraint("width > 0 AND width <= 1", name="width_range"),
        sa.CheckConstraint("height > 0 AND height <= 1", name="height_range"),
        sa.CheckConstraint("x + width <= 1", name="horizontal_bounds"),
        sa.CheckConstraint("y + height <= 1", name="vertical_bounds"),
        sa.ForeignKeyConstraint(["room_id"], ["rooms.id"], name="fk_seats_room_id_rooms"),
        sa.PrimaryKeyConstraint("id", name="pk_seats"),
        sa.UniqueConstraint("room_id", "code", name="uq_seats_room_id_code"),
    )

    op.create_table(
        "exam_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_code", sa.String(length=100), nullable=False),
        sa.Column("exam_name", sa.String(length=255), nullable=False),
        sa.Column("room_id", sa.Uuid(), nullable=False),
        sa.Column("video_asset_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("scheduled_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scheduled_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("runtime_profile", sa.String(length=100), nullable=True),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        *timestamp_columns(updated=True),
        sa.CheckConstraint(
            "status IN ('DRAFT', 'READY', 'RUNNING', 'PAUSED', 'COMPLETED', 'CANCELLED', 'ERROR')",
            name="exam_session_status",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name="fk_exam_sessions_created_by_users"
        ),
        sa.ForeignKeyConstraint(["room_id"], ["rooms.id"], name="fk_exam_sessions_room_id_rooms"),
        sa.ForeignKeyConstraint(
            ["video_asset_id"],
            ["media_assets.id"],
            name="fk_exam_sessions_video_asset_id_media_assets",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_exam_sessions"),
        sa.UniqueConstraint("session_code", name="uq_exam_sessions_session_code"),
    )

    op.create_table(
        "session_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("seat_id", sa.Uuid(), nullable=False),
        *timestamp_columns(),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["candidates.id"],
            name="fk_session_candidates_candidate_id_candidates",
        ),
        sa.ForeignKeyConstraint(
            ["seat_id"], ["seats.id"], name="fk_session_candidates_seat_id_seats"
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["exam_sessions.id"],
            name="fk_session_candidates_session_id_exam_sessions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_session_candidates"),
        sa.UniqueConstraint(
            "session_id", "candidate_id", name="uq_session_candidates_session_candidate"
        ),
        sa.UniqueConstraint("session_id", "seat_id", name="uq_session_candidates_session_seat"),
    )

    op.create_table(
        "events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("event_code", sa.String(length=100), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("source", sa.String(length=10), nullable=False),
        sa.Column("behavior_type", sa.String(length=40), nullable=True),
        sa.Column("start_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_ms", sa.BigInteger(), nullable=False),
        sa.Column("peak_ms", sa.BigInteger(), nullable=True),
        sa.Column("ai_confidence", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        *timestamp_columns(updated=True),
        sa.CheckConstraint("source IN ('AI', 'MANUAL')", name="event_source"),
        sa.CheckConstraint(
            "behavior_type IS NULL OR behavior_type IN "
            "('SUSPICIOUS_LOOKING', 'COMMUNICATING', 'EXCHANGE_OBJECT', "
            "'USING_PHONE_CHEAT_SHEET', 'OTHER')",
            name="behavior_type",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING_REVIEW', 'CONFIRMED', 'DISMISSED', "
            "'NEEDS_REVIEW', 'DISPUTED', 'RESOLVED')",
            name="event_status",
        ),
        sa.CheckConstraint("start_ms >= 0", name="start_nonnegative"),
        sa.CheckConstraint("end_ms >= start_ms", name="time_range"),
        sa.CheckConstraint(
            "peak_ms IS NULL OR (peak_ms >= start_ms AND peak_ms <= end_ms)",
            name="peak_range",
        ),
        sa.CheckConstraint(
            "ai_confidence IS NULL OR (ai_confidence >= 0 AND ai_confidence <= 1)",
            name="ai_confidence_range",
        ),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_events_created_by_users"),
        sa.ForeignKeyConstraint(
            ["session_id"], ["exam_sessions.id"], name="fk_events_session_id_exam_sessions"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_events"),
        sa.UniqueConstraint("event_code", name="uq_events_event_code"),
    )
    op.create_index("ix_events_session_id_start_ms", "events", ["session_id", "start_ms"])
    op.create_index("ix_events_session_id_status", "events", ["session_id", "status"])
    op.create_index("ix_events_session_id_behavior_type", "events", ["session_id", "behavior_type"])

    op.create_table(
        "event_actors",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("session_candidate_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=100), nullable=True),
        *timestamp_columns(),
        sa.ForeignKeyConstraint(
            ["event_id"], ["events.id"], name="fk_event_actors_event_id_events"
        ),
        sa.ForeignKeyConstraint(
            ["session_candidate_id"],
            ["session_candidates.id"],
            name="fk_event_actors_session_candidate_id_session_candidates",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_event_actors"),
        sa.UniqueConstraint(
            "event_id", "session_candidate_id", name="uq_event_actors_event_candidate"
        ),
    )

    op.create_table(
        "event_reviews",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("reviewer_id", sa.Uuid(), nullable=False),
        sa.Column("decision", sa.String(length=20), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        *timestamp_columns(),
        sa.CheckConstraint(
            "decision IN ('CONFIRM', 'DISMISS', 'NEEDS_REVIEW')",
            name="review_decision",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"], ["events.id"], name="fk_event_reviews_event_id_events"
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"], ["users.id"], name="fk_event_reviews_reviewer_id_users"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_event_reviews"),
    )

    op.create_table(
        "evidence_assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("start_ms", sa.BigInteger(), nullable=True),
        sa.Column("end_ms", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("locked", sa.Boolean(), server_default=sa.false(), nullable=False),
        *timestamp_columns(),
        sa.CheckConstraint(
            "kind IN ('CONTEXT_VIDEO', 'FOCUSED_VIDEO', 'SNAPSHOT', 'REPORT')",
            name="evidence_kind",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"], ["events.id"], name="fk_evidence_assets_event_id_events"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evidence_assets"),
    )

    op.create_table(
        "appeal_cases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("case_code", sa.String(length=100), nullable=False),
        sa.Column("session_candidate_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("resolution", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        sa.Column("resolved_by", sa.Uuid(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        *timestamp_columns(updated=True),
        sa.CheckConstraint(
            "status IN ('OPEN', 'UNDER_REVIEW', 'UPHELD', 'OVERTURNED', 'CLOSED')",
            name="appeal_status",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name="fk_appeal_cases_created_by_users"
        ),
        sa.ForeignKeyConstraint(
            ["resolved_by"], ["users.id"], name="fk_appeal_cases_resolved_by_users"
        ),
        sa.ForeignKeyConstraint(
            ["session_candidate_id"],
            ["session_candidates.id"],
            name="fk_appeal_cases_session_candidate_id_session_candidates",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_appeal_cases"),
        sa.UniqueConstraint("case_code", name="uq_appeal_cases_case_code"),
    )

    op.create_table(
        "appeal_events",
        sa.Column("appeal_id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["appeal_id"], ["appeal_cases.id"], name="fk_appeal_events_appeal_id_appeal_cases"
        ),
        sa.ForeignKeyConstraint(
            ["event_id"], ["events.id"], name="fk_appeal_events_event_id_events"
        ),
        sa.PrimaryKeyConstraint("appeal_id", "event_id", name="pk_appeal_events"),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=100), nullable=False),
        sa.Column("entity_id", sa.Uuid(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        *timestamp_columns(),
        sa.ForeignKeyConstraint(
            ["actor_user_id"], ["users.id"], name="fk_audit_logs_actor_user_id_users"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
    )
    op.create_index(
        "ix_audit_logs_entity_type_entity_id",
        "audit_logs",
        ["entity_type", "entity_id"],
    )
    op.create_index(
        "ix_audit_logs_actor_user_id_created_at",
        "audit_logs",
        ["actor_user_id", "created_at"],
    )
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])


def downgrade() -> None:
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_user_id_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_entity_type_entity_id", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_table("appeal_events")
    op.drop_table("appeal_cases")
    op.drop_table("evidence_assets")
    op.drop_table("event_reviews")
    op.drop_table("event_actors")
    op.drop_index("ix_events_session_id_behavior_type", table_name="events")
    op.drop_index("ix_events_session_id_status", table_name="events")
    op.drop_index("ix_events_session_id_start_ms", table_name="events")
    op.drop_table("events")
    op.drop_table("session_candidates")
    op.drop_table("exam_sessions")
    op.drop_table("seats")
    op.drop_table("media_assets")
    op.drop_table("candidates")
    op.drop_table("rooms")
    op.drop_table("users")
