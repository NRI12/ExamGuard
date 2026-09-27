"""camera_sources

Revision ID: 20260924_0002
Revises: 20260920_0001
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260924_0002"
down_revision: str | None = "20260920_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cameras",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("room_id", sa.Uuid(), nullable=False),
        sa.Column("source_media_asset_id", sa.Uuid(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["room_id"], ["rooms.id"], name="fk_cameras_room_id_rooms"
        ),
        sa.ForeignKeyConstraint(
            ["source_media_asset_id"],
            ["media_assets.id"],
            name="fk_cameras_source_media_asset_id_media_assets",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cameras"),
    )
    op.create_index("ix_cameras_room_id", "cameras", ["room_id"])
    op.add_column(
        "exam_sessions",
        sa.Column(
            "source_type",
            sa.String(length=20),
            server_default="VIDEO_UPLOAD",
            nullable=False,
        ),
    )
    op.add_column("exam_sessions", sa.Column("camera_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_exam_sessions_camera_id_cameras",
        "exam_sessions",
        "cameras",
        ["camera_id"],
        ["id"],
    )
    op.create_check_constraint(
        "exam_session_source_type",
        "exam_sessions",
        "source_type IN ('CAMERA', 'VIDEO_UPLOAD')",
    )
    op.create_check_constraint(
        "exam_session_source_consistency",
        "exam_sessions",
        "(source_type = 'VIDEO_UPLOAD' AND camera_id IS NULL) OR "
        "(source_type = 'CAMERA' AND camera_id IS NOT NULL AND video_asset_id IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_exam_sessions_exam_session_source_consistency"),
        "exam_sessions",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_exam_sessions_exam_session_source_type"),
        "exam_sessions",
        type_="check",
    )
    op.drop_constraint(
        "fk_exam_sessions_camera_id_cameras",
        "exam_sessions",
        type_="foreignkey",
    )
    op.drop_column("exam_sessions", "camera_id")
    op.drop_column("exam_sessions", "source_type")
    op.drop_index("ix_cameras_room_id", table_name="cameras")
    op.drop_table("cameras")
