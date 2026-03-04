"""Add experiment design tables (Sprint 4)

Revision ID: b2c3d4e5f6a7
Revises: 4a7c92e1b3d5
Create Date: 2026-03-16 12:00:00.000000

"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import JSON, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.compiler import compiles

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "4a7c92e1b3d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


class JSONB(JSON):
    # See https://docs.sqlalchemy.org/en/20/core/custom_types.html
    __visit_name__ = "JSONB"


@compiles(JSONB, "sqlite")
def _(*args: Any, **kwargs: Any) -> str:
    # See https://docs.sqlalchemy.org/en/20/core/custom_types.html
    return "JSONB"


JSON_ = (
    JSON()
    .with_variant(
        postgresql.JSONB(),
        "postgresql",
    )
    .with_variant(
        JSONB(),
        "sqlite",
    )
)


def upgrade() -> None:
    # --- Experiment Designs ---
    op.create_table(
        "experiment_designs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("description", sa.String, nullable=True),
        sa.Column(
            "design_type",
            sa.String,
            nullable=False,
            server_default=text("'full_factorial'"),
        ),
        sa.Column(
            "status",
            sa.String,
            nullable=False,
            server_default=text("'draft'"),
        ),
        sa.Column("metadata", JSON_, nullable=False, server_default=text("'{}'")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "design_type IN ('full_factorial', 'fractional', 'custom')",
            name="valid_design_type",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'cells_generated', 'running', 'completed', 'failed')",
            name="valid_design_status",
        ),
    )

    # --- Experiment Factors ---
    op.create_table(
        "experiment_factors",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "design_id",
            sa.Integer,
            sa.ForeignKey("experiment_designs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String, nullable=False),
        sa.Column(
            "factor_type",
            sa.String,
            nullable=False,
            server_default=text("'custom'"),
        ),
        sa.Column("levels", JSON_, nullable=False, server_default=text("'[]'")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "factor_type IN ('embedding', 'reranker', 'judge_llm', "
            "'rag_llm', 'testset_llm', 'custom')",
            name="valid_factor_type",
        ),
    )

    # --- Experiment Design Cells ---
    op.create_table(
        "experiment_design_cells",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "design_id",
            sa.Integer,
            sa.ForeignKey("experiment_designs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("combination", JSON_, nullable=False, server_default=text("'{}'")),
        sa.Column(
            "status",
            sa.String,
            nullable=False,
            server_default=text("'pending'"),
        ),
        sa.Column(
            "experiment_id",
            sa.Integer,
            sa.ForeignKey("experiments.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("result_summary", JSON_, nullable=False, server_default=text("'{}'")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="valid_cell_status",
        ),
    )


def downgrade() -> None:
    op.drop_table("experiment_design_cells")
    op.drop_table("experiment_factors")
    op.drop_table("experiment_designs")
