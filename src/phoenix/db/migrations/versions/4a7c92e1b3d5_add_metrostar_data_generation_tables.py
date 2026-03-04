"""Add MetroStar data generation tables

Revision ID: 4a7c92e1b3d5
Revises: f1a6b2f0c9d5
Create Date: 2026-03-02 12:00:00.000000

"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import JSON, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.compiler import compiles

# revision identifiers, used by Alembic.
revision: str = "4a7c92e1b3d5"
down_revision: Union[str, None] = "f1a6b2f0c9d5"
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
    # --- LLM Adapters ---
    op.create_table(
        "llm_adapters",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False, unique=True),
        sa.Column("provider", sa.String, nullable=False),
        sa.Column("model_name", sa.String, nullable=False),
        sa.Column("endpoint", sa.String, nullable=True),
        sa.Column("api_key_env_var", sa.String, nullable=True),
        sa.Column("can_embed", sa.Boolean, nullable=False, server_default=text("0")),
        sa.Column("can_generate", sa.Boolean, nullable=False, server_default=text("0")),
        sa.Column("can_judge", sa.Boolean, nullable=False, server_default=text("0")),
        sa.Column("can_multimodal", sa.Boolean, nullable=False, server_default=text("0")),
        sa.Column("can_rerank", sa.Boolean, nullable=False, server_default=text("0")),
        sa.Column("cost_per_1k_input_tokens", sa.Float, nullable=True),
        sa.Column("cost_per_1k_output_tokens", sa.Float, nullable=True),
        sa.Column("max_context_tokens", sa.Integer, nullable=True),
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
    )

    # --- Data Generation Jobs ---
    op.create_table(
        "data_generation_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column(
            "status",
            sa.String,
            nullable=False,
            server_default=text("'pending'"),
        ),
        sa.Column("corpus_source", sa.String, nullable=False),
        sa.Column("corpus_config", JSON_, nullable=False, server_default=text("'{}'")),
        sa.Column(
            "sampling_strategy", sa.String, nullable=False, server_default=text("'random'")
        ),
        sa.Column("sample_size", sa.Integer, nullable=False, server_default=text("50")),
        sa.Column(
            "testset_llm_adapter_id",
            sa.Integer,
            sa.ForeignKey("llm_adapters.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "transform_llm_adapter_id",
            sa.Integer,
            sa.ForeignKey("llm_adapters.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("llm_config", JSON_, nullable=False, server_default=text("'{}'")),
        sa.Column("is_multimodal", sa.Boolean, nullable=False, server_default=text("0")),
        sa.Column("output_dataset_name", sa.String, nullable=True),
        sa.Column("artifacts", JSON_, nullable=False, server_default=text("'{}'")),
        sa.Column("error_message", sa.String, nullable=True),
        sa.Column("seed", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="valid_data_gen_status",
        ),
    )

    # --- Evaluation Criteria ---
    op.create_table(
        "evaluation_criteria",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False, unique=True),
        sa.Column("description", sa.String, nullable=True),
        sa.Column("category", sa.String, nullable=False, server_default=text("'custom'")),
        sa.Column("prompt_template", sa.String, nullable=False),
        sa.Column("score_type", sa.String, nullable=False, server_default=text("'binary'")),
        sa.Column("score_labels", JSON_, nullable=False, server_default=text("'{}'")),
        sa.Column(
            "default_judge_adapter_id",
            sa.Integer,
            sa.ForeignKey("llm_adapters.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("is_builtin", sa.Boolean, nullable=False, server_default=text("0")),
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
    )


def downgrade() -> None:
    op.drop_table("evaluation_criteria")
    op.drop_table("data_generation_jobs")
    op.drop_table("llm_adapters")
