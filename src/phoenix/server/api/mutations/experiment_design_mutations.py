"""Strawberry GraphQL mutations for Experiment Design management."""

import itertools
from typing import Any, Optional

import strawberry
from sqlalchemy import delete, select
from strawberry.relay import GlobalID
from strawberry.scalars import JSON
from strawberry.types import Info

from phoenix.db import models
from phoenix.server.api.auth import IsNotReadOnly, IsNotViewer
from phoenix.server.api.context import Context
from phoenix.server.api.exceptions import BadRequest, NotFound
from phoenix.server.api.types.ExperimentDesign import (
    ExperimentDesign,
    ExperimentDesignCell,
    ExperimentFactor,
    to_gql_experiment_design,
    to_gql_experiment_design_cell,
    to_gql_experiment_factor,
)
from phoenix.server.api.types.node import from_global_id_with_expected_type


# --- Input types ---


@strawberry.input
class CreateExperimentDesignInput:
    name: str
    description: Optional[str] = None
    design_type: str = "full_factorial"
    metadata: JSON = strawberry.field(default_factory=dict)


@strawberry.input
class DeleteExperimentDesignInput:
    design_id: GlobalID


@strawberry.input
class AddExperimentFactorInput:
    design_id: GlobalID
    name: str
    factor_type: str = "custom"
    levels: JSON = strawberry.field(default_factory=list)


@strawberry.input
class RemoveExperimentFactorInput:
    factor_id: GlobalID


@strawberry.input
class GenerateExperimentCellsInput:
    design_id: GlobalID


@strawberry.input
class RunExperimentDesignInput:
    design_id: GlobalID


# --- Payload types ---


@strawberry.type
class ExperimentDesignMutationPayload:
    design: ExperimentDesign


@strawberry.type
class ExperimentFactorMutationPayload:
    factor: ExperimentFactor


@strawberry.type
class GenerateCellsPayload:
    cells: list[ExperimentDesignCell]
    design: ExperimentDesign


# --- Mutation mixin ---


@strawberry.type
class ExperimentDesignMutationMixin:
    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def create_experiment_design(
        self,
        info: Info[Context, None],
        input: CreateExperimentDesignInput,
    ) -> ExperimentDesignMutationPayload:
        valid_types = ("full_factorial", "fractional", "custom")
        if input.design_type not in valid_types:
            raise BadRequest(f"design_type must be one of {valid_types}")

        async with info.context.db() as session:
            design = models.ExperimentDesign(
                name=input.name,
                description=input.description,
                design_type=input.design_type,
                metadata_=input.metadata or {},
            )
            session.add(design)
            await session.flush()
            await session.refresh(design, ["created_at", "updated_at"])
            return ExperimentDesignMutationPayload(
                design=to_gql_experiment_design(design)
            )

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def delete_experiment_design(
        self,
        info: Info[Context, None],
        input: DeleteExperimentDesignInput,
    ) -> ExperimentDesignMutationPayload:
        design_id = from_global_id_with_expected_type(
            input.design_id, ExperimentDesign.__name__
        )
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, int(design_id))
            if design is None:
                raise NotFound(f"ExperimentDesign {design_id} not found")
            gql_design = to_gql_experiment_design(design)
            await session.delete(design)
            await session.flush()
            return ExperimentDesignMutationPayload(design=gql_design)

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def add_experiment_factor(
        self,
        info: Info[Context, None],
        input: AddExperimentFactorInput,
    ) -> ExperimentFactorMutationPayload:
        design_id = from_global_id_with_expected_type(
            input.design_id, ExperimentDesign.__name__
        )
        valid_types = ("embedding", "reranker", "judge_llm", "rag_llm", "testset_llm", "custom")
        if input.factor_type not in valid_types:
            raise BadRequest(f"factor_type must be one of {valid_types}")

        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, int(design_id))
            if design is None:
                raise NotFound(f"ExperimentDesign {design_id} not found")
            factor = models.ExperimentFactor(
                design_id=int(design_id),
                name=input.name,
                factor_type=input.factor_type,
                levels=input.levels or [],
            )
            session.add(factor)
            await session.flush()
            await session.refresh(factor, ["created_at"])
            return ExperimentFactorMutationPayload(
                factor=to_gql_experiment_factor(factor)
            )

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def remove_experiment_factor(
        self,
        info: Info[Context, None],
        input: RemoveExperimentFactorInput,
    ) -> ExperimentFactorMutationPayload:
        factor_id = from_global_id_with_expected_type(
            input.factor_id, ExperimentFactor.__name__
        )
        async with info.context.db() as session:
            factor = await session.get(models.ExperimentFactor, int(factor_id))
            if factor is None:
                raise NotFound(f"ExperimentFactor {factor_id} not found")
            gql_factor = to_gql_experiment_factor(factor)
            await session.delete(factor)
            await session.flush()
            return ExperimentFactorMutationPayload(factor=gql_factor)

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def generate_experiment_cells(
        self,
        info: Info[Context, None],
        input: GenerateExperimentCellsInput,
    ) -> GenerateCellsPayload:
        design_id = from_global_id_with_expected_type(
            input.design_id, ExperimentDesign.__name__
        )
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, int(design_id))
            if design is None:
                raise NotFound(f"ExperimentDesign {design_id} not found")

            if not design.factors:
                raise BadRequest("Design must have at least one factor with levels")

            for f in design.factors:
                if not f.levels:
                    raise BadRequest(f"Factor '{f.name}' has no levels defined")

            # Clear existing cells
            await session.execute(
                delete(models.ExperimentDesignCell).where(
                    models.ExperimentDesignCell.design_id == int(design_id)
                )
            )
            await session.flush()

            # Compute full factorial
            factor_names = [f.name for f in design.factors]
            factor_levels = [f.levels for f in design.factors]

            cells = []
            for combo in itertools.product(*factor_levels):
                combination = dict(zip(factor_names, combo))
                cell = models.ExperimentDesignCell(
                    design_id=int(design_id),
                    combination=combination,
                    result_summary={},
                )
                session.add(cell)
                cells.append(cell)

            await session.flush()
            for cell in cells:
                await session.refresh(cell, ["created_at"])

            design.status = "cells_generated"
            await session.flush()
            await session.refresh(design, ["updated_at"])

            return GenerateCellsPayload(
                cells=[to_gql_experiment_design_cell(c) for c in cells],
                design=to_gql_experiment_design(design),
            )

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def run_experiment_design(
        self,
        info: Info[Context, None],
        input: RunExperimentDesignInput,
    ) -> ExperimentDesignMutationPayload:
        design_id = from_global_id_with_expected_type(
            input.design_id, ExperimentDesign.__name__
        )
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, int(design_id))
            if design is None:
                raise NotFound(f"ExperimentDesign {design_id} not found")

            if design.status not in ("cells_generated", "failed"):
                raise BadRequest(
                    f"Cannot run design in status '{design.status}'. Generate cells first."
                )

            if not design.cells:
                raise BadRequest("No cells to run. Generate cells first.")

            design.status = "running"
            for cell in design.cells:
                if cell.status != "completed":
                    cell.status = "pending"
            await session.flush()
            await session.refresh(design, ["updated_at"])

            return ExperimentDesignMutationPayload(
                design=to_gql_experiment_design(design)
            )
