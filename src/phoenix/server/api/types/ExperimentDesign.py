"""Strawberry GraphQL type for ExperimentDesign, ExperimentFactor, and ExperimentDesignCell."""

from datetime import datetime
from typing import Any, Optional

import strawberry
from strawberry.relay import Node, NodeID
from strawberry.scalars import JSON
from strawberry.types import Info

from phoenix.db import models
from phoenix.server.api.context import Context


@strawberry.type
class ExperimentDesignCell(Node):
    id: NodeID[int]
    db_record: strawberry.Private[Optional[models.ExperimentDesignCell]] = None

    @strawberry.field
    async def design_id(self, info: Info[Context, None]) -> int:
        if self.db_record:
            return self.db_record.design_id
        async with info.context.db() as session:
            cell = await session.get(models.ExperimentDesignCell, self.id)
            assert cell is not None
            return cell.design_id

    @strawberry.field
    async def combination(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.combination
        async with info.context.db() as session:
            cell = await session.get(models.ExperimentDesignCell, self.id)
            assert cell is not None
            return cell.combination

    @strawberry.field
    async def status(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.status
        async with info.context.db() as session:
            cell = await session.get(models.ExperimentDesignCell, self.id)
            assert cell is not None
            return cell.status

    @strawberry.field
    async def experiment_id(self, info: Info[Context, None]) -> Optional[int]:
        if self.db_record:
            return self.db_record.experiment_id
        async with info.context.db() as session:
            cell = await session.get(models.ExperimentDesignCell, self.id)
            assert cell is not None
            return cell.experiment_id

    @strawberry.field
    async def result_summary(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.result_summary
        async with info.context.db() as session:
            cell = await session.get(models.ExperimentDesignCell, self.id)
            assert cell is not None
            return cell.result_summary

    @strawberry.field
    async def created_at(self, info: Info[Context, None]) -> datetime:
        if self.db_record:
            return self.db_record.created_at
        async with info.context.db() as session:
            cell = await session.get(models.ExperimentDesignCell, self.id)
            assert cell is not None
            return cell.created_at


@strawberry.type
class ExperimentFactor(Node):
    id: NodeID[int]
    db_record: strawberry.Private[Optional[models.ExperimentFactor]] = None

    @strawberry.field
    async def design_id(self, info: Info[Context, None]) -> int:
        if self.db_record:
            return self.db_record.design_id
        async with info.context.db() as session:
            factor = await session.get(models.ExperimentFactor, self.id)
            assert factor is not None
            return factor.design_id

    @strawberry.field
    async def name(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.name
        async with info.context.db() as session:
            factor = await session.get(models.ExperimentFactor, self.id)
            assert factor is not None
            return factor.name

    @strawberry.field
    async def factor_type(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.factor_type
        async with info.context.db() as session:
            factor = await session.get(models.ExperimentFactor, self.id)
            assert factor is not None
            return factor.factor_type

    @strawberry.field
    async def levels(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.levels
        async with info.context.db() as session:
            factor = await session.get(models.ExperimentFactor, self.id)
            assert factor is not None
            return factor.levels

    @strawberry.field
    async def created_at(self, info: Info[Context, None]) -> datetime:
        if self.db_record:
            return self.db_record.created_at
        async with info.context.db() as session:
            factor = await session.get(models.ExperimentFactor, self.id)
            assert factor is not None
            return factor.created_at


@strawberry.type
class ExperimentDesign(Node):
    id: NodeID[int]
    db_record: strawberry.Private[Optional[models.ExperimentDesign]] = None

    @strawberry.field
    async def name(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.name
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return design.name

    @strawberry.field
    async def description(self, info: Info[Context, None]) -> Optional[str]:
        if self.db_record:
            return self.db_record.description
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return design.description

    @strawberry.field
    async def design_type(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.design_type
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return design.design_type

    @strawberry.field
    async def status(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.status
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return design.status

    @strawberry.field
    async def metadata(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.metadata_
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return design.metadata_

    @strawberry.field
    async def created_at(self, info: Info[Context, None]) -> datetime:
        if self.db_record:
            return self.db_record.created_at
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return design.created_at

    @strawberry.field
    async def updated_at(self, info: Info[Context, None]) -> datetime:
        if self.db_record:
            return self.db_record.updated_at
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return design.updated_at

    @strawberry.field
    async def factors(self, info: Info[Context, None]) -> list[ExperimentFactor]:
        if self.db_record and self.db_record.factors is not None:
            return [to_gql_experiment_factor(f) for f in self.db_record.factors]
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return [to_gql_experiment_factor(f) for f in (design.factors or [])]

    @strawberry.field
    async def cells(self, info: Info[Context, None]) -> list[ExperimentDesignCell]:
        if self.db_record and self.db_record.cells is not None:
            return [to_gql_experiment_design_cell(c) for c in self.db_record.cells]
        async with info.context.db() as session:
            design = await session.get(models.ExperimentDesign, self.id)
            assert design is not None
            return [to_gql_experiment_design_cell(c) for c in (design.cells or [])]


def to_gql_experiment_design(design: models.ExperimentDesign) -> ExperimentDesign:
    return ExperimentDesign(id=design.id, db_record=design)


def to_gql_experiment_factor(factor: models.ExperimentFactor) -> ExperimentFactor:
    return ExperimentFactor(id=factor.id, db_record=factor)


def to_gql_experiment_design_cell(cell: models.ExperimentDesignCell) -> ExperimentDesignCell:
    return ExperimentDesignCell(id=cell.id, db_record=cell)
