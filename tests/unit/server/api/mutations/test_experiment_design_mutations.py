"""Unit tests for Experiment Design GraphQL mutations.

Tests the ExperimentDesignMutationMixin: createExperimentDesign,
deleteExperimentDesign, addExperimentFactor, removeExperimentFactor,
generateExperimentCells, runExperimentDesign.
"""

from typing import Any

import pytest
from sqlalchemy import insert, select
from strawberry.relay import GlobalID

from phoenix.db import models
from phoenix.server.types import DbSessionFactory
from tests.unit.graphql import AsyncGraphQLClient


# ── helpers ──────────────────────────────────────────────────────────────────


async def _insert_design(
    db: DbSessionFactory,
    *,
    name: str = "test-design",
    design_type: str = "full_factorial",
    status: str = "draft",
) -> int:
    async with db() as session:
        design = models.ExperimentDesign(
            name=name,
            design_type=design_type,
            status=status,
            metadata_={},
        )
        session.add(design)
        await session.flush()
        return design.id


async def _insert_factor(
    db: DbSessionFactory,
    *,
    design_id: int,
    name: str = "embedding",
    factor_type: str = "embedding",
    levels: list[Any] | None = None,
) -> int:
    async with db() as session:
        factor = models.ExperimentFactor(
            design_id=design_id,
            name=name,
            factor_type=factor_type,
            levels=levels if levels is not None else [{"name": "a"}, {"name": "b"}],
        )
        session.add(factor)
        await session.flush()
        return factor.id


def _gid(type_name: str, db_id: int) -> str:
    return str(GlobalID(type_name=type_name, node_id=str(db_id)))


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCreateExperimentDesignMutation:
    _MUTATION = """
        mutation CreateExperimentDesign($input: CreateExperimentDesignInput!) {
            createExperimentDesign(input: $input) {
                design {
                    id
                    name
                    designType
                    status
                }
            }
        }
    """

    async def test_create_design(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "name": "GQL Design",
                    "designType": "full_factorial",
                }
            },
        )
        assert not result.errors
        design = result.data["createExperimentDesign"]["design"]
        assert design["name"] == "GQL Design"
        assert design["designType"] == "full_factorial"
        assert design["status"] == "draft"

    async def test_create_invalid_type(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "name": "Bad",
                    "designType": "invalid_type",
                }
            },
        )
        assert result.errors


class TestDeleteExperimentDesignMutation:
    _MUTATION = """
        mutation DeleteExperimentDesign($input: DeleteExperimentDesignInput!) {
            deleteExperimentDesign(input: $input) {
                design {
                    id
                    name
                }
            }
        }
    """

    async def test_delete_existing(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        design_id = await _insert_design(db, name="delete-me")
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "designId": _gid("ExperimentDesign", design_id),
                }
            },
        )
        assert not result.errors
        assert result.data["deleteExperimentDesign"]["design"]["name"] == "delete-me"

    async def test_delete_not_found(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "designId": _gid("ExperimentDesign", 99999),
                }
            },
        )
        assert result.errors


class TestAddExperimentFactorMutation:
    _MUTATION = """
        mutation AddExperimentFactor($input: AddExperimentFactorInput!) {
            addExperimentFactor(input: $input) {
                factor {
                    id
                    name
                    factorType
                    levels
                }
            }
        }
    """

    async def test_add_factor(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        design_id = await _insert_design(db)
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "designId": _gid("ExperimentDesign", design_id),
                    "name": "embedding",
                    "factorType": "embedding",
                    "levels": [{"name": "ada"}, {"name": "small"}],
                }
            },
        )
        assert not result.errors
        factor = result.data["addExperimentFactor"]["factor"]
        assert factor["name"] == "embedding"
        assert factor["factorType"] == "embedding"
        assert len(factor["levels"]) == 2


class TestRemoveExperimentFactorMutation:
    _MUTATION = """
        mutation RemoveExperimentFactor($input: RemoveExperimentFactorInput!) {
            removeExperimentFactor(input: $input) {
                factor {
                    id
                    name
                }
            }
        }
    """

    async def test_remove_factor(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        design_id = await _insert_design(db)
        factor_id = await _insert_factor(db, design_id=design_id)
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "factorId": _gid("ExperimentFactor", factor_id),
                }
            },
        )
        assert not result.errors


class TestGenerateExperimentCellsMutation:
    _MUTATION = """
        mutation GenerateExperimentCells($input: GenerateExperimentCellsInput!) {
            generateExperimentCells(input: $input) {
                cells {
                    id
                    combination
                    status
                }
                design {
                    id
                    status
                }
            }
        }
    """

    async def test_generate_2x2_gives_4_cells(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        design_id = await _insert_design(db)
        await _insert_factor(
            db,
            design_id=design_id,
            name="embedding",
            levels=[{"name": "a"}, {"name": "b"}],
        )
        await _insert_factor(
            db,
            design_id=design_id,
            name="reranker",
            factor_type="reranker",
            levels=[{"name": "x"}, {"name": "y"}],
        )
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "designId": _gid("ExperimentDesign", design_id),
                }
            },
        )
        assert not result.errors
        cells = result.data["generateExperimentCells"]["cells"]
        assert len(cells) == 4
        # Design status updated
        design = result.data["generateExperimentCells"]["design"]
        assert design["status"] == "cells_generated"

    async def test_generate_no_factors_fails(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        design_id = await _insert_design(db)
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "designId": _gid("ExperimentDesign", design_id),
                }
            },
        )
        assert result.errors


class TestRunExperimentDesignMutation:
    _RUN_MUTATION = """
        mutation RunExperimentDesign($input: RunExperimentDesignInput!) {
            runExperimentDesign(input: $input) {
                design {
                    id
                    status
                }
            }
        }
    """

    _GENERATE_MUTATION = """
        mutation GenerateExperimentCells($input: GenerateExperimentCellsInput!) {
            generateExperimentCells(input: $input) {
                cells { id }
                design { id status }
            }
        }
    """

    async def test_run_after_generate(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        design_id = await _insert_design(db)
        await _insert_factor(db, design_id=design_id)
        # Generate first
        await gql_client.execute(
            self._GENERATE_MUTATION,
            variables={
                "input": {"designId": _gid("ExperimentDesign", design_id)}
            },
        )
        # Then run
        result = await gql_client.execute(
            self._RUN_MUTATION,
            variables={
                "input": {"designId": _gid("ExperimentDesign", design_id)}
            },
        )
        assert not result.errors
        assert result.data["runExperimentDesign"]["design"]["status"] == "running"

    async def test_run_draft_fails(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        design_id = await _insert_design(db, status="draft")
        result = await gql_client.execute(
            self._RUN_MUTATION,
            variables={
                "input": {"designId": _gid("ExperimentDesign", design_id)}
            },
        )
        assert result.errors
