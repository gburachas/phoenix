"""Unit tests for Data Generation GraphQL mutations.

Tests the DataGenerationMutationMixin: createLlmAdapter, patchLlmAdapter,
deleteLlmAdapter, createDataGenerationJob, cancelDataGenerationJob,
deleteDataGenerationJob.
"""

from typing import Any

import pytest
from sqlalchemy import insert, select
from strawberry.relay import GlobalID

from phoenix.db import models
from phoenix.server.types import DbSessionFactory
from tests.unit.graphql import AsyncGraphQLClient


class TestCreateLLMAdapterMutation:
    _MUTATION = """
        mutation CreateLLMAdapter($input: CreateLLMAdapterInput!) {
            createLlmAdapter(input: $input) {
                adapter {
                    id
                    name
                    provider
                    modelName
                    canGenerate
                    canJudge
                }
            }
        }
    """

    async def test_create_adapter(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "name": "gql-adapter",
                    "provider": "openai",
                    "modelName": "gpt-4o",
                    "canGenerate": True,
                    "canJudge": True,
                }
            },
        )
        assert not result.errors
        adapter = result.data["createLlmAdapter"]["adapter"]
        assert adapter["name"] == "gql-adapter"
        assert adapter["provider"] == "openai"
        assert adapter["modelName"] == "gpt-4o"
        assert adapter["canGenerate"] is True
        assert adapter["canJudge"] is True

    async def test_duplicate_name_returns_error(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        # Create the first adapter
        await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "name": "dup-name",
                    "provider": "openai",
                    "modelName": "gpt-4o",
                }
            },
        )
        # Try to create a second with the same name
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "name": "dup-name",
                    "provider": "azure_openai",
                    "modelName": "gpt-4o-mini",
                }
            },
        )
        assert result.errors


class TestPatchLLMAdapterMutation:
    _MUTATION = """
        mutation PatchLLMAdapter($input: PatchLLMAdapterInput!) {
            patchLlmAdapter(input: $input) {
                adapter {
                    id
                    name
                    canJudge
                }
            }
        }
    """

    async def test_patch_adapter(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        # Insert directly into DB
        async with db() as session:
            adapter_id = await session.scalar(
                insert(models.LLMAdapter)
                .values(
                    name="before",
                    provider="openai",
                    model_name="gpt-4o",
                    metadata_={},
                )
                .returning(models.LLMAdapter.id)
            )

        global_id = str(GlobalID("LLMAdapter", str(adapter_id)))
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "adapterId": global_id,
                    "name": "after",
                    "canJudge": True,
                }
            },
        )
        assert not result.errors
        adapter = result.data["patchLlmAdapter"]["adapter"]
        assert adapter["name"] == "after"
        assert adapter["canJudge"] is True

    async def test_patch_nonexistent_returns_error(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        global_id = str(GlobalID("LLMAdapter", "99999"))
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "adapterId": global_id,
                    "name": "x",
                }
            },
        )
        assert result.errors


class TestDeleteLLMAdapterMutation:
    _MUTATION = """
        mutation DeleteLLMAdapter($input: DeleteLLMAdapterInput!) {
            deleteLlmAdapter(input: $input) {
                adapter {
                    id
                    name
                }
            }
        }
    """

    async def test_delete_adapter(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        async with db() as session:
            adapter_id = await session.scalar(
                insert(models.LLMAdapter)
                .values(
                    name="delete-me",
                    provider="openai",
                    model_name="gpt-4o",
                    metadata_={},
                )
                .returning(models.LLMAdapter.id)
            )

        global_id = str(GlobalID("LLMAdapter", str(adapter_id)))
        result = await gql_client.execute(
            self._MUTATION,
            variables={"input": {"adapterId": global_id}},
        )
        assert not result.errors
        assert result.data["deleteLlmAdapter"]["adapter"]["name"] == "delete-me"

        # Verify deletion
        async with db() as session:
            remaining = await session.get(models.LLMAdapter, adapter_id)
        assert remaining is None


class TestCreateDataGenerationJobMutation:
    _MUTATION = """
        mutation CreateDataGenerationJob($input: CreateDataGenerationJobInput!) {
            createDataGenerationJob(input: $input) {
                job {
                    id
                    name
                    status
                    corpusSource
                    samplingStrategy
                    sampleSize
                    seed
                }
            }
        }
    """

    async def test_create_job(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        result = await gql_client.execute(
            self._MUTATION,
            variables={
                "input": {
                    "name": "gql-job",
                    "corpusSource": "test-corpus",
                    "samplingStrategy": "random",
                    "sampleSize": 25,
                    "seed": 42,
                }
            },
        )
        assert not result.errors
        job = result.data["createDataGenerationJob"]["job"]
        assert job["name"] == "gql-job"
        assert job["status"] == "pending"
        assert job["corpusSource"] == "test-corpus"
        assert job["sampleSize"] == 25
        assert job["seed"] == 42


class TestCancelDataGenerationJobMutation:
    _MUTATION = """
        mutation CancelDataGenerationJob($input: CancelDataGenerationJobInput!) {
            cancelDataGenerationJob(input: $input) {
                job {
                    id
                    status
                }
            }
        }
    """

    async def test_cancel_pending_job(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        async with db() as session:
            job_id = await session.scalar(
                insert(models.DataGenerationJob)
                .values(
                    name="cancel-me",
                    status="pending",
                    corpus_source="test",
                    corpus_config={},
                    sampling_strategy="random",
                    sample_size=10,
                    llm_config={},
                    artifacts={},
                )
                .returning(models.DataGenerationJob.id)
            )

        global_id = str(GlobalID("DataGenerationJob", str(job_id)))
        result = await gql_client.execute(
            self._MUTATION,
            variables={"input": {"jobId": global_id}},
        )
        assert not result.errors
        assert result.data["cancelDataGenerationJob"]["job"]["status"] == "cancelled"

    async def test_cancel_completed_job_returns_error(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        async with db() as session:
            job_id = await session.scalar(
                insert(models.DataGenerationJob)
                .values(
                    name="done-job",
                    status="completed",
                    corpus_source="test",
                    corpus_config={},
                    sampling_strategy="random",
                    sample_size=10,
                    llm_config={},
                    artifacts={},
                )
                .returning(models.DataGenerationJob.id)
            )

        global_id = str(GlobalID("DataGenerationJob", str(job_id)))
        result = await gql_client.execute(
            self._MUTATION,
            variables={"input": {"jobId": global_id}},
        )
        assert result.errors


class TestDeleteDataGenerationJobMutation:
    _MUTATION = """
        mutation DeleteDataGenerationJob($input: DeleteDataGenerationJobInput!) {
            deleteDataGenerationJob(input: $input) {
                job {
                    id
                    name
                }
            }
        }
    """

    async def test_delete_job(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        async with db() as session:
            job_id = await session.scalar(
                insert(models.DataGenerationJob)
                .values(
                    name="delete-me",
                    status="pending",
                    corpus_source="test",
                    corpus_config={},
                    sampling_strategy="random",
                    sample_size=10,
                    llm_config={},
                    artifacts={},
                )
                .returning(models.DataGenerationJob.id)
            )

        global_id = str(GlobalID("DataGenerationJob", str(job_id)))
        result = await gql_client.execute(
            self._MUTATION,
            variables={"input": {"jobId": global_id}},
        )
        assert not result.errors
        assert result.data["deleteDataGenerationJob"]["job"]["name"] == "delete-me"

        # Verify deletion
        async with db() as session:
            remaining = await session.get(models.DataGenerationJob, job_id)
        assert remaining is None

    async def test_delete_nonexistent_returns_error(
        self,
        db: DbSessionFactory,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        global_id = str(GlobalID("DataGenerationJob", "99999"))
        result = await gql_client.execute(
            self._MUTATION,
            variables={"input": {"jobId": global_id}},
        )
        assert result.errors
