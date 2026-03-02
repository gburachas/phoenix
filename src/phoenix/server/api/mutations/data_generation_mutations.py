"""Strawberry GraphQL mutations for LLM Adapter and Data Generation."""

from typing import Optional

import strawberry
from sqlalchemy import insert, select, update
from strawberry import UNSET
from strawberry.relay import GlobalID
from strawberry.scalars import JSON
from strawberry.types import Info

from phoenix.db import models
from phoenix.server.api.auth import IsNotReadOnly, IsNotViewer
from phoenix.server.api.context import Context
from phoenix.server.api.exceptions import BadRequest, NotFound
from phoenix.server.api.types.DataGenerationJob import (
    DataGenerationJob,
    to_gql_data_generation_job,
)
from phoenix.server.api.types.LLMAdapter import LLMAdapter, to_gql_llm_adapter
from phoenix.server.api.types.node import from_global_id_with_expected_type


# --- Input types ---


@strawberry.input
class CreateLLMAdapterInput:
    name: str
    provider: str
    model_name: str
    endpoint: Optional[str] = None
    api_key_env_var: Optional[str] = None
    can_embed: bool = False
    can_generate: bool = False
    can_judge: bool = False
    can_multimodal: bool = False
    can_rerank: bool = False
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None
    max_context_tokens: Optional[int] = None
    metadata: JSON = strawberry.field(default_factory=dict)


@strawberry.input
class PatchLLMAdapterInput:
    adapter_id: GlobalID
    name: Optional[str] = UNSET
    provider: Optional[str] = UNSET
    model_name: Optional[str] = UNSET
    endpoint: Optional[str] = UNSET
    can_embed: Optional[bool] = UNSET
    can_generate: Optional[bool] = UNSET
    can_judge: Optional[bool] = UNSET
    can_multimodal: Optional[bool] = UNSET
    can_rerank: Optional[bool] = UNSET


@strawberry.input
class DeleteLLMAdapterInput:
    adapter_id: GlobalID


@strawberry.input
class CreateDataGenerationJobInput:
    name: str
    corpus_source: str
    corpus_config: JSON = strawberry.field(default_factory=dict)
    sampling_strategy: str = "random"
    sample_size: int = 50
    testset_llm_adapter_id: Optional[GlobalID] = None
    transform_llm_adapter_id: Optional[GlobalID] = None
    llm_config: JSON = strawberry.field(default_factory=dict)
    is_multimodal: bool = False
    output_dataset_name: Optional[str] = None
    seed: Optional[int] = None


@strawberry.input
class CancelDataGenerationJobInput:
    job_id: GlobalID


@strawberry.input
class DeleteDataGenerationJobInput:
    job_id: GlobalID


# --- Payload types ---


@strawberry.type
class LLMAdapterMutationPayload:
    adapter: LLMAdapter


@strawberry.type
class DataGenerationJobMutationPayload:
    job: DataGenerationJob


# --- Mutation mixin ---


@strawberry.type
class DataGenerationMutationMixin:
    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def create_llm_adapter(
        self,
        info: Info[Context, None],
        input: CreateLLMAdapterInput,
    ) -> LLMAdapterMutationPayload:
        async with info.context.db() as session:
            # Check for duplicate name
            existing = await session.scalar(
                select(models.LLMAdapter).where(models.LLMAdapter.name == input.name)
            )
            if existing:
                raise BadRequest(f"LLM adapter with name '{input.name}' already exists")
            adapter = await session.scalar(
                insert(models.LLMAdapter)
                .values(
                    name=input.name,
                    provider=input.provider,
                    model_name=input.model_name,
                    endpoint=input.endpoint,
                    api_key_env_var=input.api_key_env_var,
                    can_embed=input.can_embed,
                    can_generate=input.can_generate,
                    can_judge=input.can_judge,
                    can_multimodal=input.can_multimodal,
                    can_rerank=input.can_rerank,
                    cost_per_1k_input_tokens=input.cost_per_1k_input_tokens,
                    cost_per_1k_output_tokens=input.cost_per_1k_output_tokens,
                    max_context_tokens=input.max_context_tokens,
                    metadata_=input.metadata or {},
                )
                .returning(models.LLMAdapter)
            )
            assert adapter is not None
        return LLMAdapterMutationPayload(adapter=to_gql_llm_adapter(adapter))

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def patch_llm_adapter(
        self,
        info: Info[Context, None],
        input: PatchLLMAdapterInput,
    ) -> LLMAdapterMutationPayload:
        adapter_id = from_global_id_with_expected_type(
            global_id=input.adapter_id, expected_type_name=LLMAdapter.__name__
        )
        patch = {}
        for attr, col in (
            ("name", "name"),
            ("provider", "provider"),
            ("model_name", "model_name"),
            ("endpoint", "endpoint"),
            ("can_embed", "can_embed"),
            ("can_generate", "can_generate"),
            ("can_judge", "can_judge"),
            ("can_multimodal", "can_multimodal"),
            ("can_rerank", "can_rerank"),
        ):
            val = getattr(input, attr)
            if val is not UNSET:
                patch[col] = val
        if not patch:
            raise BadRequest("No fields to update")
        async with info.context.db() as session:
            adapter = await session.scalar(
                update(models.LLMAdapter)
                .where(models.LLMAdapter.id == adapter_id)
                .returning(models.LLMAdapter)
                .values(**patch)
            )
            if adapter is None:
                raise NotFound(f"LLM adapter with id {adapter_id} not found")
        return LLMAdapterMutationPayload(adapter=to_gql_llm_adapter(adapter))

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def delete_llm_adapter(
        self,
        info: Info[Context, None],
        input: DeleteLLMAdapterInput,
    ) -> LLMAdapterMutationPayload:
        adapter_id = from_global_id_with_expected_type(
            global_id=input.adapter_id, expected_type_name=LLMAdapter.__name__
        )
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, adapter_id)
            if adapter is None:
                raise NotFound(f"LLM adapter with id {adapter_id} not found")
            gql_adapter = to_gql_llm_adapter(adapter)
            await session.delete(adapter)
        return LLMAdapterMutationPayload(adapter=gql_adapter)

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def create_data_generation_job(
        self,
        info: Info[Context, None],
        input: CreateDataGenerationJobInput,
    ) -> DataGenerationJobMutationPayload:
        testset_id = None
        if input.testset_llm_adapter_id:
            testset_id = from_global_id_with_expected_type(
                global_id=input.testset_llm_adapter_id, expected_type_name=LLMAdapter.__name__
            )
        transform_id = None
        if input.transform_llm_adapter_id:
            transform_id = from_global_id_with_expected_type(
                global_id=input.transform_llm_adapter_id, expected_type_name=LLMAdapter.__name__
            )
        async with info.context.db() as session:
            job = await session.scalar(
                insert(models.DataGenerationJob)
                .values(
                    name=input.name,
                    corpus_source=input.corpus_source,
                    corpus_config=input.corpus_config or {},
                    sampling_strategy=input.sampling_strategy,
                    sample_size=input.sample_size,
                    testset_llm_adapter_id=testset_id,
                    transform_llm_adapter_id=transform_id,
                    llm_config=input.llm_config or {},
                    is_multimodal=input.is_multimodal,
                    output_dataset_name=input.output_dataset_name,
                    seed=input.seed,
                    artifacts={},
                )
                .returning(models.DataGenerationJob)
            )
            assert job is not None
        return DataGenerationJobMutationPayload(job=to_gql_data_generation_job(job))

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def cancel_data_generation_job(
        self,
        info: Info[Context, None],
        input: CancelDataGenerationJobInput,
    ) -> DataGenerationJobMutationPayload:
        job_id = from_global_id_with_expected_type(
            global_id=input.job_id, expected_type_name=DataGenerationJob.__name__
        )
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, job_id)
            if job is None:
                raise NotFound(f"Data generation job with id {job_id} not found")
            if job.status not in ("pending", "running"):
                raise BadRequest(f"Cannot cancel job with status '{job.status}'")
            job = await session.scalar(
                update(models.DataGenerationJob)
                .where(models.DataGenerationJob.id == job_id)
                .values(status="cancelled")
                .returning(models.DataGenerationJob)
            )
            assert job is not None
        return DataGenerationJobMutationPayload(job=to_gql_data_generation_job(job))

    @strawberry.mutation(permission_classes=[IsNotReadOnly, IsNotViewer])  # type: ignore
    async def delete_data_generation_job(
        self,
        info: Info[Context, None],
        input: DeleteDataGenerationJobInput,
    ) -> DataGenerationJobMutationPayload:
        job_id = from_global_id_with_expected_type(
            global_id=input.job_id, expected_type_name=DataGenerationJob.__name__
        )
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, job_id)
            if job is None:
                raise NotFound(f"Data generation job with id {job_id} not found")
            gql_job = to_gql_data_generation_job(job)
            await session.delete(job)
        return DataGenerationJobMutationPayload(job=gql_job)
