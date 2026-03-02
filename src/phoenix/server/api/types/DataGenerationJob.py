"""Strawberry GraphQL type for DataGenerationJob."""

from datetime import datetime
from typing import Any, Optional

import strawberry
from strawberry.relay import Node, NodeID
from strawberry.scalars import JSON
from strawberry.types import Info

from phoenix.db import models
from phoenix.server.api.context import Context
from phoenix.server.api.types.LLMAdapter import LLMAdapter, to_gql_llm_adapter


@strawberry.type
class DataGenerationJob(Node):
    id: NodeID[int]
    db_record: strawberry.Private[Optional[models.DataGenerationJob]] = None

    @strawberry.field
    async def name(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.name
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.name

    @strawberry.field
    async def status(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.status
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.status

    @strawberry.field
    async def corpus_source(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.corpus_source
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.corpus_source

    @strawberry.field
    async def corpus_config(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.corpus_config
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.corpus_config

    @strawberry.field
    async def sampling_strategy(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.sampling_strategy
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.sampling_strategy

    @strawberry.field
    async def sample_size(self, info: Info[Context, None]) -> int:
        if self.db_record:
            return self.db_record.sample_size
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.sample_size

    @strawberry.field
    async def llm_config(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.llm_config
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.llm_config

    @strawberry.field
    async def is_multimodal(self, info: Info[Context, None]) -> bool:
        if self.db_record:
            return self.db_record.is_multimodal
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.is_multimodal

    @strawberry.field
    async def output_dataset_name(self, info: Info[Context, None]) -> Optional[str]:
        if self.db_record:
            return self.db_record.output_dataset_name
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.output_dataset_name

    @strawberry.field
    async def artifacts(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.artifacts
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.artifacts

    @strawberry.field
    async def error_message(self, info: Info[Context, None]) -> Optional[str]:
        if self.db_record:
            return self.db_record.error_message
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.error_message

    @strawberry.field
    async def seed(self, info: Info[Context, None]) -> Optional[int]:
        if self.db_record:
            return self.db_record.seed
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.seed

    @strawberry.field
    async def created_at(self, info: Info[Context, None]) -> datetime:
        if self.db_record:
            return self.db_record.created_at
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.created_at

    @strawberry.field
    async def started_at(self, info: Info[Context, None]) -> Optional[datetime]:
        if self.db_record:
            return self.db_record.started_at
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.started_at

    @strawberry.field
    async def completed_at(self, info: Info[Context, None]) -> Optional[datetime]:
        if self.db_record:
            return self.db_record.completed_at
        async with info.context.db() as session:
            job = await session.get(models.DataGenerationJob, self.id)
            assert job is not None
            return job.completed_at

    @strawberry.field
    async def testset_llm_adapter(self, info: Info[Context, None]) -> Optional[LLMAdapter]:
        if self.db_record and self.db_record.testset_llm_adapter_id is None:
            return None
        adapter_id = (
            self.db_record.testset_llm_adapter_id if self.db_record else None
        )
        if adapter_id is None:
            async with info.context.db() as session:
                job = await session.get(models.DataGenerationJob, self.id)
                assert job is not None
                adapter_id = job.testset_llm_adapter_id
        if adapter_id is None:
            return None
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, adapter_id)
        return to_gql_llm_adapter(adapter) if adapter else None

    @strawberry.field
    async def transform_llm_adapter(self, info: Info[Context, None]) -> Optional[LLMAdapter]:
        if self.db_record and self.db_record.transform_llm_adapter_id is None:
            return None
        adapter_id = (
            self.db_record.transform_llm_adapter_id if self.db_record else None
        )
        if adapter_id is None:
            async with info.context.db() as session:
                job = await session.get(models.DataGenerationJob, self.id)
                assert job is not None
                adapter_id = job.transform_llm_adapter_id
        if adapter_id is None:
            return None
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, adapter_id)
        return to_gql_llm_adapter(adapter) if adapter else None


def to_gql_data_generation_job(job: models.DataGenerationJob) -> DataGenerationJob:
    return DataGenerationJob(id=job.id, db_record=job)
