"""Strawberry GraphQL type for LLMAdapter."""

from datetime import datetime
from typing import Any, Optional

import strawberry
from strawberry import UNSET
from strawberry.relay import Node, NodeID
from strawberry.scalars import JSON
from strawberry.types import Info

from phoenix.db import models
from phoenix.server.api.context import Context


@strawberry.type
class LLMAdapter(Node):
    id: NodeID[int]
    db_record: strawberry.Private[Optional[models.LLMAdapter]] = None

    @strawberry.field
    async def name(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.name
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.name

    @strawberry.field
    async def provider(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.provider
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.provider

    @strawberry.field
    async def model_name(self, info: Info[Context, None]) -> str:
        if self.db_record:
            return self.db_record.model_name
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.model_name

    @strawberry.field
    async def endpoint(self, info: Info[Context, None]) -> Optional[str]:
        if self.db_record:
            return self.db_record.endpoint
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.endpoint

    @strawberry.field
    async def can_embed(self, info: Info[Context, None]) -> bool:
        if self.db_record:
            return self.db_record.can_embed
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.can_embed

    @strawberry.field
    async def can_generate(self, info: Info[Context, None]) -> bool:
        if self.db_record:
            return self.db_record.can_generate
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.can_generate

    @strawberry.field
    async def can_judge(self, info: Info[Context, None]) -> bool:
        if self.db_record:
            return self.db_record.can_judge
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.can_judge

    @strawberry.field
    async def can_multimodal(self, info: Info[Context, None]) -> bool:
        if self.db_record:
            return self.db_record.can_multimodal
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.can_multimodal

    @strawberry.field
    async def can_rerank(self, info: Info[Context, None]) -> bool:
        if self.db_record:
            return self.db_record.can_rerank
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.can_rerank

    @strawberry.field
    async def cost_per_1k_input_tokens(self, info: Info[Context, None]) -> Optional[float]:
        if self.db_record:
            return self.db_record.cost_per_1k_input_tokens
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.cost_per_1k_input_tokens

    @strawberry.field
    async def cost_per_1k_output_tokens(self, info: Info[Context, None]) -> Optional[float]:
        if self.db_record:
            return self.db_record.cost_per_1k_output_tokens
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.cost_per_1k_output_tokens

    @strawberry.field
    async def max_context_tokens(self, info: Info[Context, None]) -> Optional[int]:
        if self.db_record:
            return self.db_record.max_context_tokens
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.max_context_tokens

    @strawberry.field
    async def metadata(self, info: Info[Context, None]) -> JSON:
        if self.db_record:
            return self.db_record.metadata_
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.metadata_

    @strawberry.field
    async def created_at(self, info: Info[Context, None]) -> datetime:
        if self.db_record:
            return self.db_record.created_at
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.created_at

    @strawberry.field
    async def updated_at(self, info: Info[Context, None]) -> datetime:
        if self.db_record:
            return self.db_record.updated_at
        async with info.context.db() as session:
            adapter = await session.get(models.LLMAdapter, self.id)
            assert adapter is not None
            return adapter.updated_at


def to_gql_llm_adapter(adapter: models.LLMAdapter) -> LLMAdapter:
    return LLMAdapter(id=adapter.id, db_record=adapter)
