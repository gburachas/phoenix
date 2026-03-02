"""REST API router for LLM Adapter management.

Provides CRUD operations and connectivity testing for registered LLM adapters.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

from phoenix.db import models

from .models import V1RoutesBaseModel
from .utils import ResponseBody, add_errors_to_responses

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-adapters", tags=["llm-adapters"])


class LLMAdapter(V1RoutesBaseModel):
    id: int
    name: str
    provider: str
    model_name: str
    endpoint: Optional[str]
    api_key_env_var: Optional[str]
    can_embed: bool
    can_generate: bool
    can_judge: bool
    can_multimodal: bool
    can_rerank: bool
    cost_per_1k_input_tokens: Optional[float]
    cost_per_1k_output_tokens: Optional[float]
    max_context_tokens: Optional[int]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class CreateLLMAdapterRequest(V1RoutesBaseModel):
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
    metadata: dict[str, Any] = {}


class UpdateLLMAdapterRequest(V1RoutesBaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    endpoint: Optional[str] = None
    api_key_env_var: Optional[str] = None
    can_embed: Optional[bool] = None
    can_generate: Optional[bool] = None
    can_judge: Optional[bool] = None
    can_multimodal: Optional[bool] = None
    can_rerank: Optional[bool] = None
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None
    max_context_tokens: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


class LLMAdapterTestResult(V1RoutesBaseModel):
    adapter_id: int
    reachable: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


def _to_response(adapter: models.LLMAdapter) -> LLMAdapter:
    return LLMAdapter(
        id=adapter.id,
        name=adapter.name,
        provider=adapter.provider,
        model_name=adapter.model_name,
        endpoint=adapter.endpoint,
        api_key_env_var=adapter.api_key_env_var,
        can_embed=adapter.can_embed,
        can_generate=adapter.can_generate,
        can_judge=adapter.can_judge,
        can_multimodal=adapter.can_multimodal,
        can_rerank=adapter.can_rerank,
        cost_per_1k_input_tokens=adapter.cost_per_1k_input_tokens,
        cost_per_1k_output_tokens=adapter.cost_per_1k_output_tokens,
        max_context_tokens=adapter.max_context_tokens,
        metadata=adapter.metadata_ or {},
        created_at=adapter.created_at,
        updated_at=adapter.updated_at,
    )


@router.get(
    "",
    operation_id="listLLMAdapters",
    summary="List LLM adapters",
    responses=add_errors_to_responses([]),
)
async def list_llm_adapters(
    request: Request,
    capability: Optional[str] = Query(
        None,
        description="Filter by capability: embed, generate, judge, multimodal, rerank",
    ),
) -> ResponseBody[list[LLMAdapter]]:
    async with request.app.state.db() as session:
        stmt = select(models.LLMAdapter).order_by(models.LLMAdapter.name)
        if capability:
            cap_col = {
                "embed": models.LLMAdapter.can_embed,
                "generate": models.LLMAdapter.can_generate,
                "judge": models.LLMAdapter.can_judge,
                "multimodal": models.LLMAdapter.can_multimodal,
                "rerank": models.LLMAdapter.can_rerank,
            }.get(capability)
            if cap_col is not None:
                stmt = stmt.where(cap_col.is_(True))
        result = await session.execute(stmt)
        adapters = result.scalars().all()
    return ResponseBody(data=[_to_response(a) for a in adapters])


@router.get(
    "/{adapter_id}",
    operation_id="getLLMAdapter",
    summary="Get an LLM adapter by ID",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def get_llm_adapter(
    request: Request,
    adapter_id: int,
) -> ResponseBody[LLMAdapter]:
    async with request.app.state.db() as session:
        adapter = await session.get(models.LLMAdapter, adapter_id)
    if adapter is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="LLM adapter not found")
    return ResponseBody(data=_to_response(adapter))


@router.post(
    "",
    operation_id="createLLMAdapter",
    summary="Register a new LLM adapter",
    responses=add_errors_to_responses([HTTP_409_CONFLICT]),
)
async def create_llm_adapter(
    request: Request,
    body: CreateLLMAdapterRequest,
) -> ResponseBody[LLMAdapter]:
    async with request.app.state.db() as session:
        # Check for duplicate name
        existing = await session.execute(
            select(models.LLMAdapter).where(models.LLMAdapter.name == body.name)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=HTTP_409_CONFLICT,
                detail=f"LLM adapter with name '{body.name}' already exists",
            )
        adapter = models.LLMAdapter(
            name=body.name,
            provider=body.provider,
            model_name=body.model_name,
            endpoint=body.endpoint,
            api_key_env_var=body.api_key_env_var,
            can_embed=body.can_embed,
            can_generate=body.can_generate,
            can_judge=body.can_judge,
            can_multimodal=body.can_multimodal,
            can_rerank=body.can_rerank,
            cost_per_1k_input_tokens=body.cost_per_1k_input_tokens,
            cost_per_1k_output_tokens=body.cost_per_1k_output_tokens,
            max_context_tokens=body.max_context_tokens,
            metadata_=body.metadata,
        )
        session.add(adapter)
        await session.flush()
        await session.refresh(adapter, ["created_at", "updated_at"])
        response = _to_response(adapter)
    return ResponseBody(data=response)


@router.patch(
    "/{adapter_id}",
    operation_id="updateLLMAdapter",
    summary="Update an existing LLM adapter",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def update_llm_adapter(
    request: Request,
    adapter_id: int,
    body: UpdateLLMAdapterRequest,
) -> ResponseBody[LLMAdapter]:
    async with request.app.state.db() as session:
        adapter = await session.get(models.LLMAdapter, adapter_id)
        if adapter is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="LLM adapter not found"
            )
        update_data = body.model_dump(exclude_none=True)
        if "metadata" in update_data:
            update_data["metadata_"] = update_data.pop("metadata")
        for field, value in update_data.items():
            setattr(adapter, field, value)
        await session.flush()
        await session.refresh(adapter, ["updated_at"])
        response = _to_response(adapter)
    return ResponseBody(data=response)


@router.delete(
    "/{adapter_id}",
    operation_id="deleteLLMAdapter",
    summary="Delete an LLM adapter",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def delete_llm_adapter(
    request: Request,
    adapter_id: int,
) -> ResponseBody[dict[str, bool]]:
    async with request.app.state.db() as session:
        adapter = await session.get(models.LLMAdapter, adapter_id)
        if adapter is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="LLM adapter not found"
            )
        await session.delete(adapter)
    return ResponseBody(data={"deleted": True})


@router.post(
    "/{adapter_id}/test",
    operation_id="testLLMAdapter",
    summary="Test connectivity to an LLM adapter",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def test_llm_adapter(
    request: Request,
    adapter_id: int,
) -> ResponseBody[LLMAdapterTestResult]:
    import time

    import httpx

    async with request.app.state.db() as session:
        adapter = await session.get(models.LLMAdapter, adapter_id)
    if adapter is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="LLM adapter not found")

    start = time.monotonic()
    try:
        if adapter.provider == "ollama":
            url = (adapter.endpoint or "http://localhost:11434").rstrip("/")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/api/tags")
                resp.raise_for_status()
        elif adapter.provider in ("openai", "azure_openai"):
            url = (adapter.endpoint or "https://api.openai.com/v1").rstrip("/")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/models")
                # 401 is OK — it means the endpoint is reachable
        else:
            # Generic HTTP health check
            if adapter.endpoint:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(adapter.endpoint)
            else:
                return ResponseBody(
                    data=LLMAdapterTestResult(
                        adapter_id=adapter_id,
                        reachable=False,
                        error="No endpoint configured",
                    )
                )
        elapsed = (time.monotonic() - start) * 1000
        return ResponseBody(
            data=LLMAdapterTestResult(
                adapter_id=adapter_id,
                reachable=True,
                latency_ms=round(elapsed, 1),
            )
        )
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return ResponseBody(
            data=LLMAdapterTestResult(
                adapter_id=adapter_id,
                reachable=False,
                latency_ms=round(elapsed, 1),
                error=str(e),
            )
        )
