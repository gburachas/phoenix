"""Unit tests for the LLM Adapters REST API router (/v1/llm-adapters)."""

from typing import Any

import httpx
import pytest

from phoenix.db import models
from phoenix.server.types import DbSessionFactory


async def _insert_adapter(
    db: DbSessionFactory,
    *,
    name: str = "test-adapter",
    provider: str = "openai",
    model_name: str = "gpt-4o",
    can_generate: bool = True,
    can_judge: bool = False,
    can_embed: bool = False,
    **kwargs: Any,
) -> int:
    async with db() as session:
        adapter = models.LLMAdapter(
            name=name,
            provider=provider,
            model_name=model_name,
            can_generate=can_generate,
            can_judge=can_judge,
            can_embed=can_embed,
            can_multimodal=kwargs.get("can_multimodal", False),
            can_rerank=kwargs.get("can_rerank", False),
            metadata_=kwargs.get("metadata_", {}),
        )
        session.add(adapter)
        await session.flush()
        return adapter.id


class TestListLLMAdapters:
    async def test_empty_list(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.get("v1/llm-adapters")
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []

    async def test_returns_all_adapters(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        await _insert_adapter(db, name="adapter-a", provider="openai", model_name="gpt-4o")
        await _insert_adapter(db, name="adapter-b", provider="ollama", model_name="llama3")

        response = await httpx_client.get("v1/llm-adapters")
        assert response.status_code == 200
        adapters = response.json()["data"]
        assert len(adapters) == 2
        names = {a["name"] for a in adapters}
        assert names == {"adapter-a", "adapter-b"}

    async def test_filter_by_capability(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        await _insert_adapter(
            db, name="gen-only", can_generate=True, can_judge=False, can_embed=False
        )
        await _insert_adapter(
            db, name="judge-only", can_generate=False, can_judge=True, can_embed=False
        )

        response = await httpx_client.get("v1/llm-adapters?capability=judge")
        assert response.status_code == 200
        adapters = response.json()["data"]
        assert len(adapters) == 1
        assert adapters[0]["name"] == "judge-only"


class TestGetLLMAdapter:
    async def test_get_existing_adapter(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        adapter_id = await _insert_adapter(db, name="my-adapter", provider="openai")

        response = await httpx_client.get(f"v1/llm-adapters/{adapter_id}")
        assert response.status_code == 200
        adapter = response.json()["data"]
        assert adapter["name"] == "my-adapter"
        assert adapter["provider"] == "openai"

    async def test_get_nonexistent_adapter(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.get("v1/llm-adapters/99999")
        assert response.status_code == 404


class TestCreateLLMAdapter:
    async def test_create_adapter(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        body = {
            "name": "new-adapter",
            "provider": "azure_openai",
            "model_name": "gpt-4o-mini",
            "can_generate": True,
            "can_judge": True,
        }
        response = await httpx_client.post("v1/llm-adapters", json=body)
        assert response.status_code == 200
        adapter = response.json()["data"]
        assert adapter["name"] == "new-adapter"
        assert adapter["provider"] == "azure_openai"
        assert adapter["model_name"] == "gpt-4o-mini"
        assert adapter["can_generate"] is True
        assert adapter["can_judge"] is True
        assert adapter["id"] > 0

    async def test_create_duplicate_name_returns_409(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        await _insert_adapter(db, name="unique-name")

        body = {
            "name": "unique-name",
            "provider": "openai",
            "model_name": "gpt-4o",
        }
        response = await httpx_client.post("v1/llm-adapters", json=body)
        assert response.status_code == 409


class TestUpdateLLMAdapter:
    async def test_update_adapter(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        adapter_id = await _insert_adapter(db, name="before-update")

        body = {"name": "after-update", "can_judge": True}
        response = await httpx_client.patch(f"v1/llm-adapters/{adapter_id}", json=body)
        assert response.status_code == 200
        adapter = response.json()["data"]
        assert adapter["name"] == "after-update"
        assert adapter["can_judge"] is True

    async def test_update_nonexistent_returns_404(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.patch("v1/llm-adapters/99999", json={"name": "x"})
        assert response.status_code == 404


class TestDeleteLLMAdapter:
    async def test_delete_adapter(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        adapter_id = await _insert_adapter(db, name="to-delete")

        response = await httpx_client.delete(f"v1/llm-adapters/{adapter_id}")
        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True

        # Verify it's actually gone
        response2 = await httpx_client.get(f"v1/llm-adapters/{adapter_id}")
        assert response2.status_code == 404

    async def test_delete_nonexistent_returns_404(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.delete("v1/llm-adapters/99999")
        assert response.status_code == 404
