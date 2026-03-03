"""Unit tests for the Experiment Design REST API router (/v1/experiment-designs)."""

from typing import Any

import httpx
import pytest

from phoenix.db import models
from phoenix.server.types import DbSessionFactory


# ── helpers ──────────────────────────────────────────────────────────────────


async def _insert_design(
    db: DbSessionFactory,
    *,
    name: str = "test-design",
    design_type: str = "full_factorial",
    status: str = "draft",
    **kwargs: Any,
) -> int:
    async with db() as session:
        design = models.ExperimentDesign(
            name=name,
            design_type=design_type,
            status=status,
            metadata_=kwargs.get("metadata_", {}),
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


# ── CRUD Tests ───────────────────────────────────────────────────────────────


class TestCreateExperimentDesign:
    async def test_create_design(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post(
            "v1/experiment-designs",
            json={"name": "My Design", "design_type": "full_factorial"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["name"] == "My Design"
        assert data["design_type"] == "full_factorial"
        assert data["status"] == "draft"

    async def test_create_design_invalid_type(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post(
            "v1/experiment-designs",
            json={"name": "Bad", "design_type": "invalid"},
        )
        assert response.status_code == 422


class TestListExperimentDesigns:
    async def test_empty_list(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.get("v1/experiment-designs")
        assert response.status_code == 200
        assert response.json()["data"] == []

    async def test_returns_designs(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        await _insert_design(db, name="design-a")
        await _insert_design(db, name="design-b")
        response = await httpx_client.get("v1/experiment-designs")
        assert response.status_code == 200
        assert len(response.json()["data"]) == 2

    async def test_filter_by_status(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        await _insert_design(db, name="d1", status="draft")
        await _insert_design(db, name="d2", status="completed")
        response = await httpx_client.get(
            "v1/experiment-designs", params={"status": "draft"}
        )
        assert response.status_code == 200
        assert len(response.json()["data"]) == 1
        assert response.json()["data"][0]["name"] == "d1"


class TestGetExperimentDesign:
    async def test_get_existing(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db, name="get-me")
        response = await httpx_client.get(f"v1/experiment-designs/{design_id}")
        assert response.status_code == 200
        assert response.json()["data"]["name"] == "get-me"

    async def test_get_not_found(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.get("v1/experiment-designs/99999")
        assert response.status_code == 404


class TestDeleteExperimentDesign:
    async def test_delete_existing(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db, name="delete-me")
        response = await httpx_client.delete(
            f"v1/experiment-designs/{design_id}"
        )
        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True
        # Verify gone
        response = await httpx_client.get(f"v1/experiment-designs/{design_id}")
        assert response.status_code == 404

    async def test_delete_not_found(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.delete("v1/experiment-designs/99999")
        assert response.status_code == 404


# ── Factor management ────────────────────────────────────────────────────────


class TestAddFactor:
    async def test_add_factor(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/factors",
            json={
                "name": "embedding",
                "factor_type": "embedding",
                "levels": [{"name": "ada"}, {"name": "small"}],
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["name"] == "embedding"
        assert data["factor_type"] == "embedding"
        assert len(data["levels"]) == 2

    async def test_add_factor_design_not_found(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post(
            "v1/experiment-designs/99999/factors",
            json={"name": "x", "factor_type": "custom", "levels": []},
        )
        assert response.status_code == 404

    async def test_add_factor_invalid_type(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/factors",
            json={"name": "x", "factor_type": "INVALID_TYPE", "levels": []},
        )
        assert response.status_code == 422


class TestRemoveFactor:
    async def test_remove_factor(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        factor_id = await _insert_factor(db, design_id=design_id)
        response = await httpx_client.delete(
            f"v1/experiment-designs/{design_id}/factors/{factor_id}"
        )
        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True

    async def test_remove_factor_not_found(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        response = await httpx_client.delete(
            f"v1/experiment-designs/{design_id}/factors/99999"
        )
        assert response.status_code == 404


# ── Cell generation ──────────────────────────────────────────────────────────


class TestGenerateCells:
    async def test_generate_2x2x2_gives_8_cells(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        """Three binary factors → 2³ = 8 cells."""
        design_id = await _insert_design(db)
        await _insert_factor(
            db,
            design_id=design_id,
            name="embedding",
            factor_type="embedding",
            levels=[{"name": "ada"}, {"name": "small"}],
        )
        await _insert_factor(
            db,
            design_id=design_id,
            name="reranker",
            factor_type="reranker",
            levels=[{"name": "cohere"}, {"name": "none"}],
        )
        await _insert_factor(
            db,
            design_id=design_id,
            name="rag_llm",
            factor_type="rag_llm",
            levels=[{"name": "gpt-4o"}, {"name": "gpt-4o-mini"}],
        )

        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert response.status_code == 200
        cells = response.json()["data"]
        assert len(cells) == 8

        # Verify unique combinations (use json serialization for hashability)
        import json
        combos = [json.dumps(c["combination"], sort_keys=True) for c in cells]
        assert len(set(combos)) == 8

    async def test_generate_single_factor_3_levels(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        """Single factor with 3 levels → 3 cells."""
        design_id = await _insert_design(db)
        await _insert_factor(
            db,
            design_id=design_id,
            name="embedding",
            factor_type="embedding",
            levels=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
        )
        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert response.status_code == 200
        assert len(response.json()["data"]) == 3

    async def test_generate_updates_status(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        await _insert_factor(db, design_id=design_id)
        await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        # Check design status updated
        response = await httpx_client.get(f"v1/experiment-designs/{design_id}")
        assert response.json()["data"]["status"] == "cells_generated"

    async def test_generate_no_factors_fails(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert response.status_code == 422

    async def test_generate_factor_no_levels_fails(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        await _insert_factor(
            db, design_id=design_id, name="empty", levels=[]
        )
        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert response.status_code == 422

    async def test_regenerate_clears_old_cells(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        """Re-generating cells replaces old ones."""
        design_id = await _insert_design(db)
        await _insert_factor(db, design_id=design_id)
        # First generation
        r1 = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert len(r1.json()["data"]) == 2
        # Second generation
        r2 = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert len(r2.json()["data"]) == 2
        # Verify only 2 cells total
        detail = await httpx_client.get(f"v1/experiment-designs/{design_id}")
        assert len(detail.json()["data"]["cells"]) == 2


# ── Run ──────────────────────────────────────────────────────────────────────


class TestRunDesign:
    async def test_run_after_generate(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db)
        await _insert_factor(db, design_id=design_id)
        await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/run"
        )
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "running"

    async def test_run_draft_fails(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        design_id = await _insert_design(db, status="draft")
        response = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/run"
        )
        assert response.status_code == 409

    async def test_run_not_found(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post("v1/experiment-designs/99999/run")
        assert response.status_code == 404


# ── Templates ────────────────────────────────────────────────────────────────


class TestTemplates:
    async def test_list_templates(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.get("v1/experiment-designs/templates")
        assert response.status_code == 200
        templates = response.json()["data"]
        assert len(templates) >= 3
        names = [t["name"] for t in templates]
        assert "8-Way 3-Factor RAG" in names
        assert "Embedding Comparison" in names
        assert "Ablation Study" in names

    async def test_create_from_template_8way(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post(
            "v1/experiment-designs/from-template",
            json={"template_name": "8-way-3-factor-rag"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["name"] == "8-Way 3-Factor RAG"
        assert len(data["factors"]) == 3
        assert data["design_type"] == "full_factorial"

    async def test_create_from_template_not_found(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post(
            "v1/experiment-designs/from-template",
            json={"template_name": "nonexistent"},
        )
        assert response.status_code == 404

    async def test_template_generates_8_cells(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        """8-Way template → 3 binary factors → 8 cells."""
        create = await httpx_client.post(
            "v1/experiment-designs/from-template",
            json={"template_name": "8-way-3-factor-rag"},
        )
        design_id = create.json()["data"]["id"]
        gen = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert gen.status_code == 200
        assert len(gen.json()["data"]) == 8

    async def test_ablation_template_generates_6_cells(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        """Ablation template → 3×2 = 6 cells."""
        create = await httpx_client.post(
            "v1/experiment-designs/from-template",
            json={"template_name": "ablation-study"},
        )
        design_id = create.json()["data"]["id"]
        gen = await httpx_client.post(
            f"v1/experiment-designs/{design_id}/generate-cells"
        )
        assert gen.status_code == 200
        assert len(gen.json()["data"]) == 6
