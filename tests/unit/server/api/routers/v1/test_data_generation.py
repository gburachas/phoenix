"""Unit tests for the Data Generation REST API router (/v1/data-generation)."""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from phoenix.db import models
from phoenix.server.types import DbSessionFactory

# The background runner opens its own DB sessions which conflict with the
# SAVEPOINT-based test connection.  Patch it to a no-op for the whole module.
_RUNNER_PATH = (
    "phoenix.server.api.routers.v1.data_generation_runner.run_data_generation_job"
)


@pytest.fixture(autouse=True)
def _no_background_runner() -> Any:
    """Prevent the data-generation background runner from executing in tests."""
    with patch(_RUNNER_PATH, new_callable=AsyncMock) as _mock:
        yield _mock


async def _insert_job(
    db: DbSessionFactory,
    *,
    name: str = "test-job",
    status: str = "pending",
    corpus_source: str = "test-corpus",
    sampling_strategy: str = "random",
    sample_size: int = 10,
    **kwargs: Any,
) -> int:
    async with db() as session:
        job = models.DataGenerationJob(
            name=name,
            status=status,
            corpus_source=corpus_source,
            corpus_config=kwargs.get("corpus_config", {}),
            sampling_strategy=sampling_strategy,
            sample_size=sample_size,
            llm_config=kwargs.get("llm_config", {}),
            artifacts=kwargs.get("artifacts", {}),
            seed=kwargs.get("seed"),
        )
        session.add(job)
        await session.flush()
        return job.id


class TestListDataGenerationJobs:
    async def test_empty_list(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.get("v1/data-generation/jobs")
        assert response.status_code == 200
        assert response.json()["data"] == []

    async def test_returns_jobs(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        await _insert_job(db, name="job-a")
        await _insert_job(db, name="job-b")

        response = await httpx_client.get("v1/data-generation/jobs")
        assert response.status_code == 200
        jobs = response.json()["data"]
        assert len(jobs) == 2

    async def test_filter_by_status(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        await _insert_job(db, name="pending-job", status="pending")
        await _insert_job(db, name="completed-job", status="completed")

        response = await httpx_client.get("v1/data-generation/jobs?status=completed")
        assert response.status_code == 200
        jobs = response.json()["data"]
        assert len(jobs) == 1
        assert jobs[0]["name"] == "completed-job"


class TestGetDataGenerationJob:
    async def test_get_existing_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="my-job", corpus_source="wiki")

        response = await httpx_client.get(f"v1/data-generation/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()["data"]
        assert job["name"] == "my-job"
        assert job["corpus_source"] == "wiki"
        assert job["status"] == "pending"

    async def test_get_nonexistent_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.get("v1/data-generation/jobs/99999")
        assert response.status_code == 404


class TestCreateDataGenerationJob:
    async def test_create_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        body = {
            "name": "new-job",
            "corpus_source": "test-docs",
            "sampling_strategy": "random",
            "sample_size": 50,
            "corpus_config": {"source_type": "directory"},
            "llm_config": {"temperature": 0.7},
        }
        response = await httpx_client.post("v1/data-generation/jobs", json=body)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        job = response.json()["data"]
        assert job["name"] == "new-job"
        assert job["corpus_source"] == "test-docs"
        assert job["sample_size"] == 50
        assert job["id"] > 0

        # Verify it persisted
        async with db() as session:
            db_job = await session.get(models.DataGenerationJob, job["id"])
        assert db_job is not None
        assert db_job.name == "new-job"

    async def test_create_job_with_seed(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        body = {
            "name": "seeded-job",
            "corpus_source": "docs",
            "seed": 42,
        }
        response = await httpx_client.post("v1/data-generation/jobs", json=body)
        assert response.status_code == 200
        job = response.json()["data"]
        assert job["seed"] == 42


class TestDeleteDataGenerationJob:
    async def test_delete_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="delete-me")

        response = await httpx_client.delete(f"v1/data-generation/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True

        response2 = await httpx_client.get(f"v1/data-generation/jobs/{job_id}")
        assert response2.status_code == 404

    async def test_delete_nonexistent_returns_404(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.delete("v1/data-generation/jobs/99999")
        assert response.status_code == 404


class TestCancelDataGenerationJob:
    async def test_cancel_pending_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="cancel-me", status="pending")

        response = await httpx_client.post(f"v1/data-generation/jobs/{job_id}/cancel")
        assert response.status_code == 200
        job = response.json()["data"]
        assert job["status"] == "cancelled"

    async def test_cancel_running_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="running-job", status="running")

        response = await httpx_client.post(f"v1/data-generation/jobs/{job_id}/cancel")
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "cancelled"

    async def test_cancel_completed_job_returns_422(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="done-job", status="completed")

        response = await httpx_client.post(f"v1/data-generation/jobs/{job_id}/cancel")
        assert response.status_code == 422

    async def test_cancel_nonexistent_returns_404(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post("v1/data-generation/jobs/99999/cancel")
        assert response.status_code == 404


class TestRetryDataGenerationJob:
    async def test_retry_failed_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="failed-job", status="failed")

        response = await httpx_client.post(f"v1/data-generation/jobs/{job_id}/run")
        assert response.status_code == 200
        job = response.json()["data"]
        # After retry, status resets to pending (background task will set to running)
        assert job["status"] == "pending"

    async def test_retry_cancelled_job(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="cancelled-job", status="cancelled")

        response = await httpx_client.post(f"v1/data-generation/jobs/{job_id}/run")
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "pending"

    async def test_retry_running_job_returns_422(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        job_id = await _insert_job(db, name="running-job", status="running")

        response = await httpx_client.post(f"v1/data-generation/jobs/{job_id}/run")
        assert response.status_code == 422

    async def test_retry_nonexistent_returns_404(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        response = await httpx_client.post("v1/data-generation/jobs/99999/run")
        assert response.status_code == 404
