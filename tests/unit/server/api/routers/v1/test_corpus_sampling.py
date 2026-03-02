"""Unit tests for the Corpus Sampling REST API router (/v1/corpus).

Validates:
- Discovery of local directory sources
- Discovery of Phoenix dataset sources
- Random sampling from the entire corpus (the key design constraint)
- Deterministic sampling with seed parameter
"""

import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import insert

from phoenix.db import models
from phoenix.server.types import DbSessionFactory


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp data directory with a sub-folder containing sample files."""
    corpus = tmp_path / "test_corpus"
    corpus.mkdir()
    for i in range(20):
        (corpus / f"doc_{i:03d}.txt").write_text(f"Content of document {i}")
    monkeypatch.setenv("METROSTAR_DATA_DIR", str(tmp_path))
    return tmp_path


class TestListCorpusSources:
    async def test_discovers_directories(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        data_dir: Path,
    ) -> None:
        response = await httpx_client.get("v1/corpus/sources")
        assert response.status_code == 200
        sources = response.json()["data"]
        dir_sources = [s for s in sources if s["source_type"] == "directory"]
        assert len(dir_sources) >= 1
        assert dir_sources[0]["name"] == "test_corpus"
        assert dir_sources[0]["doc_count"] == 20

    async def test_discovers_datasets(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        async with db() as session:
            await session.execute(
                insert(models.Dataset).values(
                    name="my-eval-dataset",
                    description="Test evaluation dataset",
                    metadata_={},
                )
            )

        response = await httpx_client.get("v1/corpus/sources")
        assert response.status_code == 200
        sources = response.json()["data"]
        dataset_sources = [s for s in sources if s["source_type"] == "dataset"]
        assert len(dataset_sources) >= 1
        assert any(s["name"] == "my-eval-dataset" for s in dataset_sources)

    async def test_no_data_dir_returns_empty(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METROSTAR_DATA_DIR", "/nonexistent/path")
        response = await httpx_client.get("v1/corpus/sources")
        assert response.status_code == 200
        sources = response.json()["data"]
        dir_sources = [s for s in sources if s["source_type"] == "directory"]
        assert len(dir_sources) == 0


class TestSampleCorpus:
    async def test_sample_from_directory(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        data_dir: Path,
    ) -> None:
        corpus_path = str(data_dir / "test_corpus")
        body = {
            "source_name": "test_corpus",
            "source_type": "directory",
            "strategy": "random",
            "sample_size": 5,
            "location": corpus_path,
        }
        response = await httpx_client.post("v1/corpus/sample", json=body)
        assert response.status_code == 200
        docs = response.json()["data"]
        assert len(docs) == 5
        for doc in docs:
            assert "content" in doc
            assert "metadata" in doc
            assert doc["metadata"]["filename"].endswith(".txt")

    async def test_sample_entire_corpus_when_size_exceeds(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        data_dir: Path,
    ) -> None:
        """When sample_size > corpus size, return all documents."""
        corpus_path = str(data_dir / "test_corpus")
        body = {
            "source_name": "test_corpus",
            "source_type": "directory",
            "strategy": "random",
            "sample_size": 1000,
            "location": corpus_path,
        }
        response = await httpx_client.post("v1/corpus/sample", json=body)
        assert response.status_code == 200
        docs = response.json()["data"]
        assert len(docs) == 20  # All files returned

    async def test_random_sample_is_uniform_from_entire_corpus(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        data_dir: Path,
    ) -> None:
        """Key design constraint: random sampling draws from entire corpus.

        Running with seed=42 produces a deterministic sample. Verify that
        it doesn't just take the first N files (which would be doc_000 to doc_009).
        """
        corpus_path = str(data_dir / "test_corpus")
        body = {
            "source_name": "test_corpus",
            "source_type": "directory",
            "strategy": "random",
            "sample_size": 5,
            "seed": 42,
            "location": corpus_path,
        }
        response = await httpx_client.post("v1/corpus/sample", json=body)
        assert response.status_code == 200
        docs = response.json()["data"]
        assert len(docs) == 5
        filenames = [d["metadata"]["filename"] for d in docs]
        # The 5 sampled filenames should NOT be exactly the first 5 alphabetically
        first_five = [f"doc_{i:03d}.txt" for i in range(5)]
        assert sorted(filenames) != first_five, (
            "Random sampling should draw from the entire corpus, not just the first N files"
        )

    async def test_seeded_sample_is_reproducible(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        data_dir: Path,
    ) -> None:
        """Same seed + same corpus = same sample."""
        corpus_path = str(data_dir / "test_corpus")
        body = {
            "source_name": "test_corpus",
            "source_type": "directory",
            "strategy": "random",
            "sample_size": 5,
            "seed": 123,
            "location": corpus_path,
        }
        resp1 = await httpx_client.post("v1/corpus/sample", json=body)
        resp2 = await httpx_client.post("v1/corpus/sample", json=body)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        ids1 = [d["id"] for d in resp1.json()["data"]]
        ids2 = [d["id"] for d in resp2.json()["data"]]
        assert ids1 == ids2

    async def test_head_strategy(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        data_dir: Path,
    ) -> None:
        corpus_path = str(data_dir / "test_corpus")
        body = {
            "source_name": "test_corpus",
            "source_type": "directory",
            "strategy": "head",
            "sample_size": 3,
            "location": corpus_path,
        }
        response = await httpx_client.post("v1/corpus/sample", json=body)
        assert response.status_code == 200
        docs = response.json()["data"]
        assert len(docs) == 3
        filenames = sorted(d["metadata"]["filename"] for d in docs)
        assert filenames == ["doc_000.txt", "doc_001.txt", "doc_002.txt"]

    async def test_tail_strategy(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        data_dir: Path,
    ) -> None:
        corpus_path = str(data_dir / "test_corpus")
        body = {
            "source_name": "test_corpus",
            "source_type": "directory",
            "strategy": "tail",
            "sample_size": 3,
            "location": corpus_path,
        }
        response = await httpx_client.post("v1/corpus/sample", json=body)
        assert response.status_code == 200
        docs = response.json()["data"]
        assert len(docs) == 3
        filenames = sorted(d["metadata"]["filename"] for d in docs)
        assert filenames == ["doc_017.txt", "doc_018.txt", "doc_019.txt"]

    async def test_nonexistent_directory_returns_404(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
    ) -> None:
        body = {
            "source_name": "nope",
            "source_type": "directory",
            "strategy": "random",
            "sample_size": 5,
            "location": "/nonexistent/path",
        }
        response = await httpx_client.post("v1/corpus/sample", json=body)
        assert response.status_code == 404
