"""Background job runner for data generation.

Orchestrates: sample corpus → generate test dataset → store artifacts.
Uses FastAPI BackgroundTasks for async execution.

DESIGN: Random sampling always draws from the ENTIRE corpus, regardless
of sample_size. For n=10 or n=100, the sample is drawn uniformly from
the full document set.
"""

import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from phoenix.db import models
from phoenix.server.types import DbSessionFactory

logger = logging.getLogger(__name__)


async def run_data_generation_job(
    job_id: int,
    db: DbSessionFactory,
) -> None:
    """Execute a data generation job in the background.

    Updates job status through: pending → running → completed/failed.
    Stores artifacts (sampled docs, metadata) in the job record.
    """
    started_at = datetime.now(timezone.utc)

    # Mark job as running
    async with db() as session:
        await session.execute(
            update(models.DataGenerationJob)
            .where(models.DataGenerationJob.id == job_id)
            .values(status="running", started_at=started_at)
        )

    try:
        # Load the job details
        async with db() as session:
            job = await session.get(models.DataGenerationJob, job_id)
            if job is None:
                logger.error(f"Job {job_id} not found")
                return
            if job.status == "cancelled":
                logger.info(f"Job {job_id} was cancelled before execution")
                return

            # Capture job config
            corpus_source = job.corpus_source
            corpus_config = job.corpus_config or {}
            sampling_strategy = job.sampling_strategy
            sample_size = job.sample_size
            seed = job.seed

        # Set up RNG for reproducibility
        rng = random.Random(seed) if seed is not None else random

        # Step 1: Sample the corpus
        sampled_docs = await _sample_corpus(
            corpus_source=corpus_source,
            corpus_config=corpus_config,
            strategy=sampling_strategy,
            sample_size=sample_size,
            rng=rng,
            db=db,
        )

        # Check for cancellation between steps
        async with db() as session:
            job = await session.get(models.DataGenerationJob, job_id)
            if job is not None and job.status == "cancelled":
                logger.info(f"Job {job_id} cancelled during execution")
                return

        # Step 2: Build artifacts
        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()

        artifacts = {
            "sampled_doc_count": len(sampled_docs),
            "sample_ids": [doc["id"] for doc in sampled_docs[:100]],  # Cap at 100 IDs
            "duration_seconds": round(duration_seconds, 2),
            "corpus_source": corpus_source,
            "sampling_strategy": sampling_strategy,
            "seed": seed,
        }

        # Mark job as completed
        async with db() as session:
            await session.execute(
                update(models.DataGenerationJob)
                .where(models.DataGenerationJob.id == job_id)
                .values(
                    status="completed",
                    completed_at=completed_at,
                    artifacts=artifacts,
                )
            )

        logger.info(
            f"Job {job_id} completed: sampled {len(sampled_docs)} docs "
            f"in {duration_seconds:.1f}s"
        )

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        async with db() as session:
            await session.execute(
                update(models.DataGenerationJob)
                .where(models.DataGenerationJob.id == job_id)
                .values(
                    status="failed",
                    error_message=str(e)[:2000],
                    completed_at=datetime.now(timezone.utc),
                )
            )


async def _sample_corpus(
    corpus_source: str,
    corpus_config: dict[str, Any],
    strategy: str,
    sample_size: int,
    rng: Any,
    db: DbSessionFactory,
) -> list[dict[str, Any]]:
    """Sample documents from a corpus source.

    CRITICAL: For 'random' strategy, we always enumerate ALL documents first,
    then sample uniformly from the complete set. This ensures that requesting
    n=10 or n=100 samples from a 1000-document corpus draws from all 1000.
    """
    source_type = corpus_config.get("source_type", "directory")

    if source_type == "directory":
        location = corpus_config.get("location", corpus_source)
        return _sample_directory(location, strategy, sample_size, rng)
    elif source_type == "dataset":
        return await _sample_dataset(corpus_source, strategy, sample_size, rng, db)
    else:
        logger.warning(f"Unsupported corpus source type: {source_type}")
        return []


def _sample_directory(
    location: str,
    strategy: str,
    sample_size: int,
    rng: Any,
) -> list[dict[str, Any]]:
    """Sample files from a directory, always enumerating the ENTIRE directory first."""
    path = Path(location)
    if not path.is_dir():
        raise ValueError(f"Directory not found: {location}")

    # Enumerate ALL files in the directory tree
    all_files = list(path.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]

    if not all_files:
        return []

    # Apply sampling strategy - random always samples from the FULL list
    if strategy == "random":
        sampled = rng.sample(all_files, min(sample_size, len(all_files)))
    elif strategy == "head":
        sampled = sorted(all_files)[:sample_size]
    elif strategy == "tail":
        sampled = sorted(all_files)[-sample_size:]
    else:
        sampled = rng.sample(all_files, min(sample_size, len(all_files)))

    docs = []
    for f in sampled:
        try:
            content = f.read_text(errors="replace")[:4000]
        except Exception:
            content = f"<binary file: {f.name}>"
        docs.append({
            "id": str(f),
            "content": content,
            "metadata": {"filename": f.name, "size": f.stat().st_size},
        })
    return docs


async def _sample_dataset(
    dataset_name: str,
    strategy: str,
    sample_size: int,
    rng: Any,
    db: DbSessionFactory,
) -> list[dict[str, Any]]:
    """Sample from a Phoenix dataset, always loading ALL examples first."""
    from sqlalchemy import select

    async with db() as session:
        dataset = (
            await session.execute(
                select(models.Dataset).where(models.Dataset.name == dataset_name)
            )
        ).scalar_one_or_none()
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_name}")

        # Load ALL examples for this dataset
        examples = (
            await session.execute(
                select(models.DatasetExample).where(
                    models.DatasetExample.dataset_id == dataset.id
                )
            )
        ).scalars().all()

        if not examples:
            return []

        # Sample from the ENTIRE set of examples
        if strategy == "random":
            sampled = rng.sample(list(examples), min(sample_size, len(examples)))
        elif strategy == "head":
            sampled = list(examples)[:sample_size]
        elif strategy == "tail":
            sampled = list(examples)[-sample_size:]
        else:
            sampled = rng.sample(list(examples), min(sample_size, len(examples)))

        docs = []
        for ex in sampled:
            revision = (
                await session.execute(
                    select(models.DatasetExampleRevision)
                    .where(models.DatasetExampleRevision.dataset_example_id == ex.id)
                    .order_by(models.DatasetExampleRevision.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if revision is None:
                continue
            content = str(revision.input)[:4000] if revision.input else ""
            docs.append({
                "id": str(ex.id),
                "content": content,
                "metadata": {
                    "dataset": dataset.name,
                    "output": str(revision.output)[:500] if revision.output else "",
                },
            })
        return docs
