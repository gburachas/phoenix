"""REST API router for Corpus Sampling.

Provides discovery of corpus sources and document sampling from connected
vectorstores, datasets, and local directories.

IMPORTANT DESIGN CONSTRAINT: Random sampling must always draw from the ENTIRE
corpus regardless of sample_size. For example, requesting n=10 from a corpus of
1000 documents must select 10 documents uniformly at random from all 1000 — never
just the first 10.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from starlette.requests import Request
from starlette.status import HTTP_404_NOT_FOUND, HTTP_501_NOT_IMPLEMENTED

from .models import V1RoutesBaseModel
from .utils import ResponseBody, add_errors_to_responses

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpus", tags=["corpus"])


class CorpusSource(V1RoutesBaseModel):
    name: str
    source_type: str  # directory | qdrant | dataset
    location: str
    doc_count: Optional[int] = None
    metadata: dict[str, Any] = {}


class SampleRequest(V1RoutesBaseModel):
    source_name: str
    source_type: str
    strategy: str = "random"  # random | head | tail
    sample_size: int = 10
    seed: Optional[int] = None
    location: Optional[str] = None


class SampledDocument(V1RoutesBaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = {}


@router.get(
    "/sources",
    operation_id="listCorpusSources",
    summary="Discover available corpus sources",
    responses=add_errors_to_responses([]),
)
async def list_corpus_sources(request: Request) -> ResponseBody[list[CorpusSource]]:
    sources: list[CorpusSource] = []

    # Discover local data directories
    data_dir = os.environ.get("METROSTAR_DATA_DIR", "./data")
    if os.path.isdir(data_dir):
        for entry in os.scandir(data_dir):
            if entry.is_dir():
                file_count = sum(1 for f in Path(entry.path).rglob("*") if f.is_file())
                sources.append(
                    CorpusSource(
                        name=entry.name,
                        source_type="directory",
                        location=entry.path,
                        doc_count=file_count,
                    )
                )

    # Discover Qdrant collections (if configured)
    qdrant_url = os.environ.get("METROSTAR_QDRANT_URL")
    if qdrant_url:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{qdrant_url}/collections")
                if resp.status_code == 200:
                    data = resp.json()
                    for coll in data.get("result", {}).get("collections", []):
                        sources.append(
                            CorpusSource(
                                name=coll["name"],
                                source_type="qdrant",
                                location=qdrant_url,
                            )
                        )
        except Exception as e:
            logger.warning(f"Failed to discover Qdrant collections: {e}")

    # Discover Phoenix datasets via the DB
    try:
        from sqlalchemy import func, select

        from phoenix.db import models as db_models

        async with request.app.state.db() as session:
            result = await session.execute(
                select(db_models.Dataset.id, db_models.Dataset.name).order_by(
                    db_models.Dataset.name
                )
            )
            for dataset_id, dataset_name in result.all():
                sources.append(
                    CorpusSource(
                        name=dataset_name,
                        source_type="dataset",
                        location=f"phoenix://datasets/{dataset_id}",
                    )
                )
    except Exception as e:
        logger.warning(f"Failed to discover Phoenix datasets: {e}")

    return ResponseBody(data=sources)


@router.post(
    "/sample",
    operation_id="sampleCorpus",
    summary="Sample documents from a corpus source",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND, HTTP_501_NOT_IMPLEMENTED]),
)
async def sample_corpus(
    request: Request,
    body: SampleRequest,
) -> ResponseBody[list[SampledDocument]]:
    # If a seed is provided, use a local Random instance for reproducibility;
    # otherwise fall back to the module-level random which draws from /dev/urandom.
    rng = random.Random(body.seed) if body.seed is not None else random

    if body.source_type == "directory":
        location = body.location
        if not location or not os.path.isdir(location):
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {location}",
            )
        # Enumerate the ENTIRE corpus first, then sample from the full set.
        all_files = [f for f in Path(location).rglob("*") if f.is_file()]
        if not all_files:
            return ResponseBody(data=[])

        if body.strategy == "random":
            # CRITICAL: random.sample draws uniformly from the ENTIRE list,
            # regardless of whether sample_size is 10 or 100.
            sampled = rng.sample(all_files, min(body.sample_size, len(all_files)))
        elif body.strategy == "head":
            sampled = sorted(all_files)[:body.sample_size]
        elif body.strategy == "tail":
            sampled = sorted(all_files)[-body.sample_size:]
        else:
            sampled = rng.sample(all_files, min(body.sample_size, len(all_files)))

        docs = []
        for f in sampled:
            try:
                content = f.read_text(errors="replace")[:2000]  # Truncate for preview
            except Exception:
                content = f"<binary file: {f.name}>"
            docs.append(
                SampledDocument(
                    id=str(f),
                    content=content,
                    metadata={"filename": f.name, "size": f.stat().st_size},
                )
            )
        return ResponseBody(data=docs)

    elif body.source_type == "dataset":
        # Sample from a Phoenix dataset — read ALL examples, then sample randomly.
        try:
            from sqlalchemy import func, select

            from phoenix.db import models as db_models

            async with request.app.state.db() as session:
                # Resolve the dataset by name
                dataset = (
                    await session.execute(
                        select(db_models.Dataset).where(
                            db_models.Dataset.name == body.source_name
                        )
                    )
                ).scalar_one_or_none()
                if dataset is None:
                    raise HTTPException(
                        status_code=HTTP_404_NOT_FOUND,
                        detail=f"Dataset not found: {body.source_name}",
                    )

                # Get the latest version
                latest_version = (
                    await session.execute(
                        select(db_models.DatasetVersion)
                        .where(db_models.DatasetVersion.dataset_id == dataset.id)
                        .order_by(db_models.DatasetVersion.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if latest_version is None:
                    return ResponseBody(data=[])

                # Load ALL examples for this dataset version so we can sample randomly
                examples = (
                    await session.execute(
                        select(db_models.DatasetExample)
                        .join(
                            db_models.DatasetExampleRevision,
                            db_models.DatasetExampleRevision.dataset_example_id
                            == db_models.DatasetExample.id,
                        )
                        .where(
                            db_models.DatasetExample.dataset_id == dataset.id,
                        )
                    )
                ).scalars().all()

                if not examples:
                    return ResponseBody(data=[])

                # Sample randomly from the ENTIRE set of examples
                if body.strategy == "random":
                    sampled = rng.sample(
                        list(examples), min(body.sample_size, len(examples))
                    )
                elif body.strategy == "head":
                    sampled = list(examples)[:body.sample_size]
                elif body.strategy == "tail":
                    sampled = list(examples)[-body.sample_size:]
                else:
                    sampled = rng.sample(
                        list(examples), min(body.sample_size, len(examples))
                    )

                docs = []
                for ex in sampled:
                    # Get the latest revision for this example
                    revision = (
                        await session.execute(
                            select(db_models.DatasetExampleRevision)
                            .where(
                                db_models.DatasetExampleRevision.dataset_example_id == ex.id
                            )
                            .order_by(db_models.DatasetExampleRevision.id.desc())
                            .limit(1)
                        )
                    ).scalar_one_or_none()
                    if revision is None:
                        continue
                    content = str(revision.input)[:2000] if revision.input else ""
                    docs.append(
                        SampledDocument(
                            id=str(ex.id),
                            content=content,
                            metadata={
                                "dataset": dataset.name,
                                "output": str(revision.output)[:500] if revision.output else "",
                                "metadata": revision.metadata_ or {},
                            },
                        )
                    )
                return ResponseBody(data=docs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to sample from dataset: {e}")
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Failed to sample from dataset: {e}",
            )

    elif body.source_type == "qdrant":
        qdrant_url = os.environ.get("METROSTAR_QDRANT_URL")
        if not qdrant_url:
            raise HTTPException(
                status_code=HTTP_501_NOT_IMPLEMENTED,
                detail="Qdrant URL not configured (set METROSTAR_QDRANT_URL)",
            )
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get total point count first
                coll_resp = await client.get(
                    f"{qdrant_url}/collections/{body.source_name}"
                )
                if coll_resp.status_code != 200:
                    raise HTTPException(
                        status_code=HTTP_404_NOT_FOUND,
                        detail=f"Qdrant collection not found: {body.source_name}",
                    )
                total_points = (
                    coll_resp.json()
                    .get("result", {})
                    .get("points_count", 0)
                )
                if total_points == 0:
                    return ResponseBody(data=[])

                # Scroll ALL point IDs so we can sample randomly from the entire corpus
                all_ids: list[Any] = []
                offset = None
                while True:
                    scroll_body: dict[str, Any] = {
                        "limit": 1000,
                        "with_payload": False,
                        "with_vector": False,
                    }
                    if offset is not None:
                        scroll_body["offset"] = offset
                    scroll_resp = await client.post(
                        f"{qdrant_url}/collections/{body.source_name}/points/scroll",
                        json=scroll_body,
                    )
                    if scroll_resp.status_code != 200:
                        break
                    result = scroll_resp.json().get("result", {})
                    points = result.get("points", [])
                    all_ids.extend(p["id"] for p in points)
                    offset = result.get("next_page_offset")
                    if offset is None or not points:
                        break

                # Randomly sample from ALL collected IDs
                sampled_ids = rng.sample(all_ids, min(body.sample_size, len(all_ids)))

                # Fetch payloads for sampled IDs
                get_resp = await client.post(
                    f"{qdrant_url}/collections/{body.source_name}/points",
                    json={"ids": sampled_ids, "with_payload": True, "with_vector": False},
                )
                if get_resp.status_code != 200:
                    return ResponseBody(data=[])

                docs = []
                for pt in get_resp.json().get("result", []):
                    payload = pt.get("payload", {})
                    content = payload.get("text", payload.get("content", str(payload)))[:2000]
                    docs.append(
                        SampledDocument(
                            id=str(pt["id"]),
                            content=content,
                            metadata=payload,
                        )
                    )
                return ResponseBody(data=docs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to sample from Qdrant: {e}")
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Failed to sample from Qdrant collection: {e}",
            )
    else:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Unknown source type: {body.source_type}",
        )
