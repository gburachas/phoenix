"""REST API router for Corpus Sampling.

Provides discovery of corpus sources and document sampling from connected
vectorstores, datasets, and local directories.
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
    if body.source_type == "directory":
        location = body.location
        if not location or not os.path.isdir(location):
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {location}",
            )
        files = [f for f in Path(location).rglob("*") if f.is_file()]
        if not files:
            return ResponseBody(data=[])

        if body.strategy == "random":
            sampled = random.sample(files, min(body.sample_size, len(files)))
        elif body.strategy == "head":
            sampled = sorted(files)[:body.sample_size]
        elif body.strategy == "tail":
            sampled = sorted(files)[-body.sample_size:]
        else:
            sampled = files[:body.sample_size]

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

    elif body.source_type == "qdrant":
        raise HTTPException(
            status_code=HTTP_501_NOT_IMPLEMENTED,
            detail="Qdrant sampling not yet implemented",
        )
    elif body.source_type == "dataset":
        raise HTTPException(
            status_code=HTTP_501_NOT_IMPLEMENTED,
            detail="Phoenix dataset sampling not yet implemented",
        )
    else:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Unknown source type: {body.source_type}",
        )
