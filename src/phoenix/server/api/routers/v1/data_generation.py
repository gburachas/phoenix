"""REST API router for Data Generation job management.

Provides CRUD operations and lifecycle management for test-data generation jobs.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from starlette.requests import Request
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from phoenix.db import models

from .models import V1RoutesBaseModel
from .utils import ResponseBody, add_errors_to_responses

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-generation", tags=["data-generation"])


class DataGenerationJob(V1RoutesBaseModel):
    id: int
    name: str
    status: str
    corpus_source: str
    corpus_config: dict[str, Any]
    sampling_strategy: str
    sample_size: int
    testset_llm_adapter_id: Optional[int]
    transform_llm_adapter_id: Optional[int]
    llm_config: dict[str, Any]
    is_multimodal: bool
    output_dataset_name: Optional[str]
    artifacts: dict[str, Any]
    error_message: Optional[str]
    seed: Optional[int]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class CreateDataGenerationJobRequest(V1RoutesBaseModel):
    name: str
    corpus_source: str
    corpus_config: dict[str, Any] = {}
    sampling_strategy: str = "random"
    sample_size: int = 50
    testset_llm_adapter_id: Optional[int] = None
    transform_llm_adapter_id: Optional[int] = None
    llm_config: dict[str, Any] = {}
    is_multimodal: bool = False
    output_dataset_name: Optional[str] = None
    seed: Optional[int] = None


def _to_response(job: models.DataGenerationJob) -> DataGenerationJob:
    return DataGenerationJob(
        id=job.id,
        name=job.name,
        status=job.status,
        corpus_source=job.corpus_source,
        corpus_config=job.corpus_config or {},
        sampling_strategy=job.sampling_strategy,
        sample_size=job.sample_size,
        testset_llm_adapter_id=job.testset_llm_adapter_id,
        transform_llm_adapter_id=job.transform_llm_adapter_id,
        llm_config=job.llm_config or {},
        is_multimodal=job.is_multimodal,
        output_dataset_name=job.output_dataset_name,
        artifacts=job.artifacts or {},
        error_message=job.error_message,
        seed=job.seed,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get(
    "/jobs",
    operation_id="listDataGenerationJobs",
    summary="List data generation jobs",
    responses=add_errors_to_responses([]),
)
async def list_data_generation_jobs(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status"),
) -> ResponseBody[list[DataGenerationJob]]:
    async with request.app.state.db() as session:
        stmt = select(models.DataGenerationJob).order_by(
            models.DataGenerationJob.created_at.desc()
        )
        if status:
            stmt = stmt.where(models.DataGenerationJob.status == status)
        result = await session.execute(stmt)
        jobs = result.scalars().all()
    return ResponseBody(data=[_to_response(j) for j in jobs])


@router.get(
    "/jobs/{job_id}",
    operation_id="getDataGenerationJob",
    summary="Get a data generation job by ID",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def get_data_generation_job(
    request: Request,
    job_id: int,
) -> ResponseBody[DataGenerationJob]:
    async with request.app.state.db() as session:
        job = await session.get(models.DataGenerationJob, job_id)
    if job is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Job not found")
    return ResponseBody(data=_to_response(job))


@router.post(
    "/jobs",
    operation_id="createDataGenerationJob",
    summary="Create a new data generation job",
    responses=add_errors_to_responses([HTTP_409_CONFLICT]),
)
async def create_data_generation_job(
    request: Request,
    body: CreateDataGenerationJobRequest,
) -> ResponseBody[DataGenerationJob]:
    async with request.app.state.db() as session:
        job = models.DataGenerationJob(
            name=body.name,
            corpus_source=body.corpus_source,
            corpus_config=body.corpus_config,
            sampling_strategy=body.sampling_strategy,
            sample_size=body.sample_size,
            testset_llm_adapter_id=body.testset_llm_adapter_id,
            transform_llm_adapter_id=body.transform_llm_adapter_id,
            llm_config=body.llm_config,
            is_multimodal=body.is_multimodal,
            output_dataset_name=body.output_dataset_name,
            seed=body.seed,
        )
        session.add(job)
        await session.flush()
        await session.refresh(job)
        await session.commit()
    return ResponseBody(data=_to_response(job))


@router.delete(
    "/jobs/{job_id}",
    operation_id="deleteDataGenerationJob",
    summary="Delete a data generation job",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def delete_data_generation_job(
    request: Request,
    job_id: int,
) -> ResponseBody[dict[str, bool]]:
    async with request.app.state.db() as session:
        job = await session.get(models.DataGenerationJob, job_id)
        if job is None:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Job not found")
        await session.delete(job)
        await session.commit()
    return ResponseBody(data={"deleted": True})


@router.post(
    "/jobs/{job_id}/cancel",
    operation_id="cancelDataGenerationJob",
    summary="Cancel a running or pending data generation job",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY]),
)
async def cancel_data_generation_job(
    request: Request,
    job_id: int,
) -> ResponseBody[DataGenerationJob]:
    async with request.app.state.db() as session:
        job = await session.get(models.DataGenerationJob, job_id)
        if job is None:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Job not found")
        if job.status not in ("pending", "running"):
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot cancel job with status '{job.status}'",
            )
        job.status = "cancelled"
        await session.commit()
        await session.refresh(job)
    return ResponseBody(data=_to_response(job))
