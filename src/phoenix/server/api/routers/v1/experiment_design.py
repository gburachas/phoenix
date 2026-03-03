"""REST API router for Experiment Design management.

Provides CRUD operations, factorial cell generation, and run-all for
experiment designs. See Sprint 4 of the MetroStar dev plan.
"""

import itertools
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload
from starlette.requests import Request
from starlette.status import (
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_ENTITY,
)

from phoenix.db import models

from .models import V1RoutesBaseModel
from .utils import ResponseBody, add_errors_to_responses

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiment-designs", tags=["experiment-designs"])


# ── Pydantic response / request models ──────────────────────────────────────


class ExperimentFactor(V1RoutesBaseModel):
    id: int
    design_id: int
    name: str
    factor_type: str
    levels: list[Any]
    created_at: datetime


class ExperimentDesignCell(V1RoutesBaseModel):
    id: int
    design_id: int
    combination: dict[str, Any]
    status: str
    experiment_id: Optional[int]
    result_summary: dict[str, Any]
    created_at: datetime


class ExperimentDesign(V1RoutesBaseModel):
    id: int
    name: str
    description: Optional[str]
    design_type: str
    status: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    factors: list[ExperimentFactor]
    cells: list[ExperimentDesignCell]


class CreateExperimentDesignRequest(V1RoutesBaseModel):
    name: str
    description: Optional[str] = None
    design_type: str = "full_factorial"
    metadata: dict[str, Any] = {}


class AddFactorRequest(V1RoutesBaseModel):
    name: str
    factor_type: str = "custom"
    levels: list[Any] = []


class CreateFromTemplateRequest(V1RoutesBaseModel):
    template_name: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def _factor_to_response(f: models.ExperimentFactor) -> ExperimentFactor:
    return ExperimentFactor(
        id=f.id,
        design_id=f.design_id,
        name=f.name,
        factor_type=f.factor_type,
        levels=f.levels if f.levels else [],
        created_at=f.created_at,
    )


def _cell_to_response(c: models.ExperimentDesignCell) -> ExperimentDesignCell:
    return ExperimentDesignCell(
        id=c.id,
        design_id=c.design_id,
        combination=c.combination or {},
        status=c.status,
        experiment_id=c.experiment_id,
        result_summary=c.result_summary or {},
        created_at=c.created_at,
    )


def _to_response(d: models.ExperimentDesign) -> ExperimentDesign:
    return ExperimentDesign(
        id=d.id,
        name=d.name,
        description=d.description,
        design_type=d.design_type,
        status=d.status,
        metadata=d.metadata_ or {},
        created_at=d.created_at,
        updated_at=d.updated_at,
        factors=[_factor_to_response(f) for f in (d.factors or [])],
        cells=[_cell_to_response(c) for c in (d.cells or [])],
    )


async def _load_design_with_rels(
    session: Any, design_id: int
) -> models.ExperimentDesign | None:
    """Load an ExperimentDesign with factors and cells eagerly loaded.

    Uses an explicit SELECT with selectinload to guarantee relationships
    are populated, avoiding MissingGreenlet errors from lazy loading.
    """
    stmt = (
        select(models.ExperimentDesign)
        .where(models.ExperimentDesign.id == design_id)
        .options(
            selectinload(models.ExperimentDesign.factors),
            selectinload(models.ExperimentDesign.cells),
        )
        .execution_options(populate_existing=True)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ── CRUD Endpoints ──────────────────────────────────────────────────────────


@router.post(
    "",
    operation_id="createExperimentDesign",
    summary="Create a new experiment design",
    responses=add_errors_to_responses([]),
)
async def create_experiment_design(
    request: Request,
    body: CreateExperimentDesignRequest,
) -> ResponseBody[ExperimentDesign]:
    valid_types = ("full_factorial", "fractional", "custom")
    if body.design_type not in valid_types:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"design_type must be one of {valid_types}",
        )
    async with request.app.state.db() as session:
        design = models.ExperimentDesign(
            name=body.name,
            description=body.description,
            design_type=body.design_type,
            metadata_=body.metadata or {},
        )
        session.add(design)
        await session.flush()
        # Re-fetch with eager loading for relationships
        design = await _load_design_with_rels(session, design.id)
        assert design is not None
        return ResponseBody(data=_to_response(design))


@router.get(
    "",
    operation_id="listExperimentDesigns",
    summary="List experiment designs",
    responses=add_errors_to_responses([]),
)
async def list_experiment_designs(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status"),
) -> ResponseBody[list[ExperimentDesign]]:
    async with request.app.state.db() as session:
        stmt = (
            select(models.ExperimentDesign)
            .order_by(models.ExperimentDesign.created_at.desc())
            .options(
                selectinload(models.ExperimentDesign.factors),
                selectinload(models.ExperimentDesign.cells),
            )
        )
        if status:
            stmt = stmt.where(models.ExperimentDesign.status == status)
        result = await session.execute(stmt)
        designs = result.scalars().unique().all()
        return ResponseBody(data=[_to_response(d) for d in designs])


# ── Template routes (MUST come before /{design_id} to avoid path conflict) ──

# Template definitions
_TEMPLATES: dict[str, dict[str, Any]] = {
    "8-way-3-factor-rag": {
        "name": "8-Way 3-Factor RAG",
        "description": "Full factorial 2×2×2 design: embedding × reranker × LLM",
        "design_type": "full_factorial",
        "factors": [
            {
                "name": "embedding",
                "factor_type": "embedding",
                "levels": [
                    {"name": "text-embedding-3-small"},
                    {"name": "text-embedding-ada-002"},
                ],
            },
            {
                "name": "reranker",
                "factor_type": "reranker",
                "levels": [
                    {"name": "cohere-rerank-v3"},
                    {"name": "none", "ablation": True},
                ],
            },
            {
                "name": "rag_llm",
                "factor_type": "rag_llm",
                "levels": [
                    {"name": "gpt-4o"},
                    {"name": "gpt-4o-mini"},
                ],
            },
        ],
    },
    "embedding-comparison": {
        "name": "Embedding Comparison",
        "description": "Single-factor comparison of embedding models",
        "design_type": "full_factorial",
        "factors": [
            {
                "name": "embedding",
                "factor_type": "embedding",
                "levels": [
                    {"name": "text-embedding-3-small"},
                    {"name": "text-embedding-3-large"},
                    {"name": "text-embedding-ada-002"},
                ],
            },
        ],
    },
    "ablation-study": {
        "name": "Ablation Study",
        "description": "Reranker and retrieval ablation baselines",
        "design_type": "full_factorial",
        "factors": [
            {
                "name": "reranker",
                "factor_type": "reranker",
                "levels": [
                    {"name": "cohere-rerank-v3"},
                    {"name": "bge-reranker-v2-m3"},
                    {"name": "none", "ablation": True},
                ],
            },
            {
                "name": "rag_llm",
                "factor_type": "rag_llm",
                "levels": [
                    {"name": "gpt-4o"},
                    {"name": "none", "ablation": True},
                ],
            },
        ],
    },
}


@router.get(
    "/templates",
    operation_id="listExperimentDesignTemplates",
    summary="List available experiment design templates",
    responses=add_errors_to_responses([]),
)
async def list_templates(
    request: Request,
) -> ResponseBody[list[dict[str, Any]]]:
    summaries = [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in _TEMPLATES.items()
    ]
    return ResponseBody(data=summaries)


@router.post(
    "/from-template",
    operation_id="createExperimentDesignFromTemplate",
    summary="Create an experiment design from a pre-built template",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def create_from_template(
    request: Request,
    body: CreateFromTemplateRequest,
) -> ResponseBody[ExperimentDesign]:
    tpl = _TEMPLATES.get(body.template_name)
    if tpl is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Template '{body.template_name}' not found. "
            f"Available: {list(_TEMPLATES.keys())}",
        )

    async with request.app.state.db() as session:
        design = models.ExperimentDesign(
            name=tpl["name"],
            description=tpl["description"],
            design_type=tpl["design_type"],
            metadata_={"template": body.template_name},
        )
        session.add(design)
        await session.flush()

        for fdef in tpl["factors"]:
            factor = models.ExperimentFactor(
                design_id=design.id,
                name=fdef["name"],
                factor_type=fdef["factor_type"],
                levels=fdef["levels"],
            )
            session.add(factor)

        await session.flush()
        # Re-fetch with eager loading to get relationships
        design = await _load_design_with_rels(session, design.id)
        assert design is not None
        return ResponseBody(data=_to_response(design))


# ── Individual design routes (after static routes) ──────────────────────────


@router.get(
    "/{design_id}",
    operation_id="getExperimentDesign",
    summary="Get an experiment design by ID (with factors and cells)",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def get_experiment_design(
    request: Request,
    design_id: int,
) -> ResponseBody[ExperimentDesign]:
    async with request.app.state.db() as session:
        design = await _load_design_with_rels(session, design_id)
        if design is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="Experiment design not found"
            )
        return ResponseBody(data=_to_response(design))


@router.delete(
    "/{design_id}",
    operation_id="deleteExperimentDesign",
    summary="Delete an experiment design",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def delete_experiment_design(
    request: Request,
    design_id: int,
) -> ResponseBody[dict[str, bool]]:
    async with request.app.state.db() as session:
        design = await session.get(models.ExperimentDesign, design_id)
        if design is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="Experiment design not found"
            )
        await session.delete(design)
        await session.flush()
        return ResponseBody(data={"deleted": True})


# ── Factor management ────────────────────────────────────────────────────────


@router.post(
    "/{design_id}/factors",
    operation_id="addExperimentFactor",
    summary="Add a factor to an experiment design",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def add_factor(
    request: Request,
    design_id: int,
    body: AddFactorRequest,
) -> ResponseBody[ExperimentFactor]:
    valid_types = ("embedding", "reranker", "judge_llm", "rag_llm", "testset_llm", "custom")
    if body.factor_type not in valid_types:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"factor_type must be one of {valid_types}",
        )
    async with request.app.state.db() as session:
        design = await session.get(models.ExperimentDesign, design_id)
        if design is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="Experiment design not found"
            )
        factor = models.ExperimentFactor(
            design_id=design_id,
            name=body.name,
            factor_type=body.factor_type,
            levels=body.levels or [],
        )
        session.add(factor)
        await session.flush()
        await session.refresh(factor, ["created_at"])
        return ResponseBody(data=_factor_to_response(factor))


@router.delete(
    "/{design_id}/factors/{factor_id}",
    operation_id="removeExperimentFactor",
    summary="Remove a factor from an experiment design",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND]),
)
async def remove_factor(
    request: Request,
    design_id: int,
    factor_id: int,
) -> ResponseBody[dict[str, bool]]:
    async with request.app.state.db() as session:
        factor = await session.get(models.ExperimentFactor, factor_id)
        if factor is None or factor.design_id != design_id:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="Factor not found in this design"
            )
        await session.delete(factor)
        await session.flush()
        return ResponseBody(data={"deleted": True})


# ── Cell generation ──────────────────────────────────────────────────────────


@router.post(
    "/{design_id}/generate-cells",
    operation_id="generateExperimentCells",
    summary="Compute factorial combinations and create cells",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND, HTTP_409_CONFLICT]),
)
async def generate_cells(
    request: Request,
    design_id: int,
) -> ResponseBody[list[ExperimentDesignCell]]:
    async with request.app.state.db() as session:
        design = await _load_design_with_rels(session, design_id)
        if design is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="Experiment design not found"
            )

        if not design.factors:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Design must have at least one factor with levels",
            )

        # Validate each factor has at least one level
        for f in design.factors:
            if not f.levels:
                raise HTTPException(
                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Factor '{f.name}' has no levels defined",
                )

        # Clear existing cells
        await session.execute(
            delete(models.ExperimentDesignCell).where(
                models.ExperimentDesignCell.design_id == design_id
            )
        )
        await session.flush()

        # Compute full factorial
        factor_names = [f.name for f in design.factors]
        factor_levels = [f.levels for f in design.factors]

        cells = []
        for combo in itertools.product(*factor_levels):
            combination = dict(zip(factor_names, combo))
            cell = models.ExperimentDesignCell(
                design_id=design_id,
                combination=combination,
                result_summary={},
            )
            session.add(cell)
            cells.append(cell)

        await session.flush()

        # Refresh cells to get server defaults
        for cell in cells:
            await session.refresh(cell, ["created_at"])

        # Update design status
        design.status = "cells_generated"
        await session.flush()

        return ResponseBody(data=[_cell_to_response(c) for c in cells])


# ── Run ──────────────────────────────────────────────────────────────────────


@router.post(
    "/{design_id}/run",
    operation_id="runExperimentDesign",
    summary="Queue all cells for execution",
    responses=add_errors_to_responses([HTTP_404_NOT_FOUND, HTTP_409_CONFLICT]),
)
async def run_experiment_design(
    request: Request,
    design_id: int,
) -> ResponseBody[ExperimentDesign]:
    async with request.app.state.db() as session:
        design = await _load_design_with_rels(session, design_id)
        if design is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND, detail="Experiment design not found"
            )

        if design.status not in ("cells_generated", "failed"):
            raise HTTPException(
                status_code=HTTP_409_CONFLICT,
                detail=f"Cannot run design in status '{design.status}'. "
                "Generate cells first.",
            )

        if not design.cells:
            raise HTTPException(
                status_code=HTTP_409_CONFLICT,
                detail="No cells to run. Generate cells first.",
            )

        design.status = "running"
        for cell in design.cells:
            if cell.status != "completed":
                cell.status = "pending"
        await session.flush()
        # Re-fetch with eager loading after status change
        design = await _load_design_with_rels(session, design.id)
        assert design is not None
        return ResponseBody(data=_to_response(design))
