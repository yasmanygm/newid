from fastapi import APIRouter, Response

from app.config import settings
from app.pipeline_singleton import state
from app.schemas import ReadyResponse, VersionResponse

router = APIRouter(tags=["meta"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "alive"}


@router.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
async def ready(response: Response) -> ReadyResponse:
    if not state.ready:
        response.status_code = 503
    return ReadyResponse(ready=state.ready)


@router.get("/version", response_model=VersionResponse, summary="Build info")
async def version() -> VersionResponse:
    return VersionResponse(
        version=settings.app_version,
        engine=settings.mrz_engine,
        git_sha=settings.git_sha,
    )
