import asyncio
import base64
import binascii
import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.config import settings
from app.pipeline_singleton import state
from app.schemas import Base64Request, MRZResponse

router = APIRouter(tags=["mrz"])
log = logging.getLogger(__name__)

_ACCEPTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _result_to_response(result, t0: float, request_id: str) -> MRZResponse:
    elapsed = time.perf_counter() - t0
    if result is None:
        return MRZResponse(
            document_type=None,
            valid=False,
            mrz=None,
            fields={},
            error="no_mrz_found",
            processing_time_s=round(elapsed, 3),
            request_id=request_id,
        )
    return MRZResponse(
        document_type=result.document_type,
        valid=result.valid,
        mrz=result.corrected_mrz,
        fields=result.fields,
        processing_time_s=round(elapsed, 3),
        request_id=request_id,
    )


async def _run_pipeline_on_bytes(
    data: bytes,
    suffix: str,
    request_id: str,
    max_image_dim: int | None = None,
) -> MRZResponse:
    if not state.ready or state.pipeline is None:
        raise HTTPException(status_code=503, detail={"error": "not_ready", "request_id": request_id})

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        async with state.lock:
            try:
                result = await asyncio.to_thread(state.pipeline.process, tmp.name, max_image_dim)
            except Exception as exc:
                log.exception("inference_failed", extra={"request_id": request_id})
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "inference_failed",
                        "detail": type(exc).__name__,
                        "request_id": request_id,
                    },
                ) from exc
    return _result_to_response(result, t0, request_id)


@router.post("/mrz", response_model=MRZResponse, summary="Read MRZ from uploaded image file")
async def read_mrz(
    request: Request,
    file: UploadFile = File(...),
    max_image_dim: int | None = Form(None, ge=256, le=4096),
) -> MRZResponse:
    request_id = request.state.request_id
    suffix = Path(file.filename or "").suffix.lower() or ".bin"
    if suffix not in _ACCEPTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_media_type",
                "detail": f"expected one of {sorted(_ACCEPTED_SUFFIXES)}",
                "request_id": request_id,
            },
        )

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large", "request_id": request_id},
        )
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_body", "request_id": request_id},
        )

    return await _run_pipeline_on_bytes(data, suffix, request_id, max_image_dim)


@router.post(
    "/mrz/base64",
    response_model=MRZResponse,
    summary="Read MRZ from a base64-encoded image",
)
async def read_mrz_base64(request: Request, body: Base64Request) -> MRZResponse:
    request_id = request.state.request_id
    try:
        data = base64.b64decode(body.image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_base64",
                "detail": type(exc).__name__,
                "request_id": request_id,
            },
        ) from exc

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large", "request_id": request_id},
        )
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_body", "request_id": request_id},
        )

    # base64 payloads don't carry a filename; .jpg is a safe default suffix for
    # the temp file since cv2 inspects the magic bytes, not the extension.
    return await _run_pipeline_on_bytes(data, ".jpg", request_id, body.max_image_dim)
