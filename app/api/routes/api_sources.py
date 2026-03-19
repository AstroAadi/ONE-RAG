import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.api_upload import APISourceInput
from app.core.dependencies import get_user_id
from app.services.api_source_service import APISourceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["Sources"])


@router.post("/api/extract")
async def extract_api_source(
    payload: APISourceInput,
    user_id: str = Depends(get_user_id),
):
    """
    Call the configured API endpoint and return the extracted JSON data.

    The response is already filtered by `json_path` and `field_paths`, so the
    returned `data` and `markdown_text` are ready for downstream chunking.
    """
    try:
        service = APISourceService()
        result = await service.fetch_and_extract(payload)
        return {
            **result,
            "user_id": user_id,
        }
    except httpx.HTTPStatusError as exc:
        logger.error("API source request failed: %s", exc, exc_info=True)
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(
            status_code=exc.response.status_code if exc.response is not None else 502,
            detail=f"API source request failed: {detail}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("API source extraction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"API source extraction failed: {exc}",
        ) from exc
