import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.api_upload import APISourceInput, AuthType, PaginationType
from app.services.extraction.extractors.json_extractor import JSONExtractor

logger = logging.getLogger(__name__)


class APISourceService:
    """Fetch JSON from an external API and extract only the required content."""

    DEFAULT_TIMEOUT_SECONDS = 30.0
    MAX_PAGES = 50

    async def fetch_and_extract(self, payload: APISourceInput) -> Dict[str, Any]:
        headers, auth = await self._build_auth(payload)
        extractor = JSONExtractor(
            {
                "fmt": "json",
                "json_path": payload.json_path,
                "field_paths": payload.field_paths,
            }
        )

        aggregated_entries: List[Any] = []
        pages_fetched = 0
        page_marker = payload.pagination.start_page

        async with httpx.AsyncClient(
            timeout=self.DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            for attempt in range(self.MAX_PAGES):
                params = self._build_pagination_params(payload, page_marker)
                response = await client.request(
                    method=payload.http_method.value,
                    url=str(payload.url),
                    headers=headers,
                    params=params,
                    json=payload.request_body,
                    auth=auth,
                )
                response.raise_for_status()

                try:
                    response_data = response.json()
                except ValueError as exc:
                    raise ValueError("API response is not valid JSON") from exc

                result = extractor.extract_data_result(
                    response_data,
                    source_name=payload.source_name,
                    base_metadata={
                        "api_url": str(payload.url),
                        "page_index": attempt + 1,
                    },
                )

                if not result.success:
                    if pages_fetched > 0 and "No data matched" in (result.error or ""):
                        break
                    raise ValueError(result.error or "Failed to extract API response")

                page_entries = [entry.raw_object for entry in result.entries]
                if not page_entries:
                    break

                aggregated_entries.extend(page_entries)
                pages_fetched += 1

                if payload.pagination.pagination_type == PaginationType.NONE:
                    break

                if payload.pagination.pagination_type == PaginationType.CURSOR:
                    raise ValueError(
                        "Cursor pagination is not supported by the current API source schema"
                    )

                page_size = payload.pagination.page_size_value or len(page_entries)
                if len(page_entries) < page_size:
                    break

                page_marker = self._next_page_marker(payload, page_marker, page_size)

        if not aggregated_entries:
            raise ValueError("No entries were extracted from the API response")

        markdown_text = json.dumps(aggregated_entries, indent=2, ensure_ascii=False)
        return {
            "source_name": payload.source_name,
            "url": str(payload.url),
            "http_method": payload.http_method.value,
            "json_path": payload.json_path,
            "field_paths": payload.field_paths,
            "pages_fetched": pages_fetched,
            "entries_extracted": len(aggregated_entries),
            "data": aggregated_entries,
            "markdown_text": markdown_text,
        }

    async def _build_auth(
        self,
        payload: APISourceInput,
    ) -> Tuple[Dict[str, str], Optional[httpx.Auth]]:
        headers = {
            "Accept": "application/json",
            **payload.headers,
        }
        auth: Optional[httpx.Auth] = None

        if payload.auth.type == AuthType.BASIC:
            auth = httpx.BasicAuth(
                payload.auth.username,
                payload.auth.password.get_secret_value(),
            )
        elif payload.auth.type == AuthType.BEARER:
            headers["Authorization"] = (
                f"Bearer {payload.auth.token.get_secret_value()}"
            )
        elif payload.auth.type == AuthType.API_KEY:
            headers[payload.auth.header_name] = payload.auth.api_key.get_secret_value()
        elif payload.auth.type == AuthType.OAUTH2:
            headers["Authorization"] = (
                f"Bearer {await self._get_oauth2_token(payload)}"
            )

        return headers, auth

    async def _get_oauth2_token(self, payload: APISourceInput) -> str:
        auth = payload.auth
        if auth.type != AuthType.OAUTH2:
            raise ValueError("OAuth2 token requested for non-OAuth2 configuration")

        request_data = {"grant_type": "client_credentials"}
        if auth.scope:
            request_data["scope"] = auth.scope

        async with httpx.AsyncClient(
            timeout=self.DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                str(auth.token_url),
                data=request_data,
                auth=httpx.BasicAuth(
                    auth.client_id,
                    auth.client_secret.get_secret_value(),
                ),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_payload = response.json()

        access_token = token_payload.get("access_token")
        if not access_token:
            raise ValueError("OAuth2 token response did not include access_token")
        return access_token

    def _build_pagination_params(
        self,
        payload: APISourceInput,
        page_marker: int,
    ) -> Dict[str, Any]:
        if payload.pagination.pagination_type == PaginationType.NONE:
            return {}

        params: Dict[str, Any] = {}
        page_param = payload.pagination.page_param_name
        size_param = payload.pagination.page_size_param_name
        page_size = payload.pagination.page_size_value

        if not page_param or not size_param or page_size is None:
            raise ValueError("Pagination parameters are incomplete")

        params[size_param] = page_size
        params[page_param] = page_marker
        return params

    def _next_page_marker(
        self,
        payload: APISourceInput,
        page_marker: int,
        page_size: int,
    ) -> int:
        if payload.pagination.pagination_type == PaginationType.PAGE_NUMBER:
            return page_marker + 1
        if payload.pagination.pagination_type == PaginationType.OFFSET:
            return page_marker + page_size
        return page_marker
