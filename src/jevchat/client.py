"""Thin async wrapper around the TypeSafe System One client."""

from __future__ import annotations

from typing import Any, Protocol

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy


class SystemOneClient(Protocol):
    async def evaluate(
        self,
        state: object,
        questions: dict[str, Any],
        *,
        model: str,
    ) -> Any: ...


class JevClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            timeout=timeout,
            retry=RetryPolicy(max_retries=4, backoff_initial=0.4, backoff_max=8.0),
        )

    async def evaluate(
        self,
        state: object,
        questions: dict[str, Any],
        *,
        model: str,
    ) -> Any:
        return await self._client.system_one(
            state=state,
            questions=questions,
            model=model,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()
