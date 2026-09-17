"""Local demo server: SSE chat over the System One decoder."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jevchat.client import JevClient
from jevchat.decoder import DecodeConfig, SystemOneDecoder
from jevchat.types import ChatMessage

STATIC_DIR = Path(__file__).parent / "static"


class IncomingMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[IncomingMessage] = Field(min_length=1)
    temperature: float = 0.7
    max_calls: int = 14


def create_app() -> FastAPI:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    app = FastAPI(title="jev-chat", version="0.1.0")
    client = JevClient(api_key=os.environ.get("TYPESAFE_API_KEY"))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise HTTPException(status_code=500, detail="TYPESAFE_API_KEY is not set")
        messages = [
            ChatMessage(role=message.role, content=message.content)  # type: ignore[arg-type]
            for message in body.messages
            if message.role in {"user", "assistant", "system"}
        ]
        if not messages or messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from the user")
        decoder = SystemOneDecoder(
            client,
            config=DecodeConfig(
                model=os.environ.get("TYPESAFE_MODEL", "jev-latest"),
                temperature=body.temperature,
                max_calls=min(body.max_calls, 24),
            ),
        )

        async def events():
            try:
                async for event in decoder.generate(messages):
                    payload = {"text": event.text, **event.data}
                    yield f"event: {event.kind}\ndata: {json.dumps(payload)}\n\n"
            except Exception as exc:  # noqa: BLE001 — surface decoder failures to the UI
                yield (
                    "event: error\ndata: "
                    + json.dumps({"text": str(exc), "error": type(exc).__name__})
                    + "\n\n"
                )

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    uvicorn.run(
        "jevchat.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


app = create_app()
