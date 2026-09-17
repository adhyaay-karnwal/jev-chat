"""Command-line interface for jev-chat."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from jevchat.client import JevClient
from jevchat.decoder import DecodeConfig, SystemOneDecoder
from jevchat.types import ChatMessage


def _load_env() -> None:
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevchat",
        description="Chat with Jev via hierarchical speculative decoding.",
    )
    parser.add_argument("prompt", nargs="?", help="Single-turn user prompt")
    parser.add_argument("--serve", action="store_true", help="Run the local demo server")
    parser.add_argument("--host", default=os.environ.get("JEVCHAT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JEVCHAT_PORT", "8765")))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-calls", type=int, default=14)
    parser.add_argument("--model", default=os.environ.get("TYPESAFE_MODEL", "jev-latest"))
    parser.add_argument("--seed", type=int, default=None)
    return parser


async def _run_prompt(prompt: str, config: DecodeConfig) -> int:
    client = JevClient(api_key=os.environ.get("TYPESAFE_API_KEY"))
    decoder = SystemOneDecoder(client, config=config)
    messages = [ChatMessage(role="user", content=prompt)]
    try:
        async for event in decoder.generate(messages):
            if event.kind == "plan":
                act = event.data.get("act")
                print(f"# plan  act={act}", file=sys.stderr)
            elif event.kind == "call":
                print(
                    f"# call {event.data['call']}  "
                    f"{event.data['latency_s']}s  "
                    f"in={event.data['input_tokens']} out={event.data['output_tokens']}",
                    file=sys.stderr,
                )
            elif event.kind == "token":
                sys.stdout.write(event.text)
                sys.stdout.flush()
            elif event.kind == "done":
                if not event.text.endswith("\n"):
                    sys.stdout.write("\n")
                usage = event.data.get("usage", {})
                print(
                    f"# done  calls={usage.get('calls')}  "
                    f"tokens={usage.get('input_tokens')}+{usage.get('output_tokens')}",
                    file=sys.stderr,
                )
            elif event.kind == "error":
                print(event.text or event.data, file=sys.stderr)
                return 1
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            await close()
    return 0


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = build_parser().parse_args(argv)
    if args.serve:
        from jevchat.server import serve

        serve(host=args.host, port=args.port)
        return
    if not args.prompt:
        build_parser().print_help()
        raise SystemExit(2)
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY in the environment or a .env file.", file=sys.stderr)
        raise SystemExit(2)
    config = DecodeConfig(
        model=args.model,
        temperature=args.temperature,
        max_calls=args.max_calls,
        seed=args.seed,
    )
    raise SystemExit(asyncio.run(_run_prompt(args.prompt, config)))
