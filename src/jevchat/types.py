"""Shared types for System One language modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "system"]
PieceKind = Literal["eos", "phrase", "word", "punct", "copy", "char", "other"]
EventKind = Literal["plan", "call", "token", "done", "error"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Piece:
    """One candidate continuation.

    ``surface`` is the exact text appended to the prefix.
    ``id`` is the Choice key sent to Jev; it is never shown to the user.
    """

    id: str
    surface: str
    kind: PieceKind
    label: str
    description: str


@dataclass(frozen=True)
class Plan:
    act: str
    act_confidence: float
    length_score: float
    grounded: float


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, latency_s: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        self.latency_s += latency_s


@dataclass
class DecodeEvent:
    kind: EventKind
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
