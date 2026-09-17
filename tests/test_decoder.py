from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jevchat.codebook import Codebook
from jevchat.decoder import DecodeConfig, SystemOneDecoder
from jevchat.types import ChatMessage


@dataclass
class FakeChoice:
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.9
    type: str = "choice"


@dataclass
class FakeNoul:
    noul: float
    type: str = "noul"


@dataclass
class FakeScore:
    score: float
    type: str = "score"
    confidence: float = 0.8


@dataclass
class FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 4


@dataclass
class FakeResponse:
    answers: dict[str, Any]
    usage: FakeUsage = field(default_factory=FakeUsage)


class ScriptedClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def evaluate(self, state: object, questions: dict[str, Any], *, model: str) -> FakeResponse:
        self.calls.append({"state": state, "questions": questions, "model": model})
        if not self.responses:
            raise AssertionError("unexpected extra Jev call")
        return self.responses.pop(0)


def _next_dist(questions: dict[str, Any], winner_kind: str) -> tuple[str, dict[str, float]]:
    criteria = questions["next"].criteria
    ids = list(criteria)
    winner = ids[0]
    for key, spec in criteria.items():
        text = spec if isinstance(spec, str) else str(spec)
        if winner_kind == "eos" and text.startswith("End"):
            winner = key
            break
        if text.startswith(winner_kind + " "):
            winner = key
            break
    mass = {key: 0.02 for key in ids}
    mass[winner] = max(0.4, 1.0 - 0.02 * (len(ids) - 1))
    total = sum(mass.values())
    mass = {key: value / total for key, value in mass.items()}
    return winner, mass


class KindClient:
    """Picks a planned sequence of kinds, then EOS."""

    def __init__(self, kinds: list[str]) -> None:
        self.kinds = list(kinds)
        self.calls = 0

    async def evaluate(self, state: object, questions: dict[str, Any], *, model: str) -> FakeResponse:
        self.calls += 1
        answers: dict[str, Any] = {
            "done": FakeNoul(0.05 if self.kinds else 0.95),
        }
        if "act" in questions:
            answers["act"] = FakeChoice("answer", {"answer": 1.0}, 0.9)
            answers["length"] = FakeScore(1.0)
            answers["grounded"] = FakeNoul(0.8)
        kind = self.kinds.pop(0) if self.kinds else "eos"
        winner, mass = _next_dist(questions, kind)
        answers["next"] = FakeChoice(winner, mass, 0.7)
        for key, question in questions.items():
            if key.startswith("then_"):
                ids = list(question.criteria)
                answers[key] = FakeChoice(ids[0], {ids[0]: 1.0}, 0.1)
        return FakeResponse(answers)


async def test_decoder_stops_on_eos(tmp_path: Path) -> None:
    words = tmp_path / "words.txt"
    words.write_text("hello\nworld\n", encoding="utf-8")
    client = KindClient(["phrase", "eos"])
    decoder = SystemOneDecoder(
        client,
        codebook=Codebook(words),
        config=DecodeConfig(temperature=0.0, max_calls=6, seed=0, speculative_confidence=1.1),
    )
    events = [
        event
        async for event in decoder.generate([ChatMessage(role="user", content="Hi")])
    ]
    kinds = [event.kind for event in events]
    assert kinds[0] == "call"
    assert "plan" in kinds
    assert "token" in kinds
    assert kinds[-1] == "done"
    done = events[-1]
    assert done.text
    assert done.data["usage"]["calls"] >= 1
