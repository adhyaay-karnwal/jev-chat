"""Adaptive hierarchical speculative decoding with Jev."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from random import Random
from typing import Any

from typesafe_sdk import Choice

from jevchat.codebook import Codebook, looks_complete
from jevchat.sampling import sample_from, top_items
from jevchat.state import conversation_state, piece_criteria, step_questions
from jevchat.types import ChatMessage, DecodeEvent, Piece, Plan, Usage


def _with_ids(pieces: list[Piece], prefix: str) -> list[Piece]:
    return [
        Piece(
            id=f"{prefix}{index}",
            surface=piece.surface,
            kind=piece.kind,
            label=piece.label,
            description=piece.description,
        )
        for index, piece in enumerate(pieces)
    ]


@dataclass(frozen=True)
class DecodeConfig:
    model: str = "jev-latest"
    temperature: float = 0.4
    top_p: float = 0.92
    eos_threshold: float = 0.72
    complete_eos_threshold: float = 0.40
    other_threshold: float = 0.35
    speculative_confidence: float = 0.42
    max_calls: int = 14
    max_chars: int = 480
    seed: int | None = None
    block_repeat: bool = True
    speculative: bool = True
    mode: str = "decode"  # "decode" or "select"


class SystemOneDecoder:
    """Turn calibrated Choice distributions into streamed assistant text."""

    def __init__(
        self,
        client: Any,
        *,
        codebook: Codebook | None = None,
        config: DecodeConfig | None = None,
    ) -> None:
        self.client = client
        self.codebook = codebook or Codebook()
        self.config = config or DecodeConfig()

    async def generate(self, messages: list[ChatMessage]) -> AsyncIterator[DecodeEvent]:
        if self.config.mode == "select":
            async for event in self._select(messages):
                yield event
            return

        config = self.config
        rng = Random(config.seed)
        usage = Usage()
        prefix = ""
        plan: Plan | None = None
        used_labels: list[str] = []
        stop = "max_calls"

        for _ in range(config.max_calls):
            if len(prefix) >= config.max_chars:
                stop = "max_chars"
                break

            banned = frozenset(used_labels[-6:] if config.block_repeat else ())
            primary, expansions = self.codebook.propose(
                messages, prefix, banned_labels=banned
            )
            speculative = (
                self.codebook.speculative_targets(primary) if config.speculative else []
            )
            follow_up_sets = {
                piece.id: _with_ids(
                    self.codebook.propose(messages, prefix + piece.surface)[0][:24],
                    f"f{piece.id}_",
                )
                for piece in speculative
            }
            questions = step_questions(
                primary,
                {},
                speculative,
                include_plan=plan is None,
                follow_up_sets=follow_up_sets,
            )
            state = conversation_state(messages, prefix, plan)
            by_id = {piece.id: piece for piece in primary}
            for pieces in expansions.values():
                by_id.update({piece.id: piece for piece in pieces})
            for pieces in follow_up_sets.values():
                by_id.update({piece.id: piece for piece in pieces})

            started = time.perf_counter()
            response = await self.client.evaluate(state, questions, model=config.model)
            latency = time.perf_counter() - started
            input_tokens, output_tokens = _usage_of(response)
            usage.add(input_tokens, output_tokens, latency)
            answers = response.answers

            yield DecodeEvent(
                kind="call",
                data={
                    "call": usage.calls,
                    "latency_s": round(latency, 3),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "questions": list(questions),
                    "top": _named_top(primary, answers["next"].probabilities),
                    "done_p": round(float(answers["done"].noul), 4),
                },
            )

            if plan is None and "act" in answers:
                plan = Plan(
                    act=answers["act"].choice,
                    act_confidence=float(answers["act"].confidence),
                    length_score=float(answers["length"].score),
                    grounded=float(answers["grounded"].noul),
                )
                yield DecodeEvent(
                    kind="plan",
                    data={
                        "act": plan.act,
                        "act_confidence": plan.act_confidence,
                        "length_score": plan.length_score,
                        "grounded": plan.grounded,
                    },
                )

            done_p = float(answers["done"].noul)
            eos_cut = (
                config.complete_eos_threshold
                if looks_complete(prefix)
                else config.eos_threshold
            )
            if prefix and done_p >= eos_cut:
                stop = "done_noul"
                break

            next_id = sample_from(
                answers["next"].probabilities,
                temperature=config.temperature,
                top_p=config.top_p,
                rng=rng,
            )
            piece = by_id[next_id]

            if piece.kind == "eos":
                if prefix:
                    stop = "eos_choice"
                    break
                next_id = _mode_excluding(answers["next"].probabilities, {piece.id})
                piece = by_id[next_id]

            if piece.kind == "other":
                recovered = await self._expand_other(
                    messages, prefix, expansions, usage, rng
                )
                if recovered is None:
                    stop = "other_fail"
                    break
                piece = recovered
                by_id[piece.id] = piece

            prefix += piece.surface
            used_labels.append(piece.label)
            yield DecodeEvent(
                kind="token",
                text=piece.surface,
                data=_token_data(
                    piece, answers["next"].probabilities.get(piece.id, 0.0), False
                ),
            )

            then_key = f"then_{piece.id}"
            if then_key in answers and len(prefix) < config.max_chars:
                follow = answers[then_key]
                if float(follow.confidence) >= config.speculative_confidence:
                    follow_id = sample_from(
                        follow.probabilities,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        rng=rng,
                    )
                    follow_piece = by_id.get(follow_id)
                    if (
                        follow_piece is not None
                        and follow_piece.kind not in {"eos", "other"}
                        and follow_piece.surface
                    ):
                        prefix += follow_piece.surface
                        used_labels.append(follow_piece.label)
                        yield DecodeEvent(
                            kind="token",
                            text=follow_piece.surface,
                            data=_token_data(
                                follow_piece,
                                follow.probabilities.get(follow_id, 0.0),
                                True,
                            ),
                        )

        yield DecodeEvent(
            kind="done",
            text=prefix.strip(),
            data={
                "stop": stop,
                "usage": {
                    "calls": usage.calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "latency_s": round(usage.latency_s, 3),
                },
                "plan": None
                if plan is None
                else {
                    "act": plan.act,
                    "act_confidence": plan.act_confidence,
                    "length_score": plan.length_score,
                    "grounded": plan.grounded,
                },
            },
        )

    async def _select(self, messages: list[ChatMessage]) -> AsyncIterator[DecodeEvent]:
        config = self.config
        rng = Random(config.seed)
        usage = Usage()
        replies = self.codebook.complete_replies(messages)
        questions = {
            "reply": Choice(
                instructions={
                    "question": "Which complete assistant reply should be sent?",
                    "focus": "Choose one full reply. Do not continue after this choice.",
                    "constraint": "Prefer a correct short answer over a greeting or a copied question span.",
                },
                criteria=piece_criteria(replies),
            )
        }
        state = conversation_state(messages, "", None)
        started = time.perf_counter()
        response = await self.client.evaluate(state, questions, model=config.model)
        usage.add(*_usage_of(response), time.perf_counter() - started)
        answer = response.answers["reply"]
        yield DecodeEvent(
            kind="call",
            data={
                "call": 1,
                "latency_s": round(usage.latency_s, 3),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "questions": ["reply"],
                "top": _named_top(replies, answer.probabilities),
            },
        )
        chosen_id = sample_from(
            answer.probabilities,
            temperature=config.temperature,
            top_p=config.top_p,
            rng=rng,
        )
        by_id = {piece.id: piece for piece in replies}
        piece = by_id[chosen_id]
        yield DecodeEvent(
            kind="token",
            text=piece.surface,
            data=_token_data(piece, answer.probabilities.get(chosen_id, 0.0), False),
        )
        yield DecodeEvent(
            kind="done",
            text=piece.surface,
            data={
                "stop": "select",
                "usage": {
                    "calls": usage.calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "latency_s": round(usage.latency_s, 3),
                },
                "plan": None,
            },
        )

    async def _expand_other(
        self,
        messages: list[ChatMessage],
        prefix: str,
        expansions: dict[str, list[Piece]],
        usage: Usage,
        rng: Random,
    ) -> Piece | None:
        if not expansions:
            return None
        state = conversation_state(messages, prefix, None)
        letter_question = {
            "letter": Choice(
                instructions={
                    "question": (
                        "The listed continuations were insufficient. "
                        "What does the next word start with?"
                    ),
                    "focus": "Choose the first letter of the word a careful assistant would write next.",
                },
                criteria={
                    letter: f"A word starting with '{letter}'" for letter in expansions
                },
            )
        }
        letter_response = await self._call(state, letter_question, usage)
        letter = letter_response.answers["letter"].choice
        if letter not in expansions:
            letter = max(
                expansions,
                key=lambda item: letter_response.answers["letter"].probabilities.get(
                    item, 0.0
                ),
            )
        pieces = expansions[letter]
        word_question = {
            "word": Choice(
                instructions={
                    "question": (
                        f"Assume the next word starts with '{letter}'. "
                        "Which word should be appended to `assistant_prefix`?"
                    )
                },
                criteria=piece_criteria(pieces),
            )
        }
        word_response = await self._call(state, word_question, usage)
        chosen = sample_from(
            word_response.answers["word"].probabilities,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            rng=rng,
        )
        for piece in pieces:
            if piece.id == chosen:
                return piece
        return None

    async def _call(self, state: object, questions: dict[str, Any], usage: Usage) -> Any:
        started = time.perf_counter()
        response = await self.client.evaluate(
            state, questions, model=self.config.model
        )
        usage.add(*_usage_of(response), time.perf_counter() - started)
        return response


def _usage_of(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(
        getattr(usage, "output_tokens", 0) or 0
    )


def _named_top(
    primary: list[Piece], probabilities: dict[str, float]
) -> list[dict[str, object]]:
    labels = {piece.id: piece.label for piece in primary}
    return [
        {"id": key, "label": labels.get(key, key), "p": round(prob, 4)}
        for key, prob in top_items(probabilities, 8)
    ]


def _token_data(piece: Piece, probability: float, speculative: bool) -> dict[str, object]:
    return {
        "id": piece.id,
        "kind": piece.kind,
        "label": piece.label,
        "p": round(float(probability), 4),
        "speculative": speculative,
    }


def _mode_excluding(probabilities: dict[str, float], excluded: set[str]) -> str:
    rest = {key: prob for key, prob in probabilities.items() if key not in excluded}
    if not rest:
        return next(iter(probabilities))
    return max(rest, key=rest.get)  # type: ignore[arg-type]


