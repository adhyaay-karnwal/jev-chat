"""Build TypeSafe state and questions for one decode step."""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, Score

from jevchat.codebook import looks_complete
from jevchat.types import ChatMessage, Piece, Plan

ACT_CRITERIA = {
    "greet": {
        "what": "A greeting or social opener",
        "not_for": "Answering a question",
    },
    "answer": {
        "what": "Directly answer a question in the last user message",
        "not_for": "A greeting or a refusal",
    },
    "explain": {
        "what": "Explain a concept or mechanism",
        "not_for": "A one-word factual answer",
    },
    "clarify": {
        "what": "Ask a clarifying question because the request is underspecified",
        "not_for": "Answering with the available information",
    },
    "refuse": {
        "what": "Decline because the request is disallowed, unsafe, or impossible",
        "not_for": "Ordinary uncertainty about a benign topic",
    },
    "smalltalk": {
        "what": "Casual conversation with no information request",
        "not_for": "A factual or how-to question",
    },
    "list": {
        "what": "Give a short enumerated list",
        "not_for": "A single-sentence definition",
    },
    "correct": {
        "what": "Correct a mistaken premise in the user message",
        "not_for": "Agreeing or answering as asked",
    },
}


def conversation_state(
    messages: list[ChatMessage],
    prefix: str,
    plan: Plan | None,
) -> dict[str, object]:
    return {
        "task": (
            "Judge the next surface unit of a careful assistant reply. "
            "Code will append your choice to `assistant_prefix`. "
            "Do not write a full message; choose only the next unit."
        ),
        "conversation": [
            {"role": message.role, "text": message.content} for message in messages
        ],
        "assistant_prefix": prefix,
        "prefix_is_empty": not prefix,
        "prefix_looks_complete": looks_complete(prefix),
        "plan": None
        if plan is None
        else {
            "act": plan.act,
            "length_score": plan.length_score,
            "grounded": plan.grounded,
        },
    }


def piece_criteria(pieces: list[Piece]) -> dict[str, str]:
    criteria: dict[str, str] = {}
    for piece in pieces:
        if piece.kind == "eos":
            criteria[piece.id] = "End the assistant reply."
        elif piece.kind == "other":
            criteria[piece.id] = "None of the listed units is right; a rarer word is needed."
        else:
            criteria[piece.id] = f"{piece.kind} {piece.label!r} appends {piece.surface!r}"
    return criteria


def step_questions(
    primary: list[Piece],
    expansions: dict[str, list[Piece]],
    speculative: list[Piece],
    *,
    include_plan: bool,
    follow_up_sets: dict[str, list[Piece]],
) -> dict[str, Choice | Noul | Score]:
    questions: dict[str, Choice | Noul | Score] = {
        "done": Noul(
            instructions={
                "question": "Is `assistant_prefix` already a complete assistant reply?",
                "true": "The prefix is grammatical, addresses the user, and should not be extended.",
                "false": "The prefix is empty, mid-sentence, or otherwise incomplete.",
            }
        ),
        "next": Choice(
            instructions={
                "question": "Which surface unit should be appended next to `assistant_prefix`?",
                "focus": (
                    "Pick the continuation a careful assistant would write next, "
                    "given `conversation` and `plan`."
                ),
                "constraint": (
                    "Prefer a longer accurate phrase over spelling the same text "
                    "word by word. Use EOS only when the reply is complete. "
                    "Use other only if no listed unit is acceptable."
                ),
            },
            criteria=piece_criteria(primary),
        ),
    }

    for letter, pieces in expansions.items():
        questions[f"words_{letter}"] = Choice(
            instructions={
                "question": (
                    f"Assume the next word starts with '{letter}'. "
                    "Which of these words should be appended to `assistant_prefix`?"
                ),
                "focus": "This question is used only if the primary choice is a rarer word.",
            },
            criteria=piece_criteria(pieces),
        )

    for piece in speculative:
        follow = follow_up_sets.get(piece.id, [])
        if len(follow) < 2:
            continue
        questions[f"then_{piece.id}"] = Choice(
            instructions={
                "question": (
                    "Assume `assistant_prefix` has just been extended by exactly "
                    f"{piece.surface!r}. What comes immediately after?"
                ),
                "premise": f"The next unit is {piece.label!r} ({piece.kind}).",
            },
            criteria=piece_criteria(follow),
        )

    if include_plan:
        questions["act"] = Choice(
            instructions={
                "question": "What should the assistant's next reply do?",
                "focus": "Classify the speech act of the full reply, not of a single token.",
            },
            criteria=ACT_CRITERIA,
        )
        questions["length"] = Score(
            instructions="How long should the full assistant reply be?",
            criteria=[
                "A few words",
                "One short sentence",
                "Two sentences",
                "A short paragraph",
            ],
        )
        questions["grounded"] = Noul(
            instructions=(
                "Can a correct reply be produced from `conversation` plus ordinary "
                "common knowledge, without tools or private data?"
            )
        )

    return questions
