"""Context-conditioned candidate set for the next surface unit.

A Choice question accepts at most 255 options. English does not. The codebook
builds a *primary* set of up to ``MAX_PRIMARY`` mixed pieces (phrases, copy
spans, function words, content words, punctuation, EOS) and a *tail* of
per-letter expansions used only if the primary set votes ``other``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import cached_property
from pathlib import Path

from jevchat.types import ChatMessage, Piece, PieceKind

MAX_PRIMARY = 200
MAX_LETTER = 220
MAX_COPY = 24
MAX_PHRASES = 36
MAX_SPECULATIVE = 6
WORD_INITIAL = tuple("tasoiwcbphfmdrenlg")
DATA_DIR = Path(__file__).parent / "data"
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z0-9]")

FUNCTION_WORDS: tuple[str, ...] = (
    "the", "a", "an", "and", "or", "but", "if", "then", "when", "as", "at",
    "by", "for", "from", "in", "into", "of", "on", "to", "with", "without",
    "about", "after", "before", "between", "through", "during", "against",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "there", "here",
    "not", "no", "yes", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should", "may", "might", "must",
    "what", "which", "who", "where", "why", "how",
    "so", "too", "very", "just", "also", "only", "even", "still",
    "one", "two", "three", "up", "out", "off", "over", "under",
    "other", "another", "than", "because", "though",
    "let", "get", "make", "know", "think", "see", "want", "need",
)

OPENERS: tuple[str, ...] = (
    "Hello.", "Hi.", "Hey.", "Yes.", "No.", "Sure.", "Of course.",
    "I think", "I don't think", "I'm not sure.", "Good question.",
    "In short,", "The short answer is", "It depends.", "Thanks.",
    "Sorry,", "I can't help with that.", "That's right.", "Not exactly.",
    "Possibly.", "Probably.", "Unlikely.", "I agree.", "I disagree.",
    "One way to think about it:", "A simple example:", "Here's the idea.",
    "Let's be precise.", "In practice,", "The usual answer is",
)

MID_PHRASES: tuple[str, ...] = (
    "for example", "in other words", "on the other hand", "that said",
    "as a result", "in practice", "at least", "of course", "in general",
    "more precisely", "rather than", "such as", "as well as", "in this case",
    "at the same time", "in particular", "to be clear", "in fact",
    "and so on", "and then", "which means", "because of that",
)

PUNCT: tuple[str, ...] = (".", "?", "!", ",", ";", ":", "—", "...")
DIGITS: tuple[str, ...] = tuple(str(n) for n in range(10))


def _closed(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[-1].isspace():
        return stripped
    if stripped[-1] in ".?!,;:":
        return stripped + " "
    return stripped + " "


def current_partial(prefix: str) -> str:
    index = len(prefix)
    while index > 0 and prefix[index - 1].isalpha():
        index -= 1
    return prefix[index:]


def needs_joiner(prefix: str) -> bool:
    if not prefix:
        return False
    return prefix[-1].isalnum() or prefix[-1] in "')]}>\""


def looks_complete(prefix: str) -> bool:
    stripped = prefix.rstrip()
    return bool(stripped) and stripped[-1] in ".?!"


class Codebook:
    def __init__(
        self,
        words_path: Path | None = None,
        *,
        allow_phrases: bool = True,
        reopen_after_complete: bool = False,
    ) -> None:
        path = words_path or DATA_DIR / "words.txt"
        raw = path.read_text(encoding="utf-8").splitlines()
        self.words: tuple[str, ...] = tuple(
            word.strip() for word in raw if word.strip() and word.strip().isalpha()
        )
        by_letter: dict[str, list[str]] = defaultdict(list)
        for word in self.words:
            by_letter[word[0]].append(word)
        self.by_letter: dict[str, tuple[str, ...]] = {
            letter: tuple(values) for letter, values in by_letter.items()
        }
        self.allow_phrases = allow_phrases
        self.reopen_after_complete = reopen_after_complete

    @cached_property
    def function_set(self) -> set[str]:
        return {word.lower() for word in FUNCTION_WORDS}

    def propose(
        self,
        messages: list[ChatMessage],
        prefix: str,
        *,
        banned_labels: frozenset[str] | None = None,
    ) -> tuple[list[Piece], dict[str, list[Piece]]]:
        """Return ``(primary, letter_expansions)``."""

        partial = current_partial(prefix)
        pieces: list[Piece] = []
        used_surfaces: set[str] = set()
        banned = {label.lower() for label in (banned_labels or frozenset())}

        def add(surface: str, kind: PieceKind, label: str, description: str) -> None:
            if label.lower() in banned and kind != "eos":
                return
            if surface in used_surfaces and kind != "eos":
                return
            if len(pieces) >= MAX_PRIMARY - 1 and kind != "other":
                return
            used_surfaces.add(surface)
            pieces.append(
                Piece(
                    id=f"u{len(pieces)}",
                    surface=surface,
                    kind=kind,
                    label=label,
                    description=description,
                )
            )

        add("", "eos", "EOS", "End the assistant reply. Use only when the prefix is a complete answer.")

        if partial:
            for word in self._completions(partial)[: MAX_PRIMARY - 8]:
                suffix = word[len(partial):]
                if not suffix:
                    continue
                add(suffix + " ", "word", word, f"Finish the current word as '{word}'.")
            for char in "abcdefghijklmnopqrstuvwxyz'":
                add(char, "char", char, f"Append the character '{char}' to the current word.")
        else:
            if self.allow_phrases and not prefix:
                for phrase in OPENERS:
                    add(_closed(phrase), "phrase", phrase, f"Open the reply with {phrase!r}.")
            elif self.allow_phrases and looks_complete(prefix) and self.reopen_after_complete:
                for phrase in OPENERS[:12]:
                    add(_closed(phrase), "phrase", phrase, f"Start a new sentence with {phrase!r}.")
            elif self.allow_phrases and not looks_complete(prefix):
                for phrase in MID_PHRASES[:MAX_PHRASES]:
                    add(_closed(phrase), "phrase", phrase, f"Insert the phrase {phrase!r}.")

            for mark in PUNCT:
                add(_closed(mark.strip()), "punct", mark.strip(), f"Append punctuation {mark!r}.")

            for span in self._copy_spans(messages)[:MAX_COPY]:
                add(_closed(span), "copy", span, f"Reuse {span!r} from the conversation.")

            for word in FUNCTION_WORDS:
                add(_closed(word), "word", word, f"The function word '{word}'.")
            for digit in DIGITS:
                add(_closed(digit), "word", digit, f"The digit {digit}.")

            for word in self.words[:80]:
                if word.lower() in self.function_set:
                    continue
                add(_closed(word), "word", word, f"The word '{word}'.")

        add("[other]", "other", "other", "None of the listed continuations is right; use a rarer word.")

        expansions = self._letter_expansions(prefix, partial)
        return pieces, expansions

    def complete_replies(self, messages: list[ChatMessage]) -> list[Piece]:
        """Candidate *full* replies for one-shot selection."""

        pieces: list[Piece] = []
        seen: set[str] = set()

        def add(text: str, kind: PieceKind, description: str) -> None:
            closed = text.strip()
            if not closed or closed.lower() in seen or len(pieces) >= MAX_PRIMARY:
                return
            seen.add(closed.lower())
            pieces.append(
                Piece(
                    id=f"r{len(pieces)}",
                    surface=closed,
                    kind=kind,
                    label=closed,
                    description=description,
                )
            )

        for phrase in (
            "Hello.",
            "Hi.",
            "Yes.",
            "No.",
            "I don't know.",
            "I can't help with that.",
            "OK.",
        ):
            add(phrase, "phrase", f"Reply with {phrase!r}.")
        for digit in DIGITS:
            add(digit, "word", f"Reply with the number {digit}.")
        for span in self._copy_spans(messages)[:12]:
            add(span, "copy", f"Reply by repeating {span!r} from the user.")
        for word in (
            "Paris",
            "London",
            "Tuesday",
            "red",
            "blue",
            "yellow",
            "cat",
            "two",
            "four",
            "yes",
            "no",
        ):
            add(word, "word", f"Reply with {word!r}.")
        add("2", "word", "The integer two.")
        add("4", "word", "The integer four.")
        return pieces

    def speculative_targets(self, primary: list[Piece]) -> list[Piece]:
        ranked: list[Piece] = []
        for kind in ("phrase", "copy", "word", "punct"):
            ranked.extend(piece for piece in primary if piece.kind == kind)
        return ranked[:MAX_SPECULATIVE]

    def _completions(self, partial: str) -> list[str]:
        needle = partial.lower()
        matches = [word for word in self.words if word.startswith(needle) and word != needle]
        for word in FUNCTION_WORDS:
            lower = word.lower()
            if lower.startswith(needle) and lower != needle and lower not in matches:
                matches.append(word)
        return matches

    def _word_surface(self, word: str, prefix: str) -> str:
        partial = current_partial(prefix)
        if partial and word.lower().startswith(partial.lower()):
            return _closed(word[len(partial):])
        return _closed(word)

    def _informative(self, token: str, *, position: int) -> bool:
        if token.isdigit():
            return len(token) >= 2
        if not token.isalpha() or len(token) < 2:
            return False
        if token.lower() in self.function_set:
            return False
        if token[0].isupper() and position > 0:
            return True
        return len(token) >= 4

    def _copy_spans(self, messages: list[ChatMessage]) -> list[str]:
        texts = [message.content for message in messages if message.role == "user"]
        if not texts:
            return []
        latest = texts[-1]
        tokens = TOKEN_RE.findall(latest)
        spans: list[str] = []
        seen: set[str] = set()
        limit = max(8, len(latest) // 2)
        for index, token in enumerate(tokens):
            if not self._informative(token, position=index):
                continue
            if token.lower() in seen:
                continue
            seen.add(token.lower())
            spans.append(token)
        for index in range(len(tokens) - 1):
            left, right = tokens[index], tokens[index + 1]
            if not (
                self._informative(left, position=index)
                and self._informative(right, position=index + 1)
            ):
                continue
            span = f"{left} {right}"
            if span.lower() in seen or len(span) > limit:
                continue
            seen.add(span.lower())
            spans.append(span)
        return spans

    def _letter_expansions(self, prefix: str, partial: str) -> dict[str, list[Piece]]:
        expansions: dict[str, list[Piece]] = {}
        if partial:
            return expansions
        for letter in WORD_INITIAL:
            bucket = self.by_letter.get(letter, ())
            pieces: list[Piece] = []
            for word in bucket[:MAX_LETTER]:
                pieces.append(
                    Piece(
                        id=f"{letter}{len(pieces)}",
                        surface=_closed(word),
                        kind="word",
                        label=word,
                        description=f"The word '{word}'.",
                    )
                )
            if pieces:
                expansions[letter] = pieces
        return expansions
