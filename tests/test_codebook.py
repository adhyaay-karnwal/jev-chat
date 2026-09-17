from pathlib import Path

from jevchat.codebook import Codebook, current_partial, needs_joiner
from jevchat.types import ChatMessage


def test_partial_and_joiner() -> None:
    assert current_partial("Hel") == "Hel"
    assert current_partial("Hello ") == ""
    assert needs_joiner("Hello")
    assert not needs_joiner("")
    assert not needs_joiner("Hello ")


def test_copy_and_midword(tmp_path: Path) -> None:
    words = tmp_path / "words.txt"
    words.write_text("hello\nhelp\nhelium\nworld\nfrance\nparis\n", encoding="utf-8")
    codebook = Codebook(words)
    messages = [ChatMessage(role="user", content="What is the capital of France?")]
    primary, _expansions = codebook.propose(messages, "")
    labels = {piece.label.lower() for piece in primary}
    assert "france" in labels
    assert any(piece.kind == "eos" for piece in primary)
    assert any(piece.kind == "other" for piece in primary)

    mid, _ = codebook.propose(messages, "Par")
    assert any(piece.label == "paris" and piece.surface.strip() == "is" for piece in mid)
