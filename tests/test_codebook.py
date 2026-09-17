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


def test_no_reopen_after_complete(tmp_path: Path) -> None:
    words = tmp_path / "words.txt"
    words.write_text("hello\nworld\n", encoding="utf-8")
    codebook = Codebook(words, reopen_after_complete=False)
    messages = [ChatMessage(role="user", content="Hi")]
    primary, _ = codebook.propose(messages, "Hello. ")
    labels = {piece.label for piece in primary}
    assert "Hello." not in labels
    assert "EOS" in labels


def test_complete_replies_include_digits(tmp_path: Path) -> None:
    words = tmp_path / "words.txt"
    words.write_text("hello\n", encoding="utf-8")
    replies = Codebook(words).complete_replies(
        [ChatMessage(role="user", content="What is 1+1")]
    )
    labels = {piece.label for piece in replies}
    assert "2" in labels
    assert "Hello." in labels
