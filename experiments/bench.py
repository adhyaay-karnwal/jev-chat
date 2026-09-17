"""Live decoding ablations against Jev.

Run from the repo root:

    uv run --with matplotlib python experiments/bench.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from jevchat.client import JevClient
from jevchat.codebook import Codebook
from jevchat.decoder import DecodeConfig, SystemOneDecoder
from jevchat.types import ChatMessage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"
FIG = ROOT / "paper" / "figures"

PROMPTS: list[dict[str, str]] = [
    {"id": "greet", "prompt": "Hello, this is a test", "expect": "single_hello"},
    {"id": "add1", "prompt": "What is 1+1", "expect": "2"},
    {"id": "add2", "prompt": "What is 2+2? Reply with just the number.", "expect": "4"},
    {"id": "france", "prompt": "What is the capital of France? One sentence.", "expect": "paris"},
    {"id": "water", "prompt": "Is water wet? Yes or no.", "expect": "yes"},
    {"id": "color", "prompt": "Name one primary color.", "expect": "color"},
    {"id": "monday", "prompt": "What day comes after Monday?", "expect": "tuesday"},
    {"id": "cat", "prompt": "Spell the word cat in letters.", "expect": "cat"},
]


def score(expect: str, text: str) -> bool:
    t = text.strip().lower()
    if expect == "single_hello":
        return t.count("hello") == 1 and t.startswith("hello")
    if expect == "2":
        return bool(re.search(r"\b2\b", t)) and not t.endswith("1")
    if expect == "4":
        return bool(re.search(r"\b4\b", t))
    if expect == "paris":
        return "paris" in t
    if expect == "yes":
        return t.startswith("yes") or t == "yes."
    if expect == "color":
        return any(c in t for c in ("red", "blue", "yellow"))
    if expect == "tuesday":
        return "tuesday" in t
    if expect == "cat":
        return "cat" in t.replace(" ", "")
    return False


def repetition(text: str) -> float:
    tokens = text.lower().split()
    if len(tokens) < 2:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


STRATEGIES = {
    "naive_ar": {
        "codebook": {"reopen_after_complete": True, "allow_phrases": True},
        "config": DecodeConfig(
            temperature=0.7,
            block_repeat=False,
            complete_eos_threshold=0.72,
            eos_threshold=0.72,
            speculative=True,
            max_calls=8,
            max_chars=240,
            seed=0,
            mode="decode",
        ),
    },
    "constrained_ar": {
        "codebook": {"reopen_after_complete": False, "allow_phrases": True},
        "config": DecodeConfig(
            temperature=0.3,
            block_repeat=True,
            complete_eos_threshold=0.40,
            eos_threshold=0.72,
            speculative=True,
            max_calls=8,
            max_chars=240,
            seed=0,
            mode="decode",
        ),
    },
    "greedy_ar": {
        "codebook": {"reopen_after_complete": False, "allow_phrases": True},
        "config": DecodeConfig(
            temperature=0.0,
            block_repeat=True,
            complete_eos_threshold=0.40,
            eos_threshold=0.72,
            speculative=False,
            max_calls=8,
            max_chars=240,
            seed=0,
            mode="decode",
        ),
    },
    "select": {
        "codebook": {"reopen_after_complete": False, "allow_phrases": True},
        "config": DecodeConfig(
            temperature=0.0,
            max_calls=1,
            seed=0,
            mode="select",
        ),
    },
}


@dataclass
class Row:
    strategy: str
    prompt_id: str
    prompt: str
    text: str
    ok: bool
    calls: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    stop: str
    repetition: float
    done_curve: list[float]


async def run_one(client: JevClient, strategy: str, item: dict[str, str]) -> Row:
    spec = STRATEGIES[strategy]
    codebook = Codebook(**spec["codebook"])
    decoder = SystemOneDecoder(client, codebook=codebook, config=spec["config"])
    messages = [ChatMessage(role="user", content=item["prompt"])]
    text = ""
    usage = {"calls": 0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0}
    stop = "unknown"
    done_curve: list[float] = []
    async for event in decoder.generate(messages):
        if event.kind == "call":
            if "done_p" in event.data:
                done_curve.append(float(event.data["done_p"]))
        elif event.kind == "done":
            text = event.text
            usage = event.data.get("usage", usage)
            stop = str(event.data.get("stop", "unknown"))
    return Row(
        strategy=strategy,
        prompt_id=item["id"],
        prompt=item["prompt"],
        text=text,
        ok=score(item["expect"], text),
        calls=int(usage.get("calls", 0)),
        latency_s=float(usage.get("latency_s", 0)),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        stop=stop,
        repetition=round(repetition(text), 3),
        done_curve=done_curve,
    )


async def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY required")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    client = JevClient(api_key=os.environ["TYPESAFE_API_KEY"])
    rows: list[Row] = []
    try:
        for strategy in STRATEGIES:
            for item in PROMPTS:
                print(f"→ {strategy:16} {item['id']:8}", flush=True)
                row = await run_one(client, strategy, item)
                rows.append(row)
                print(f"  ok={row.ok} calls={row.calls} {row.text!r}", flush=True)
    finally:
        await client.aclose()

    payload = [asdict(row) for row in rows]
    (OUT / "runs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(summary, rows)
    print(json.dumps(summary, indent=2))


def summarize(rows: list[Row]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        group = [row for row in rows if row.strategy == strategy]
        out[strategy] = {
            "n": len(group),
            "accuracy": mean(row.ok for row in group),
            "calls": mean(row.calls for row in group),
            "latency_s": mean(row.latency_s for row in group),
            "input_tokens": mean(row.input_tokens for row in group),
            "repetition": mean(row.repetition for row in group),
        }
    return out


def mean(values) -> float:
    seq = [float(v) for v in values]
    return round(statistics.fmean(seq), 4) if seq else 0.0


def plot(summary: dict[str, dict[str, float]], rows: list[Row]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    names = list(STRATEGIES)
    labels = ["naive AR", "constrained AR", "greedy AR", "select"]
    colors = ["#4a4a4a", "#2b6cb0", "#2f855a", "#c05621"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    metrics = [
        ("accuracy", "Task accuracy"),
        ("calls", "API calls"),
        ("latency_s", "Latency (s)"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        vals = [summary[name][key] for name in names]
        ax.bar(labels, vals, color=colors, width=0.72)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)
        if key == "accuracy":
            ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG / "strategy_bars.pdf")
    fig.savefig(FIG / "strategy_bars.png")
    plt.close(fig)

    # Per-prompt heatmap-like table as grouped bars for accuracy
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    prompt_ids = [item["id"] for item in PROMPTS]
    x = list(range(len(prompt_ids)))
    width = 0.2
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        vals = [
            1.0
            if any(row.prompt_id == pid and row.strategy == name and row.ok for row in rows)
            else 0.0
            for pid in prompt_ids
        ]
        ax.bar([xi + (i - 1.5) * width for xi in x], vals, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_ids)
    ax.set_ylabel("Correct")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    ax.set_title("Correctness by prompt")
    fig.tight_layout()
    fig.savefig(FIG / "per_prompt.pdf")
    fig.savefig(FIG / "per_prompt.png")
    plt.close(fig)

    greet = next((row for row in rows if row.strategy == "naive_ar" and row.prompt_id == "greet"), None)
    constrained = next((row for row in rows if row.strategy == "constrained_ar" and row.prompt_id == "greet"), None)
    if greet and greet.done_curve:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        ax.plot(range(1, len(greet.done_curve) + 1), greet.done_curve, color="#4a4a4a", marker="o", label="naive AR")
        if constrained and constrained.done_curve:
            ax.plot(
                range(1, len(constrained.done_curve) + 1),
                constrained.done_curve,
                color="#2b6cb0",
                marker="o",
                label="constrained AR",
            )
        ax.axhline(0.72, color="#999", ls="--", lw=0.8, label="EOS threshold")
        ax.set_xlabel("Decode step")
        ax.set_ylabel(r"$P(\mathrm{done})$")
        ax.set_title("Greeting: termination mass")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "done_curve.pdf")
        fig.savefig(FIG / "done_curve.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.scatter(
        [summary[n]["latency_s"] for n in names],
        [summary[n]["accuracy"] for n in names],
        c=colors,
        s=48,
        zorder=3,
    )
    for n, lab in zip(names, labels):
        ax.annotate(lab, (summary[n]["latency_s"], summary[n]["accuracy"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Mean latency (s)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accuracy–latency")
    fig.tight_layout()
    fig.savefig(FIG / "pareto.pdf")
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)


if __name__ == "__main__":
    asyncio.run(main())
