"""Nucleus sampling over a Choice distribution."""

from __future__ import annotations

import math
import random


def sample_from(
    probabilities: dict[str, float],
    *,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> str:
    """Draw a key from a probability map.

    ``temperature <= 0`` selects the mode. ``top_p`` is applied after the
    temperature softmax.
    """

    items = [(key, max(float(prob), 1e-12)) for key, prob in probabilities.items()]
    if not items:
        raise ValueError("cannot sample from an empty distribution")
    if temperature <= 0:
        return max(items, key=lambda item: item[1])[0]

    logits = [math.log(prob) / temperature for _, prob in items]
    offset = max(logits)
    weights = [math.exp(logit - offset) for logit in logits]
    total = sum(weights)
    probs = [weight / total for weight in weights]

    ranked = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)
    kept: list[int] = []
    cumulative = 0.0
    threshold = min(max(top_p, 1e-6), 1.0)
    for index in ranked:
        kept.append(index)
        cumulative += probs[index]
        if cumulative >= threshold:
            break

    mass = sum(probs[index] for index in kept)
    draw = rng.random() * mass
    running = 0.0
    for index in kept:
        running += probs[index]
        if running >= draw:
            return items[index][0]
    return items[kept[-1]][0]


def top_items(probabilities: dict[str, float], k: int) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]
