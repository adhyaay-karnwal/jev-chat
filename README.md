# jev-chat

A research decoder that treats Jev (a System One **decision** model) as if it were a language model.

[Jev](https://typesafe.ai) returns `Choice` / `Score` / `Noul` distributions. It does not emit tokens. This repo asks for the next surface unit anyway, samples, and appends. Autoregression over that interface is the wrong use of the model; one-shot **selection** of a complete reply is the TypeSafe-native one. The paper measures both.

Paper: [paper/main.pdf](paper/main.pdf) · traces: [experiments/results/runs.json](experiments/results/runs.json)

## Method

Documented in [`paper/METHOD.md`](paper/METHOD.md) and the paper:

1. **Hierarchical codebook.** A `Choice` has at most 255 options. The primary set mixes phrases, copy spans from the user, function words, and frequent content words, plus `other`. The long tail is a letter, then a word — asked in parallel only when `other` wins.
2. **Speculative fan-out.** Independent questions share one state. Each call asks for the next unit *and* hypothetical follow-ups (“assume the prefix was just extended by \(u\)”). Code stitches an accepted path.
3. **Adaptive unit size.** Phrases when they fit, words otherwise, characters only to finish a partial word. Copy n-grams so names and numbers are not spelled.

Jev never emits the reply. The string is assembled in code.

## Install

Requires Python 3.11+ and a TypeSafe API key.

```bash
git clone https://github.com/adhyaay-karnwal/jev-chat
cd jev-chat
uv sync --extra dev
cp .env.example .env   # set TYPESAFE_API_KEY
```

## Use

```bash
uv run jevchat "What is a System One model?"
uv run jevchat --serve          # http://127.0.0.1:8765
uv run pytest
uv run --with matplotlib python experiments/bench.py
```

The demo UI streams tokens as they are sampled and shows the per-call `Choice` mass in a side trace. That trace is the system: there is no hidden generator behind it.

## Layout

```
src/jevchat/          decoder, codebook, TypeSafe client, demo server
paper/METHOD.md       method note for a later paper
tests/                codebook, sampling, and a scripted decoder
.agents/skills/       TypeSafe agent skill (project-local)
```

## Limits

Naive autoregression loops greetings and copies operands (`1+1 → 1`). Greedy decoding recovers short facts and still emits ungrammatical strings. Select is grammatical when the answer is in the candidate list, and silent otherwise. See the paper.

## License

MIT. Not affiliated with TypeSafe AI.
