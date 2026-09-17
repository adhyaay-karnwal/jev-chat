# jev-chat

A working chatbot whose next token is a **typed decision**, not a generated string.

[Jev](https://typesafe.ai) is TypeSafe's System One model. It evaluates `Choice`, `Score`, and `Noul` questions against a state and returns calibrated probabilities. It does not write text. This repository turns that constraint into an autoregressive decoder: code proposes a bounded set of next surface units, Jev returns \(P(u_t \mid \text{conversation}, \text{prefix})\), code samples and appends.

Naive next-letter decoding is linear in response length in API calls. The decoder here is built to stay sublinear.

## Method

Three uses of the System One programming model, documented in [`paper/METHOD.md`](paper/METHOD.md):

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

Coverage of the candidate set is the binding constraint. If the right continuation is not in the codebook and tail recovery fails, the decoder stops. Jev is trained for calibrated decisions (RLCD), not open-ended generation; this is an estimator on top of that interface, not a claim that System One models are decoder-only transformers.

## License

MIT. Not affiliated with TypeSafe AI.
