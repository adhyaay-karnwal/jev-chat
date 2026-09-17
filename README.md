# jev-chat

A working chatbot whose next token is a **typed decision**, not a generated string.

[Jev](https://typesafe.ai) is TypeSafe's System One model: it returns calibrated probabilities over `Choice`, `Score`, and `Noul` questions. It does not emit text. This repository turns that constraint into a language model by asking Jev, in parallel, for the distribution over the next surface unit, then sampling.

Naive next-letter decoding is too slow and too expensive. The decoder here uses three ideas from the System One programming model:

1. **Hierarchical codebook** — a 255-option `Choice` cannot cover English. We factor \(P(x_t \mid x_{<t})\) into a cluster, then a piece, then (rarely) a character.
2. **Speculative fan-out** — independent questions over the same state run in one call. The decoder asks the next unit *and* hypothetical follow-ups, then stitches an accepted path in code.
3. **Adaptive unit size** — phrases when the model is confident, words otherwise, characters only as a fallback. Copy spans from the conversation so names and numbers do not have to be spelled.

This is research software intended for an eventual paper. See [`paper/METHOD.md`](paper/METHOD.md) for the method.

## Status

Early public prototype. The model never writes a sentence; code does, from sampled decisions.

## License

MIT
