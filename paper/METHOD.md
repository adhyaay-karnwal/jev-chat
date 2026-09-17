# Decision-native language modeling

Jev is a System One model: given a state and a set of typed questions, it returns calibrated probabilities. It does not emit tokens. This note records how we still obtain an autoregressive chatbot from that interface, and why a naive next-letter loop is the wrong estimator.

## Problem

Let \(x_{<t}\) be the assistant prefix and \(V\) a surface vocabulary (phrases, words, punctuation, copy spans, characters). We want

\[
x_t \sim P_\theta(\cdot \mid x_{<t}, C)
\]

where \(C\) is the conversation and \(\theta\) is Jev. TypeSafe exposes \(P_\theta\) only as:

- `Choice` — a categorical distribution over at most 255 named options
- `Score` — an ordered rubric
- `Noul` — \(P(\text{yes})\)

Questions in one request are independent, evaluated in parallel against the same state. Extra questions cost tokens; they barely cost wall-clock. Serial requests cost both.

Character-level `Choice` over an alphabet is a valid language model. It is also linear in response length in API calls, which is too slow and too expensive for chat.

## Method

**S1LM** (System One Language Model) estimates \(P_\theta\) at an adaptive unit size, with a hierarchical tail for coverage and speculative fan-out for multi-unit progress per call.

### 1. Context-conditioned codebook

Code, not the model, proposes a primary candidate set \(U_t\) with \(|U_t| \le 200\):

- EOS
- discourse phrases that fit the prefix (openers vs mid-sentence glue)
- copy n-grams from the user message (pointer-generator)
- function words and digits
- high-frequency content words, or morphological completions if the prefix is mid-word
- an `other` option

This is the same move as TypeSafe's "select instead of generate" pattern: the model never invents a string outside \(U_t\).

### 2. One `Choice` is the next-token distribution

The primary decode step is a single `Choice` over \(U_t\). Sampling (temperature + nucleus) is ordinary. A companion `Noul` asks whether the prefix is already a complete reply; code stops when that probability exceeds a threshold after at least one unit.

On the first call only, the same request also asks for a speech-act `Choice`, a length `Score`, and a groundedness `Noul`. Those answers are written into subsequent state as `plan`. They do not generate text; they condition later decisions.

### 3. Hierarchical tail

If `other` wins, a second request factors the long tail:

\[
P(w) = P(\text{letter}(w))\,P(w \mid \text{letter}(w))
\]

The letter question and every per-letter word `Choice` are asked together (speculative fan-out). Code consumes only the matching bin. This is hierarchical softmax with TypeSafe as the scorer, and it is the same traversal idea as the hierarchical classification cookbook, pointed at a lexicon instead of a taxonomy.

### 4. Speculative fan-out of the next-next unit

Because questions cannot see one another, a follow-up `Choice` must state its premise in instructions: *assume the prefix was just extended by \(u\)*. The first request includes \(K\) such hypotheticals for the locally most likely units. After sampling \(u_t\), if a matching follow-up exists and its confidence is high enough, code also commits \(u_{t+1}\) without a round trip.

This is speculative decoding where the draft model is the same System One model, run independently rather than autoregressively.

## Complexity

Let \(T\) be response length in words and \(k\) mean words per committed unit (phrases plus accepted speculative follow-ups). API calls are \(\Theta(T/k)\) plus a rare tail call, not \(\Theta\)(characters). The conversation state is sent once per call; batching questions over that state is the cost win documented in TypeSafe's parallel-questions cookbook.

## What this is not

It is not a hidden LLM. Jev never produces the reply string. It is not guaranteed to match a decoder-only transformer trained with next-token prediction; RLCD trains calibrated decisions, not open-ended generation. Coverage of \(U_t\) is the binding constraint: if the right continuation is absent and `other` recovery fails, the decoder stops.

## Planned measurements

- Calls, tokens, and latency vs. a character-level ablation and a word-only ablation
- Acceptance rate of speculative follow-ups
- Human preference against a small instruction-tuned LM on short chat
- Calibration of `done` vs. annotator judgments of completeness

Those experiments are out of scope for the prototype; the implementation is instrumented to collect them.
