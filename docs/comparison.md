# Text comparison protocols

SpeechScope compares text after a selected normalization profile. Normalization and tokenization are part of the scoring protocol: scores are comparable only when the same profiles, token boundaries, and language conventions are used.

## Normalization profiles

- `basic-v1`: Unicode NFKC, casefolding, punctuation converted to separators, and whitespace collapsed. Accents remain; underscores and internal apostrophes remain.
- `strict-v1`: Unicode NFKC and whitespace collapse, preserving case and punctuation.
- `readio-compat-v1`: the documented compatibility punctuation behavior.

These profiles are stable identifiers. They do not expand numbers, contractions, abbreviations, spelling variants, or compounds, and they do not perform language-specific segmentation. New semantics require a new profile.

## Strict word metrics

`wer`, `mer`, `wil`, `wip`, and `word_matches` / `word_substitutions` / `word_deletions` / `word_insertions` retain conventional whitespace-tokenized Levenshtein semantics after normalization. They are unchanged by the compound-aware scorer. WER can exceed 1 when insertions outnumber reference tokens.

## Compound-aware word metrics

`compound_wer` is a second, explicit protocol. It uses the same normalized tokens and a bounded dynamic program that allows a zero-cost boundary match only when adjacent tokens on one side concatenate exactly to a single token on the other side. The multi-token span has two or three parts; unrestricted many-to-many regrouping is not allowed. Examples:

```text
audiobook  <-> audio book   accepted boundary variant
whitepaper <-> white paper  accepted boundary variant
audiobook  <-> audio books  not accepted
["a", "bc"] <-> ["ab", "c"] not accepted
```

A boundary event has zero lexical cost but is not hidden: `word_boundary_match_count` counts events and the report includes `word_boundary_variant` diagnostics with zero-based, half-open token spans and the normalized text. `compound_word_matches` counts the reference tokens consumed by exact and boundary matches. Compound substitutions, deletions, and insertions remain errors. `compound_wer` is their sum divided by the strict reference token count; it is not ordinary WER. `content_match` is true exactly when all three compound lexical error counts are zero.

No text-only rule can always decide whether a boundary changes meaning. `nowhere` / `now here` can be a zero-error compound match, but it remains visible as a boundary variant. Review diagnostics or use strict thresholds when orthographic boundaries are semantically significant.

## Character error rate

CER compares normalized character streams after removing ordinary spaces. It complements word metrics when tokenization is unstable, but it deliberately cannot report whitespace differences. A zero CER alone is not a claim that every word boundary is harmless; use the explicit compound diagnostics as well.

## Threshold examples

For exact spoken-content validation that tolerates only the bounded compound-boundary cases:

```json
{
  "schema_version": "1",
  "max_compound_wer": 0.0,
  "max_cer": 0.0,
  "missing_timing": "review"
}
```

For the reported `audiobook` / `audio book` case, strict WER remains `2/19`, compound WER is `0`, and CER is `0`; the policy above passes while exposing one boundary variant. Do not substitute `max_wer: 0.0` for `max_compound_wer`: strict WER remains a useful diagnostic and an intentional exact-boundary gate. Configured rules are conjunctive; any failing rule makes the evaluation fail.

## Multilingual limitations

WER depends on language-specific orthography and segmentation. Languages commonly written without whitespace may produce one token under `basic-v1`; this compound protocol is not a general segmenter. CER can be a useful complement for such text, but is not a universal semantic metric. Do not compare scores across languages or normalization protocols without a documented, matching tokenization convention.
