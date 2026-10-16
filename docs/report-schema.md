# SpeechScope report schema v1

Reports are serialized as JSON using the packaged `speechscope/schemas/report-v1.schema.json`. The report's `schema_version` is independent of the package version. The v1 contract includes transcription-only reports (`reference`, `metrics` are `null`; `alignment` is empty) and analysis reports.

## Metrics

Analysis reports retain the original normalized reference and transcript; compound scoring never rewrites either input. The conventional `wer`, `mer`, `wil`, `wip`, and `word_*` counts are whitespace-tokenized diagnostics and keep their existing semantics. Counts are non-negative integers; rates are non-negative numbers and may exceed 1 where the metric definition permits it.

The additional compound-protocol fields are:

- `compound_word_matches`, `compound_word_substitutions`, `compound_word_deletions`, `compound_word_insertions`: reference-token match/error counts under the bounded split/join alignment. A boundary match counts the reference tokens it consumes as matched.
- `compound_wer`: compound-protocol lexical errors divided by the strict reference token count. It is not ordinary WER and does not replace it.
- `word_boundary_match_count`: number of explicit adjacent split/join equivalences used by the selected alignment.
- `content_match`: true when compound substitutions, deletions, and insertions are all zero. Boundary variants remain visible and do not make this value false.

`diagnostics` records each `word_boundary_variant` with `reference_span`, `hypothesis_span`, `reference_text`, and `hypothesis_text`. Spans are zero-based and half-open (`[start, end)`), and texts are the exact normalized tokens joined with spaces. This records scoring equivalence, not a linguistic correction or a claim of semantic equivalence; ambiguous cases such as `nowhere` / `now here` remain visible.

## Provenance and timing

`provenance.backend_audio_input` is required because it is emitted for both analysis and transcription-only reports. It identifies the audio representation selected for the backend separately from loader sample transformations. Word timestamps remain conservative ASR estimates: normalized comparison tokens do not inherit one native timestamp if their individual spans are unknown. Nullable timing values are seconds; sample rates are hertz.

## Validation

Use a Draft 2020-12 validator with the packaged schema. Unknown fields are rejected in enumerated stable objects so consumers can detect contract drift. Schema changes made before the first public release remain under schema version `1`; freeze the contract before publishing v0.1.0.
