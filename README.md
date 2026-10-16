# SpeechScope

Independent Python library and CLI for **WAV transcription with ASR word timestamps** and **reference-text speech analysis**. SpeechScope evaluates recognition accuracy; it does not synthesize speech, identify speakers, or claim forced alignment.

## Install

Install the core package (no ASR model download):

```bash
pip install speechscope
```

For Parakeet Redux transcription:

```bash
pip install 'speechscope[redux]'
```

Python 3.11+ is supported. The optional Redux integration uses Moondream/Parakeet Redux; its first inference may download model weights to Moondream's managed cache. Import and backend discovery do not initialize a model.

## CLI

```bash
speechscope backends list --format json
speechscope doctor --backend redux
speechscope metrics list
speechscope transcribe speech.wav --backend redux --format json --srt speech.srt
speechscope analyze speech.wav --text-file expected.txt --format json --json report.json
speechscope analyze speech.wav --text-file expected.txt --thresholds examples/policy-exact-content.json
```

`verify` is a compatibility alias for `analyze`. Transcription-only reports omit reference metrics and alignment. No threshold policy means `not_evaluated`, not `pass`. JSON stdout is one complete report; diagnostics go to stderr. Exit codes: 0 success/review/not-evaluated, 1 policy failure, 2 invalid input/export, 3 backend unavailable/unsupported, 4 inference failure, 5 unexpected error. Redux `doctor` checks dependency metadata only; it does not initialize Photon, access weights, or verify hardware readiness.

## Strict and compound-aware comparison

SpeechScope keeps ordinary whitespace-tokenized `wer`, `mer`, `wil`, `wip`, and `word_*` counts unchanged. It also reports `compound_wer`, which allows a bounded adjacent split/join equivalence after the selected normalization profile. Each equivalence remains visible in `diagnostics`; true substitutions, deletions, and insertions remain errors.

For example, with `basic-v1`, `audiobook` versus `audio book` can have strict WER `2/19` while compound WER and CER are `0`. Use `max_compound_wer: 0` with `max_cer: 0` when validating exact spoken content despite compound word-boundary spelling variation. Do not set `max_wer: 0` for that use case; it remains a strict orthographic tokenization gate. If exact word boundaries matter, use `max_wer` intentionally. Compound matching can also accept ambiguous pairs such as `nowhere` / `now here`, so inspect boundary diagnostics when semantics matter.

See [comparison protocol](docs/comparison.md) for metric semantics, multilingual limits, and policy examples.

## Python API

```python
from speechscope import ThresholdPolicy, VerificationRequest, analyze, Verifier

report = analyze(
    VerificationRequest(
        "speech.wav",
        reference_text="Hello there",
        thresholds=ThresholdPolicy(max_compound_wer=0.0, max_cer=0.0),
    )
)
print(report.metrics.wer, report.metrics.compound_wer, report.metrics.cer)
print(report.metrics.content_match, report.diagnostics)
report.write_json("report.json")

with Verifier(backend="redux", device="cpu") as session:
    for wav in ("a.wav", "b.wav"):
        print(session.transcribe(wav).text)
```

Third-party providers can register a factory with `register_backend(name, factory)` or install an entry point in `speechscope.backends`. See [backend integration](docs/backends.md) and `examples/custom_backend.py` for the minimal protocol.

## Normalization, alignment, and timing

`basic-v1` applies Unicode NFKC/casefolding, punctuation-to-separator normalization, and whitespace collapse; it retains underscores and internal apostrophes. `strict-v1` preserves case and punctuation while collapsing whitespace. `readio-compat-v1` preserves the documented legacy punctuation behavior. These versioned profiles are stable: changes require a new profile ID. They do not expand numbers, contractions, spelling variants, or compounds. WER depends on normalization and tokenization and is not language-independent; scripts without spaces may remain one token. See [comparison protocol](docs/comparison.md).

Word alignment work is bounded by `(reference length + 1) * (hypothesis length + 1)` dynamic-programming cells. Strict alignment retains its backtrace up to 1,000,000 cells; compound alignment uses the same budget and permits at most three tokens on one side of a split/join match. CER uses a two-row algorithm up to 5,000,000 cells. Larger comparisons raise `AlignmentTooLargeError` before allocation.

Word timestamps are **ASR estimates**, not ground-truth forced alignment. Only unambiguous one-to-one normalized ASR word/token correspondence permits timestamp projection. For example, a native `end-to-end` timestamp does not establish separate timestamps for normalized `end`, `to`, `end` tokens; those token timings remain unavailable. Invalid timing is diagnosed. `reference_timestamp_coverage_ratio` measures usable ASR timing, not textual correctness. Original audio time is preserved.

Reports use the packaged [report-v1 schema](docs/report-schema.md). Transcription-only mode has `reference=null`, `metrics=null`, and an empty alignment. `provenance.backend_audio_input` records which audio representation a backend received. CSV/SRT/VTT use recognized text, and individual file writes are atomic.

## Redux evidence

The supplied Piper report records a live run with Moondream 2.6.1, `moondream/parakeet-redux`, CPU inference, WAV input, and native word timestamps. This is evidence for that reported setup only; it does not establish compatibility across all operating systems, devices, or model/runtime combinations. SpeechScope's offline suite uses fake backends and its Redux SDK tests mock the SDK contract. The rights-cleared real-model integration remains an explicit opt-in release gate; see [Redux integration](docs/redux-integration.md) and [release readiness](docs/v0.1.0-release-readiness.md).

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q -m 'not integration'
python -m ruff check .
python -m ruff format --check .
python -m mypy speechscope
```

The package uses a flat layout (`speechscope/`, not `src/`) and derives versions from Git tags with `setuptools-scm`. `speechscope/py.typed` and the report schema are package data. Short runnable scripts are in [`examples/`](examples/README.md).

## Release status

SpeechScope v0.1.0 remains unreleased ([changelog](docs/changelog.md)); no tag or PyPI publication has been made. Hosted CI, clean Linux artifact/Twine checks, and reproducible rights-cleared Redux inference must pass before release; see the [release-readiness record](docs/v0.1.0-release-readiness.md).
