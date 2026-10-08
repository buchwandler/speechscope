# SpeechScope (v0.1 MVP)

Independent Python library and CLI for **WAV transcription with word timestamps** and **reference-text speech analysis**. Analyzes recognition accuracy; it does not synthesize speech, verify speaker identity, or infer independently verified word timings. No dependency on Readio or `speechonnxmetrics`.

## Install

```bash
pip install -e .                    # lightweight core, no ASR model download
pip install -e '.[redux]'          # optional Moondream/Parakeet Redux
pip install -e '.[dev]'            # offline tests
```

Python 3.11+. The core requires `audiosig>=0.1.6` for deterministic downmixing and signal preprocessing. The import and discovery paths never initialize an ASR model. First Redux transcription may download model weights to its managed cache. `moondream>=2.4.1` is optional; Redux supports only `moondream/parakeet-redux`, with CI contract coverage targeting the 2.4.1 minimum and the 2.6.1 release reviewed for v0.1.0.

## CLI

```bash
speechscope backends list --format json
speechscope backends inspect redux --format json
speechscope doctor --backend redux
speechscope metrics list
speechscope transcribe speech.wav --backend redux --format json --srt speech.srt
speechscope analyze speech.wav --text-file expected.txt --format json --json report.json
speechscope analyze speech.wav --text 'Hello there' --words alignment.csv --vtt transcript.vtt
speechscope analyze speech.wav --text-file expected.txt --thresholds policy.json
```

`verify` is a compatibility alias for `analyze`. `transcribe` omits reference metrics and alignment. No threshold policy means `not_evaluated`, not `pass`. JSON stdout is a single complete object; diagnostic messages go to stderr. Exit codes: 0 success/review/not-evaluated, 1 policy fail, 2 invalid input/export, 3 backend unavailable or unsupported, 4 inference failure, 5 unexpected error. Redux `doctor` checks installed Moondream version metadata (minimum 2.4.1); it does not initialize Photon, access weights, or validate hardware/model readiness.

Redux uses `moondream.photon('moondream/parakeet-redux', device='cpu')` with `transcribe(audio=..., timestamps='word')`, matching Readio's verified integration and current Moondream docs. This adapter **has not been exercised against live weights in this offline MVP**. Parakeet Redux detects language automatically and rejects an explicit `--language` hint. `--cache-dir` is deliberately rejected until an upstream-supported contract is confirmed; model caching follows Moondream's own settings. Device `auto` leaves selection to Photon; explicit devices do not silently fall back in SpeechScope. First real use may download large model weights; plan for storage and memory use, and expect CPU inference to be slow. Check Moondream's current release notes for quantitative footprint and hardware requirements; SpeechScope makes no numeric performance promise.

The mocked Photon SDK contract and the explicitly opted-in, rights-cleared real CPU integration procedure are documented in [`docs/redux-integration.md`](docs/redux-integration.md). Offline tests do not import Redux or download model weights.

## Python API

```python
from speechscope import VerificationRequest, analyze, transcribe, Verifier

report = analyze(VerificationRequest("speech.wav", reference_text="Hello there"))
print(report.metrics.wer, report.metrics.cer, report.metrics.wip)
print(report.aligned_words)
report.write_json("report.json")
transcript = transcribe("speech.wav")
print(transcript.words)

with Verifier(backend="redux", device="cpu") as session:
    for wav in ["a.wav", "b.wav"]:
        print(session.transcribe(wav).text)
```

Third-party providers register a factory with `register_backend(name, factory)` or publish an entry-point in `speechscope.backends`. Factories accept `device`, `model`, `cache_dir` and return an object with `info()`, `transcribe(AudioInput, *, language=None)` and `close()` methods. Duplicate backend names are rejected. Unit tests use a fake, offline backend.

## Normalization and timing

`basic-v1` applies Unicode NFKC and casefolding, turns punctuation into separators, collapses whitespace, retains underscores and internal apostrophes, and does not remove accents or perform stemming, transliteration, or language-specific segmentation. `strict-v1` applies NFKC and whitespace collapse while preserving case/punctuation. `readio-compat-v1` retains Readio's punctuation behavior for migration. These versioned profiles are stable; semantic changes require a new profile ID. Word metrics use whitespace tokenization after normalization, so scripts without spaces (for example, unsegmented Chinese) may remain one token. Do not treat WER as language-independent or directly compare languages without matching tokenization conventions. Word alignment uses a single Levenshtein path with tie preference match > substitution > deletion > insertion. WER/CER can exceed 1.0. MER/WIL/WIP are derived from the same word path.

Alignment work is bounded by `(reference length + 1) * (hypothesis length + 1)` dynamic-programming cells. Word alignment retains its backtrace matrix up to 1,000,000 cells; character-error counts use a two-row algorithm up to 5,000,000 cells. Larger comparisons raise `AlignmentTooLargeError` before the DP allocation (CLI exit code 2). For similarly sized inputs these limits are about 999 words per side and 2,235 non-space Unicode code points per side; the exact limit depends on both lengths. Transcription-only mode is not subject to these comparison limits.
Word timestamps are **ASR estimates, not ground-truth forced alignment**. Only unambiguous one-to-one correspondence between normalized ASR word entries and transcript tokens permits timestamp projection; otherwise timestamps are null. Invalid timing is diagnosed and not used for coverage. `reference_timestamp_coverage_ratio` measures timestamped reference alignment positions, including substitutions; it is not a correctness score or forced alignment, so pair it with text-accuracy thresholds when correctness matters. A missing timestamp cannot be used to make an SRT/VTT caption. Original-audio time is preserved; no silence trimming.

Redux receives the original encoded WAV path. SpeechScope's `audio_transformations` records changes to the loader's mono sample representation (such as Audiosig downmix), not necessarily the signal consumed by Redux; report provenance adds `backend_audio_input` to distinguish this pass-through path. Moondream may preprocess multichannel encoded audio independently.

Reports follow `speechscope/schemas/report-v1.schema.json`. Transcription-only mode has `reference=null`, `metrics=null`, `alignment=[]`. Schema versioning is separate from package version. CSV/SRT/VTT use recognized ASR text, not reference words presented as spoken. All individual output writes are atomic. Caption grouping uses consecutive word timings and duration/character limits, not language-aware sentence segmentation; punctuation and UTF-8 text are preserved, and a single long word is not split. Cross-file transactionality is not guaranteed in this MVP.

`processing_time_s` and `realtime_factor` use the same boundary for analysis and transcription-only reports: processing begins after the backend context/model initialization and WAV loading, and ends after inference and transcript normalization; analysis additionally includes alignment and metric calculation, but not threshold evaluation or JSON serialization. They exclude initialization, audio loading, and backend teardown; `realtime_factor` is `processing_time_s / audio_duration_s`. These values are not first-call end-to-end latency, which may include a model download or load.

## Dynamic versioning and flat layout

Versions are derived from Git tags by `setuptools-scm` (for example `v0.1.0`), with `0.1.dev0` only as an archive-without-Git fallback. Wheel and sdist versions are determined at build time; `speechscope/_version.py` is generated during the build. Package sources live in `speechscope/` at repository root, **without a `src/` directory**.

## Scope

MVP implementation scope: WAV loading, ASR via optional Redux, reusable sessions, text alignment, five text metrics, timing plausibility, policy thresholds, JSON/CSV/SRT/VTT, and offline backend contracts. Neural MOS (UTMOS/DNSMOS/SIGMOS/NISQA), STOI, and spectral reference-audio metrics are **not implemented**. SpeechScope remains independent of Readio and does not vendor its source. The real Redux model has not been exercised in this implementation.

## Release verification

No `v0.1.0` tag was created and nothing was published. The staged `0.1.0` artifacts are validation artifacts, not a release. Real Redux inference, GitHub Actions, Twine checks, and a clean Linux dependency install remain external gates; see [`docs/v0.1.0-release-readiness.md`](docs/v0.1.0-release-readiness.md) for the exact evidence and blockers.
