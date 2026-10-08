# SpeechScope (v0.1 MVP)

Independent Python library and CLI for **WAV transcription with word timestamps** and **reference-text speech analysis**. Analyzes recognition accuracy; it does not synthesize speech, verify speaker identity, or infer independently verified word timings. No dependency on Readio or `speechonnxmetrics`.

## Install

```bash
pip install -e .                    # lightweight core, no ASR model download
pip install -e '.[redux]'          # optional Moondream/Parakeet Redux
pip install -e '.[dev]'            # offline tests
```

Python 3.11+. The core requires `audiosig>=0.1.6` for deterministic downmixing and signal preprocessing. The import and discovery paths never initialize an ASR model. First Redux transcription may download model weights to its managed cache. `moondream>=2.4.0` is optional.

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

`verify` is a compatibility alias for `analyze`. `transcribe` omits reference metrics and alignment. No threshold policy means `not_evaluated`, not `pass`. JSON stdout is a single complete object; diagnostic messages go to stderr. Exit codes: 0 success/review/not-evaluated, 1 policy fail, 2 invalid input/export, 3 backend unavailable or unsupported, 4 inference failure, 5 unexpected error.

Redux uses `moondream.photon('moondream/parakeet-redux', device='cpu')` with `transcribe(audio=..., timestamps='word')`, matching Readio's verified integration and current Moondream docs. This adapter **has not been exercised against live weights in this offline MVP**. Parakeet Redux detects language automatically and rejects an explicit `--language` hint. `--cache-dir` is deliberately rejected until an upstream-supported contract is confirmed; model caching follows Moondream's own settings. Device `auto` leaves selection to Photon; explicit devices do not silently fall back in SpeechScope.

## Python API

```python
from speechscope import VerificationRequest, analyze, transcribe, Verifier

report = analyze(VerificationRequest('speech.wav', reference_text='Hello there'))
print(report.metrics.wer, report.metrics.cer, report.metrics.wip)
print(report.aligned_words)
report.write_json('report.json')
transcript = transcribe('speech.wav')
print(transcript.words)

with Verifier(backend='redux', device='cpu') as session:
    for wav in ['a.wav', 'b.wav']:
        print(session.transcribe(wav).text)
```

Third-party providers register a factory with `register_backend(name, factory)` or publish an entry-point in `speechscope.backends`. Factories accept `device`, `model`, `cache_dir` and return an object with `info()`, `transcribe(AudioInput, *, language=None)` and `close()` methods. Duplicate backend names are rejected. Unit tests use a fake, offline backend.

## Normalization and timing

`basic-v1` applies Unicode NFKC, casefolding, punctuation splitting, and whitespace collapse; inside-word apostrophes are retained. `strict-v1` retains case/punctuation with whitespace normalization. `readio-compat-v1` retains Readio's current punctuation behavior for migration fixtures. Word alignment uses a single Levenshtein path with tie preference match > substitution > deletion > insertion. WER/CER can exceed 1.0. MER/WIL/WIP are derived from the same word path.

Word timestamps are **ASR estimates, not ground-truth forced alignment**. Only unambiguous one-to-one correspondence between normalized ASR word entries and transcript tokens permits timestamp projection; otherwise timestamps are null. Invalid timing is diagnosed and not used for coverage. A missing timestamp cannot be used to make an SRT/VTT caption. Original-audio time is preserved; no silence trimming.

Reports follow `speechscope/schemas/report-v1.schema.json`. Transcription-only mode has `reference=null`, `metrics=null`, `alignment=[]`. Schema versioning is separate from package version. CSV/SRT/VTT use recognized ASR text, not reference words presented as spoken. All individual output writes are atomic. Cross-file transactionality is not guaranteed in this MVP.

## Dynamic versioning and flat layout

Versions are derived from Git tags by `setuptools-scm` (for example `v0.1.0`), with `0.1.dev0` only as an archive-without-Git fallback. Wheel and sdist versions are determined at build time; `speechscope/_version.py` is generated during the build. Package sources live in `speechscope/` at repository root, **without a `src/` directory**.

## Scope

MVP: WAV loading, ASR via optional Redux, reusable sessions, text alignment, five text metrics, timing plausibility, policy thresholds, JSON/CSV/SRT/VTT, offline backend contracts. Neural MOS (UTMOS/DNSMOS/SIGMOS/NISQA), STOI, and spectral reference-audio metrics are **not implemented in this MVP**; the accompanying `01_audiosig_implementation_brief.md` proposes audio-only DSP ownership and later models. No Readio integration has been changed.
