# whisper.cpp integration

SpeechScope's `whispercpp` backend runs a separately installed `whisper-cli` executable with either an explicitly supplied local model file or an already-installed, catalog-pinned managed model. SpeechScope does not bundle whisper.cpp or model weights, and adds no Python dependency for this backend. Model installation and real inference are explicit operator actions.

## Install whisper.cpp

Build whisper.cpp using its upstream instructions. A typical Linux build is:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release
```

The executable is commonly at `build/bin/whisper-cli`. SpeechScope supports the model artifact formats accepted by the local build. For the managed catalog path, provision a catalog-pinned model separately with `speechscope models install whispercpp:<id>`; alternatively, supply your own local model file. Model installation is never triggered by transcription.

## Configure and run

Provision a managed model explicitly; SpeechScope never installs or updates weights during transcription. The executable defaults to `whisper-cli` on `PATH`, and its override remains independent from model storage:

```bash
speechscope models install whispercpp:tiny.en-q5_1
export SPEECHSCOPE_WHISPERCPP_BIN="$PWD/build/bin/whisper-cli"
export SPEECHSCOPE_WHISPERCPP_MODEL="whispercpp:tiny.en-q5_1"
speechscope doctor --backend whispercpp --model "$SPEECHSCOPE_WHISPERCPP_MODEL" --device cpu --format json
speechscope transcribe input.wav --backend whispercpp --device cpu --format json
```

`--cache-dir` or API `cache_dir` overrides `SPEECHSCOPE_MODEL_HOME`, which overrides the platform cache directory. For example, set `SPEECHSCOPE_MODEL_HOME` before the install command to choose a dedicated store. The inference and doctor paths use the packaged catalog snapshot and do not fetch `SPEECHSCOPE_MODEL_CATALOG` even if it names a remote URL.

An explicit API `model=` value or CLI `--model` option takes precedence over `SPEECHSCOPE_WHISPERCPP_MODEL`. Managed refs support `cache_dir`; explicit model file paths remain an escape hatch and reject `cache_dir`:

```bash
speechscope transcribe input.wav \
  --backend whispercpp \
  --model /path/to/ggml-model.bin \
  --language en \
  --device cpu
```

Omitting `--language` asks whisper.cpp to auto-detect. `cpu` passes `--no-gpu`; `auto` omits it and lets the local whisper.cpp build choose its compiled acceleration. SpeechScope currently supports only `cpu` and `auto`; it does not advertise CUDA, MPS, Vulkan, or other upstream possibilities without adapter evidence. The adapter currently reports no word or segment timing.

## Audio input and report provenance

SpeechScope loads the source WAV through its normal audio boundary, then always materializes an ephemeral mono 16 kHz WAV with signed PCM16 encoding from `AudioInput.samples`. It uses the public `audiosig.resample()` API only when the source sample rate differs, with filter width 32, rolloff 0.945, and round length mode. SoundFile writes explicit WAV/PCM_16 metadata, which SpeechScope verifies before inference. The adapter does not trim, pad, normalize, apply gain, time-stretch, or run VAD. No FFmpeg conversion is needed for ordinary SpeechScope WAV input.

The temporary WAV remains available only while `whisper-cli` runs and is removed after success or failure. Reports retain the original audio path and duration; `provenance.backend_audio_input` records the generated representation and its transformation profile, not its temporary pathname. PCM16 range clipping is reported as a diagnostic and is not peak normalization.

## Preflight and testing

```bash
speechscope backends list --format json
speechscope backends inspect whispercpp --format json
speechscope doctor --backend whispercpp
```

Backend listing/inspection does not execute the CLI or inspect a model. `doctor` checks the selected executable with `--version` or, conservatively, `--help`. With `--model <ref>`, it also validates device support, catalog membership, managed installation integrity, artifact readability, and JSON readiness data. Doctor does not download or run inference; verifying a model does read its bytes to recompute the digest.

The ordinary suite uses synthetic WAVs and a fake CLI, so it needs no model weights. Real inference is opt-in and requires an already-installed managed model, an operator-supplied rights-cleared WAV, the fixture transcript, and a rights-evidence string. Provision the selected model separately before running the integration test:

```bash
export SPEECHSCOPE_MODEL_HOME="$PWD/.speechscope-models"
speechscope models install whispercpp:tiny.en-q5_1
export SPEECHSCOPE_WHISPERCPP_BIN="/path/to/whisper-cli"
export SPEECHSCOPE_WHISPERCPP_MODEL="whispercpp:tiny.en-q5_1"
export SPEECHSCOPE_WHISPERCPP_FIXTURE="/path/to/authorized.wav"
export SPEECHSCOPE_WHISPERCPP_FIXTURE_LICENSE="license or permission record"
export SPEECHSCOPE_WHISPERCPP_REFERENCE="known transcript for the fixture"
speechscope doctor --backend whispercpp --model "$SPEECHSCOPE_WHISPERCPP_MODEL" --device cpu --format json
export SPEECHSCOPE_RUN_WHISPERCPP_INTEGRATION=1
python -m pytest -q -m integration tests/test_whispercpp.py
```

The integration test asserts a non-empty transcript and backend/model/device provenance, but does not impose an exact recognition score.

This task has not established real model-inference or ARM64 Termux evidence. Do not infer general Linux or Termux/ARM64 support from fake-executable tests or from the presence of an upstream executable; record the exact whisper.cpp version/build, model, device, fixture rights, and target platform for any live smoke evidence.
