# whisper.cpp integration

SpeechScope's `whispercpp` backend runs a separately installed `whisper-cli` executable with a local Whisper model. SpeechScope does not bundle whisper.cpp, download model weights, or add a Python dependency for this backend.

## Install whisper.cpp

Build whisper.cpp using its upstream instructions. A typical Linux build is:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release
```

The executable is commonly at `build/bin/whisper-cli`. Obtain a model supported by that build separately and keep its local filesystem path available. SpeechScope accepts model paths; it does not download them or reinterpret `cache_dir` as a model directory.

## Configure and run

The executable defaults to `whisper-cli` on `PATH`. Override it and select a local model with environment variables:

```bash
export SPEECHSCOPE_WHISPERCPP_BIN="$PWD/build/bin/whisper-cli"
export SPEECHSCOPE_WHISPERCPP_MODEL="/path/to/ggml-tiny.en.bin"
speechscope doctor --backend whispercpp
speechscope transcribe input.wav --backend whispercpp --device cpu --format json
```

The explicit API `model=` value or CLI `--model` option takes precedence over `SPEECHSCOPE_WHISPERCPP_MODEL`:

```bash
speechscope transcribe input.wav \
  --backend whispercpp \
  --model /path/to/ggml-model.bin \
  --language en \
  --device cpu
```

Omitting `--language` asks whisper.cpp to auto-detect. `cpu` passes `--no-gpu`; `auto` omits it and lets the local whisper.cpp build choose its compiled acceleration. SpeechScope currently supports only `cpu` and `auto`; it does not advertise CUDA, MPS, Vulkan, or other upstream possibilities without adapter evidence. `cache_dir` is unsupported.

## Audio input and report provenance

SpeechScope loads the source WAV through its normal audio boundary, then always materializes an ephemeral mono 16 kHz WAV with signed PCM16 encoding from `AudioInput.samples`. It uses the public `audiosig.resample()` API only when the source sample rate differs, with filter width 32, rolloff 0.945, and round length mode. SoundFile writes explicit WAV/PCM_16 metadata, which SpeechScope verifies before inference. The adapter does not trim, pad, normalize, apply gain, time-stretch, or run VAD. No FFmpeg conversion is needed for ordinary SpeechScope WAV input.

The temporary WAV remains available only while `whisper-cli` runs and is removed after success or failure. Reports retain the original audio path and duration; `provenance.backend_audio_input` records the generated representation and its transformation profile, not its temporary pathname. PCM16 range clipping is reported as a diagnostic and is not peak normalization.

## Preflight and testing

```bash
speechscope backends list --format json
speechscope backends inspect whispercpp --format json
speechscope doctor --backend whispercpp
```

Backend listing/inspection does not execute the CLI or access a model. `doctor` checks that the selected executable is runnable and that `--version` or, conservatively, `--help` succeeds. It does not require a model, load weights, transcribe audio, or access the network.

The ordinary test suite uses synthetic WAVs and a fake CLI, so it does not need model weights. A real integration test is opt-in and requires an operator-supplied rights-cleared WAV, an installed `whisper-cli`, a local model, the known fixture transcript, and an evidence string for the fixture rights:

```bash
export SPEECHSCOPE_WHISPERCPP_BIN="/path/to/whisper-cli"
export SPEECHSCOPE_WHISPERCPP_MODEL="/path/to/model.bin"
export SPEECHSCOPE_WHISPERCPP_FIXTURE="/path/to/authorized.wav"
export SPEECHSCOPE_WHISPERCPP_FIXTURE_LICENSE="license or permission record"
export SPEECHSCOPE_WHISPERCPP_REFERENCE="known transcript for the fixture"
export SPEECHSCOPE_RUN_WHISPERCPP_INTEGRATION=1
python -m pytest -q -m integration tests/test_whispercpp.py
```

This task has not established real model-inference or ARM64 Termux evidence. Do not infer general Linux or Termux/ARM64 support from fake-executable tests or from the presence of an upstream executable; record the exact whisper.cpp version/build, model, device, fixture rights, and target platform for any live smoke evidence.
