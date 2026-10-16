# Backends and provider integrations

SpeechScope separates WAV loading, comparison, timing inspection, and reporting from ASR providers. Built-in Redux support is optional; importing SpeechScope and listing backend metadata do not initialize a model or download weights.

## Redux

Install the extra with `pip install 'speechscope[redux]'`. The supported model is `moondream/parakeet-redux`; the current declared Moondream floor is 2.4.1, with a `<3` compatibility boundary. Redux detects language automatically, so a language hint is not accepted. `speechscope doctor --backend redux` checks dependency metadata only, not Photon readiness, hardware, or model weights. The first inference may download weights to the SDK-managed cache.

Redux receives the original encoded WAV path and requests native word timestamps. SpeechScope's mono float64 `AudioInput` samples are not necessarily the array consumed by Redux. Reports distinguish loader transformations from the backend input in `provenance.backend_audio_input`.

## Provider protocol

A backend factory receives `device`, `model`, and `cache_dir` keyword options and returns an object with:

```python
info() -> BackendInfo
transcribe(audio: AudioInput, *, language: str | None = None) -> TranscriptionResult
close() -> None
```

`AudioInput.samples` is mono float64 data at `sample_rate` Hz, while `duration_s` and all word start/end values are seconds from the original audio. The original WAV `path` is included as a possible optimization. A provider must return its recognized text, optional `TimedWord` values, backend/model/version/device identifiers, and a truthful `timing_source`. Missing or malformed timing remains unavailable; SpeechScope validates provider results and does not infer forced alignment.

For an application-local provider, register a factory before calling the API:

```python
from speechscope import register_backend

register_backend("my-provider", lambda **options: MyTranscriber(**options))
```

For an installed package, publish an entry point in the `speechscope.backends` group. Backend names are unique; `redux` is reserved. Discovery exposes metadata but does not create a provider instance. See `examples/custom_backend.py` for a short runnable fake provider.

## Session lifecycle and testing

`analyze()` and `transcribe()` create a backend session per call. Use `Verifier` as a context manager to initialize a model once and reuse it across files; it closes the provider at context exit. Offline tests should use a fake provider. The mocked Redux SDK tests verify the call contract without weights. Real Redux inference is a separate opt-in integration gate requiring an operator-supplied, rights-cleared WAV, reference text, and rights record; see [Redux integration](redux-integration.md).
