# Optional Parakeet Redux CPU integration test

The optional dependency floor is `moondream>=2.4.1`, following the Redux vendor installation guidance. `.github/workflows/redux-sdk-compat.yml` runs the no-weights mocked/API-surface contract against 2.4.1 and the v0.1.0 review's current 2.6.1 release; only completed CI runs count as compatibility evidence.
The default test suite and mocked SDK-contract tests never download model weights. Run a real-model smoke test only in an explicitly enabled environment with the optional SpeechScope extra installed and a local WAV fixture whose recording rights permit this use. The integration test requires an operator-supplied transcript and records an operator-supplied license/rights reference; it does not commit or download speech samples.

```bash
python -m pip install -e '.[redux,dev]'
export SPEECHSCOPE_RUN_REDUX_INTEGRATION=1
export SPEECHSCOPE_REDUX_FIXTURE=/absolute/path/to/rights-cleared.wav
export SPEECHSCOPE_REDUX_REFERENCE='the known reference transcript'
export SPEECHSCOPE_REDUX_FIXTURE_LICENSE='license or permission record'
python -m pytest -q -m integration tests/test_redux.py
```

This explicitly opted-in test may download Parakeet Redux model weights through Moondream into its managed cache and will initialize CPU inference several times to exercise the Python API, `transcribe`, `analyze`, and `doctor`. Keep the fixture short and speech-only, and do not supply sensitive or unlicensed recordings. The test asserts recognized nonempty text and plausible native word timestamps without requiring an exact transcript match. It reports the installed Moondream version and fixture-rights reference. A passing mocked contract or a skipped integration test is not evidence of a successful real inference.
