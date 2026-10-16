# Runnable examples

Run from the repository root after installing the project (`python -m pip install -e .`). The analysis/session examples use Redux by default and require `python -m pip install -e '.[redux]'` plus a local WAV. They may initialize the model and download weights on first use.

```bash
python examples/analyze_reference.py speech.wav --reference "Expected words"
python examples/reusable_session.py first.wav second.wav --device cpu
```

The custom provider is offline: it returns the supplied transcript through the public provider protocol while SpeechScope loads the WAV and computes metrics.

```bash
python examples/custom_backend.py speech.wav \
  --reference "Expected words" --transcript "Recognized words"
```

`policy-exact-content.json` is an example policy for a tolerant spoken-content gate: it requires zero compound WER and zero CER, while leaving strict WER available as a diagnostic. Boundary variants remain visible in report diagnostics.
