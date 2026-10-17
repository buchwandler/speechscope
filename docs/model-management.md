# Model catalog and management

SpeechScope ships a packaged snapshot of the normalized [SpeechScope model catalog](https://github.com/buchwandler/speechscope-model-catalog). Catalog metadata pins upstream revisions, artifact URLs, sizes, and digests. Model acquisition is explicit: normal transcription and model-aware doctor checks use local catalog metadata and never download or refresh it.

## Ownership

- `whispercpp` GGML artifacts are **managed** by SpeechScope. An explicit `speechscope models install` downloads them, verifies exact size and SHA-256, and activates a manifest-owned installation.
- `redux:parakeet-redux` is **runtime-managed** by Photon. SpeechScope can display its catalog metadata, but cannot install, enumerate, resolve, verify, or remove Photon-owned files.
- A user-supplied whisper.cpp model path remains supported and is never copied into SpeechScope's managed store.

## Model home

The managed model root is selected in this order:

1. An explicit `--cache-dir` CLI option or API `cache_dir` value.
2. `SPEECHSCOPE_MODEL_HOME`.
3. The platform cache directory selected by `platformdirs`.

The configured directory is the model home itself. A managed install has a stable structure:

```text
<SPEECHSCOPE_MODEL_HOME>/
  whispercpp/
    tiny.en-q5_1/
      install.json
      ggml-tiny.en-q5_1.bin
```

`--cache-dir` and `cache_dir` remain supported as explicit model-home overrides. Explicit per-command catalog downloads are cached separately below `<model-home>/catalogs/`.

## Commands

```bash
speechscope models list
speechscope models list --backend whispercpp --format json
speechscope models show whispercpp:tiny.en-q5_1
speechscope models install whispercpp:tiny.en-q5_1
speechscope models verify whispercpp:tiny.en-q5_1
speechscope models path whispercpp:tiny.en-q5_1
speechscope models remove whispercpp:tiny.en-q5_1

speechscope doctor \
  --backend whispercpp \
  --model whispercpp:tiny.en-q5_1 \
  --device cpu \
  --format json
```

The canonical reference is `<backend>:<id>`. Unambiguous aliases may be used interactively; JSON output always reports the canonical ref. `models install --force` replaces an existing owned install only after downloading and verifying the replacement in staging. A valid existing install is idempotent and does not download the weights again.

The catalog's `whispercpp:tiny.en-q5_1` model is the smallest English-specific quantized candidate in the supplied catalog (about 30.68 MiB). It is a lightweight smoke-test candidate, not an automatic default or a guarantee that it passes a downstream content gate. The caller selects and pins the model used for production validation. SpeechScope never silently upgrades or falls back to a larger model.

## Catalog snapshot and refresh

The separate `speechscope-model-catalog` repository is the metadata source of truth. `speechscope/data/model-catalog-v1.json` is a generated, byte-for-byte snapshot of its normalized `catalog/models.json`; do not maintain a second independent catalog by hand. When adopting a catalog release, from the SpeechScope checkout refresh it from the sibling catalog checkout, for example:

```bash
cp ../speechscope-model-catalog/catalog/models.json speechscope/data/model-catalog-v1.json
python -m json.tool speechscope/data/model-catalog-v1.json >/dev/null
python -m pytest -q tests/test_models.py tests/test_packaging.py
```

Review the catalog diff and package test results with the catalog release. The snapshot is included in both wheel and sdist; model weights are never package data.

By default, `models list` and `models show` read the packaged snapshot without network access. Catalog-management commands accept `--catalog PATH_OR_HTTPS_URL`, `--offline`, and `--refresh`. `SPEECHSCOPE_MODEL_CATALOG` may override that source for catalog-management CLI/API calls; an explicit HTTPS source can be fetched when the caller runs a catalog operation. `--refresh` explicitly requests a fresh remote catalog and cannot be combined with `--offline`. Whisper.cpp inference and `doctor` always use the packaged catalog, even if that environment variable names a remote URL.

## Integrity, activation, and removal

Managed catalog artifacts must have safe relative paths, HTTPS Hugging Face URLs pinned to the full source revision, positive byte sizes, and SHA-256 digests. Installation streams each artifact to a staging directory, checks its exact byte count and digest, and writes `install.json` with the canonical ref, catalog schema, source repository/revision, artifact paths/sizes/digests, install timestamp, and SpeechScope version. The final model directory is activated only after every artifact has passed verification; failed downloads clean their staging data.

`models verify` re-hashes installed bytes instead of trusting the manifest. `models path` also verifies the installation before returning the readable primary artifact. An invalid existing installation fails by default; `--force` can replace it only when a valid ownership manifest identifies the existing directory. An unowned directory is never recursively deleted. `models remove` deletes only the manifest-owned model directory and refuses runtime-managed or arbitrary paths.

Catalog integrity is relative to the selected catalog snapshot; it does not authenticate a compromised catalog source. Keep the catalog repository and its release review in the trust chain.

## Inference and local paths

Install weights during an explicit provisioning step, then run offline:

```bash
export SPEECHSCOPE_MODEL_HOME="$PWD/.ci-cache/speechscope-models"
speechscope models install whispercpp:tiny.en-q5_1
speechscope transcribe speech.wav \
  --backend whispercpp \
  --model whispercpp:tiny.en-q5_1 \
  --device cpu
```

A canonical model ref resolves only to a catalogued, already-installed, verified managed model. Inference does not fetch the catalog or download weights. Use an explicit local file path as an escape hatch for models outside the packaged catalog:

```bash
speechscope transcribe speech.wav \
  --backend whispercpp \
  --model /absolute/path/to/ggml-model.bin \
  --device cpu
```

`whisper-cli` is an external runtime and must be installed separately; see [whisper.cpp integration](whispercpp-integration.md). `doctor --backend whispercpp --model ... --format json` checks executable availability, device support, packaged catalog membership, managed installation integrity, and primary artifact readability. It does not download or run inference; when a model is selected, it reads the installed artifact to recompute its digest.

Redux inference is unchanged and remains Photon-managed:

```bash
speechscope transcribe speech.wav --backend redux
```

Do not infer Photon cache paths or treat SpeechScope's installed-model inventory as a Photon inventory.
