# Model catalog and management

SpeechScope uses a metadata-only catalog to describe compatible models. Model downloads are explicit: inference does not fetch or refresh catalog metadata and never downloads model weights.

## Ownership

- `whispercpp` GGML artifacts are **managed** by SpeechScope. It downloads pinned artifacts, verifies size and SHA-256, writes an ownership manifest, and supports inventory, path lookup, verification, and removal.
- `redux:parakeet-redux` is **runtime-managed** by Photon. SpeechScope can show its catalog metadata, but cannot install it, enumerate the Photon cache, resolve a SpeechScope-owned path, verify Photon files, or remove them. Redux inference continues to pass `moondream/parakeet-redux` to Photon.
- User-supplied whisper.cpp model paths remain supported and are not imported into SpeechScope's managed store.

The MVP catalog is [speechscope-model-catalog](https://github.com/buchwandler/speechscope-model-catalog). It pins upstream repository revisions, license identifiers, artifact URLs, sizes, and hashes. Catalog membership is not an assertion that SpeechScope owns the runtime's cache.

## Commands

```bash
speechscope models list
speechscope models list --backend whispercpp --format json
speechscope models show whispercpp:base.en-q5_1
speechscope models install whispercpp:base.en-q5_1
speechscope models installed
speechscope models path whispercpp:base.en-q5_1
speechscope models verify whispercpp:base.en-q5_1
speechscope models remove whispercpp:base.en-q5_1

speechscope transcribe speech.wav \
  --backend whispercpp \
  --model whispercpp:base.en-q5_1 \
  --device cpu
```

The canonical reference form is `<backend>:<id>`. Unambiguous catalog aliases (such as `base.en`) are accepted by catalog operations; use canonical refs to avoid ambiguity. Model-management errors are printed as user-facing errors rather than tracebacks.

Catalog-reading commands (`list`, `show`, and `install`) accept `--catalog PATH_OR_HTTPS_URL`, `--cache-dir DIR`, `--offline`, and `--refresh`; `list`, `show`, and `install` also accept `--backend` and `--format human|json`. `install` accepts `--force` to replace an existing SpeechScope-managed installation after the replacement has been downloaded and verified. `installed`, `path`, `verify`, and `remove` also accept `--cache-dir` and `--format`; `path`, `verify`, and `remove` accept `--backend`. These commands operate only on SpeechScope manifests and never remove an arbitrary path.

For development and controlled deployments, `--catalog` can name a local JSON file. `SPEECHSCOPE_MODEL_CATALOG` provides the same default-source override for CLI and API calls. The built-in URL is:

```text
https://raw.githubusercontent.com/buchwandler/speechscope-model-catalog/main/catalog/models.json
```

The default remote catalog is cached at `<cache-root>/catalogs/models.json`; overridden remote URLs use URL-specific hashed cache filenames. `--offline` uses that cached catalog and fails clearly if none exists; a local catalog path is read directly. `--refresh` explicitly fetches the remote catalog instead of using a cached copy and cannot be combined with `--offline`. The cache root defaults to the platform-specific SpeechScope user cache directory (`platformdirs`); `--cache-dir` or the API's `cache_dir` overrides it. Managed files are installed below `<cache-root>/models/<backend>/<id>/`.

## Integrity and removal

Managed catalog artifacts must use HTTPS Hugging Face URLs pinned to the model's full upstream revision, safe relative paths, positive byte sizes, and SHA-256 digests. Installation streams each artifact into a staging directory and verifies exact size and hash before activating the install. Failed or mismatched downloads are removed without activating an installation.

Each installation has a `.speechscope-model.json` manifest containing the canonical ref and verified artifact metadata. `models installed` enumerates only directories with a valid SpeechScope manifest. `models verify` recomputes every installed artifact's size and SHA-256. `models remove` deletes only the corresponding manifest-owned model directory; it will not follow symbolic links or recursively delete an arbitrary supplied path. Catalog integrity is relative to the catalog snapshot and does not authenticate a compromised catalog source.

## Inference and local paths

Installing a model is an explicit provisioning step. A server can install weights during image/build setup and then run offline:

```bash
speechscope models install whispercpp:tiny-q5_1
speechscope transcribe speech.wav --backend whispercpp --model whispercpp:tiny-q5_1
```

When the backend receives a `whispercpp:` ref, it resolves the already-installed primary artifact from local store state. It does not load, download, or refresh the catalog and does not download weights. With a ref, `--cache-dir` selects the SpeechScope cache root. With an explicit model file path, existing behavior remains: `--cache-dir` is unsupported.

Redux inference is unchanged and remains Photon-managed:

```bash
speechscope transcribe speech.wav --backend redux
```

Do not infer Photon cache paths or treat `speechscope models installed` as a Photon inventory. Runtime cache status is intentionally not part of this MVP.
