from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from speechscope.errors import (
    ModelCatalogError,
    ModelIntegrityError,
    ModelNotInstalledError,
    ModelNotManagedError,
)
from speechscope.models import MODEL_HOME_ENV, ModelCatalogClient, ModelManager, ModelStore

REVISION = "a" * 40
CONTENT = b"tiny model artifact"


def managed_model(*, backend: str = "whispercpp", model_id: str = "tiny-test", alias: str = "tiny"):
    ref = f"{backend}:{model_id}"
    filename = f"{model_id}.bin"
    return {
        "ref": ref,
        "backend": backend,
        "id": model_id,
        "aliases": [alias],
        "kind": "model",
        "format": "ggml",
        "languages": ["en"],
        "description": "Test model",
        "acquisition": {"mode": "managed"},
        "runtime": {"model_argument": "primary_artifact"},
        "source": {
            "provider": "huggingface",
            "repository": "test/model",
            "revision": REVISION,
            "license": "mit",
        },
        "artifacts": [
            {
                "filename": filename,
                "path": filename,
                "primary": True,
                "role": "model",
                "size": len(CONTENT),
                "sha256": hashlib.sha256(CONTENT).hexdigest(),
                "url": f"https://huggingface.co/test/model/resolve/{REVISION}/{filename}?download=true",
            }
        ],
    }


def runtime_model():
    model = managed_model(backend="redux", model_id="parakeet-redux", alias="shared")
    model["acquisition"] = {
        "mode": "runtime-managed",
        "reason": "Photon owns its model cache.",
        "runtime_locator": "moondream/parakeet-redux",
    }
    model["artifacts"][0]["size"] = None
    model["artifacts"][0]["sha256"] = None
    return model


def catalog_data(*models):
    models = models or (managed_model(),)
    return {
        "schema": 1,
        "kind": "speechscope-model-catalog",
        "source": {"provider": "huggingface", "model_count": len(models)},
        "models": {model["ref"]: model for model in models},
    }


def write_catalog(path: Path, *models) -> Path:
    path.write_text(json.dumps(catalog_data(*models)), encoding="utf-8")
    return path


def test_local_catalog_resolves_canonical_refs_and_aliases(tmp_path):
    path = write_catalog(tmp_path / "models.json", managed_model())
    client = ModelCatalogClient(path)

    assert client.resolve("whispercpp:tiny-test")["id"] == "tiny-test"
    assert client.resolve("tiny", backend="whispercpp")["ref"] == "whispercpp:tiny-test"
    assert client.list(backend="whispercpp")[0]["ref"] == "whispercpp:tiny-test"


def test_ambiguous_alias_requires_backend_or_canonical_ref(tmp_path):
    first = managed_model(alias="shared")
    second = managed_model(backend="otherbackend", model_id="tiny-two", alias="shared")
    client = ModelCatalogClient(write_catalog(tmp_path / "models.json", first, second))

    with pytest.raises(ModelCatalogError, match="ambiguous"):
        client.resolve("shared")
    assert client.resolve("shared", backend="whispercpp")["ref"] == first["ref"]


def test_remote_catalog_is_cached_and_available_offline(tmp_path):
    payload = json.dumps(catalog_data()).encode()
    with patch("speechscope.models.urlopen", return_value=BytesIO(payload)) as fetch:
        online = ModelCatalogClient("https://catalog.example/models.json", cache_dir=tmp_path)
        assert online.load()["source"]["model_count"] == 1
        assert fetch.call_count == 1

    with patch("speechscope.models.urlopen") as fetch:
        offline = ModelCatalogClient(
            "https://catalog.example/models.json", cache_dir=tmp_path, offline=True
        )
        assert offline.load()["models"]["whispercpp:tiny-test"]["id"] == "tiny-test"
        fetch.assert_not_called()


def test_offline_and_refresh_cannot_be_combined(tmp_path):
    client = ModelCatalogClient(
        "https://catalog.example/models.json",
        cache_dir=tmp_path,
        offline=True,
        refresh=True,
    )
    with pytest.raises(ModelCatalogError, match="cannot be used together"):
        client.load()


def test_offline_catalog_without_cache_fails_clearly(tmp_path):
    client = ModelCatalogClient(
        "https://catalog.example/models.json", cache_dir=tmp_path, offline=True
    )
    with pytest.raises(ModelCatalogError, match="No cached model catalog"):
        client.load()


def test_install_verifies_bytes_and_manages_inventory_path_and_remove(tmp_path):
    model = managed_model()
    catalog = write_catalog(tmp_path / "catalog.json", model)
    manager = ModelManager(catalog=catalog, cache_dir=tmp_path / "cache")
    with patch("speechscope.models.urlopen", return_value=BytesIO(CONTENT)):
        installed_path = manager.install(model["ref"])

    assert installed_path.read_bytes() == CONTENT
    assert manager.path(model["ref"]) == installed_path
    assert manager.installed()[0]["ref"] == model["ref"]
    assert manager.verify(model["ref"])["verified"] is True
    manager.remove(model["ref"])
    assert manager.installed() == []
    assert not installed_path.parent.exists()


def test_bad_download_is_removed_before_install_activation(tmp_path):
    model = managed_model()
    store = ModelStore(tmp_path)
    with (
        patch("speechscope.models.urlopen", return_value=BytesIO(b"wrong bytes")),
        pytest.raises(ModelIntegrityError, match="Integrity check failed"),
    ):
        store.install(model)

    parent = tmp_path / "models" / "whispercpp"
    assert not (parent / "tiny-test").exists()
    assert not list(parent.glob("*.stage-*"))


def test_verify_detects_installed_tampering(tmp_path):
    model = managed_model()
    store = ModelStore(tmp_path)
    with patch("speechscope.models.urlopen", return_value=BytesIO(CONTENT)):
        artifact = store.install(model)
    artifact.write_bytes(b"modified")

    with pytest.raises(ModelIntegrityError, match="failed integrity verification"):
        store.verify(model)


def test_runtime_managed_models_are_not_installed_or_removed(tmp_path):
    model = runtime_model()
    store = ModelStore(tmp_path)

    with pytest.raises(ModelNotManagedError, match="Photon owns"):
        store.install(model)
    with pytest.raises(ModelNotManagedError):
        store.remove(model)
    assert not (tmp_path / "models").exists()


def test_remove_refuses_unowned_directory(tmp_path):
    directory = tmp_path / "whispercpp" / "tiny-test"
    directory.mkdir(parents=True)
    sentinel = directory / "user-file"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ModelNotInstalledError, match="manifest"):
        ModelStore(tmp_path).remove("whispercpp:tiny-test")
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("bad_path", ["../escape.bin", ".", "./weights.bin", "a//b"])
def test_catalog_rejects_unsafe_artifact_paths(tmp_path, bad_path):
    model = managed_model()
    model["artifacts"][0]["path"] = bad_path
    path = write_catalog(tmp_path / "models.json", model)

    with pytest.raises(ModelCatalogError, match="Unsafe catalog artifact path"):
        ModelCatalogClient(path).load()


def test_force_replacement_is_staged_and_requires_owned_existing_install(tmp_path):
    model = managed_model()
    store = ModelStore(tmp_path)
    destination = tmp_path / "whispercpp" / "tiny-test"
    destination.mkdir(parents=True)
    sentinel = destination / "user-file"
    sentinel.write_text("not managed", encoding="utf-8")

    with (
        patch("speechscope.models.urlopen", return_value=BytesIO(CONTENT)),
        pytest.raises(ModelNotInstalledError),
    ):
        store.install(model, force=True)
    assert sentinel.exists()


def test_packaged_catalog_contains_pinned_tiny_english_model_without_network():
    with patch(
        "speechscope.models.urlopen", side_effect=AssertionError("default catalog must be local")
    ):
        model = ModelCatalogClient().resolve("whispercpp:tiny.en-q5_1")

    assert model["ref"] == "whispercpp:tiny.en-q5_1"
    assert model["backend"] == "whispercpp"
    assert model["source"]["revision"] == "5359861c739e955e79d9a303bcbc70fb988958b1"
    artifact = model["artifacts"][0]
    assert artifact["path"] == "ggml-tiny.en-q5_1.bin"
    assert artifact["size"] == 32_166_155
    assert artifact["sha256"] == "c77c5766f1cef09b6b7d47f21b546cbddd4157886b3b5d6d4f709e91e66c7c2b"
    assert "/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/" in artifact["url"]


def test_model_home_explicit_override_precedes_environment(tmp_path, monkeypatch):
    environment_home = tmp_path / "environment-home"
    explicit_home = tmp_path / "explicit-home"
    monkeypatch.setenv(MODEL_HOME_ENV, str(environment_home))

    assert ModelStore().root == environment_home
    assert ModelStore(explicit_home).root == explicit_home


def test_valid_install_is_idempotent_and_writes_required_manifest(tmp_path):
    model = managed_model()
    store = ModelStore(tmp_path)
    with patch("speechscope.models.urlopen", return_value=BytesIO(CONTENT)) as fetch:
        installed = store.install(model)
        assert store.install(model) == installed

    assert fetch.call_count == 1
    manifest = json.loads((installed.parent / "install.json").read_text(encoding="utf-8"))
    assert manifest["ref"] == model["ref"]
    assert manifest["catalog_schema"] == 1
    assert manifest["source_repository"] == "test/model"
    assert manifest["source_revision"] == REVISION
    assert manifest["installed_at"]
    assert manifest["speechscope_version"]
    assert manifest["artifacts"] == [
        {
            "path": "tiny-test.bin",
            "size": len(CONTENT),
            "sha256": hashlib.sha256(CONTENT).hexdigest(),
            "primary": True,
        }
    ]


def test_corrupt_install_requires_force_and_force_activates_verified_replacement(tmp_path):
    model = managed_model()
    store = ModelStore(tmp_path)
    with patch("speechscope.models.urlopen", return_value=BytesIO(CONTENT)):
        artifact = store.install(model)
    artifact.write_bytes(b"corrupted")

    with pytest.raises(ModelIntegrityError, match="pass --force"):
        store.install(model)

    with patch("speechscope.models.urlopen", return_value=BytesIO(CONTENT)) as fetch:
        replacement = store.install(model, force=True)
    assert fetch.call_count == 1
    assert replacement.read_bytes() == CONTENT
    assert store.verify(model)["verified"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: model["artifacts"][0].update(url="http://huggingface.co/file"), "HTTPS"),
        (
            lambda model: model["artifacts"][0].update(
                url=f"https://huggingface.co/test/model/resolve/{'c' * 40}/tiny-test.bin"
            ),
            "pinned to its source revision",
        ),
    ],
)
def test_install_rejects_insecure_or_unpinned_urls_before_download(tmp_path, mutation, message):
    model = managed_model()
    mutation(model)
    with (
        patch("speechscope.models.urlopen", side_effect=AssertionError("must not download")),
        pytest.raises(ModelCatalogError, match=message),
    ):
        ModelStore(tmp_path).install(model)


def test_install_rejects_exact_size_and_digest_mismatches_and_cleans_staging(tmp_path):
    for field, value, message in (
        ("size", len(CONTENT) + 1, "Integrity check failed"),
        ("sha256", "0" * 64, "Integrity check failed"),
    ):
        model = managed_model(model_id=f"tiny-{field}")
        model["artifacts"][0][field] = value
        store = ModelStore(tmp_path / field)
        with (
            patch("speechscope.models.urlopen", return_value=BytesIO(CONTENT)),
            pytest.raises(ModelIntegrityError, match=message),
        ):
            store.install(model)
        backend_dir = store.models_dir / "whispercpp"
        assert not (backend_dir / model["id"]).exists()
        assert not list(backend_dir.glob("*.stage-*"))
