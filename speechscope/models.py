"""Catalog-backed management of SpeechScope-owned speech models.

Model acquisition is explicit: this module is never imported by an inference path
unless the caller selected a managed model reference.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from platformdirs import user_cache_dir

from .errors import (
    ModelCatalogError,
    ModelIntegrityError,
    ModelNotFoundError,
    ModelNotInstalledError,
    ModelNotManagedError,
)

DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/buchwandler/speechscope-model-catalog/"
    "main/catalog/models.json"
)
CATALOG_ENV = "SPEECHSCOPE_MODEL_CATALOG"
MODEL_HOME_ENV = "SPEECHSCOPE_MODEL_HOME"
MANIFEST_NAME = "install.json"
_MANIFEST_SCHEMA = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def default_cache_dir() -> Path:
    """Return the platform-specific SpeechScope cache root."""
    return Path(user_cache_dir("speechscope"))


def default_model_home() -> Path:
    """Resolve the environment-configured model home or platform cache default."""
    configured = os.environ.get(MODEL_HOME_ENV)
    return Path(configured).expanduser() if configured else default_cache_dir()


def _safe_component(value: str, what: str) -> str:
    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value) or value in {".", ".."}:
        raise ModelCatalogError(f"Unsafe {what}: {value!r}")
    return value


def _safe_artifact_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModelCatalogError("Catalog artifact path must be a non-empty relative path")
    parts = value.split("/")
    if value.startswith("/") or ":" in value or any(part in {"", ".", ".."} for part in parts):
        raise ModelCatalogError(f"Unsafe catalog artifact path: {value!r}")
    return Path(*parts)


def _artifact_file(directory: Path, value: object) -> Path:
    relative = _safe_artifact_path(value)
    current = directory
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ModelIntegrityError(
                f"Installed artifact path contains a symbolic link: {current}"
            )
    path = directory / relative
    if path.is_symlink():
        raise ModelIntegrityError(f"Installed artifact must not be a symbolic link: {path}")
    return path


def _validate_catalog(data: object) -> dict[str, Any]:
    if (
        not isinstance(data, dict)
        or isinstance(data.get("schema"), bool)
        or data.get("schema") != 1
    ):
        raise ModelCatalogError("Model catalog must be an object with schema version 1")
    if data.get("kind") != "speechscope-model-catalog":
        raise ModelCatalogError("Unrecognized model catalog kind")
    models = data.get("models")
    source = data.get("source")
    if not isinstance(models, dict) or not isinstance(source, dict):
        raise ModelCatalogError("Model catalog must contain source metadata and a models object")
    count = source.get("model_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(models):
        raise ModelCatalogError("Model catalog model_count does not match its entries")

    aliases_by_backend: set[tuple[str, str]] = set()
    for ref, model in models.items():
        if not isinstance(ref, str) or not isinstance(model, dict):
            raise ModelCatalogError("Catalog model entries must be keyed objects")
        backend = model.get("backend")
        model_id = model.get("id")
        if not isinstance(backend, str) or not isinstance(model_id, str):
            raise ModelCatalogError(f"Catalog model {ref!r} has no backend or id")
        _safe_component(backend, "backend")
        _safe_component(model_id, "model id")
        if ref != f"{backend}:{model_id}" or model.get("ref") != ref:
            raise ModelCatalogError(f"Catalog key/ref/backend/id mismatch for {ref!r}")
        model_source = model.get("source")
        if (
            not isinstance(model_source, dict)
            or not isinstance(model_source.get("revision"), str)
            or not _REVISION_RE.fullmatch(model_source["revision"])
            or not isinstance(model_source.get("repository"), str)
            or not model_source["repository"]
        ):
            raise ModelCatalogError(f"Catalog model {ref!r} must pin a source repository revision")
        acquisition = model.get("acquisition")
        mode = acquisition.get("mode") if isinstance(acquisition, dict) else None
        if mode not in {"managed", "runtime-managed"}:
            raise ModelCatalogError(f"Catalog model {ref!r} has an invalid acquisition mode")
        artifacts = model.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ModelCatalogError(f"Catalog model {ref!r} has no artifacts")
        primary_count = 0
        artifact_paths: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ModelCatalogError(f"Catalog model {ref!r} has an invalid artifact")
            artifact_path = _safe_artifact_path(artifact.get("path")).as_posix()
            if artifact_path in artifact_paths:
                raise ModelCatalogError(
                    f"Catalog model {ref!r} repeats artifact path {artifact_path!r}"
                )
            artifact_paths.add(artifact_path)
            if not isinstance(artifact.get("primary"), bool):
                raise ModelCatalogError(
                    f"Catalog artifact for {ref!r} must declare primary as boolean"
                )
            if artifact["primary"]:
                primary_count += 1
            artifact_url = artifact.get("url")
            if not isinstance(artifact_url, str):
                raise ModelCatalogError(f"Catalog artifact for {ref!r} needs a URL")
            parsed = urlparse(artifact_url)
            if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
                raise ModelCatalogError(f"Catalog artifact for {ref!r} must use Hugging Face HTTPS")
            if f"/resolve/{model_source['revision']}/" not in parsed.path:
                raise ModelCatalogError(
                    f"Catalog artifact for {ref!r} is not pinned to its source revision"
                )
            size = artifact.get("size")

            digest = artifact.get("sha256")
            if mode == "managed":
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ModelCatalogError(f"Managed artifact for {ref!r} needs a positive size")
                if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                    raise ModelCatalogError(f"Managed artifact for {ref!r} needs a SHA-256 digest")
            elif digest is not None and (
                not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
            ):
                raise ModelCatalogError(
                    f"Runtime-managed artifact for {ref!r} has an invalid digest"
                )
        if primary_count != 1:
            raise ModelCatalogError(f"Catalog model {ref!r} must have exactly one primary artifact")
        aliases = model.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias for alias in aliases
        ):
            raise ModelCatalogError(f"Catalog model {ref!r} has invalid aliases")
        for alias in aliases:
            key = (backend, alias)
            if key in aliases_by_backend:
                raise ModelCatalogError(f"Duplicate model alias {alias!r} for backend {backend!r}")
            aliases_by_backend.add(key)
    return data


class ModelCatalogClient:
    """Load and resolve a local or remote model catalog."""

    def __init__(
        self,
        catalog: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        refresh: bool = False,
        timeout: float = 30,
        packaged: bool = False,
    ):
        selected = (
            None
            if packaged
            else str(catalog)
            if catalog is not None
            else os.environ.get(CATALOG_ENV)
        )
        self.catalog = selected
        self.cache_dir = (
            Path(cache_dir).expanduser() if cache_dir is not None else default_model_home()
        )
        self.offline = offline
        self.refresh = refresh
        self.timeout = timeout
        self._data: dict[str, Any] | None = None

    @property
    def is_local(self) -> bool:
        if self.catalog is None:
            return True
        parsed = urlparse(self.catalog)
        return (
            not parsed.scheme
            or parsed.scheme == "file"
            or (len(parsed.scheme) == 1 and self.catalog[1] == ":")
        )

    def load(self) -> dict[str, Any]:
        if self.offline and self.refresh:
            raise ModelCatalogError("--offline and --refresh cannot be used together")
        if self._data is not None:
            return self._data
        if self.catalog is None:
            try:
                payload = (
                    resources.files("speechscope")
                    .joinpath("data/model-catalog-v1.json")
                    .read_text(encoding="utf-8")
                )
                self._data = _validate_catalog(json.loads(payload))
            except ModelCatalogError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ModelCatalogError(f"Could not read packaged model catalog: {exc}") from exc
            return self._data
        if self.is_local:
            path = Path(self.catalog.removeprefix("file://")).expanduser()
            self._data = self._read_catalog(path)
            return self._data
        parsed = urlparse(self.catalog)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ModelCatalogError("Catalog URL must use HTTPS or refer to a local file")
        cache_name = (
            "models.json"
            if self.catalog == DEFAULT_CATALOG_URL
            else f"models-{hashlib.sha256(self.catalog.encode()).hexdigest()}.json"
        )
        cache_file = self.cache_dir / "catalogs" / cache_name
        if self.offline:
            if not cache_file.is_file():
                raise ModelCatalogError(
                    f"No cached model catalog is available for offline use: {cache_file}"
                )
            self._data = self._read_catalog(cache_file)
            return self._data
        if not self.refresh and cache_file.is_file():
            try:
                self._data = self._read_catalog(cache_file)
                return self._data
            except ModelCatalogError:
                # Replace a corrupt cache only after a successful remote fetch.
                pass
        try:
            request = Request(self.catalog, headers={"User-Agent": "SpeechScope model catalog"})
            with urlopen(request, timeout=self.timeout) as response:
                if (
                    urlparse(
                        response.geturl() if hasattr(response, "geturl") else self.catalog
                    ).scheme
                    != "https"
                ):
                    raise ModelCatalogError("Catalog fetch redirected away from HTTPS")
                payload = response.read()
            data = _validate_catalog(json.loads(payload.decode("utf-8")))
        except ModelCatalogError:
            raise
        except (
            HTTPError,
            URLError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TimeoutError,
        ) as exc:
            raise ModelCatalogError(f"Could not load model catalog {self.catalog}: {exc}") from exc
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary = cache_file.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(cache_file)
        except OSError as exc:
            raise ModelCatalogError(
                f"Could not cache model catalog at {cache_file}: {exc}"
            ) from exc
        self._data = data
        return data

    @staticmethod
    def _read_catalog(path: Path) -> dict[str, Any]:
        try:
            return _validate_catalog(json.loads(path.read_text(encoding="utf-8")))
        except ModelCatalogError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelCatalogError(f"Could not read model catalog {path}: {exc}") from exc

    def list(self, *, backend: str | None = None) -> builtins.list[dict[str, Any]]:
        models = self.load()["models"]
        return [
            model for model in models.values() if backend is None or model["backend"] == backend
        ]

    def resolve(self, ref: str, *, backend: str | None = None) -> dict[str, Any]:
        models = self.load()["models"]
        if ref in models:
            model = models[ref]
            if backend is None or model["backend"] == backend:
                return model
            raise ModelNotFoundError(f"Model {ref!r} is not in backend {backend!r}")
        matches = [
            model
            for model in models.values()
            if (backend is None or model["backend"] == backend) and ref in model.get("aliases", [])
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ", ".join(model["ref"] for model in matches)
            raise ModelCatalogError(f"Model alias {ref!r} is ambiguous; use one of: {choices}")
        raise ModelNotFoundError(f"Model {ref!r} was not found in the catalog")


class ModelStore:
    """Install and manage model files below a SpeechScope-owned cache root."""

    def __init__(self, cache_dir: str | Path | None = None, *, timeout: float = 60):
        self.root = Path(cache_dir).expanduser() if cache_dir is not None else default_model_home()
        self.models_dir = self.root
        self.timeout = timeout

    def install(self, model: dict[str, Any], *, force: bool = False) -> Path:
        _validate_catalog(
            {
                "schema": 1,
                "kind": "speechscope-model-catalog",
                "source": {"model_count": 1},
                "models": {model.get("ref"): model},
            }
        )
        self._require_managed(model)
        backend = _safe_component(model["backend"], "backend")
        model_id = _safe_component(model["id"], "model id")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self.models_dir.is_symlink():
            raise ModelCatalogError(
                f"Model store directory must not be a symbolic link: {self.models_dir}"
            )
        destination_parent = self.models_dir / backend
        if destination_parent.is_symlink():
            raise ModelCatalogError(
                f"Model backend directory must not be a symbolic link: {destination_parent}"
            )
        destination_parent.mkdir(parents=True, exist_ok=True)
        if destination_parent.is_symlink():
            raise ModelCatalogError(
                f"Model backend directory must not be a symbolic link: {destination_parent}"
            )
        destination = destination_parent / model_id
        if destination.is_symlink():
            raise ModelIntegrityError(
                f"Refusing to replace a symbolic-link model directory: {destination}"
            )
        if destination.exists() and not force:
            try:
                return self.path(model)
            except (ModelIntegrityError, ModelNotInstalledError) as exc:
                raise ModelIntegrityError(
                    f"Existing installation of {model['ref']} is invalid; "
                    "pass --force to replace it"
                ) from exc
        if destination.exists():
            # Never let --force delete a directory that lacks a valid ownership manifest.
            self._read_manifest(destination, expected_ref=model["ref"])

        stage = Path(tempfile.mkdtemp(prefix=f".{model_id}.stage-", dir=destination_parent))
        backup: Path | None = None
        try:
            installed_artifacts: list[dict[str, Any]] = []
            for artifact in model["artifacts"]:
                relative = _safe_artifact_path(artifact["path"])
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                self._download_and_verify(artifact, target, model["ref"])
                installed_artifacts.append(
                    {
                        "path": relative.as_posix(),
                        "size": artifact["size"],
                        "sha256": artifact["sha256"],
                        "primary": artifact.get("primary") is True,
                    }
                )
            from . import __version__

            manifest = {
                "schema": _MANIFEST_SCHEMA,
                "catalog_kind": "speechscope-model-catalog",
                "catalog_schema": 1,
                "ref": model["ref"],
                "backend": backend,
                "id": model_id,
                "source": model["source"],
                "source_repository": model["source"]["repository"],
                "source_revision": model["source"]["revision"],
                "installed_at": datetime.now(UTC).isoformat(),
                "speechscope_version": __version__,
                "artifacts": installed_artifacts,
            }
            (stage / MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if destination.exists():
                backup = Path(
                    tempfile.mkdtemp(prefix=f".{model_id}.backup-", dir=destination_parent)
                )
                backup.rmdir()
                destination.replace(backup)
            try:
                stage.replace(destination)
            except Exception:
                if backup is not None and backup.exists() and not destination.exists():
                    backup.replace(destination)
                raise
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            if backup is not None and backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        return self.path(model)

    @staticmethod
    def _require_managed(model: dict[str, Any]) -> None:
        acquisition = model.get("acquisition", {})
        if acquisition.get("mode") != "managed":
            runtime = acquisition.get("runtime_locator") or model.get("runtime", {}).get(
                "model_argument"
            )
            owner = acquisition.get("reason") or "the model runtime"
            detail = f" Runtime locator: {runtime}." if runtime else ""
            raise ModelNotManagedError(
                f"{model['ref']} is runtime-managed: {owner} SpeechScope does not install or remove it.{detail}"
            )

    def _download_and_verify(self, artifact: dict[str, Any], target: Path, ref: str) -> None:
        digest = hashlib.sha256()
        size = 0
        try:
            request = Request(
                str(artifact["url"]), headers={"User-Agent": "SpeechScope model manager"}
            )
            with urlopen(request, timeout=self.timeout) as response:
                final_url = getattr(response, "geturl", lambda: artifact["url"])()
                if urlparse(final_url).scheme != "https":
                    raise ModelIntegrityError(f"Artifact for {ref} redirected away from HTTPS")
                with target.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            target.unlink(missing_ok=True)
            raise ModelIntegrityError(f"Could not download artifact for {ref}: {exc}") from exc
        expected_size = artifact["size"]
        expected_hash = artifact["sha256"]
        if size != expected_size or digest.hexdigest() != expected_hash:
            target.unlink(missing_ok=True)
            raise ModelIntegrityError(
                f"Integrity check failed for {ref}: expected {expected_size} bytes / {expected_hash}, "
                f"received {size} bytes / {digest.hexdigest()}"
            )

    def _install_dir(self, ref: str) -> Path:
        if not isinstance(ref, str) or ref.count(":") != 1:
            raise ModelNotFoundError(f"Invalid canonical model reference: {ref!r}")
        backend, model_id = ref.split(":", 1)
        return (
            self.models_dir
            / _safe_component(backend, "backend")
            / _safe_component(model_id, "model id")
        )

    def _read_manifest(self, directory: Path, *, expected_ref: str | None = None) -> dict[str, Any]:
        manifest_path = directory / MANIFEST_NAME
        if (
            self.models_dir.is_symlink()
            or directory.parent.is_symlink()
            or directory.is_symlink()
            or not directory.is_dir()
        ):
            raise ModelNotInstalledError(
                f"No SpeechScope-managed model directory or manifest at {directory}"
            )
        if manifest_path.is_symlink():
            raise ModelNotInstalledError(
                f"SpeechScope model manifest must not be a symbolic link: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelNotInstalledError(
                f"No valid SpeechScope model manifest in {directory}"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or isinstance(manifest.get("schema"), bool)
            or manifest.get("schema") != _MANIFEST_SCHEMA
            or manifest.get("catalog_kind") != "speechscope-model-catalog"
            or isinstance(manifest.get("catalog_schema"), bool)
            or manifest.get("catalog_schema") != 1
            or not isinstance(manifest.get("source_repository"), str)
            or not isinstance(manifest.get("source_revision"), str)
            or not isinstance(manifest.get("installed_at"), str)
            or not isinstance(manifest.get("speechscope_version"), str)
        ):
            raise ModelNotInstalledError(f"No valid SpeechScope model manifest in {directory}")
        ref = manifest.get("ref")
        if not isinstance(ref, str):
            raise ModelNotInstalledError(f"Invalid SpeechScope model manifest in {directory}")
        try:
            expected_directory = self._install_dir(ref)
        except (ModelCatalogError, ModelNotFoundError) as exc:
            raise ModelNotInstalledError(
                f"Invalid SpeechScope model manifest in {directory}"
            ) from exc
        if directory != expected_directory:
            raise ModelNotInstalledError(f"SpeechScope model manifest does not own {directory}")
        backend, model_id = ref.split(":", 1)
        source = manifest.get("source")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("revision"), str)
            or not _REVISION_RE.fullmatch(source["revision"])
            or not isinstance(source.get("repository"), str)
            or manifest["source_repository"] != source["repository"]
            or manifest["source_revision"] != source["revision"]
        ):
            raise ModelNotInstalledError(f"SpeechScope model manifest in {directory} is invalid")
        if manifest.get("backend") != backend or manifest.get("id") != model_id:
            raise ModelNotInstalledError(f"SpeechScope model manifest in {directory} is invalid")
        if expected_ref is not None and ref != expected_ref:
            raise ModelNotInstalledError(
                f"Manifest in {directory} belongs to {ref}, not {expected_ref}"
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ModelNotInstalledError(f"SpeechScope model manifest in {directory} is invalid")
        primary_count = 0
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ModelNotInstalledError(
                    f"SpeechScope model manifest in {directory} is invalid"
                )
            try:
                _safe_artifact_path(artifact.get("path"))
            except ModelCatalogError as exc:
                raise ModelNotInstalledError(
                    f"SpeechScope model manifest in {directory} is invalid"
                ) from exc
            size, digest = artifact.get("size"), artifact.get("sha256")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or not _SHA256_RE.fullmatch(digest)
                or not isinstance(artifact.get("primary"), bool)
            ):
                raise ModelNotInstalledError(
                    f"SpeechScope model manifest in {directory} is invalid"
                )
            primary_count += artifact["primary"]
        if primary_count != 1:
            raise ModelNotInstalledError(f"SpeechScope model manifest in {directory} is invalid")
        return manifest

    @staticmethod
    def _verify_catalog_metadata(manifest: dict[str, Any], model: dict[str, Any]) -> None:
        ModelStore._require_managed(model)
        expected_artifacts = [
            {
                "path": _safe_artifact_path(item["path"]).as_posix(),
                "size": item["size"],
                "sha256": item["sha256"],
                "primary": item.get("primary") is True,
            }
            for item in model["artifacts"]
        ]
        actual_artifacts = [
            {
                "path": item["path"],
                "size": item["size"],
                "sha256": item["sha256"],
                "primary": item["primary"],
            }
            for item in manifest["artifacts"]
        ]
        if (
            manifest.get("catalog_kind") != "speechscope-model-catalog"
            or manifest.get("catalog_schema") != 1
            or manifest.get("source") != model["source"]
            or actual_artifacts != expected_artifacts
        ):
            raise ModelIntegrityError(
                f"Installed model {model['ref']} manifest does not match the current catalog"
            )

    def installed(self) -> builtins.list[dict[str, Any]]:
        result: builtins.list[dict[str, Any]] = []
        if not self.models_dir.is_dir():
            return result
        for manifest_path in self.models_dir.glob("*/*/" + MANIFEST_NAME):
            directory = manifest_path.parent
            try:
                result.append(self._read_manifest(directory))
            except ModelNotInstalledError:
                continue
        return sorted(result, key=lambda item: item["ref"])

    def path(self, model: dict[str, Any] | str) -> Path:
        if isinstance(model, dict):
            self._require_managed(model)
        ref = model if isinstance(model, str) else model["ref"]
        self.verify(model)
        directory = self._install_dir(ref)
        manifest = self._read_manifest(directory, expected_ref=ref)
        primary = next((a for a in manifest["artifacts"] if a.get("primary") is True), None)
        if primary is None:
            raise ModelNotInstalledError(f"Installed model {ref} has no primary artifact")
        path = _artifact_file(directory, primary["path"])
        if not path.is_file():
            raise ModelNotInstalledError(f"Primary artifact for {ref} is missing: {path}")
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except OSError as exc:
            raise ModelIntegrityError(
                f"Primary artifact for {ref} is not readable: {path}"
            ) from exc
        return path

    def verify(self, model: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(model, dict):
            self._require_managed(model)
        ref = model if isinstance(model, str) else model["ref"]
        directory = self._install_dir(ref)
        manifest = self._read_manifest(directory, expected_ref=ref)

        if isinstance(model, dict):
            self._verify_catalog_metadata(manifest, model)
        results = []
        for artifact in manifest["artifacts"]:
            path = _artifact_file(directory, artifact["path"])
            try:
                actual_size = path.stat().st_size
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise ModelIntegrityError(
                    f"Could not verify installed artifact {path}: {exc}"
                ) from exc
            actual_hash = digest.hexdigest()
            if actual_size != artifact["size"] or actual_hash != artifact["sha256"]:
                raise ModelIntegrityError(
                    f"Installed model {ref} failed integrity verification: {path}"
                )
            results.append({"path": artifact["path"], "size": actual_size, "sha256": actual_hash})
        return {"ref": ref, "verified": True, "artifacts": results}

    def remove(self, model: dict[str, Any] | str) -> None:
        if isinstance(model, dict):
            self._require_managed(model)
        ref = model if isinstance(model, str) else model["ref"]
        directory = self._install_dir(ref)
        self._read_manifest(directory, expected_ref=ref)
        shutil.rmtree(directory)


class ModelManager:
    """Small facade combining catalog lookup and SpeechScope-owned storage."""

    def __init__(
        self,
        *,
        catalog: str | Path | None = None,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        refresh: bool = False,
        packaged_catalog: bool = False,
    ):
        self.catalog = ModelCatalogClient(
            catalog,
            cache_dir=cache_dir,
            offline=offline,
            refresh=refresh,
            packaged=packaged_catalog,
        )
        self.store = ModelStore(cache_dir)

    def list(self, *, backend: str | None = None) -> builtins.list[dict[str, Any]]:
        return self.catalog.list(backend=backend)

    def show(self, ref: str, *, backend: str | None = None) -> dict[str, Any]:
        return self.catalog.resolve(ref, backend=backend)

    def install(self, ref: str, *, backend: str | None = None, force: bool = False) -> Path:
        return self.store.install(self.show(ref, backend=backend), force=force)

    def installed(self) -> builtins.list[dict[str, Any]]:
        return self.store.installed()

    def _installed_selector(self, ref: str, backend: str | None):
        model = self.show(ref, backend=backend)
        ModelStore._require_managed(model)
        return model

    def path(self, ref: str, *, backend: str | None = None) -> Path:
        return self.store.path(self._installed_selector(ref, backend))

    def verify(self, ref: str, *, backend: str | None = None) -> dict[str, Any]:
        return self.store.verify(self._installed_selector(ref, backend))

    def remove(self, ref: str, *, backend: str | None = None) -> None:
        model = self.show(ref, backend=backend)
        self.store.remove(model)
