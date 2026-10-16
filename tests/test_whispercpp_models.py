from __future__ import annotations

import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from speechscope.backends.whispercpp import EXECUTABLE_ENV
from speechscope.cli import main
from speechscope.models import ModelStore

REVISION = "b" * 40
WEIGHTS = b"verified model bytes"


def model_entry(*, backend="whispercpp", model_id="tiny-test", mode="managed"):
    ref = f"{backend}:{model_id}"
    filename = f"{model_id}.bin"
    artifact = {
        "filename": filename,
        "path": filename,
        "primary": True,
        "role": "model",
        "size": len(WEIGHTS) if mode == "managed" else None,
        "sha256": hashlib.sha256(WEIGHTS).hexdigest() if mode == "managed" else None,
        "url": f"https://huggingface.co/test/model/resolve/{REVISION}/{filename}?download=true",
    }
    acquisition = {"mode": mode}
    if mode == "runtime-managed":
        acquisition.update(
            reason="Photon owns the cache.", runtime_locator="moondream/parakeet-redux"
        )
    return {
        "ref": ref,
        "backend": backend,
        "id": model_id,
        "aliases": [model_id],
        "kind": "model",
        "format": "ggml" if backend == "whispercpp" else "photon-parakeet-redux",
        "languages": ["en"],
        "description": "Test model entry",
        "acquisition": acquisition,
        "runtime": {"model_argument": "primary_artifact"},
        "source": {
            "provider": "huggingface",
            "repository": "test/model",
            "revision": REVISION,
            "license": "mit",
        },
        "artifacts": [artifact],
    }


def write_catalog(path: Path, *models):
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "speechscope-model-catalog",
                "source": {"provider": "huggingface", "model_count": len(models)},
                "models": {model["ref"]: model for model in models},
            }
        ),
        encoding="utf-8",
    )
    return path


def fake_whisper_cli(tmp_path: Path, monkeypatch):
    executable = tmp_path / "whisper-cli-fake"
    args_file = tmp_path / "args.json"
    executable.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        "args=sys.argv[1:]\n"
        "def option(name): return args[args.index(name)+1]\n"
        "with open(os.environ['FAKE_ARGS_FILE'], 'w') as f: json.dump(args, f)\n"
        "prefix=option('--output-file')\n"
        "with open(prefix+'.json', 'w') as f: json.dump({'result': {'text': 'hello', 'language': 'en'}, 'transcription': [{'text': ' hello', 'offsets': {'from': 0, 'to': 400}}]}, f)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv(EXECUTABLE_ENV, str(executable))
    monkeypatch.setenv("FAKE_ARGS_FILE", str(args_file))
    return args_file


def test_managed_whispercpp_reference_transcribes_without_catalog_or_weight_network(
    tmp_path, monkeypatch
):
    model = model_entry()
    cache = tmp_path / "cache"
    with patch("speechscope.models.urlopen", return_value=BytesIO(WEIGHTS)):
        installed_path = ModelStore(cache).install(model)
    assert installed_path.read_bytes() == WEIGHTS

    args_file = fake_whisper_cli(tmp_path, monkeypatch)
    source = tmp_path / "speech.wav"
    sf.write(
        source,
        np.zeros(1600, dtype=np.float32),
        16_000,
        format="WAV",
        subtype="PCM_16",
    )
    with patch(
        "speechscope.models.urlopen",
        side_effect=AssertionError("inference must not use network"),
    ):
        from speechscope.api import transcribe_report

        report = transcribe_report(
            source,
            backend="whispercpp",
            model=model["ref"],
            cache_dir=cache,
        )

    argv = json.loads(args_file.read_text(encoding="utf-8"))
    assert argv[argv.index("--model") + 1] == str(installed_path)
    assert report.transcription.model == model["ref"]
    assert report.transcription.text == "hello"


def test_models_list_and_show_support_local_catalog_json(tmp_path, capsys):
    model = model_entry()
    catalog = write_catalog(tmp_path / "models.json", model)

    assert main(["models", "list", "--catalog", str(catalog), "--format", "json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [entry["ref"] for entry in listed] == [model["ref"]]

    assert (
        main(["models", "show", model["ref"], "--catalog", str(catalog), "--format", "json"]) == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["source"]["revision"] == REVISION


def test_managed_models_cli_lifecycle(tmp_path, capsys):
    model = model_entry()
    catalog = write_catalog(tmp_path / "models.json", model)
    cache = tmp_path / "cache"
    common = ["--cache-dir", str(cache), "--format", "json"]

    with patch("speechscope.models.urlopen", return_value=BytesIO(WEIGHTS)):
        assert main(["models", "install", model["ref"], "--catalog", str(catalog), *common]) == 0
    installed = json.loads(capsys.readouterr().out)
    assert installed["ref"] == model["ref"]

    assert main(["models", "installed", *common]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory[0]["ref"] == model["ref"]

    assert main(["models", "path", model["ref"], *common]) == 0
    model_path = json.loads(capsys.readouterr().out)["path"]
    assert Path(model_path).is_file()

    assert main(["models", "verify", model["ref"], *common]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True

    assert main(["models", "remove", model["ref"], "--catalog", str(catalog), *common]) == 0
    assert json.loads(capsys.readouterr().out)["removed"] is True
    assert main(["models", "installed", *common]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_runtime_managed_install_is_rejected_as_user_error(tmp_path, capsys):
    model = model_entry(backend="redux", model_id="parakeet-redux", mode="runtime-managed")
    catalog = write_catalog(tmp_path / "models.json", model)

    status = main(
        [
            "models",
            "install",
            model["ref"],
            "--catalog",
            str(catalog),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    captured = capsys.readouterr()
    assert status == 3
    assert "Photon owns the cache" in captured.err
    assert "internal error" not in captured.err
    assert not (tmp_path / "cache" / "models").exists()
