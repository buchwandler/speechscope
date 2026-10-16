from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _metadata_version(content: bytes) -> tuple[str, str, str]:
    message = Parser().parsestr(content.decode("utf-8"))
    return (
        message["Version"],
        message.get_payload(),
        message.get("License-Expression", ""),
    )


def test_built_wheel_and_sdist_include_public_contracts(tmp_path):
    pytest.importorskip("build", reason="install the project dev extras to build artifacts")
    pytest.importorskip(
        "setuptools_scm", reason="install the project dev extras to build artifacts"
    )
    pytest.importorskip("wheel", reason="install the project dev extras to build artifacts")
    build_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert build_result.returncode == 0, f"{build_result.stdout}\n{build_result.stderr}"
    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())
        assert "speechscope/py.typed" in members
        assert "speechscope/schemas/report-v1.schema.json" in members
        assert "speechscope/backend_audio.py" in members
        assert "speechscope/backends/whispercpp.py" in members
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in members)
        metadata_path = next(name for name in members if name.endswith(".dist-info/METADATA"))
        version, description, license_expression = _metadata_version(wheel.read(metadata_path))
        assert wheel.read(
            next(name for name in members if name.endswith(".dist-info/entry_points.txt"))
        ) == (b"[console_scripts]\nspeechscope = speechscope.cli:main\n")
        metadata = Parser().parsestr(wheel.read(metadata_path).decode("utf-8"))
        assert metadata["Name"] == "speechscope"
        assert metadata.get("Description-Content-Type", "").startswith("text/markdown")
        assert "# SpeechScope" in description
        assert license_expression == "Apache-2.0"

    with tarfile.open(sdists[0], "r:gz") as sdist:
        members = sdist.getnames()
        assert any(name.endswith("/LICENSE") for name in members)
        assert any(name.endswith("/README.md") for name in members)
        assert any(name.endswith("/speechscope/py.typed") for name in members)
        assert any(name.endswith("/speechscope/schemas/report-v1.schema.json") for name in members)
        pkg_info = next(name for name in members if name.endswith("/PKG-INFO"))
        version_sdist, _, _ = _metadata_version(sdist.extractfile(pkg_info).read())

    assert version == version_sdist
    expected_version = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION")
    if expected_version:
        assert version == expected_version


def test_whispercpp_adds_no_python_or_ffmpeg_dependency():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = [item.lower() for item in project["dependencies"]]
    assert "numpy>=1.24" in dependencies
    assert "soundfile>=0.12" in dependencies
    assert "audiosig>=0.1.6,<0.2" in dependencies
    assert not any("whispercpp" in item or "ffmpeg" in item for item in dependencies)
    assert "whispercpp" not in project["optional-dependencies"]
