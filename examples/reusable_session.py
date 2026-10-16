"""Reuse one backend/model session to transcribe multiple WAV files."""

import argparse
from pathlib import Path

from speechscope import Verifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--backend", default="redux")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with Verifier(backend=args.backend, device=args.device) as session:
        for path in args.audio:
            result = session.transcribe(path)
            print(f"{path}: {result.text}")
            print(f"timing source: {result.timing_source}; words: {len(result.words)}")


if __name__ == "__main__":
    main()
