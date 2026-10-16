"""Analyze one WAV against a supplied reference using the selected ASR backend."""

import argparse
from pathlib import Path

from speechscope import VerificationRequest, analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--backend", default="redux")
    parser.add_argument("--normalization", default="basic-v1")
    args = parser.parse_args()

    result = analyze(
        VerificationRequest(
            audio=args.audio,
            reference_text=args.reference,
            backend=args.backend,
            normalization=args.normalization,
        )
    )
    metrics = result.metrics
    if metrics is None:
        raise RuntimeError("Analysis unexpectedly returned no comparison metrics")
    print(f"WER: {metrics.wer:.4f}")
    print(f"Compound-WER: {metrics.compound_wer:.4f}")
    print(f"CER: {metrics.cer:.4f}")
    print(f"Boundary variants: {metrics.word_boundary_match_count}")


if __name__ == "__main__":
    main()
