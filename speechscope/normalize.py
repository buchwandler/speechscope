"""Versioned Unicode normalization; strict profile preserves punctuation and case."""

import re
import unicodedata

from .errors import InvalidReferenceError

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "＇": "'"})


def normalize(text: str, profile: str = "basic-v1") -> str:
    if profile not in ("basic-v1", "strict-v1", "readio-compat-v1"):
        raise InvalidReferenceError(f"Unknown normalization profile: {profile}")
    text = unicodedata.normalize("NFKC", text)
    if profile == "strict-v1":
        return " ".join(text.split())
    text = text.casefold().translate(_APOSTROPHES)
    if profile == "readio-compat-v1":
        text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    else:
        # Apostrophes retained only between word characters; punctuation becomes a token boundary.
        text = "".join(
            ch
            if (
                ch.isalnum()
                or ch == "_"
                or ch.isspace()
                or (
                    ch == "'"
                    and 0 < i < len(text) - 1
                    and text[i - 1].isalnum()
                    and text[i + 1].isalnum()
                )
            )
            else " "
            for i, ch in enumerate(text)
        )
    return " ".join(text.split())
