from collections import Counter
from itertools import product

import pytest

from speechscope.errors import AlignmentTooLargeError, InvalidReferenceError
from speechscope.metrics import (
    MAX_CHARACTER_CELLS,
    MAX_EDIT_PATH_CELLS,
    char_rates,
    edit_path,
    word_rates,
)
from speechscope.normalize import normalize


def score(ref, hyp):
    r, h = ref.split(), hyp.split()
    return word_rates(edit_path(r, h), len(r), len(h))


def test_exact():
    result = score("a b", "a b")
    assert result["wer"] == result["mer"] == result["wil"] == 0
    assert result["wip"] == 1


def test_sub_del_ins_and_wer_above_one():
    assert score("a b", "a c")["word_substitutions"] == 1
    assert score("a b", "a")["word_deletions"] == 1
    assert score("a", "a b")["word_insertions"] == 1
    assert score("a", "b c d")["wer"] == 3


def test_empty_hypothesis():
    result = score("some words", "")
    assert result["wer"] == 1.0
    assert result["wip"] == 0 and result["wil"] == 1


def test_empty_reference_invalid():
    with pytest.raises(InvalidReferenceError):
        score("", "hello")


def test_ambiguous_alignment_tie_break():
    assert [(x.tag, x.ref, x.hyp) for x in edit_path(["a", "a"], ["a"])] == [
        ("deletion", 0, None),
        ("match", 1, 0),
    ]


def test_cer_above_one():
    assert char_rates("a", "abcd")["cer"] == 3


def test_normalization_unicode():
    assert normalize(" HéLLo,   WORLD! ") == "héllo world"
    assert normalize("DoN’T") == "don't"
    assert normalize("A, b!", "strict-v1") == "A, b!"
    assert normalize("Hi...There") == "hi there"
    assert normalize("A, B", "readio-compat-v1") == "a b"


def test_character_counts_match_full_edit_path_tie_break():
    for ref_length in range(1, 4):
        for ref in product("ab", repeat=ref_length):
            for hyp_length in range(4):
                for hyp in product("ab", repeat=hyp_length):
                    rates = char_rates("".join(ref), "".join(hyp))
                    counts = Counter(edit.tag for edit in edit_path(ref, hyp))
                    assert rates["character_substitutions"] == counts["substitution"]
                    assert rates["character_deletions"] == counts["deletion"]
                    assert rates["character_insertions"] == counts["insertion"]


def test_alignment_cell_budgets_reject_oversized_inputs():
    assert MAX_EDIT_PATH_CELLS == 1_000_000
    assert MAX_CHARACTER_CELLS == 5_000_000
    with pytest.raises(AlignmentTooLargeError, match="1,002,001"):
        edit_path(["a"] * 1000, ["a"] * 1000)
    with pytest.raises(AlignmentTooLargeError, match="5,004,169"):
        char_rates("a" * 2236, "a" * 2236)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Café, déjà vu!", "café déjà vu"),
        ("Привет, мир!", "привет мир"),
        ("你好世界", "你好世界"),
        ("こんにちは。世界！", "こんにちは 世界"),
        ("مرحبا، بالعالم", "مرحبا بالعالم"),
    ],
)
def test_basic_normalization_multilingual_token_boundaries(text, expected):
    assert normalize(text) == expected


def test_basic_normalization_keeps_accents_and_removes_punctuation_only_tokens():
    assert normalize("é") == "é"
    assert normalize("!!!") == ""
