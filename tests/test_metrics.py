import pytest

from speechscope.errors import InvalidReferenceError
from speechscope.metrics import char_rates, edit_path, word_rates
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
