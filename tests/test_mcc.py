import random
from tranx.synth.mcc import CATEGORY_MCC, mcc_for
from tranx import config


def test_every_category_has_mcc_options():
    for cat in config.CATEGORIES:
        assert cat in CATEGORY_MCC
        assert len(CATEGORY_MCC[cat]) >= 1


def test_mcc_for_is_deterministic_with_seed():
    a = mcc_for("Food & Dining", random.Random(1))
    b = mcc_for("Food & Dining", random.Random(1))
    assert a == b
    assert a in CATEGORY_MCC["Food & Dining"]


def test_mcc_for_unknown_category_returns_none():
    assert mcc_for("Nonexistent", random.Random(1)) is None
