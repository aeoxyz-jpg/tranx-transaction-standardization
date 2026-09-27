import random
from tranx.synth.hard import hard_descriptor
from tranx.synth.canonical import derive_canonical


def _gen(seed, canonical="McDonald's", category="Food & Dining", country="USA"):
    return hard_descriptor(canonical, category, country, random.Random(seed))


def test_hard_descriptor_is_deterministic():
    assert _gen(7) == _gen(7)


def test_hard_descriptor_is_nonempty_string():
    out = _gen(1)
    assert isinstance(out, str) and len(out) > 0


def test_hard_descriptor_varies_across_seeds():
    outs = {_gen(s) for s in range(40)}
    assert len(outs) > 5  # real variety, not one fixed template


def test_hard_descriptor_defeats_naive_stripper():
    # The whole point: the simple regex stripper that builds the silver label
    # should NOT cleanly recover the canonical name from a hard descriptor.
    n = 100
    recovered = sum(1 for s in range(n) if derive_canonical(_gen(s)) == "McDonald's")
    assert recovered < 0.5 * n


def test_hard_descriptor_keeps_some_merchant_signal():
    # It must stay parseable in principle: the merchant's leading letters survive
    # in at least some fraction of samples (a smart parser/LLM can still recover it).
    n = 100
    hits = sum(1 for s in range(n) if "MCDONALD" in _gen(s).upper() or "McDonald" in _gen(s))
    assert hits > 0.5 * n


def test_truncation_never_removes_the_merchant():
    import random, re
    from tranx.synth.hard import hard_descriptor_flags
    rng = random.Random(0)
    for _ in range(2000):
        d, abbrev, _ = hard_descriptor_flags("Kentucky Fried Chicken", "Food & Dining",
                                             "USA", rng)
        letters = re.sub(r"[^a-z]", "", d.lower())
        # abbreviation drops vowels on purpose (KNTCKY); only its first letter survives
        assert ("kntcky" if abbrev else "kentuck") in letters, d


def test_abbreviate_drops_vowels_after_first_letter():
    from tranx.synth.hard import _abbreviate
    assert _abbreviate("BLUE HERON BAKERY") == "BLUE HRN BKRY"
    assert _abbreviate("ACME OAK ELM") == "ACME OAK ELM"  # words <= 4 chars untouched
    assert _abbreviate("ORCHARD") == "ORCHRD"  # first letter kept even if a vowel


def test_flags_recorded_and_wrapper_matches():
    from tranx import config
    from tranx.synth.hard import hard_descriptor_flags
    abbrevs, truncs = 0, 0
    for s in range(2000):
        d, a, t = hard_descriptor_flags("Blue Heron Bakery", "Food & Dining", "USA",
                                        random.Random(s))
        assert d == hard_descriptor("Blue Heron Bakery", "Food & Dining", "USA",
                                    random.Random(s))
        if a:
            assert "HRN" in d.upper() or "Hrn" in d
            assert "HERON" not in d.upper()
        abbrevs += a
        truncs += t
    assert abs(abbrevs / 2000 - config.ABBREV_P) < 0.03
    assert truncs == 0  # 17-char name never cut by a 20-25 char limit
    long_name = "Pembrook Chiropractic Associates"
    cut = sum(hard_descriptor_flags(long_name, "x", "USA", random.Random(s))[2]
              for s in range(500))
    assert 0.35 * 500 < cut < 0.65 * 500


def test_trunc_flag_only_when_name_is_cut():
    from tranx.synth.hard import hard_descriptor_flags
    for s in range(2000):
        d, a, t = hard_descriptor_flags("Pembrook Chiropractic Associates", "x", "USA",
                                        random.Random(s))
        if not t and not a:
            assert "ASSOCIATES" in d.upper().replace(" ", "") or "Associates" in d, d
