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
    from tranx.synth.hard import hard_descriptor
    rng = random.Random(0)
    for _ in range(2000):
        d = hard_descriptor("Kentucky Fried Chicken", "Food & Dining", "USA", rng)
        letters = re.sub(r"[^a-z]", "", d.lower())
        assert "kentuck" in letters, d
