from tranx import config
from tranx.synth.families import _tokens, brand_families
from tranx.synth.local import local_merchants, LOCAL_CATEGORIES

_SOURCE = ["McDonald's", "Walmart", "Walmart Pharmacy", "Family Dental", "Blue Bottle",
           "Nordstrom", "Nordstrom Rack", "Pharmacy", "Kroger Pharmacy",
           "Safeway Pharmacy", "Fox Theater", "Pizza Hut", "Grill House"]


def test_local_merchants_deterministic_and_sized():
    a = local_merchants(_SOURCE, seed=42, n=800)
    b = local_merchants(_SOURCE, seed=42, n=800)
    assert a == b
    assert len(a) == 800 and len({n for n, _ in a}) == 800
    assert local_merchants(_SOURCE, seed=7, n=800) != a


def test_local_merchants_share_no_token_with_source():
    banned = set().union(*[_tokens(s) for s in _SOURCE])
    for name, _ in local_merchants(_SOURCE, seed=42, n=800):
        assert not (_tokens(name) & banned), name


def test_local_categories_are_merchant_categories():
    cats = {c for _, c in local_merchants(_SOURCE, seed=42, n=800)}
    assert cats == set(LOCAL_CATEGORIES)
    assert cats <= set(config.CATEGORIES)
    assert not cats & {"Income", "Financial Services"}


def test_source_families_unchanged_by_locals():
    names = [n for n, _ in local_merchants(_SOURCE, seed=42, n=800)]
    before = brand_families(_SOURCE)
    after = brand_families(_SOURCE + names)
    assert {m: after[m] for m in _SOURCE} == before
    # no local lands in a source family
    source_keys = set(before.values())
    assert not {after[n] for n in names} & source_keys
