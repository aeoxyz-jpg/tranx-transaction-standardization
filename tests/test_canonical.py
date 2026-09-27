import pytest
from tranx.synth.canonical import derive_canonical, strip_coverage


@pytest.mark.parametrize("raw, expected", [
    ("McDonald's #111", "McDonald's"),
    ("McDonald's #121", "McDonald's"),
    ("Arby's (Contactless)", "Arby's"),
    ("Amazon - AUSTRALIA", "Amazon"),
    ("UCLA Medical #7731 - UK Center TXN176162", "UCLA Medical"),
    ("Starbucks - AUSTRALIA Store - Evening", "Starbucks"),
    ("Tim Hortons #1434 Branch (App)", "Tim Hortons"),
    ("Walgreens #8780", "Walgreens"),
    ("Target - USA Store - Rush Hour", "Target"),
    ("Disney+ #2115", "Disney+"),
    ("Barnes & Noble #9059", "Barnes & Noble"),
    ("Wage", "Wage"),
    # source-template location/time suffixes (split brands into fake merchants before)
    ("Walmart Pharmacy Shopping Center", "Walmart Pharmacy"),
    ("Rally's Business District", "Rally's"),
    ("Delta Airport", "Delta"),
    ("MBTA Campus", "MBTA"),
    ("Domino's - Weekday", "Domino's"),
    ("Burlington Store Branch Hospital", "Burlington"),
    ("Donation Hospital", "Donation"),
    # genuine hospital merchants survive
    ("Hospital", "Hospital"),
    ("Hospital #9559", "Hospital"),
    ("Hospital Airport", "Hospital"),
    ("Children's Hospital Airport", "Children's Hospital"),
    ("Children's Hospital", "Children's Hospital"),
])
def test_derive_canonical(raw, expected):
    assert derive_canonical(raw) == expected


def test_strip_coverage_returns_fraction():
    raws = ["McDonald's #111", "Amazon - AUSTRALIA", "Wage"]
    cov = strip_coverage(raws)
    assert 0.0 <= cov <= 1.0
    assert cov == pytest.approx(1.0)  # all reduce to clean alpha names


@pytest.mark.parametrize("raw, expected", [
    # names that contain a noise word keep it
    ("Community Center - Weekday", "Community Center"),
    ("Senior Center #12 Airport", "Senior Center"),
    ("Convention Center Business District", "Convention Center"),
    ("Online Bank Store", "Online Bank"),
    ("Holiday Pay - Evening", "Holiday Pay"),
    # the noise words are still stripped elsewhere
    ("Bank Online", "Bank"),
    ("Target Center", "Target"),
])
def test_protected_names(raw, expected):
    assert derive_canonical(raw) == expected


def test_gold_merchant_rules():
    from tranx.synth.canonical import gold_merchant
    assert gold_merchant("Salary", "Income") == (None, "Salary")
    assert gold_merchant("Holiday Pay", "Income") == (None, "Holiday Pay")
    assert gold_merchant("Cane's", "Food & Dining") == ("Raising Cane's", None)
    assert gold_merchant("Frontier", "Transportation") == ("Frontier Airlines", None)
    assert gold_merchant("Frontier", "Utilities & Services") == ("Frontier Communications", None)
    assert gold_merchant("Starbucks", "Food & Dining") == ("Starbucks", None)
