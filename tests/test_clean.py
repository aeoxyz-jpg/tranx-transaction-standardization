from tranx.pipeline.clean import clean_description, strip_processor_prefix


def test_lowercases_and_strips_ids():
    assert clean_description("McDonald's #111") == "mcdonald's"
    assert clean_description("UCLA Medical TXN176162") == "ucla medical"


def test_collapses_whitespace():
    assert clean_description("  BP   on   Buford  ") == "bp on buford"


def test_keeps_alphanumeric_core():
    assert clean_description("Disney+ #2115") == "disney+"


def test_strip_processor_prefix_spaced():
    assert strip_processor_prefix("SQ *COFFEE") == "COFFEE"
    assert strip_processor_prefix("PAYPAL *CAPITAL ONE") == "CAPITAL ONE"
    assert strip_processor_prefix("POS DEBIT WALMART") == "WALMART"
    assert strip_processor_prefix("PURCHASE DIRECT DEPOSIT") == "DIRECT DEPOSIT"


def test_strip_processor_prefix_concatenated():
    # star wrappers carry their own separator, so fused merchants are still exposed
    assert strip_processor_prefix("PP*CAPITALGAINS") == "CAPITALGAINS"
    assert strip_processor_prefix("PP*DOUGHNOTTS") == "DOUGHNOTTS"


def test_word_prefix_needs_boundary():
    # a word prefix fused to letters is part of a name, not a prefix
    assert strip_processor_prefix("POSDEBITMATERNITY") == "POSDEBITMATERNITY"
    assert strip_processor_prefix("ACHILLES SHOES") == "ACHILLES SHOES"
    assert strip_processor_prefix("PURCHASED GOODS LTD") == "PURCHASED GOODS LTD"
    assert strip_processor_prefix("ACH ELECTRIC CO") == "ELECTRIC CO"


def test_amazon_marketplace_is_not_a_wrapper():
    # real statement descriptors: the merchant IS Amazon
    for d in ("AMZNMktplace", "AMZNMKTPLACE AMAZO", "AMZN Mktp UK*MI5TU", "AMZN MKTP UK AMAZO"):
        assert strip_processor_prefix(d) == d


def test_strip_processor_prefix_noop_without_prefix():
    assert strip_processor_prefix("McDonald's #111") == "McDonald's #111"
