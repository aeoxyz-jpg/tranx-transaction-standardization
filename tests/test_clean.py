from tranx.pipeline.clean import clean_description, strip_processor_prefix


def test_lowercases_and_strips_ids():
    assert clean_description("McDonald's #111") == "mcdonald's"
    assert clean_description("UCLA Medical TXN176162") == "ucla medical"


def test_collapses_whitespace():
    assert clean_description("  BP   on   Buford  ") == "bp on buford"


def test_keeps_alphanumeric_core():
    assert clean_description("Disney+ #2115") == "disney+"


def test_strip_processor_prefix_spaced():
    assert strip_processor_prefix("AMZN MKTP HOME DEPOT") == "HOME DEPOT"
    assert strip_processor_prefix("SQ *COFFEE") == "COFFEE"
    assert strip_processor_prefix("PAYPAL *CAPITAL ONE") == "CAPITAL ONE"
    assert strip_processor_prefix("POS DEBIT WALMART") == "WALMART"
    assert strip_processor_prefix("PURCHASE DIRECT DEPOSIT") == "DIRECT DEPOSIT"


def test_strip_processor_prefix_concatenated():
    # space-collapsed descriptors still get the fused prefix removed
    assert strip_processor_prefix("POSDEBITMATERNITY") == "MATERNITY"
    assert strip_processor_prefix("PP*CAPITALGAINS") == "CAPITALGAINS"


def test_strip_processor_prefix_noop_without_prefix():
    assert strip_processor_prefix("McDonald's #111") == "McDonald's #111"
