from tranx import config


def test_categories_has_ten():
    assert len(config.CATEGORIES) == 10
    assert "Government & Legal" in config.CATEGORIES
    assert "Food & Dining" in config.CATEGORIES


def test_payment_methods():
    assert set(config.PAYMENT_METHODS) == {
        "credit_card", "debit_card", "ach", "wire", "check", "cash_app", "transfer",
    }


def test_seed_and_paths():
    assert config.SEED == 42
    assert config.DATA_DIR.name == "data"
    assert config.REPORTS_DIR.name == "reports"
