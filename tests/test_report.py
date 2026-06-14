from tranx.eval.report import build_leaderboard_md


def test_leaderboard_md_has_rows_and_headers():
    results = [
        {"route": "rules", "split": "random", "category_acc": 0.71, "macro_f1": 0.65,
         "merchant_acc": 0.80, "dedup_ratio": 0.5, "kpi_within_tol": 0.78, "avg_ms": 0.4},
        {"route": "embedding", "split": "unseen", "category_acc": 0.83, "macro_f1": 0.79,
         "merchant_acc": 0.88, "dedup_ratio": 0.4, "kpi_within_tol": 0.85, "avg_ms": 5.0},
    ]
    md = build_leaderboard_md(results)
    assert "| route | split |" in md
    assert "Dedup Ratio" in md
    assert "rules" in md
    assert "embedding" in md
    assert "random" in md
    assert "unseen" in md
    assert "0.83" in md
