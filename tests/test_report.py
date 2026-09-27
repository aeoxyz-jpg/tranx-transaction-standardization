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


def test_leaderboard_shows_dash_for_absent_view_columns():
    results = [
        {"route": "rules", "split": "random", "category_acc": 0.71},
    ]
    md = build_leaderboard_md(results)
    lines = md.splitlines()
    row = [l for l in lines if l.startswith("| rules")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    by_header = dict(zip(header_cells, cells))
    assert by_header["view"] == "-"
    assert by_header["No-Match Rate"] == "-"
    assert by_header["Retrieval Recall"] == "-"


def test_leaderboard_spend_kpi_only_shown_for_ideal_cache_view():
    results = [
        {"route": "rules", "split": "random", "view": "model", "kpi_within_tol": 0.9},
        {"route": "rules", "split": "random", "view": "ideal_cache", "kpi_within_tol": 0.9},
    ]
    md = build_leaderboard_md(results)
    lines = [l for l in md.splitlines() if l.startswith("| rules")]
    header_cells = [c.strip() for c in md.splitlines()[0].strip("|").split("|")]

    def cell(line, name):
        cells = [c.strip() for c in line.strip("|").split("|")]
        return dict(zip(header_cells, cells))[name]

    assert cell(lines[0], "Spend KPI") == "-"       # view=model -> suppressed
    assert cell(lines[1], "Spend KPI") == "0.90"    # view=ideal_cache -> shown
