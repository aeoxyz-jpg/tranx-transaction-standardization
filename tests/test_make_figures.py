"""Tests for scripts/make_figures.py (spec D7 step 4, T8): each figure function
writes a PNG from toy inputs; a missing input file is skipped without an
exception; leaderboard "-" cells are ignored rather than crashing the plot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_figures as mf  # noqa: E402


def _row(route, split, view="model", **overrides):
    row = {"route": route, "split": split, "view": view,
          "category_acc": 0.8, "category_ci": 0.05,
          "merchant_norm": 0.7, "merchant_norm_ci": 0.1,
          "avg_ms": 500.0, "no_match_rate": 0.1, "retrieval_recall": 0.9}
    row.update(overrides)
    return row


def toy_leaderboard():
    rows = []
    for split in ("random", "unseen"):
        rows.append(_row("rules", split, merchant_norm=0.3, category_acc=0.6, avg_ms=0.5))
        rows.append(_row("embedding", split, merchant_norm=0.5, category_acc=0.7, avg_ms=8.0))
        rows.append(_row("slm_fewshot", split, merchant_norm=0.75, category_acc=0.72, avg_ms=560.0))
        rows.append(_row("jev_merchant", split, merchant_norm=0.8, category_acc=0.74, avg_ms=300.0))
        rows.append(_row("jev_slm", split, merchant_norm=0.85, category_acc=0.76, avg_ms=480.0))
        # cleaner: no category score at all ("-", as the real route produces)
        rows.append(_row("cleaner", split, merchant_norm=0.2, category_acc="-", category_ci="-",
                         avg_ms=0.02))
        # metadata: no merchant score at all ("-")
        rows.append(_row("metadata", split, merchant_norm="-", merchant_norm_ci="-",
                         category_acc=0.68, avg_ms=1.0))
    return rows


def toy_significance():
    return {
        "primary": {
            "synthetic_unseen": {"diff": 0.05, "lo": 0.01, "hi": 0.09, "label": "primary",
                                 "n_clusters": 100, "n_rows": 200},
            "moneydata": {"diff": 0.04, "lo": 0.01, "hi": 0.09, "label": "primary",
                         "n_clusters": 261, "n_rows": 547},
        },
        "exploratory": {
            "synthetic_random_merchant": {"skipped": "missing preds: nope", "label": "exploratory"},
            "ddt_category": {"diff": 0.09, "lo": 0.06, "hi": 0.12, "label": "exploratory",
                            "n_clusters": 1000, "n_rows": 1000},
        },
        "notes": {"moneydata_filter": "168 excluded", "ci_interpretation": "..."},
    }


def toy_moneydata():
    return {
        "spend_join_coverage": 1.0,
        "derive": {"distinct": 0.18, "row_weighted": 0.16, "macro_by_merchant": 0.32,
                  "spend_weighted": 0.13},
        "jev_known": {"distinct": 0.96, "row_weighted": 0.94, "macro_by_merchant": 0.94,
                     "spend_weighted": 0.77},
        "cascade_embed0.7_to_slm": {"distinct": 0.82, "row_weighted": 0.84,
                                   "macro_by_merchant": 0.69, "spend_weighted": 0.58,
                                   "held_subset": {"distinct": 0.64}, "escalated_descriptor_share": 0.67},
        "no_spend": {"distinct": 0.5, "row_weighted": 0.5, "macro_by_merchant": 0.5,
                    "spend_weighted": None},
    }


def test_fig_merchant_norm_writes_png(tmp_path):
    mf.fig_merchant_norm(toy_leaderboard(), tmp_path)
    assert (tmp_path / "merchant_norm.png").exists()


def test_fig_category_writes_png(tmp_path):
    mf.fig_category(toy_leaderboard(), tmp_path)
    assert (tmp_path / "category.png").exists()


def test_fig_latency_writes_png(tmp_path):
    mf.fig_latency(toy_leaderboard(), tmp_path)
    assert (tmp_path / "latency.png").exists()


def test_fig_significance_writes_png(tmp_path):
    mf.fig_significance(toy_significance(), tmp_path)
    assert (tmp_path / "significance.png").exists()


def test_fig_significance_omits_skipped(tmp_path):
    sig = toy_significance()
    rows, skipped = mf._forest_rows(sig)
    keys = [k for k, _, _ in rows]
    assert "synthetic_random_merchant" not in keys
    assert any("synthetic_random_merchant" in s for s in skipped)
    assert "synthetic_unseen" in keys and "moneydata" in keys and "ddt_category" in keys


def test_fig_moneydata_writes_png(tmp_path):
    mf.fig_moneydata(toy_moneydata(), tmp_path)
    assert (tmp_path / "moneydata.png").exists()


def test_fig_moneydata_skips_none_spend_weighted(tmp_path):
    # "no_spend" has spend_weighted=None; it must not appear and must not crash.
    mf.fig_moneydata(toy_moneydata(), tmp_path)
    assert (tmp_path / "moneydata.png").exists()


def test_missing_input_file_skipped_without_exception(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.json"
    out = tmp_path / "figs"
    mf.run(missing, missing, missing, out)
    captured = capsys.readouterr()
    assert "WARNING missing input" in captured.out
    assert list(out.glob("*.png")) == []


def test_run_with_only_moneydata_present(tmp_path, capsys):
    import json
    md_path = tmp_path / "md.json"
    md_path.write_text(json.dumps(toy_moneydata()))
    missing = tmp_path / "nope.json"
    out = tmp_path / "figs"
    mf.run(missing, missing, md_path, out)
    assert (out / "moneydata.png").exists()
    assert not (out / "merchant_norm.png").exists()
    assert not (out / "significance.png").exists()


def test_dash_values_ignored_in_merchant_norm(tmp_path):
    # metadata route has merchant_norm == "-" for both splits; must not crash
    # and must not be drawn as a route bar (no numeric merchant_norm anywhere).
    rows = toy_leaderboard()
    mf.fig_merchant_norm(rows, tmp_path)
    idx = mf._rows_by_route_split(rows, "model")
    assert mf._num(idx["metadata"]["random"]["merchant_norm"]) is None
    assert (tmp_path / "merchant_norm.png").exists()


def test_dash_values_ignored_in_category(tmp_path):
    # cleaner route has category_acc == "-" for both splits; must not crash.
    rows = toy_leaderboard()
    mf.fig_category(rows, tmp_path)
    idx = mf._rows_by_route_split(rows, "model")
    assert mf._num(idx["cleaner"]["random"]["category_acc"]) is None
    assert (tmp_path / "category.png").exists()


def test_empty_leaderboard_all_dash_skips_without_exception(tmp_path, capsys):
    rows = [_row("metadata", "random", merchant_norm="-", merchant_norm_ci="-"),
           _row("metadata", "unseen", merchant_norm="-", merchant_norm_ci="-")]
    mf.fig_merchant_norm(rows, tmp_path)
    captured = capsys.readouterr()
    assert "skipping merchant_norm.png" in captured.out
    assert not (tmp_path / "merchant_norm.png").exists()


def test_run_refuses_leaderboard_and_significance_from_different_runs(tmp_path):
    import json
    import pytest
    from make_figures import run
    lb = tmp_path / "lb.json"
    sig = tmp_path / "sig.json"
    lb.write_text(json.dumps([{"route": "rules", "split": "random", "view": "model",
                               "merchant_norm": 0.5, "manifest_hash": "a"}]))
    sig.write_text(json.dumps({"primary": {}, "exploratory": {}, "manifest_hash": "b"}))
    with pytest.raises(SystemExit):
        run(lb, sig, tmp_path / "missing.json", tmp_path / "out")
