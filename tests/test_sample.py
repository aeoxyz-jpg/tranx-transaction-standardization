from tranx.synth.sample import stratified_sample


def test_stratified_sample_size_and_columns(fixture_df):
    out = stratified_sample(fixture_df, n=20, seed=42, by=["category", "country"])
    assert len(out) <= 20
    assert set(out.columns) == set(fixture_df.columns)


def test_stratified_sample_is_deterministic(fixture_df):
    a = stratified_sample(fixture_df, n=20, seed=42, by=["category"])
    b = stratified_sample(fixture_df, n=20, seed=42, by=["category"])
    assert a.to_dicts() == b.to_dicts()


def test_stratified_sample_covers_strata(fixture_df):
    out = stratified_sample(fixture_df, n=20, seed=42, by=["category"])
    # every category present in source appears in the sample
    assert set(out["category"].unique()) == set(fixture_df["category"].unique())
