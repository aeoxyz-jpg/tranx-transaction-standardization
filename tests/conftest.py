from pathlib import Path
import polars as pl
import pytest


@pytest.fixture
def fixture_df() -> pl.DataFrame:
    path = Path(__file__).parent / "fixtures" / "sample.parquet"
    return pl.read_parquet(path)
