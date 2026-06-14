import polars as pl
from tranx import config


def stratified_sample(df: pl.DataFrame, n: int, seed: int, by: list[str]) -> pl.DataFrame:
    """Proportional stratified sample of ~n rows, guaranteeing >=1 row per stratum."""
    total = len(df)
    if total <= n:
        return df

    out_frames = []
    for _, group in df.group_by(by, maintain_order=True):
        share = max(1, round(n * len(group) / total))
        take = min(share, len(group))
        out_frames.append(group.sample(n=take, seed=seed))
    return pl.concat(out_frames)


def load_sample(n: int = config.SAMPLE_SIZE, seed: int = config.SEED) -> pl.DataFrame:
    """Download the HF dataset parquet (cached) and return a stratified sample.

    Requires `hf auth login` / HF_TOKEN and accepted dataset terms.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=config.HF_DATASET,
        repo_type="dataset",
        filename="default/train/0000.parquet",
        revision="refs/convert/parquet",
    )
    df = pl.read_parquet(path)
    return stratified_sample(df, n=n, seed=seed, by=["category", "country"])
