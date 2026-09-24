"""Reading the supplied TSVs.

The files are tab-separated and names contain literal quote characters, so quoting is
disabled (`quote_char=None`). Every column is read as a string.
"""
import polars as pl

from . import config as C


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)


def cache_sources(force: bool = False) -> None:
    """Convert every supplied TSV to parquet once (fast reloads)."""
    C.CACHE.mkdir(parents=True, exist_ok=True)
    for split in C.SPLITS:
        for s in C.SOURCES:
            out = C.parquet_path(split, s)
            if out.exists() and not force:
                continue
            read_tsv(C.src_path(split, s)).write_parquet(out)
    gt_out = C.CACHE / "train_ground_truth.parquet"
    if force or not gt_out.exists():
        read_tsv(C.DATA / "train" / "train_ground_truth.tsv").write_parquet(gt_out)


def load_source(split: str, source: int) -> pl.DataFrame:
    return pl.read_parquet(C.parquet_path(split, source))


def load_train_pairs() -> pl.DataFrame:
    """Ground truth exploded to one row per (s1, matched id) positive pair."""
    path = C.CACHE / "train_pairs.parquet"
    if path.exists():
        return pl.read_parquet(path)
    gt = pl.read_parquet(C.CACHE / "train_ground_truth.parquet")
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
        .with_columns(pl.col("m").str.slice(0, 2).alias("src"))
    )
    pairs.write_parquet(path)
    return pairs
