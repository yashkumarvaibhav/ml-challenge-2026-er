"""Validation splits (PLAN.md V1, V3, V6).

Drawn once, with a fixed seed, before any learned artifact exists:

* holdout: 200k train S1 entities, stratified by country x match-count bucket
  (0,1,2,3,4,5,6+); split into halves A/B (same strata) for two-way threshold tuning.
* ghosts: 20% of *all* train S1 entities, stratified by country. In the 20%-ghost
  world their S1 rows vanish from the reference population while their S2/S3 records
  stay in the pool as orphans. Ghost labels are never used for learning.
* training portion: everything that is neither holdout nor ghost.

Output: cache/splits/train_s1_split.parquet and a small summary in experiments/.
"""
import hashlib
import json

import numpy as np
import polars as pl

from . import config as C
from .io import cache_sources, load_source, load_train_pairs


def _stratified_pick(ids: np.ndarray, strata: np.ndarray, frac: float, rng) -> np.ndarray:
    """Boolean mask selecting round(frac * n) ids from every stratum."""
    pick = np.zeros(len(ids), dtype=bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        k = int(round(frac * len(idx)))
        pick[rng.choice(idx, size=k, replace=False)] = True
    return pick


def make_splits() -> pl.DataFrame:
    cache_sources()
    s1 = load_source("train", 1).select("entity_id", "country")
    pairs = load_train_pairs()
    n = pairs.group_by("s1").len().rename({"s1": "entity_id", "len": "n_matches"})
    df = (
        s1.join(n, on="entity_id", how="left")
        .with_columns(pl.col("n_matches").fill_null(0).cast(pl.Int32))
        .with_columns(pl.col("n_matches").clip(upper_bound=6).alias("bucket"))
        .sort("entity_id")  # deterministic order independent of file order
    )
    ids = df["entity_id"].to_numpy()
    country = df["country"].to_numpy()
    strata = np.char.add(country.astype(str), df["bucket"].cast(pl.Utf8).to_numpy().astype(str))

    rng = np.random.default_rng(C.SEED)
    holdout = _stratified_pick(ids, strata, C.HOLDOUT_SIZE / len(ids), rng)
    half_b = np.zeros(len(ids), dtype=bool)
    hidx = np.flatnonzero(holdout)
    half_b[hidx[_stratified_pick(ids[hidx], strata[hidx], 0.5, rng)]] = True
    ghost = _stratified_pick(ids, country.astype(str), C.GHOST_FRAC, rng)

    out = df.with_columns(
        pl.Series("is_holdout", holdout),
        pl.Series("_half_b", half_b),
        pl.Series("is_ghost", ghost),
    ).with_columns(
        pl.when(pl.col("is_holdout"))
        .then(pl.when(pl.col("_half_b")).then(pl.lit("B")).otherwise(pl.lit("A")))
        .otherwise(None)
        .alias("holdout_half"),
        (~pl.col("is_holdout") & ~pl.col("is_ghost")).alias("is_train"),
    ).drop("_half_b")

    (C.CACHE / "splits").mkdir(parents=True, exist_ok=True)
    out.write_parquet(C.CACHE / "splits" / "train_s1_split.parquet")

    def digest(mask) -> str:
        return hashlib.sha256("\n".join(out.filter(mask)["entity_id"].to_list()).encode()).hexdigest()

    summary = {
        "seed": C.SEED,
        "n_s1": out.height,
        "holdout": int(out["is_holdout"].sum()),
        "holdout_A": int((out["holdout_half"] == "A").sum()),
        "holdout_B": int((out["holdout_half"] == "B").sum()),
        "ghost": int(out["is_ghost"].sum()),
        "holdout_nonghost": int((out["is_holdout"] & ~out["is_ghost"]).sum()),
        "train_portion": int(out["is_train"].sum()),
        "by_country": out.group_by("country")
        .agg(
            pl.len().alias("n"),
            pl.col("is_holdout").sum().alias("holdout"),
            pl.col("is_ghost").sum().alias("ghost"),
            pl.col("is_train").sum().alias("train"),
        )
        .sort("country")
        .to_dicts(),
        "holdout_bucket_share": out.filter(pl.col("is_holdout"))
        .group_by("bucket")
        .len()
        .sort("bucket")
        .with_columns((pl.col("len") / pl.col("len").sum()).round(4))
        .to_dicts(),
        "sha256_holdout_ids": digest(pl.col("is_holdout")),
        "sha256_ghost_ids": digest(pl.col("is_ghost")),
    }
    C.EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    (C.EXPERIMENTS / "splits.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    make_splits()
