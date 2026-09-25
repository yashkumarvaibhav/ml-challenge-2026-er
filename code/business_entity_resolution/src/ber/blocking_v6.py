"""v6 candidate generation: learned dense retrieval + exact keys (plan-discussion 2026-09-25 16:44 / 16:50 UTC).

Per country:
  (df) dense forward   S1 -> top KF S2/S3 records by bi-encoder cosine (faiss IVF, `ber.dense`)
  (dr) dense reverse   S2/S3 record -> top KR S1s (address-less / heavily noised records)
  (h)  exact core name + address number key       (as in v4 blocking; keys shared by <= 50 records)
  (j)  exact full name + address number key       (as in v4 blocking)
  (n)  address-null record whose exact normalized name is shared by <= FAN world S1s -> all of them
Final set: union, ordered by the best channel rank (then cosine), capped at CAP per S1. This union is
exactly `candidate_pairs.tsv` = the input to the first scoring model. Every pair gets its bi-encoder
cosine `d_cos` (computed directly for key-channel pairs that dense retrieval did not return).
"""
import time

import numpy as np
import polars as pl

from . import config as C
from .blocking import CORE_STOP_SHARE, KH, KH_KEYCAP, KJ, KJ_KEYCAP, load_norm
from .dense import DATA, EMB, retrieve
from .progress import step

KF, KR, CAP, FAN = 20, 5, 30, 10
BIG = 10_000


def _keys(s1w: pl.DataFrame, q: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """(h), (j) and (n) channels for one country. s1w: world S1s; q: query S1s; pool: S2/S3 records."""
    stop = (s1w.select("n_tok").explode("n_tok").group_by("n_tok").len()
            .filter(pl.col("len") > CORE_STOP_SHARE * s1w.height)["n_tok"].to_list())
    core = pl.col("n_tok").list.set_difference(pl.lit(stop, dtype=pl.List(pl.Utf8))).list.sort().list.join(" ")
    full = pl.col("n_tok").list.unique().list.sort().list.join(" ")

    def num_keys(d, name_expr, sep):
        return (d.select("entity_id", name_expr.alias("nk"), pl.col("a_tok").list.unique().alias("t"))
                .filter(pl.col("nk") != "").explode("t")
                .filter(pl.col("t").is_not_null() & pl.col("t").str.contains(r"^[0-9]+$"))
                .select("entity_id", (pl.col("nk") + sep + pl.col("t")).alias("k")))

    out = []
    for name_expr, sep, cap, kmax, col in ((core, "#", KH_KEYCAP, KH, "r_h"), (full, "@", KJ_KEYCAP, KJ, "r_j")):
        pk = num_keys(pool, name_expr, sep)
        pk = pk.join(pk.group_by("k").len().filter(pl.col("len") <= cap).select("k"), on="k")
        pairs = (num_keys(q, name_expr, sep).rename({"entity_id": "s1"}).join(pk.rename({"entity_id": "m"}), on="k")
                 .group_by("s1", "m").agg(pl.len().alias("n"))
                 .sort(["s1", "n"], descending=[False, True])
                 .with_columns(pl.int_range(1, pl.len() + 1).over("s1").cast(pl.Int16).alias(col))
                 .filter(pl.col(col) <= kmax).select("s1", "m", col))
        out.append(pairs)
    # (n) exact-name fan-out for address-null records
    nk = pl.col("n_tok").list.join(" ")
    fan = s1w.select("entity_id", nk.alias("k")).filter(pl.col("k") != "")
    small = fan.group_by("k").len().filter(pl.col("len") <= FAN).select("k")
    nullp = pool.filter(pl.col("a_tok").list.len() == 0).select(pl.col("entity_id").alias("m"), nk.alias("k")).join(small, on="k")
    nch = (fan.join(q.select("entity_id"), on="entity_id").join(small, on="k").rename({"entity_id": "s1"})
           .join(nullp, on="k").select("s1", "m", pl.lit(True).alias("in_fan")))
    res = out[0].join(out[1], on=["s1", "m"], how="full", coalesce=True).join(nch, on=["s1", "m"], how="full", coalesce=True)
    return res.with_columns(pl.col("in_fan").fill_null(False))


def _cosine(pairs: pl.DataFrame, split: str) -> np.ndarray:
    s1ids = pl.read_parquet(DATA / f"ids_{split}_s1.parquet").select(pl.col("entity_id").alias("s1")).with_row_index("i")
    pids = pl.read_parquet(DATA / f"ids_{split}_pool.parquet").select(pl.col("entity_id").alias("m")).with_row_index("j")
    d = pairs.select("s1", "m").with_row_index("_o").join(s1ids, on="s1").join(pids, on="m").sort("_o")
    assert d.height == pairs.height
    E1 = np.load(EMB / f"{split}_s1.npy", mmap_mode="r")
    EP = np.load(EMB / f"{split}_pool.npy", mmap_mode="r")
    i, j = d["i"].to_numpy(), d["j"].to_numpy()
    out = np.empty(len(i), np.float32)
    for b in range(0, len(i), 2_000_000):
        out[b:b + 2_000_000] = (E1[i[b:b + 2_000_000]].astype(np.float32) * EP[j[b:b + 2_000_000]].astype(np.float32)).sum(1)
    return out


def block(split: str, query_ids: pl.DataFrame, world_ids: pl.DataFrame, kf=KF, kr=KR, cap=CAP) -> pl.DataFrame:
    """query_ids / world_ids: one column `entity_id` (S1). Returns s1, m, d_cos, r_df, r_dr, r_h, r_j, in_fan, pri."""
    t0 = time.time()
    dc = retrieve(split, query_ids, world_ids, k_fwd=kf, k_rev=kr)
    s1 = load_norm(split, 1)
    pool = pl.concat([load_norm(split, 2), load_norm(split, 3)])
    keys = []
    for cty in sorted(s1["country"].unique().to_list()):
        w = s1.filter(pl.col("country") == cty).join(world_ids, on="entity_id")
        q = w.join(query_ids, on="entity_id")
        if q.height:
            keys.append(_keys(w, q, pool.filter(pl.col("country") == cty)))
    c = dc.join(pl.concat(keys), on=["s1", "m"], how="full", coalesce=True).with_columns(pl.col("in_fan").fill_null(False))
    c = c.with_columns(pl.Series("d_cos", _cosine(c, split)))
    c = c.with_columns(pl.min_horizontal(
        [pl.col(k).cast(pl.Int32).fill_null(BIG) for k in ("r_df", "r_dr", "r_h", "r_j")]
        + [pl.when(pl.col("in_fan")).then(pl.lit(1)).otherwise(BIG)]).alias("pri"))
    c = (c.sort(["s1", "pri", "d_cos"], descending=[False, False, True])
         .with_columns(pl.int_range(1, pl.len() + 1).over("s1").alias("_pos")).filter(pl.col("_pos") <= cap).drop("_pos"))
    step("block_v6", f"{split}: {c.height:,} pairs ({c.height / max(1, query_ids.height):.1f}/S1) in {time.time() - t0:.0f}s")
    return c.select("s1", "m", "d_cos", "r_df", "r_dr", "r_h", "r_j", "in_fan", "pri")
