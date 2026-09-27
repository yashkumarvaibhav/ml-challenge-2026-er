"""Blocking v2 = v1 features + (z) single core-name word x street word keys, with a memory-safe two-pass df.

Why z: US misses are dominated by perturbed house numbers (1115 -> 1117, 718 -> 18) plus a typo in one name word;
every number-based key (b/x/q/y) breaks, but "one intact name word + the street name" survives.
"""
import math

import polars as pl

from er import block as B

import os
B.KIND_W["z"] = 1.0
B.KIND_W["w"] = 0.9
B.CAP_S1 = int(os.environ.get("ER_CAP_S1", 40))
B.K_TOP = int(os.environ.get("ER_K_TOP", 10))
B.KIND_CODE = {k: i for i, k in enumerate(B.KIND_W)}
B.KIND_WV = {i: B.KIND_W[k] for k, i in B.KIND_CODE.items()}


def features2(df: pl.DataFrame) -> pl.DataFrame:
    base = B.features(df)
    core = df.select("i", pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(B.LEGAL) & (pl.element().str.len_chars() >= 3)))
                     .list.unique().list.head(5).alias("c"),
                     pl.col("a_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{4,}$"))).list.unique().list.head(6).alias("w"))
    z = B._h(core.explode("c").explode("w").drop_nulls().select("i", (pl.col("c") + "~" + pl.col("w")).alias("t")), "z")
    # pairs of address words (street + locality): survive perturbed house numbers and made-up names
    w = B._h(core.select("i", pl.col("w").list.sort().alias("w1"), pl.col("w").list.sort().alias("w2")).explode("w1").explode("w2")
             .filter(pl.col("w1") < pl.col("w2")).select("i", (pl.col("w1") + "^" + pl.col("w2")).alias("t")), "w")
    return pl.concat([base, z, w]).unique(["i", "f"])


def block_country(s1: pl.DataFrame, rec: pl.DataFrame, chunk: int = 250_000, s1_chunk: int = 300_000, log=print) -> pl.DataFrame:
    s1 = s1.with_row_index("i")
    rec = rec.with_row_index("i")
    n_s1 = s1.height
    # pass 1: document frequencies over S1 (chunked)
    dfc = None
    for lo in range(0, n_s1, s1_chunk):
        c = features2(s1.slice(lo, s1_chunk)).group_by("f").agg(pl.len().alias("df"), pl.col("kind").first())
        dfc = c if dfc is None else pl.concat([dfc, c]).group_by("f").agg(pl.col("df").sum(), pl.col("kind").first())
    dfc = dfc.filter(pl.col("df") <= B.CAP_S1).with_columns(
        ((pl.col("df").cast(pl.Float64).pow(-1) * n_s1).log() * pl.col("kind").replace_strict(B.KIND_WV, return_dtype=pl.Float64))
        .cast(pl.Float32).alias("w")).select("f", "w")
    # pass 2: keep only rare S1 features
    fs = pl.concat([features2(s1.slice(lo, s1_chunk)).join(dfc, on="f").select(pl.col("i").alias("si"), "f", "w")
                    for lo in range(0, n_s1, s1_chunk)])
    del dfc
    log(f"  block2 S1 features: {fs.height:,} rare (i, f) rows")
    parts = []
    for lo in range(0, rec.height, chunk):
        fr = features2(rec.slice(lo, chunk)).select(pl.col("i").alias("ri"), "f")
        pr = fr.join(fs, on="f").group_by("ri", "si").agg(pl.col("w").sum().cast(pl.Float32).alias("bscore"))
        pr = pr.with_columns(pl.col("bscore").rank("ordinal", descending=True).over("ri").cast(pl.Int32).alias("r_rank"))
        parts.append(pr.filter(pl.col("r_rank") <= 2 * B.R_TOP + 2))
        log(f"  block2 chunk {lo // chunk + 1}/{math.ceil(rec.height / chunk)}: {pr.height:,} scored pairs")
    p = pl.concat(parts)
    p = p.with_columns(pl.col("bscore").rank("ordinal", descending=True).over("si").cast(pl.Int32).alias("s_rank"))
    p = p.filter((pl.col("r_rank") <= B.R_TOP) | (pl.col("s_rank") <= B.K_TOP))
    return p.join(s1.select(pl.col("i").alias("si"), pl.col("entity_id").alias("s1")), on="si") \
        .join(rec.select(pl.col("i").alias("ri"), pl.col("entity_id").alias("m")), on="ri") \
        .select("s1", "m", "bscore", "r_rank", "s_rank")
