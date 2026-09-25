"""Pairwise features for candidate pairs (PLAN.md B3). Country is NOT a feature.

Inputs: candidate pairs (s1, m, blocking scores/ranks) + normalized records.
IDF tables come from the split's own S2+S3 pool (label-free). Name-ambiguity counts
come from the world's S1 population (recomputed per ghost world).
"""
import math
import multiprocessing as mp
import os
import shutil
import uuid

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from . import config as C
from .progress import Progress
from .text import is_abbrev as _abbr

NORM_COLS = ["entity_id", "country", "n_tok", "a_tok", "a_comp", "a_nums", "n_native", "n_native_mapped_frac",
             "n_allcaps", "n_accent", "n_domain", "n_dba", "a_null", "a_ncomma"]
SOFT_T = 0.88  # Jaro-Winkler threshold for a fuzzy token match
_G = {}

FEATURES = [
    # name
    "n_ratio", "n_partial", "n_tset", "n_tsort", "n_jw", "n_compact_ratio", "n_compact_partial", "n_compact_eq",
    "n_jacc", "n_overlap_min", "n_idf_cov1", "n_idf_cov2", "n_idf_jacc", "n_soft_cov1", "n_soft_cov2",
    "n_abbr_hits", "n_ntok1", "n_ntok2", "n_first_eq", "n_idf_max_shared", "n_idf_max_missing1",
    # address
    "a_null2", "a_ratio", "a_tset", "a_tsort", "a_partial", "a_jacc", "a_idf_cov1", "a_idf_cov2", "a_soft_cov1",
    "a_soft_cov2", "a_num_any", "a_num_jacc", "a_num_first_eq", "a_num_prefix", "a_num_conflict", "a_nnum1",
    "a_nnum2", "a_comp_cov1", "a_comp_cov2", "a_ntok1", "a_ntok2", "a_ncomma1", "a_ncomma2", "a_numfrac2",
    # record flags (S2/S3 side) + source
    "m_native", "m_native_mapped", "m_allcaps", "m_accent", "m_domain", "m_dba", "m_is_s3",
    # blocking
    "s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "r_h", "r_j", "prio",
    # context (label-free)
    "amb_s1_core", "amb_pool_core", "c_n_cand", "c_sc_rel", "c_sc_rank",
]


def _idf_sum(toks, idf, default):
    return sum(idf.get(t, default) for t in toks)


def _soft_cov(a, b, idf, default):
    """IDF-weighted share of tokens in a that have a fuzzy match (JW >= SOFT_T) in b."""
    if not a:
        return 0.0
    tot = hit = 0.0
    for t in a:
        w = idf.get(t, default)
        tot += w
        if not b:
            continue
        best = max(JaroWinkler.normalized_similarity(t, u) for u in b)
        if best >= SOFT_T:
            hit += w * best
    return hit / tot if tot else 0.0


def _pair_feats(r1, r2, blk, ctx, nidf, aidf, ndef, adef):
    n1, n2 = r1["n_tok"], r2["n_tok"]
    s1n, s2n = " ".join(n1), " ".join(n2)
    c1, c2 = s1n.replace(" ", ""), s2n.replace(" ", "")
    set1, set2 = set(n1), set(n2)
    inter = set1 & set2
    i1, i2 = _idf_sum(set1, nidf, ndef), _idf_sum(set2, nidf, ndef)
    ii = _idf_sum(inter, nidf, ndef)
    iu = _idf_sum(set1 | set2, nidf, ndef)
    miss1 = set1 - set2
    abbr = sum(1 for t in set2 - set1 if any(_abbr(t, u) for u in miss1)) if miss1 else 0
    f = [
        fuzz.ratio(s1n, s2n), fuzz.partial_ratio(s1n, s2n), fuzz.token_set_ratio(s1n, s2n), fuzz.token_sort_ratio(s1n, s2n),
        JaroWinkler.normalized_similarity(s1n, s2n), fuzz.ratio(c1, c2), fuzz.partial_ratio(c1, c2), float(c1 == c2 and c1 != ""),
        len(inter) / max(1, len(set1 | set2)), len(inter) / max(1, min(len(set1), len(set2))),
        ii / i1 if i1 else 0.0, ii / i2 if i2 else 0.0, ii / iu if iu else 0.0,
        _soft_cov(set1, set2, nidf, ndef), _soft_cov(set2, set1, nidf, ndef),
        abbr, len(n1), len(n2), float(bool(n1) and bool(n2) and n1[0] == n2[0]),
        max((nidf.get(t, ndef) for t in inter), default=0.0), max((nidf.get(t, ndef) for t in miss1), default=0.0),
    ]
    a1, a2 = r1["a_tok"], r2["a_tok"]
    if r2["a_null"] or not a2:
        nan = np.nan
        f += [1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
              len(set(r1["a_nums"])), 0, nan, nan, len(a1), 0, r1["a_ncomma"], 0, nan]
    else:
        s1a, s2a = " ".join(a1), " ".join(a2)
        aw1 = {t for t in a1 if not t.isdigit()}
        aw2 = {t for t in a2 if not t.isdigit()}
        ai = aw1 & aw2
        w1, w2, wi = _idf_sum(aw1, aidf, adef), _idf_sum(aw2, aidf, adef), _idf_sum(ai, aidf, adef)
        nm1, nm2 = r1["a_nums"], r2["a_nums"]
        ns1, ns2 = set(nm1), set(nm2)
        prefix = float(any(x != y and (x.startswith(y) or y.startswith(x)) for x in ns1 for y in ns2))
        cs1, cs2 = set(r1["a_comp"]), set(r2["a_comp"])
        f += [
            0.0, fuzz.ratio(s1a, s2a), fuzz.token_set_ratio(s1a, s2a), fuzz.token_sort_ratio(s1a, s2a), fuzz.partial_ratio(s1a, s2a),
            len(ai) / max(1, len(aw1 | aw2)), wi / w1 if w1 else 0.0, wi / w2 if w2 else 0.0,
            _soft_cov(aw1, aw2, aidf, adef), _soft_cov(aw2, aw1, aidf, adef),
            float(bool(ns1 & ns2)), len(ns1 & ns2) / max(1, len(ns1 | ns2)),
            float(bool(nm1) and bool(nm2) and nm1[0] == nm2[0]), prefix,
            float(bool(ns1) and bool(ns2) and not (ns1 & ns2)), len(ns1), len(ns2),
            len(cs1 & cs2) / max(1, len(cs1)), len(cs1 & cs2) / max(1, len(cs2)),
            len(a1), len(a2), r1["a_ncomma"], r2["a_ncomma"], len(nm2) / max(1, len(a2)),
        ]
    f += [float(r2["n_native"]), r2["n_native_mapped_frac"] if r2["n_native_mapped_frac"] is not None else np.nan,
          float(r2["n_allcaps"]), float(r2["n_accent"]), float(r2["n_domain"]), float(r2["n_dba"]), float(r2["is_s3"])]
    f += blk
    f += ctx
    return f


REC_COLS = ["country", "n_tok", "a_tok", "a_comp", "a_nums", "n_native", "n_native_mapped_frac", "n_allcaps",
            "n_accent", "n_domain", "n_dba", "a_null", "a_ncomma"]
BLK_COLS = ["s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "r_h", "r_j", "prio"]
CTX_COLS = ["amb_s1_core", "amb_pool_core", "c_n_cand", "c_sc_rel", "c_sc_rank"]


def _work(job):
    """job = (offset, path to a chunk parquet in /dev/shm). Spawned workers read and
    convert their own chunk, so the parent does no per-value Python conversion and
    no large object is shared through fork (bounded memory, no polars-after-fork)."""
    off, path = job
    P = pl.read_parquet(path).to_dict(as_series=False)
    os.remove(path)
    n = len(P["s_a"])
    out = np.empty((n, len(FEATURES)), np.float32)
    for i in range(n):
        r1 = {k: P["1_" + k][i] for k in REC_COLS}
        r2 = {k: P["2_" + k][i] for k in REC_COLS}
        r2["is_s3"] = P["is_s3"][i]
        nidf = dict(zip(r1["n_tok"], P["1_n_idf"][i]))
        nidf.update(zip(r2["n_tok"], P["2_n_idf"][i]))
        aidf = dict(zip(r1["a_tok"], P["1_a_idf"][i]))
        aidf.update(zip(r2["a_tok"], P["2_a_idf"][i]))
        blk = [P[k][i] for k in BLK_COLS]
        ctx = [P[k][i] for k in CTX_COLS]
        out[i] = _pair_feats(r1, r2, blk, ctx, nidf, aidf, 0.0, 0.0)
    return off, out


def _idf_tables(pool: pl.DataFrame):
    """Per-country (tok, idf) tables from the pool (label-free); unseen token = log(N)."""
    N = pool.group_by("country").len().rename({"len": "N"})
    out = {}
    for col in ("n_tok", "a_tok"):
        t = (pool.select("country", pl.col(col).list.unique().alias("tok")).explode("tok").drop_nulls()
             .group_by("country", "tok").len().join(N, on="country")
             .with_columns((pl.col("N") / pl.col("len")).log().cast(pl.Float32).alias("idf")).select("country", "tok", "idf"))
        out[col] = t
    return out, N.with_columns(pl.col("N").log().cast(pl.Float32).alias("unseen")).select("country", "unseen")


def _with_idf(df: pl.DataFrame, col: str, table: pl.DataFrame, unseen: pl.DataFrame, out: str) -> pl.DataFrame:
    """Adds `out`: list of idf values aligned with list column `col` (zip-safe)."""
    e = (df.select("entity_id", "country", col).explode(col).with_row_index("_o")
         .join(table, left_on=["country", col], right_on=["country", "tok"], how="left")
         .join(unseen, on="country", how="left")
         .with_columns(pl.col("idf").fill_null(pl.col("unseen")))
         .sort("_o").group_by("entity_id", maintain_order=True).agg(pl.col("idf").alias(out)))
    return df.join(e, on="entity_id", how="left")


def build(split: str, cand: pl.DataFrame, world_ids: pl.DataFrame, name: str) -> pl.DataFrame:
    """Returns cand (s1, m, ...) with FEATURES columns appended."""
    s1 = pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s1.parquet", columns=NORM_COLS)
    pool = pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=NORM_COLS) for s in (2, 3)])
    idf_tabs, unseen = _idf_tables(pool)

    # name ambiguity: how many world S1s / pool records share the sorted-token name key
    key = pl.col("n_tok").list.sort().list.join(" ")
    world = s1.join(world_ids, on="entity_id")
    s1k = world.select("country", key.alias("k")).group_by("country", "k").len().rename({"len": "amb_s1_core"})
    pk = pool.select("country", key.alias("k")).group_by("country", "k").len().rename({"len": "amb_pool_core"})

    c = cand.with_columns(
        pl.len().over("s1").alias("c_n_cand"),
        (pl.col("s_c").fill_null(0) / pl.col("s_c").fill_null(0).max().over("s1").clip(lower_bound=1e-6)).alias("c_sc_rel"),
        pl.col("s_c").fill_null(0).rank("ordinal", descending=True).over("s1").alias("c_sc_rank"),
        pl.col("in_d").cast(pl.Float32),
    )
    q = s1.join(c.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id").with_columns(key.alias("k")) \
        .join(s1k, on=["country", "k"], how="left").drop("k")
    ms = pool.join(c.select(pl.col("m").alias("entity_id")).unique(), on="entity_id").with_columns(key.alias("k")) \
        .join(pk, on=["country", "k"], how="left").drop("k") \
        .with_columns(pl.col("entity_id").str.starts_with("S3-").alias("is_s3"))
    for col, out in (("n_tok", "n_idf"), ("a_tok", "a_idf")):
        q = _with_idf(q, col, idf_tabs[col], unseen, out)
        ms = _with_idf(ms, col, idf_tabs[col], unseen, out)
    cols = REC_COLS + ["n_idf", "a_idf"]
    rec1 = q.select(pl.col("entity_id").alias("s1"), "amb_s1_core", *[pl.col(k).alias("1_" + k) for k in cols])
    rec2 = ms.select(pl.col("entity_id").alias("m"), "amb_pool_core", "is_s3", *[pl.col(k).alias("2_" + k) for k in cols])
    num = BLK_COLS + CTX_COLS

    n = c.height
    X = np.empty((n, len(FEATURES)), np.float32)
    prog = Progress(name, n, unit="pairs")
    BATCH, CHUNK = 1_000_000, 10_000
    tmp = f"/dev/shm/ber_feat_{uuid.uuid4().hex[:8]}"
    os.makedirs(tmp)
    try:
        with mp.get_context("spawn").Pool(C.N_PROCS) as pool_:
            pending = []
            for b0 in range(0, n, BATCH):
                b = c.slice(b0, BATCH).join(rec1, on="s1", how="left", maintain_order="left") \
                    .join(rec2, on="m", how="left", maintain_order="left") \
                    .with_columns([pl.col(k).cast(pl.Float32).fill_null(np.nan) for k in num]).drop("s1", "m")
                for i in range(0, b.height, CHUNK):
                    path = f"{tmp}/{b0 + i}.parquet"
                    b.slice(i, CHUNK).write_parquet(path, compression="uncompressed")
                    pending.append(pool_.apply_async(_work, ((b0 + i, path),)))
                # collect finished work continuously (smooth progress/ETA); block only when the
                # backlog is large, or at the very end
                still = []
                for r in pending:
                    if r.ready():
                        off, out = r.get()
                        X[off:off + len(out)] = out
                        prog.update(len(out))
                    else:
                        still.append(r)
                pending = still
                while len(pending) > 3 * C.N_PROCS or (b0 + BATCH >= n and pending):
                    off, out = pending.pop(0).get()
                    X[off:off + len(out)] = out
                    prog.update(len(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    prog.done()
    return c.select("s1", "m").with_columns(pl.from_numpy(X, schema=FEATURES))
