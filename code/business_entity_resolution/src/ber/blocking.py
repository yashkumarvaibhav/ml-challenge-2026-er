"""Candidate generation (PLAN.md B2).

Per country, over normalized records (cache/norm_final):

  (a) name     S1 -> pool : IDF-weighted overlap of name words, top KA
  (b) address  S1 -> pool : IDF-weighted overlap of address words + adjacent-word
                            bigrams (house number + street, street + city ...), top KB
  (c) combined S1 -> pool : (a) + (b) scores, top KC
  (d) core name S1 = pool : identical "core" name key (frequent words removed), only
                            for keys shared by <= KD pool records (address-null rescue)
  (e) reverse  pool -> S1 : each S2/S3 record's top KE S1s by combined score

IDF / document frequency come from the split's own S2+S3 pool (label-free). Words
with df above DF_CAP are ignored for retrieval (they still count in the features).
Channel quotas are applied first, then a global cap per S1 ordered by best channel
rank. The union written here is exactly the set the matcher scores.
"""
import multiprocessing as mp
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import config as C
from .progress import Progress, step

KA, KB, KC, KD, KE = 15, 15, 30, 30, 3
CAP = 60
DF_CAP = 5000
CORE_STOP_SHARE = 0.005  # name words used by > 0.5% of the country's S1 names are not "core"

_G = {}


def record_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Long table (row, tok, kind) with kind 0 = name, 1 = address."""
    n = df.select(pl.col("row"), pl.col("n_tok").list.unique().alias("tok")).explode("tok")
    a = df.select(pl.col("row"), pl.col("a_tok").list.unique().alias("tok")).explode("tok")
    b = df.select(
        pl.col("row"),
        pl.concat_list(
            pl.col("a_tok").list.eval(pl.element() + "_" + pl.element().shift(-1)).list.slice(0, pl.col("a_tok").list.len() - 1)
        ).list.unique().alias("tok"),
    ).explode("tok")
    return pl.concat([
        n.with_columns(("n:" + pl.col("tok")).alias("tok"), pl.lit(0, pl.Int8).alias("kind")),
        a.with_columns(("a:" + pl.col("tok")).alias("tok"), pl.lit(1, pl.Int8).alias("kind")),
        b.with_columns(("b:" + pl.col("tok")).alias("tok"), pl.lit(1, pl.Int8).alias("kind")),
    ]).drop_nulls()


def _csr(long: pl.DataFrame, nrows: int, ncols: int, data=None) -> sp.csr_matrix:
    r = long["row"].to_numpy()
    c = long["tid"].to_numpy()
    d = np.ones(len(r), np.float32) if data is None else data
    return sp.csr_matrix((d, (r, c)), shape=(nrows, ncols), dtype=np.float32)


def _topk_rows(S: sp.csr_matrix, k: int):
    """Top-k (col, score) per row of a CSR matrix -> (row, col, score, rank) arrays."""
    rows, cols, vals, ranks = [], [], [], []
    indptr, indices, data = S.indptr, S.indices, S.data
    for r in range(S.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        if a == b:
            continue
        v = data[a:b]
        idx = indices[a:b]
        if b - a > k:
            top = np.argpartition(-v, k)[:k]
            v, idx = v[top], idx[top]
        o = np.argsort(-v, kind="stable")
        n = len(o)
        rows.append(np.full(n, r, np.int32))
        cols.append(idx[o].astype(np.int32))
        vals.append(v[o])
        ranks.append(np.arange(1, n + 1, dtype=np.int16))
    if not rows:
        e = np.array([], np.int32)
        return e, e, np.array([], np.float32), np.array([], np.int16)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(ranks)


def _fwd_chunk(bounds):
    lo, hi = bounds
    qn, qa = _G["QN"][lo:hi], _G["QA"][lo:hi]
    sn = (qn @ _G["PNT"]).tocsr()
    sa = (qa @ _G["PAT"]).tocsr()
    sc = (sn + sa).tocsr()
    out = []
    for ch, S, k in (("a", sn, KA), ("b", sa, KB), ("c", sc, KC)):
        r, c, v, rk = _topk_rows(S, k)
        out.append((ch, r + lo, c, v, rk))
    return out


def _rev_chunk(bounds):
    lo, hi = bounds
    S = (_G["RQ"][lo:hi] @ _G["S1T"]).tocsr()
    r, c, v, rk = _topk_rows(S, KE)
    return r + lo, c, v, rk


def _run_pool(fn, n, chunk, name):
    bounds = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    prog = Progress(name, n, unit="rows")
    res = []
    with mp.get_context("fork").Pool(C.N_PROCS) as pool:
        for out, (lo, hi) in zip(pool.imap(fn, bounds, chunksize=1), bounds):
            res.append(out)
            prog.update(hi - lo)
    prog.done()
    return res


def block_country(s1q: pl.DataFrame, s1_world: pl.DataFrame, pool: pl.DataFrame, cty: str) -> pl.DataFrame:
    """Candidates for queries s1q (subset of s1_world) against pool, one country."""
    t0 = time.time()
    pool = pool.with_row_index("row")
    world = s1_world.with_row_index("row")
    q = s1q.join(world.select("entity_id", pl.col("row").alias("wrow")), on="entity_id").with_row_index("row")

    pt = record_tokens(pool)
    df = pt.group_by("tok").agg(pl.len().alias("df"), pl.col("kind").first())
    df = df.filter(pl.col("df") <= DF_CAP).with_row_index("tid").with_columns(
        (np.log(pool.height) - pl.col("df").log()).cast(pl.Float32).alias("idf"))
    V = df.height
    idf = df["idf"].to_numpy()
    pt = pt.join(df.select("tok", "tid"), on="tok")
    wt = record_tokens(world).join(df.select("tok", "tid"), on="tok")
    qt = wt.join(q.select(pl.col("wrow").alias("row"), pl.col("row").alias("qrow")), on="row").drop("row").rename({"qrow": "row"})

    def split_kind(long):
        return long.filter(pl.col("kind") == 0), long.filter(pl.col("kind") == 1)

    ptn, pta = split_kind(pt)
    qtn, qta = split_kind(qt)
    _G["PNT"] = _csr(ptn, pool.height, V).T.tocsr()
    _G["PAT"] = _csr(pta, pool.height, V).T.tocsr()
    _G["QN"] = _csr(qtn, q.height, V, idf[qtn["tid"].to_numpy()])
    _G["QA"] = _csr(qta, q.height, V, idf[qta["tid"].to_numpy()])
    step(f"block {cty}", f"matrices ready: pool={pool.height:,} queries={q.height:,} vocab={V:,} ({time.time() - t0:.0f}s)")

    parts = _run_pool(_fwd_chunk, q.height, 500, f"block {cty} forward")
    fwd = []
    for chunk in parts:
        for ch, r, c, v, rk in chunk:
            fwd.append(pl.DataFrame({"qrow": r, "prow": c, f"s_{ch}": v, f"r_{ch}": rk}))
    by_ch = {}
    for f in fwd:
        ch = [c for c in f.columns if c.startswith("r_")][0][2:]
        by_ch.setdefault(ch, []).append(f)
    cand = None
    for ch in ("a", "b", "c"):
        f = pl.concat(by_ch[ch]) if ch in by_ch else pl.DataFrame(schema={"qrow": pl.Int32, "prow": pl.Int32})
        cand = f if cand is None else cand.join(f, on=["qrow", "prow"], how="full", coalesce=True)
    for k in ("PNT", "PAT", "QN", "QA"):
        _G.pop(k, None)

    # (e) reverse: every pool record -> top KE S1s of the world (combined tokens)
    _G["RQ"] = _csr(pt, pool.height, V, idf[pt["tid"].to_numpy()])
    _G["S1T"] = _csr(wt, world.height, V).T.tocsr()
    rparts = _run_pool(_rev_chunk, pool.height, 5000, f"block {cty} reverse")
    rev = pl.concat([pl.DataFrame({"prow": r, "wrow": c, "r_e": rk}) for r, c, v, rk in rparts])
    rev = rev.join(q.select(pl.col("row").alias("qrow"), "wrow"), on="wrow").drop("wrow")
    for k in ("RQ", "S1T"):
        _G.pop(k, None)

    # (d) core-name key equality, for keys shared by <= KD pool records
    s1w = world.select("n_tok")
    stop = (s1w.explode("n_tok").group_by("n_tok").len()
            .filter(pl.col("len") > CORE_STOP_SHARE * world.height)["n_tok"].to_list())

    def core(d):
        return d.with_columns(
            pl.col("n_tok").list.set_difference(pl.lit(stop, dtype=pl.List(pl.Utf8))).list.sort().list.join(" ").alias("core"))

    pc = core(pool.select("row", "n_tok")).filter(pl.col("core") != "")
    sizes = pc.group_by("core").len().filter(pl.col("len") <= KD)
    pc = pc.join(sizes.select("core"), on="core")
    qc = core(q.select("row", "n_tok")).filter(pl.col("core") != "")
    dch = qc.select(pl.col("row").alias("qrow"), "core").join(
        pc.select(pl.col("row").alias("prow"), "core"), on="core").drop("core").with_columns(pl.lit(True).alias("in_d"))

    cand = (cand.join(rev, on=["qrow", "prow"], how="full", coalesce=True)
            .join(dch, on=["qrow", "prow"], how="full", coalesce=True))
    big = 10_000
    cand = cand.with_columns(
        pl.min_horizontal(
            pl.col("r_a").cast(pl.Int32).fill_null(big),
            pl.col("r_b").cast(pl.Int32).fill_null(big),
            pl.col("r_c").cast(pl.Int32).fill_null(big),
            pl.when(pl.col("r_e").is_not_null()).then(pl.col("r_e").cast(pl.Int32)).otherwise(big),
            pl.when(pl.col("in_d")).then(pl.lit(KD // 2)).otherwise(big),
        ).alias("prio"),
        pl.col("in_d").fill_null(False),
    ).sort(["qrow", "prio", "s_c"], descending=[False, False, True], nulls_last=True)
    cand = cand.with_columns(pl.col("prio").rank("ordinal").over("qrow").alias("pos")).filter(pl.col("pos") <= CAP)
    out = (cand.join(q.select(pl.col("row").alias("qrow"), pl.col("entity_id").alias("s1")), on="qrow")
           .join(pool.select(pl.col("row").alias("prow"), pl.col("entity_id").alias("m")), on="prow")
           .select("s1", "m", "s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "prio"))
    step(f"block {cty}", f"DONE {out.height:,} pairs ({out.height / max(1, q.height):.1f}/S1) in {time.time() - t0:.0f}s")
    return out


def load_norm(split: str, s: int) -> pl.DataFrame:
    return pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet",
                           columns=["entity_id", "country", "n_tok", "a_tok"])


def block(split: str, query_ids: pl.DataFrame, world_ids: pl.DataFrame) -> pl.DataFrame:
    """query_ids / world_ids: one column `entity_id` (S1). Countries come from the files."""
    s1 = load_norm(split, 1)
    pool = pl.concat([load_norm(split, 2), load_norm(split, 3)])
    outs = []
    for cty in sorted(s1["country"].unique().to_list()):
        w = s1.filter(pl.col("country") == cty).join(world_ids, on="entity_id")
        qq = w.join(query_ids, on="entity_id")
        if qq.height == 0:
            continue
        outs.append(block_country(qq, w, pool.filter(pl.col("country") == cty), cty))
    return pl.concat(outs)
