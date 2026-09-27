"""Candidate generation (PLAN.md B2).

Per country, over normalized records (cache/norm_final):

  (a) name     S1 -> pool : IDF-weighted overlap of name words, top KA
  (b) address  S1 -> pool : IDF-weighted overlap of address words + adjacent-word
                            bigrams (house number + street, street + city ...), top KB
  (c) combined S1 -> pool : (a) + (b) scores, top KC
  (d) core name S1 = pool : identical "core" name key (frequent words removed), only
                            for keys shared by <= KD pool records (address-null rescue)
  (e) reverse  pool -> S1 : each S2/S3 record's top KE S1s by combined score
  (f) name+locality       : core name + one address word as an exact key (keys shared by
                            <= KF_KEYCAP pool records), top KF by shared-key count
  (g) null-addr reverse   : address-null pool records -> top KG S1s by name-only overlap
                            with a higher df cap (DF_CAP_NREV)
  (j) full name+number    : full sorted name (no frequent-word removal) + one address number as
                            an exact key (keys shared by <= KJ_KEYCAP pool records), top KJ
  (h) name+house number   : core name + one address number as an exact key (keys shared by
                            <= KH_KEYCAP pool records); ties ranked by IDF of shared address words

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
KF, KF_KEYCAP = 20, 50
KG, DF_CAP_NREV = 10, 30_000
KH, KH_KEYCAP = 10, 50
KJ, KJ_KEYCAP = 10, 50
CAP = 70
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
    return hi - lo, out


def _rev_chunk(bounds):
    lo, hi = bounds
    S = (_G["RQ"][lo:hi] @ _G["S1T"]).tocsr()
    r, c, v, rk = _topk_rows(S, _G.get("K_REV", KE))
    return hi - lo, (r + lo, c, v, rk)


def _run_pool(fn, n, chunk, name):
    bounds = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    prog = Progress(name, n, unit="rows")
    res = []
    with mp.get_context("fork").Pool(C.N_PROCS) as pool:
        for n_rows, out in pool.imap_unordered(fn, bounds, chunksize=1):
            res.append(out)  # order does not matter: rows carry absolute indices
            prog.update(n_rows)
    prog.done()
    return res


def block_country(s1q: pl.DataFrame, s1_world: pl.DataFrame, pool: pl.DataFrame, cty: str) -> pl.DataFrame:
    """Candidates for queries s1q (subset of s1_world) against pool, one country."""
    t0 = time.time()
    I64 = pl.Int64
    pool = pool.with_row_index("row").with_columns(pl.col("row").cast(I64))
    world = s1_world.with_row_index("row").with_columns(pl.col("row").cast(I64))
    q = (s1q.join(world.select("entity_id", pl.col("row").alias("wrow")), on="entity_id")
         .with_row_index("row").with_columns(pl.col("row").cast(I64)))

    pt = record_tokens(pool)
    df_all = pt.group_by("tok").agg(pl.len().alias("df"), pl.col("kind").first())
    df = df_all.filter(pl.col("df") <= DF_CAP).with_row_index("tid").with_columns(
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
            fwd.append(pl.DataFrame({"qrow": r.astype(np.int64), "prow": c.astype(np.int64), f"s_{ch}": v, f"r_{ch}": rk}))
    by_ch = {}
    for f in fwd:
        ch = [c for c in f.columns if c.startswith("r_")][0][2:]
        by_ch.setdefault(ch, []).append(f)
    cand = None
    for ch in ("a", "b", "c"):
        f = pl.concat(by_ch[ch]) if ch in by_ch else pl.DataFrame(schema={"qrow": pl.Int64, "prow": pl.Int64})
        cand = f if cand is None else cand.join(f, on=["qrow", "prow"], how="full", coalesce=True)
    for k in ("PNT", "PAT", "QN", "QA"):
        _G.pop(k, None)

    # (e) reverse: every pool record -> top KE S1s of the world (combined tokens)
    _G["RQ"] = _csr(pt, pool.height, V, idf[pt["tid"].to_numpy()])
    _G["S1T"] = _csr(wt, world.height, V).T.tocsr()
    rparts = _run_pool(_rev_chunk, pool.height, 5000, f"block {cty} reverse")
    rev = pl.concat([pl.DataFrame({"prow": r.astype(np.int64), "wrow": c.astype(np.int64), "r_e": rk}) for r, c, v, rk in rparts])
    rev = rev.join(q.select(pl.col("row").alias("qrow"), "wrow"), on="wrow").drop("wrow")
    for k in ("RQ", "S1T"):
        _G.pop(k, None)

    # (g) address-null pool records -> top KG S1s by name-only IDF overlap (higher df cap)
    dfn = df_all.filter((pl.col("kind") == 0) & (pl.col("df") <= DF_CAP_NREV)).with_row_index("nid").with_columns(
        (np.log(pool.height) - pl.col("df").log()).cast(pl.Float32).alias("idf"))
    idfn = dfn["idf"].to_numpy()
    nullp = pool.filter(pl.col("a_tok").list.len() == 0).select("row", "n_tok").with_row_index("nrow") \
        .with_columns(pl.col("nrow").cast(I64))
    gch = pl.DataFrame(schema={"prow": I64, "qrow": I64, "r_g": pl.Int16})
    if nullp.height:
        pnt = (nullp.select(pl.col("nrow").alias("row"), pl.col("n_tok").list.unique().alias("tok")).explode("tok")
               .with_columns(("n:" + pl.col("tok")).alias("tok")).join(dfn.select("tok", "nid"), on="tok")
               .rename({"nid": "tid"}))
        wnt = (world.select("row", pl.col("n_tok").list.unique().alias("tok")).explode("tok")
               .with_columns(("n:" + pl.col("tok")).alias("tok")).join(dfn.select("tok", "nid"), on="tok")
               .rename({"nid": "tid"}))
        _G["RQ"] = _csr(pnt, nullp.height, dfn.height, idfn[pnt["tid"].to_numpy()])
        _G["S1T"] = _csr(wnt, world.height, dfn.height).T.tocsr()
        _G["K_REV"] = KG
        gparts = _run_pool(_rev_chunk, nullp.height, 2000, f"block {cty} null-address reverse")
        for k in ("RQ", "S1T", "K_REV"):
            _G.pop(k, None)
        g = pl.concat([pl.DataFrame({"nrow": r.astype(np.int64), "wrow": c.astype(np.int64), "r_g": rk}) for r, c, v, rk in gparts])
        gch = (g.join(nullp.select("nrow", pl.col("row").alias("prow")), on="nrow")
               .join(q.select(pl.col("row").alias("qrow"), "wrow"), on="wrow").select("prow", "qrow", "r_g"))

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

    # address-word IDF (uncapped, from the pool) for ranking the key channels
    aidf = (df_all.filter(pl.col("tok").str.starts_with("a:"))
            .select(pl.col("tok").str.slice(2).alias("t"),
                    (np.log(pool.height) - pl.col("df").log()).cast(pl.Float32).alias("w")))

    # (f) name + locality keys, ranked by the IDF of the shared locality words
    def nl_keys(d):
        return (core(d.select("row", "n_tok", "a_tok")).filter(pl.col("core") != "")
                .select("row", "core", pl.col("a_tok").list.unique().alias("t")).explode("t")
                .filter(pl.col("t").is_not_null() & (pl.col("t").str.len_chars() >= 3) & ~pl.col("t").str.contains(r"^[0-9]+$"))
                .select("row", "t", (pl.col("core") + "|" + pl.col("t")).alias("k")))

    pk = nl_keys(pool)
    pk = pk.join(pk.group_by("k").len().filter(pl.col("len") <= KF_KEYCAP).select("k"), on="k")
    fch = (nl_keys(q).rename({"row": "qrow"}).join(pk.rename({"row": "prow"}).drop("t"), on="k")
           .join(aidf, on="t", how="left")
           .group_by("qrow", "prow").agg(pl.col("w").fill_null(0).sum().alias("w_f"))
           .with_columns(pl.col("w_f").rank("ordinal", descending=True).over("qrow").cast(pl.Int16).alias("r_f"))
           .filter(pl.col("r_f") <= KF).drop("w_f"))

    # (h) exact core name + house number keys; ties ranked by IDF of shared address words
    def nn_keys(d):
        return (core(d.select("row", "n_tok", "a_tok")).filter(pl.col("core") != "")
                .select("row", "core", pl.col("a_tok").list.unique().alias("t")).explode("t")
                .filter(pl.col("t").is_not_null() & pl.col("t").str.contains(r"^[0-9]+$"))
                .select("row", (pl.col("core") + "#" + pl.col("t")).alias("k")))

    hk = nn_keys(pool)
    hk = hk.join(hk.group_by("k").len().filter(pl.col("len") <= KH_KEYCAP).select("k"), on="k")
    hpairs = (nn_keys(q).rename({"row": "qrow"}).join(hk.rename({"row": "prow"}), on="k")
              .group_by("qrow", "prow").agg(pl.len().alias("n_h")))
    if hpairs.height:
        qa = q.select(pl.col("row").alias("qrow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        pa = pool.select(pl.col("row").alias("prow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        shared = (hpairs.select("qrow", "prow").join(qa, on="qrow").join(pa, on=["prow", "t"])
                  .filter(~pl.col("t").str.contains(r"^[0-9]+$")).join(aidf, on="t", how="left")
                  .group_by("qrow", "prow").agg(pl.col("w").fill_null(0).sum().alias("w_h")))
        hch = (hpairs.join(shared, on=["qrow", "prow"], how="left").with_columns(pl.col("w_h").fill_null(0))
               .sort(["qrow", "n_h", "w_h"], descending=[False, True, True])
               .with_columns(pl.int_range(1, pl.len() + 1).over("qrow").cast(pl.Int16).alias("r_h"))
               .filter(pl.col("r_h") <= KH).select("qrow", "prow", "r_h"))
    else:
        hch = pl.DataFrame(schema={"qrow": pl.Int64, "prow": pl.Int64, "r_h": pl.Int16})

    # (j) exact full name (sorted words, nothing removed) + address number
    def fn_keys(d):
        return (d.select("row", pl.col("n_tok").list.unique().list.sort().list.join(" ").alias("fn"),
                         pl.col("a_tok").list.unique().alias("t"))
                .filter(pl.col("fn") != "").explode("t")
                .filter(pl.col("t").is_not_null() & pl.col("t").str.contains(r"^[0-9]+$"))
                .select("row", (pl.col("fn") + "#" + pl.col("t")).alias("k")))

    jk = fn_keys(pool)
    jk = jk.join(jk.group_by("k").len().filter(pl.col("len") <= KJ_KEYCAP).select("k"), on="k")
    jpairs = (fn_keys(q).rename({"row": "qrow"}).join(jk.rename({"row": "prow"}), on="k")
              .group_by("qrow", "prow").agg(pl.len().alias("n_j")))
    if jpairs.height:
        qa = q.select(pl.col("row").alias("qrow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        pa = pool.select(pl.col("row").alias("prow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        jshared = (jpairs.select("qrow", "prow").join(qa, on="qrow").join(pa, on=["prow", "t"])
                   .filter(~pl.col("t").str.contains(r"^[0-9]+$")).join(aidf, on="t", how="left")
                   .group_by("qrow", "prow").agg(pl.col("w").fill_null(0).sum().alias("w_j")))
        jch = (jpairs.join(jshared, on=["qrow", "prow"], how="left").with_columns(pl.col("w_j").fill_null(0))
               .sort(["qrow", "n_j", "w_j"], descending=[False, True, True])
               .with_columns(pl.int_range(1, pl.len() + 1).over("qrow").cast(pl.Int16).alias("r_j"))
               .filter(pl.col("r_j") <= KJ).select("qrow", "prow", "r_j"))
    else:
        jch = pl.DataFrame(schema={"qrow": pl.Int64, "prow": pl.Int64, "r_j": pl.Int16})

    cand = (cand.join(rev, on=["qrow", "prow"], how="full", coalesce=True)
            .join(dch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(fch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(gch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(hch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(jch, on=["qrow", "prow"], how="full", coalesce=True))
    big = 10_000
    cand = cand.with_columns(
        pl.min_horizontal(
            pl.col("r_a").cast(pl.Int32).fill_null(big),
            pl.col("r_b").cast(pl.Int32).fill_null(big),
            pl.col("r_c").cast(pl.Int32).fill_null(big),
            pl.when(pl.col("r_e").is_not_null()).then(pl.col("r_e").cast(pl.Int32)).otherwise(big),
            pl.when(pl.col("in_d")).then(pl.lit(KD // 2)).otherwise(big),
            pl.col("r_f").cast(pl.Int32).fill_null(big),
            pl.col("r_g").cast(pl.Int32).fill_null(big),
            pl.col("r_h").cast(pl.Int32).fill_null(big),
            pl.col("r_j").cast(pl.Int32).fill_null(big),
        ).alias("prio"),
        pl.col("in_d").fill_null(False),
    ).sort(["qrow", "prio", "s_c"], descending=[False, False, True], nulls_last=True)
    cand = cand.with_columns(pl.col("prio").rank("ordinal").over("qrow").alias("pos")).filter(pl.col("pos") <= CAP)
    out = (cand.join(q.select(pl.col("row").alias("qrow"), pl.col("entity_id").alias("s1")), on="qrow")
           .join(pool.select(pl.col("row").alias("prow"), pl.col("entity_id").alias("m")), on="prow")
           .select("s1", "m", "s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "r_h", "r_j", "prio"))
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
