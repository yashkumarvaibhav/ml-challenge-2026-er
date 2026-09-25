"""Reviewer 16:51 UTC, item 2: targeted character-trigram NAME retrieval baseline for address-null
S2/S3 records (reverse: record -> top-K world S1s by char-3-gram TF-IDF cosine of the normalized name),
to compare its marginal rescued true pairs with dense retrieval under the same final cap.

  python code/business_entity_resolution/src/ber/experiments/chargram_null.py
"""
import json
import multiprocessing as mp

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from ber import config as C
from ber.blocking import _topk_rows
from ber.io import load_train_pairs
from ber.progress import Progress, step

K = 10
_G = {}


def _chunk(b):
    lo, hi = b
    S = (_G["Q"][lo:hi] @ _G["WT"]).tocsr()
    r, c, v, rk = _topk_rows(S, K)
    return hi - lo, (r + lo, c, v, rk)


def main():
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
    world = sp.filter(~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
    r1 = pl.read_parquet(C.CACHE / "norm_final" / "train_s1.parquet", columns=["entity_id", "country", "n_str"]).rename({"entity_id": "s1"}).join(world, on="s1")
    pool = pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"train_s{s}.parquet", columns=["entity_id", "country", "n_str", "a_null"]) for s in (2, 3)])
    nullp = pool.filter(pl.col("a_null") & (pl.col("n_str") != "")).rename({"entity_id": "m"})
    outs = []
    for cty in ("US", "India"):
        w = r1.filter(pl.col("country") == cty)
        q = nullp.filter(pl.col("country") == cty)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=0.02, dtype=np.float32, sublinear_tf=True)
        W = vec.fit_transform(w["n_str"].to_list())
        _G["WT"] = W.T.tocsr()
        _G["Q"] = vec.transform(q["n_str"].to_list()).tocsr()
        bounds = [(i, min(q.height, i + 2000)) for i in range(0, q.height, 2000)]
        prog = Progress(f"chargram null-record reverse {cty}", q.height, unit="rows")
        res = []
        with mp.get_context("fork").Pool(C.N_PROCS) as p:
            for n, out in p.imap_unordered(_chunk, bounds, chunksize=1):
                res.append(out)
                prog.update(n)
        prog.done()
        g = pl.concat([pl.DataFrame({"qi": r.astype(np.int64), "wi": c.astype(np.int64), "cg": v, "r_cg": rk}) for r, c, v, rk in res])
        g = g.join(q.select(pl.col("m")).with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int64)), on="qi") \
             .join(w.select("s1").with_row_index("wi").with_columns(pl.col("wi").cast(pl.Int64)), on="wi").select("s1", "m", "cg", "r_cg")
        outs.append(g.join(hold, on="s1"))
    cg = pl.concat(outs)
    cg.write_parquet(C.CACHE / "dense" / "holdout_w20_chargram_null.parquet")
    truth = load_train_pairs().select("s1", "m").join(hold, on="s1").join(nullp.select("m"), on="m")
    old = pl.read_parquet(C.CACHE / "cand" / "train_w20.parquet", columns=["s1", "m"]).join(hold, on="s1")
    rep = {"null_record_true_pairs": truth.height, "old_found": truth.join(old, on=["s1", "m"]).height}
    for k in (1, 2, 3, 5, 10):
        x = cg.filter(pl.col("r_cg") <= k).select("s1", "m")
        rep[f"chargram top{k}"] = {"pairs_per_holdout_s1": round(x.height / hold.height, 3), "true_found": truth.join(x, on=["s1", "m"]).height,
                                   "rescued_vs_old": truth.join(x, on=["s1", "m"]).join(old, on=["s1", "m"], how="anti").height}
    (C.EXPERIMENTS / "chargram_null_w20.json").write_text(json.dumps(rep, indent=1))
    step("chargram", json.dumps(rep))


if __name__ == "__main__":
    main()
