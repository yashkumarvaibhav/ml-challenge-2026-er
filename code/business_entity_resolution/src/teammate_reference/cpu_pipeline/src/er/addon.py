"""Add-on pair features: TF-IDF character-3-gram cosine similarity of names, core names and addresses.

IDF is fitted on the split's own S1 strings (label-free). Vectors are hashed (2^20 buckets), idf-weighted and
L2-normalized; a pair's cosine is the row-wise dot product. Output: cache/addon/<split>/part<k>.parquet with
(s1, m, tf_name, tf_core, tf_addr, tf_addr_w), aligned with the feature parts of the run.

  python -m er.addon --split train
"""
import argparse
import os
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from er import features as F
from er.common import CACHE, log, norm, world_s1

FEATS_DIR = os.environ.get("ER_FEATS", "feats2")
PARTS = 16
HV_C = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20, alternate_sign=False, norm=None)
HV_W = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=2 ** 20, alternate_sign=False, norm=None)
FIELDS = [("tf_name", "n_str", HV_C), ("tf_core", "n_core_str", HV_C), ("tf_addr", "a_str", HV_C), ("tf_addr_w", "a_str", HV_W)]


def _idf(strings, hv, chunk=400_000):
    df = np.zeros(hv.n_features, dtype=np.float64)
    for lo in range(0, len(strings), chunk):
        x = hv.transform(strings[lo:lo + chunk])
        df += np.bincount(x.indices, minlength=hv.n_features)
    return np.log((len(strings) + 1) / (df + 1)).astype(np.float32) + 1.0


def _vec(strings, hv, idf):
    x = hv.transform(strings).tocsr().astype(np.float32)
    x.data = np.log1p(x.data) * idf[x.indices]
    return normalize(x, copy=False)


def _strings(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("entity_id", pl.col("n_str").fill_null(""), pl.col("a_str").fill_null(""),
                     pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(list(F.LEGAL)))).list.join(" ").fill_null("").alias("n_core_str"))


_G = {}


def _part(k):
    split, S1, idfs, out = _G["split"], _G["S1"], _G["idfs"], _G["out"]
    cols = ["entity_id", "n_str", "a_str", "n_tok"]
    pairs = pl.read_parquet(CACHE / FEATS_DIR / split / f"part{k}.parquet", columns=["s1", "m"])
    u1 = pairs.select(pl.col("s1").alias("entity_id")).unique().with_row_index("i1")
    u2 = pairs.select(pl.col("m").alias("entity_id")).unique().with_row_index("i2")
    e1 = u1.join(S1, on="entity_id", how="left").sort("i1")
    e2 = pl.concat([pl.scan_parquet(CACHE / "norm_final" / f"{split}_s{s}.parquet").select(cols) for s in (2, 3)]) \
        .join(u2.lazy(), on="entity_id", how="inner").collect()
    e2 = _strings(e2).join(u2, on="entity_id").sort("i2")
    idx = pairs.join(u1, left_on="s1", right_on="entity_id").join(u2, left_on="m", right_on="entity_id")
    i1, i2 = idx["i1"].to_numpy(), idx["i2"].to_numpy()
    res = {}
    for name, col, hv in FIELDS:
        a = _vec(e1[col].to_list(), hv, idfs[name])
        b = _vec(e2[col].to_list(), hv, idfs[name])
        res[name] = np.asarray(a[i1].multiply(b[i2]).sum(axis=1)).ravel().astype(np.float32)
    idx.select("s1", "m").with_columns([pl.Series(n, v) for n, v in res.items()]).write_parquet(out / f"part{k}.parquet")
    return k, idx.height


def _init(split, out):
    import numpy as _np
    s1 = norm(split, 1, ["entity_id", "n_str", "a_str", "n_tok"])
    if split == "train":
        s1 = s1.join(world_s1(20), on="entity_id")
    _G.update(split=split, out=out, S1=_strings(s1))
    z = _np.load(out / "idf.npz")
    _G["idfs"] = {name: z[name] for name, _, _ in FIELDS}


def run_parallel(split, workers=4):
    import multiprocessing as mp
    t0 = time.time()
    out = CACHE / "addon" / split
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "idf.npz").exists():
        s1 = norm(split, 1, ["entity_id", "n_str", "a_str", "n_tok"])
        if split == "train":
            s1 = s1.join(world_s1(20), on="entity_id")
        S1 = _strings(s1)
        np.savez(out / "idf.npz", **{name: _idf(S1[col].to_list(), hv) for name, col, hv in FIELDS})
        del s1, S1
    log(f"addon {split}: idf ready ({time.time() - t0:.0f}s)")
    todo = [k for k in range(PARTS) if not (out / f"part{k}.parquet").exists()]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(split, out)) as pool:
        for k, n in pool.imap_unordered(_part, todo):
            log(f"addon {split} part {k + 1}/{PARTS}: {n:,} pairs ({time.time() - t0:.0f}s)")


def run(split):
    t0 = time.time()
    out = CACHE / "addon" / split
    out.mkdir(parents=True, exist_ok=True)
    cols = ["entity_id", "n_str", "a_str", "n_tok"]
    s1 = norm(split, 1, cols)
    if split == "train":
        s1 = s1.join(world_s1(20), on="entity_id")
    S1 = _strings(s1)
    del s1
    idfs = {name: _idf(S1[col].to_list(), hv) for name, col, hv in FIELDS}
    log(f"addon {split}: idf fitted on {S1.height:,} S1 ({time.time() - t0:.0f}s)")
    for k in range(PARTS):
        if (out / f"part{k}.parquet").exists():
            continue
        pairs = pl.read_parquet(CACHE / FEATS_DIR / split / f"part{k}.parquet", columns=["s1", "m"])
        u1 = pairs.select(pl.col("s1").alias("entity_id")).unique().with_row_index("i1")
        u2 = pairs.select(pl.col("m").alias("entity_id")).unique().with_row_index("i2")
        e1 = u1.join(S1, on="entity_id", how="left").sort("i1")
        e2 = pl.concat([pl.scan_parquet(CACHE / "norm_final" / f"{split}_s{s}.parquet").select(cols) for s in (2, 3)]) \
            .join(u2.lazy(), on="entity_id", how="inner").collect()
        e2 = _strings(e2).join(u2, on="entity_id").sort("i2")
        idx = pairs.join(u1, left_on="s1", right_on="entity_id").join(u2, left_on="m", right_on="entity_id")
        i1, i2 = idx["i1"].to_numpy(), idx["i2"].to_numpy()
        res = {}
        for name, col, hv in FIELDS:
            a = _vec(e1[col].to_list(), hv, idfs[name])
            b = _vec(e2[col].to_list(), hv, idfs[name])
            res[name] = np.asarray(a[i1].multiply(b[i2]).sum(axis=1)).ravel().astype(np.float32)
        idx.select("s1", "m").with_columns([pl.Series(n, v) for n, v in res.items()]).write_parquet(out / f"part{k}.parquet")
        log(f"addon {split} part {k + 1}/{PARTS}: {idx.height:,} pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    run_parallel(ap.parse_args().split)
