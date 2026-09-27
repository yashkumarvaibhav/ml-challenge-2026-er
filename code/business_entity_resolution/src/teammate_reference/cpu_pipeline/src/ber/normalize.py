"""Normalize every record of a split in parallel (PLAN.md B1).

Two passes:
* `--stage raw`   : formatting rules only (input for `ber.learn_maps`), train only.
* `--stage final` : formatting rules + learned maps (training-portion only) + domain
                    segmentation with the split's own S1 name vocabulary (label-free).

Output: cache/norm_<stage>/<split>_s<k>.parquet with one row per record.
"""
import argparse
import json
import multiprocessing as mp
import time
from collections import Counter

import polars as pl

from . import config as C
from .io import load_source
from .progress import Progress
from .text import Segmenter, norm_addr, norm_name

_G = {}
COLS = ["n_tok", "n_alt", "n_native", "n_native_mapped_frac", "n_allcaps", "n_accent", "n_domain", "n_dba",
        "a_comp", "a_tok", "a_nums", "a_null", "a_native", "a_ncomma"]
LIST_COLS = ("n_tok", "a_comp", "a_tok", "a_nums")
SEP = "\x1f"  # list columns travel as SEP-joined strings (building polars lists from Python lists is slow)


def _init(maps: dict, seg_counts: dict):
    _G["native"] = (maps or {}).get("native_name", {})
    _G["comp"] = (maps or {}).get("addr_comp", {})
    _G["tok"] = (maps or {}).get("addr_tok", {})
    _G["ntok"] = (maps or {}).get("name_tok", {})
    _G["seg"] = {c: Segmenter(cnt) for c, cnt in (seg_counts or {}).items()}


def _work(job):
    idx, (names, addrs, countries) = job
    out = {k: [] for k in COLS}
    for n, a, c in zip(names, addrs, countries):
        r = norm_name(n, _G["native"], _G["seg"].get(c))
        ntok = _G["ntok"].get(c)
        if ntok:
            r["n_tok"] = [ntok.get(t, t) for t in r["n_tok"]]
        r.update(norm_addr(a, _G["comp"].get(c), _G["tok"].get(c)))
        for k in COLS:
            out[k].append(SEP.join(r[k]) if k in LIST_COLS else r[k])
    return idx, out


def normalize_df(df: pl.DataFrame, maps: dict = None, seg_counts: dict = None, chunk: int = 20_000,
                 name: str = "normalize") -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    ctys = df["country"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk], ctys[i:i + chunk]) for i in range(0, len(names), chunk)]
    prog = Progress(name, total=len(names), unit="records")
    parts = [None] * len(chunks)
    with mp.get_context("fork").Pool(C.N_PROCS, initializer=_init, initargs=(maps, seg_counts)) as pool:
        for idx, part in pool.imap_unordered(_work, enumerate(chunks), chunksize=1):
            parts[idx] = part  # re-ordered by chunk index
            prog.update(len(part["n_tok"]))
    prog.done()
    cols = {k: [v for p in parts for v in p[k]] for k in COLS}
    out = pl.DataFrame(
        {"entity_id": df["entity_id"], "country": df["country"], **cols},
        schema_overrides={k: pl.Utf8 for k in LIST_COLS} | {"n_alt": pl.Utf8, "n_native_mapped_frac": pl.Float32},
    ).with_columns(
        [pl.col(k).str.split(SEP).list.eval(pl.element().filter(pl.element() != "")) for k in LIST_COLS]
    )
    return out.with_columns(
        pl.col("n_tok").list.join(" ").alias("n_str"),
        pl.col("a_tok").list.join(" ").alias("a_str"),
    )


def s1_vocab(norm_s1: pl.DataFrame) -> dict:
    """Per-country S1 name-token counts (for domain-name segmentation)."""
    out = {}
    for (cty,), g in norm_s1.group_by("country"):
        out[cty] = dict(Counter(t for toks in g["n_tok"].to_list() for t in toks))
    return out


def run(stage: str, split: str):
    maps = None
    if stage == "final":
        maps = json.loads((C.CACHE / "maps" / "maps.json").read_text())
        # self-trained maps (pseudo-pairs from the split's own predictions; FAQ: self-training allowed)
        # add countries that have no supervised maps (France); supervised countries are never overridden
        self_path = C.CACHE / "maps" / f"self_maps_{split}.json"
        if self_path.exists():
            sm = json.loads(self_path.read_text())
            for kind in ("addr_comp", "addr_tok", "name_tok"):
                for cty, m in sm.get(kind, {}).items():
                    maps.setdefault(kind, {}).setdefault(cty, {})
                    if not maps[kind][cty]:
                        maps[kind][cty] = m
    outdir = C.CACHE / f"norm_{stage}"
    outdir.mkdir(parents=True, exist_ok=True)
    seg = None
    for s in C.SOURCES:
        t = time.time()
        df = load_source(split, s)
        res = normalize_df(df, maps, seg, name=f"normalize {stage} {split} s{s}")
        res.write_parquet(outdir / f"{split}_s{s}.parquet")
        if s == 1 and stage == "final":
            seg = s1_vocab(res)
        print(f"{stage} {split} s{s}: {res.height} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["raw", "final"], required=True)
    ap.add_argument("--split", choices=list(C.SPLITS), required=True)
    a = ap.parse_args()
    run(a.stage, a.split)
