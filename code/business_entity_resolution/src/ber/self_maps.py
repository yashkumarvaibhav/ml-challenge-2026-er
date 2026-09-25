"""Self-trained normalization maps from pseudo-pairs (France v5; FAQ: self-training is allowed).

Pseudo-pairs = high-confidence model predictions (p >= P_MIN), no labels. The same diff/threshold
machinery as the supervised `ber.learn_maps` (support, share, dominance, plausibility filter for
word maps) is applied to them.

  python -m ber.self_maps validate     # label-free miner check on US/India holdout (vs supervised maps)
  python -m ber.self_maps mine         # mine maps for countries WITHOUT supervised maps (France) from test
"""
import argparse
import json
from collections import Counter

import polars as pl

from . import config as C
from .io import load_train_pairs
from .learn_maps import _apply_comp, _diff_counts, _maps_from, plausible_token_map
from .progress import step

P_MIN = 0.99
COMMON_COMP = 1000  # a non-similar component map needs both sides to be very common whole components


def _digits(x: str) -> str:
    return "".join(ch for ch in x if ch.isdigit())


def filter_self_maps(maps: dict, comp_counts: dict) -> dict:
    """Plausibility filters for maps learned from pseudo-pairs (applied per country).

    * address maps never drop or change digits (e.g. '1bis' -> 'bis' loses the house number); name maps
      may (digit-for-letter typos such as 'c1ub' -> 'club', as in the supervised US/India maps);
    * a component map must look like a spelling/abbreviation variant word by word, OR both sides
      must be very common whole components (regions/departements are; streets are not), which
      rejects co-occurrence artefacts such as 'rue auber' -> 'hauts de france'.
    """
    out = {"addr_comp": {}, "addr_tok": {}, "name_tok": {}}
    for cty, m in maps["addr_tok"].items():
        out["addr_tok"][cty] = {v: c for v, c in m.items() if _digits(v) == _digits(c)}
    out["name_tok"] = {cty: dict(m) for cty, m in maps["name_tok"].items()}
    for cty, m in maps["addr_comp"].items():
        cc = comp_counts.get(cty, {})
        keep = {}
        for v, c in m.items():
            if _digits(v) != _digits(c):
                continue
            vw, cw = v.split(" "), c.split(" ")
            wordwise = len(vw) == len(cw) and all(a == b or plausible_token_map(a, b) for a, b in zip(vw, cw))
            common = cc.get(v, 0) >= COMMON_COMP and cc.get(c, 0) >= COMMON_COMP
            if wordwise or common:
                keep[v] = c
        out["addr_comp"][cty] = keep
    return out


def comp_counts(split: str) -> dict:
    """Per-country counts of whole address components (S1 + S2 + S3, label-free)."""
    d = pl.concat([pl.read_parquet(C.CACHE / "norm_raw" / f"{split}_s{s}.parquet", columns=["country", "a_comp"]) for s in C.SOURCES])
    g = d.explode("a_comp").drop_nulls().group_by("country", "a_comp").len()
    out = {}
    for cty, comp, n in g.iter_rows():
        out.setdefault(cty, {})[comp] = n
    return out


def _raw(split):
    r1 = pl.read_parquet(C.CACHE / "norm_raw" / f"{split}_s1.parquet", columns=["entity_id", "a_comp", "n_tok"]) \
        .rename({"entity_id": "s1", "a_comp": "c1", "n_tok": "n1"})
    r23 = pl.concat([pl.read_parquet(C.CACHE / "norm_raw" / f"{split}_s{s}.parquet", columns=["entity_id", "country", "a_comp", "n_tok"])
                     for s in (2, 3)]).rename({"entity_id": "m", "a_comp": "cm", "n_tok": "nm"})
    return r1, r23


def mine_from_pairs(pairs: pl.DataFrame, split: str) -> dict:
    """pairs: (s1, m). Returns {addr_comp, addr_tok, name_tok} per country, like learn_maps."""
    r1, r23 = _raw(split)
    tp = pairs.join(r1, on="s1").join(r23, on="m")
    combos, nvc = _diff_counts(tp.select("country", "c1", "cm"), "c1", "cm")
    comp = _maps_from(combos, nvc, exclude_digits=False)
    rows = []
    for cty, c1, cm, n1, nm in tp.select("country", "c1", "cm", "n1", "nm").iter_rows():
        cmap = comp.get(cty, {})
        rows.append((cty, [t for k in _apply_comp(c1, cmap) for t in k.split(" ")],
                     [t for k in _apply_comp(cm, cmap) for t in k.split(" ")], n1, nm))
    tokdf = pl.DataFrame(rows, schema=["country", "t1", "tm", "n1", "nm"], orient="row")
    combos, nvt = _diff_counts(tokdf.select("country", "t1", "tm"), "t1", "tm")
    addr_tok = _maps_from(combos, nvt, exclude_digits=True, plausible=plausible_token_map)
    combos, nvn = _diff_counts(tokdf.select("country", "n1", "nm"), "n1", "nm")
    name_tok = _maps_from(combos, nvn, exclude_digits=True, plausible=plausible_token_map)
    return {"addr_comp": comp, "addr_tok": addr_tok, "name_tok": name_tok}


def validate():
    """Label-free miner on US/India holdout pseudo-pairs vs the supervised maps."""
    pred = pl.read_parquet(C.CACHE / "runs" / "v4" / "train_holdout_w20.parquet").filter(pl.col("p") >= P_MIN)
    truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
    prec = pred.join(truth, on=["s1", "m"], how="left")["y"].fill_null(0).mean()
    mined = mine_from_pairs(pred.select("s1", "m"), "train")
    sup = json.loads((C.CACHE / "maps" / "maps.json").read_text())
    rep = {"pseudo_pairs": pred.height, "pseudo_pair_precision": round(prec, 4)}
    for kind in ("addr_comp", "addr_tok", "name_tok"):
        for cty, m in mined[kind].items():
            s = sup[kind].get(cty, {})
            same = sum(1 for v, c in m.items() if s.get(v) == c)
            differ = {v: (c, s[v]) for v, c in m.items() if v in s and s[v] != c}
            only = [v for v in m if v not in s]
            rep[f"{kind}/{cty}"] = {"mined": len(m), "agree_with_supervised": same, "disagree": len(differ),
                                    "not_in_supervised": len(only), "agree_rate_on_overlap": round(same / max(1, same + len(differ)), 4),
                                    "disagree_examples": dict(list(differ.items())[:8]), "new_examples": only[:8]}
    (C.EXPERIMENTS / "self_maps_validate.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    step("self_maps", "validate: " + json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in ("mined", "agree_rate_on_overlap", "not_in_supervised")}) for k, v in rep.items()}))


def mine(pred_path=None, params_path=None):
    """France (and any country without supervised maps): mine maps from test pseudo-pairs."""
    pred = pl.read_parquet(pred_path or (C.CACHE / "pred" / "test.parquet")).filter(pl.col("p") >= P_MIN)
    sup = json.loads((C.CACHE / "maps" / "maps.json").read_text())
    supervised_countries = set(sup["addr_comp"]) | set(sup["addr_tok"])
    mined = mine_from_pairs(pred.select("s1", "m"), "test")
    mined = filter_self_maps(mined, comp_counts("test"))
    out = {k: {c: m for c, m in v.items() if c not in supervised_countries} for k, v in mined.items()}
    (C.CACHE / "maps" / "self_maps_test.json").write_text(json.dumps(out, ensure_ascii=False))
    rep = {k: {c: {"n": len(m), "examples": dict(list(m.items())[:25])} for c, m in v.items()} for k, v in out.items()}
    (C.EXPERIMENTS / "self_maps_france.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    step("self_maps", "mined: " + json.dumps({k: {c: len(m) for c, m in v.items()} for k, v in out.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["validate", "mine"])
    a = ap.parse_args()
    {"validate": validate, "mine": mine}[a.what]()
