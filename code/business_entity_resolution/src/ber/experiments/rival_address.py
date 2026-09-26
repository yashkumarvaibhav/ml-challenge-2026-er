"""Rival-address rule: an accepted pair (S1 A, record R) is dropped when R's name equals A's core name AND another S1 B
of the same country has the same core name AND R's address matches B's address clearly better than A's.

Motivation (26 Sep): among France accepted pairs with standard-CE < 0.02 and 'same name, different street', 74% of
those with a same-name rival S1 have a record address closer to the rival (e.g. S1 'Lille Federation SASU | 14 Rue
Saint François' holding record '14 RUE KEPLER' while S1 'Lille Federation SASU | 14 Rue Kepler' exists). This uses
only the one-S1-per-record structure and address evidence (no CE, no labels). Harm-screened on the US/India holdout
(both ghost worlds) with the same code.
"""
import json
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from ..io import load_train_pairs
from ..metrics import per_s1_f05, summarize
from ..model import decide

LEG = ["sarl", "sas", "sasu", "sa", "eurl", "ei", "sci", "snc", "llc", "inc", "ltd", "corp", "co", "pvt", "private", "limited",
       "s", "a", "r", "l", "u", "e", "i"]
MARGIN, MIN_RIV = 10, 80


def core(c):
    return (pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}0-9 ]", " ").str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "")).list.set_difference(pl.lit(LEG)).list.sort().list.join(" "))


def na(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(t for t in re.split(r"[^a-z0-9]+", s) if t)


def flag(acc: pl.DataFrame, s1: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """acc: accepted (s1, m). s1: (s1, country, n1, a1); rec: (m, n2, a2). Returns dropped (s1, m) with evidence."""
    s1 = s1.with_columns(core("n1").alias("k"))
    rec = rec.with_columns(core("n2").alias("km")).filter(pl.col("a2").is_not_null())
    x = acc.join(s1, on="s1").join(rec, on="m").filter(pl.col("k") == pl.col("km"))
    riv = s1.select(pl.col("s1").alias("r"), "country", "k", pl.col("a1").alias("ar"))
    y = x.join(riv, on=["country", "k"]).filter(pl.col("r") != pl.col("s1"))
    if y.height == 0:
        return y.select("s1", "m")
    y = y.with_columns(pl.Series("own", [fuzz.token_set_ratio(na(a), na(b)) for a, b in zip(y["a1"], y["a2"])], dtype=pl.Float64),
                       pl.Series("riv", [fuzz.token_set_ratio(na(a), na(b)) for a, b in zip(y["ar"], y["a2"])], dtype=pl.Float64))
    best = y.sort("riv", descending=True).group_by(["s1", "m"]).first()
    return best.filter((pl.col("riv") >= MIN_RIV) & (pl.col("riv") >= pl.col("own") + MARGIN)).select("s1", "m", "own", "riv", "r")


def main():
    rep = {}
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), "country", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    p = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2_frveto.parquet")
    acc = decide(p, 0.4, 0.75)
    d = flag(acc, t1, rec)
    d.write_parquet(C.WORK / "s2" / "rival_address_drops_test.parquet")
    dc = d.join(t1.select("s1", "country"), on="s1")
    rep["test"] = {c: {"pairs": dc.filter(pl.col("country") == c).height, "S1": dc.filter(pl.col("country") == c)["s1"].n_unique()} for c in ("US", "India", "France")}
    print("test", rep["test"], flush=True)
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    truth = load_train_pairs().select("s1", "m")
    tr1 = pl.read_parquet(C.CACHE / "train_source1.parquet").select(pl.col("entity_id").alias("s1"), "country", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    trec = pl.concat([pl.read_parquet(C.CACHE / f"train_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    for w in (20, 0):
        world = sp if w == 0 else sp.filter(~pl.col("is_ghost"))
        ws1 = tr1.join(world.select(pl.col("entity_id").alias("s1")), on="s1")
        hp = pl.read_parquet(C.WORK / "s2" / f"holdout_pred_w{w}_e5base2.parquet")
        hacc = decide(hp, 0.4, 0.75).join(hold.select("s1"), on="s1")
        hd = flag(hacc, ws1, trec)
        kept = hacc.join(hd.select("s1", "m"), on=["s1", "m"], how="anti")
        yt = hd.join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left")["y"].fill_null(0)
        b, n = summarize(per_s1_f05(hacc, truth, hold))["macro_f05"], summarize(per_s1_f05(kept, truth, hold))["macro_f05"]
        rep[f"holdout w{w}"] = {"dropped": hd.height, "dropped_true_rate": round(yt.mean(), 3) if hd.height else None,
                                "base": b, "rule": n, "delta": round(n - b, 6)}
        print(f"holdout w{w}", rep[f"holdout w{w}"], flush=True)
    (C.EXP_RUN / "rival_address.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
