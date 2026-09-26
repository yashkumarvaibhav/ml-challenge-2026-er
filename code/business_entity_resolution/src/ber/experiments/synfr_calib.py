"""Compare label-free statistics of accepted pairs: real France (test, e5base2 decisions BEFORE any France rule) vs
the synthetic France world (ber.synth_france via ber.synfr_run). Used to calibrate the synthetic generator."""
import json
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from ..model import decide
from .france_category_swap import classify, toks


def na(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def stats(scored, s1, rec, cats, cand=None):
    d = decide(scored, 0.4, 0.75)
    n = s1.height
    k = s1.select("s1").join(d.group_by("s1").len(), on="s1", how="left").fill_null(0)
    a = d.join(s1, on="s1").join(rec, on="m").sample(min(150_000, d.height), seed=1)
    sw = [bool((c := classify(x, y)) and c[0] and c[1] in cats) for x, y in zip(a["n1"], a["n2"])]
    same_name = [" ".join(sorted(toks(x))) == " ".join(sorted(toks(y))) for x, y in zip(a["n1"], a["n2"])]
    def street_diff(x, y):
        if x is None or y is None:
            return False
        sx = {t for t in na(x).split() if not t.isdigit() and len(t) >= 4}
        sy = {t for t in na(y).split() if not t.isdigit() and len(t) >= 4}
        return len(sx & sy) / max(1, len(sx)) < 0.5
    sd = [street_diff(x, y) for x, y in zip(a["a1"], a["a2"])]
    nameless = [not any(max((fuzz.ratio(u, v) for v in toks(y)), default=0) >= 75 for u in toks(x)) for x, y in zip(a["n1"], a["n2"])]
    out = {"pairs_per_S1": round(d.height / n, 3), "k0": round(100 * (k["len"] == 0).mean(), 2), "k1": round(100 * (k["len"] == 1).mean(), 2),
           "swap_rate": round(sum(sw) / a.height, 4), "addr_null": round(a["a2"].is_null().mean(), 4),
           "same_name_diff_street": round(sum(1 for s, t in zip(same_name, sd) if s and t) / a.height, 4),
           "nameless": round(sum(nameless) / a.height, 4),
           "band_05_95_per_S1": round(scored.filter(pl.col("p").is_between(0.05, 0.95)).height / n, 3)}
    return out


def main():
    cats = set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"])
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    real = stats(pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2.parquet").join(t1.select("s1"), on="s1"), t1, rec, cats)
    s1 = pl.read_parquet(C.CACHE / "synfr_source1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    srec = pl.concat([pl.read_parquet(C.CACHE / f"synfr_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    syn = stats(pl.read_parquet(C.WORK / "synfr" / "pred_s2.parquet"), s1, srec, cats)
    rep = {"real_France": real, "synthetic": syn, "records_per_S1": {"real": round(1_435_000 / 259_452, 2), "synthetic": round(srec.height / s1.height, 2)}}
    print(json.dumps(rep, indent=1))
    (C.EXP_RUN / "synfr_calib.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
