"""Label-free precision scan of France accepted-pair slices (the subagent's 'k1-share' check).

For accepted pairs with stage-2 p >= P_HI, the share whose S1 has exactly one accepted record is 2.2% for all France
accepted pairs of #8, 6.2% for the public-confirmed false category swaps; in the labelled holdout false/true pairs
show 3.4%/1.8%. A slice's share maps to an implied false fraction f = (share - base) / (6.2% - base).
Run on the sub-2026-09-27-1 decisions (after both France removal rules).
"""
import json
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from .france_category_swap import toks

P_HI, BASE_FALSE = 0.95, 0.062
FILL = {"services", "service", "france", "fils", "associes", "developpement", "groupe", "international", "participations", "distribution", "holding"}


def pairs(tag):
    d = pl.read_csv(C.ROOT / "submissions" / "archive" / tag / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c = d.columns
    return d.with_columns(pl.col(c[1]).fill_null("").str.split(",")).explode(c[1]).rename({c[0]: "s1", c[1]: "m"}).filter(pl.col("m") != "")


def na(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def main(tag="sub-2026-09-27-1"):
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    acc = pairs(tag).join(t1.select("s1"), on="s1")
    p = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2.parquet").join(t1.select("s1"), on="s1")
    k = acc.group_by("s1").len().rename({"len": "k"})
    x = acc.join(p, on=["s1", "m"]).join(k, on="s1").join(t1, on="s1").join(rec, on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
    core = lambda c: pl.col(c).map_elements(lambda s: " ".join(sorted(toks(s))), return_dtype=pl.Utf8)
    x = x.with_columns(core("n1").alias("k1n"), core("n2").alias("k2n"))
    namecnt = t1.with_columns(core("n1").alias("k1n")).group_by("k1n").len().rename({"len": "n_same_name"})
    x = x.join(namecnt, on="k1n", how="left")
    rk = x.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("rk"))
    t1s, t2s = x["n1"].to_list(), x["n2"].to_list()
    nameless = [not any(max((fuzz.ratio(u, v) for v in toks(b)), default=0) >= 75 for u in toks(a)) for a, b in zip(t1s, t2s)]
    fillsub = []
    for a, b in zip(t1s, t2s):
        ta, tb = toks(a), toks(b)
        fillsub.append(bool(ta and tb and any(w in FILL and w not in ta for w in tb) and any(max(fuzz.ratio(u, v) for v in tb) < 75 for u in ta)))
    wrap = [bool(re.search(r"(?i)\b(a/k/a|doing business as|dba|formerly)\b", b or "")) for b in t2s]
    x = x.with_columns(pl.Series("nameless", nameless), pl.Series("fillsub", fillsub), pl.Series("wrap", wrap), rk["rk"])
    base = x.filter(pl.col("p") >= P_HI)
    b0 = (base["k"] == 1).mean()
    slices = {
        "addr_null": pl.col("a2").is_null(),
        "addr_null & name shared by >=2 S1": pl.col("a2").is_null() & (pl.col("n_same_name") >= 2),
        "addr_null & name shared by >=5 S1": pl.col("a2").is_null() & (pl.col("n_same_name") >= 5),
        "nameless (initials/pseudo/domain)": pl.col("nameless"),
        "filler replaces an S1 word": pl.col("fillsub"),
        "wrapper name": pl.col("wrap"),
        "source S2": pl.col("src") == "S2",
        "source S3": pl.col("src") == "S3",
        "extra (rank>1) p<0.99": (pl.col("rk") > 1) & (pl.col("p") < 0.99),
        "name shared by >=10 S1": pl.col("n_same_name") >= 10,
        "exact core name": pl.col("k1n") == pl.col("k2n"),
    }
    rep = {"base_k1_share": round(b0, 4), "slices": {}}
    for nm, cond in slices.items():
        s = x.filter(cond)
        sh = s.filter(pl.col("p") >= P_HI)
        share = (sh["k"] == 1).mean() if sh.height else None
        f = (share - b0) / (BASE_FALSE - b0) if share is not None else None
        rep["slices"][nm] = {"pairs": s.height, "hi_p_pairs": sh.height, "k1_share": round(share, 4) if share is not None else None,
                             "implied_false": round(f, 2) if f is not None else None}
        print(f"{nm:40s} pairs {s.height:7d}  k1-share {rep['slices'][nm]['k1_share']}  implied false {rep['slices'][nm]['implied_false']}", flush=True)
    (C.EXP_RUN / "k1_scan.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
