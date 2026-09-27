"""Label-free per-country comparison of two matching_results.tsv files (pred pairs/S1, empty share,
records claimed by >1 S1, pairs added/removed)."""
import json
import sys

import polars as pl

from ber import config as C


def read(path):
    d = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8})
    return d.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
        .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})


a, b = read(sys.argv[1]), read(sys.argv[2])
s1 = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
rep = {}
for cty in ("US", "India", "France"):
    n = s1.filter(pl.col("country") == cty).height
    r = {}
    for name, d in (("A", a), ("B", b)):
        x = d.join(s1, on="s1").filter(pl.col("country") == cty)
        r[name] = {"pairs_per_s1": round(x.height / n, 4), "empty_share": round(1 - x["s1"].n_unique() / n, 4),
                   "records_in_gt1_s1": int(x.group_by("m").len().filter(pl.col("len") > 1).height)}
    xa = a.join(s1, on="s1").filter(pl.col("country") == cty).select("s1", "m")
    xb = b.join(s1, on="s1").filter(pl.col("country") == cty).select("s1", "m")
    r["added_in_B"] = xb.join(xa, on=["s1", "m"], how="anti").height
    r["removed_in_B"] = xa.join(xb, on=["s1", "m"], how="anti").height
    rep[cty] = r
print(json.dumps(rep, indent=1))
