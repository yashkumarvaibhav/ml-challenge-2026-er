"""Controlled France-only variant of v6 (plan-discussion 26 Sep): drop France accepted pairs whose record shares a
house number with the S1 but no rare street token (both sides have street tokens) -- the 'sibling business'
pattern that is 8x more frequent among France S1s than US S1s and 3.4x more frequent among France accepted pairs.
US/India decisions are byte-identical to v6. candidate_pairs.tsv = the v6 candidate set (unchanged).

  BER_RUN=v6 BER_WORK=v6 python code/business_entity_resolution/src/ber/experiments/france_street_variant.py
"""
import polars as pl

from ber import config as C
from ber.io import load_source
from ber.model import decide

RARE = 0.002


def recs(split, s):
    return pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "country", "a_tok", "a_nums"])


def street_tokens(d):
    n = d.group_by("country").len().rename({"len": "N"})
    t = d.select("entity_id", "country", pl.col("a_tok").list.unique()).explode("a_tok").drop_nulls().filter(~pl.col("a_tok").str.contains(r"\d"))
    rare = t.group_by("country", "a_tok").len().join(n, on="country").filter(pl.col("len") < RARE * pl.col("N")).select("country", "a_tok")
    return t.join(rare, on=["country", "a_tok"]).group_by("entity_id").agg(pl.col("a_tok").alias("st"))


if __name__ == "__main__":
    t = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base.parquet")
    acc = decide(t, 0.4, 0.75)
    r = pl.concat([recs("test", s) for s in C.SOURCES])
    r = r.join(street_tokens(r), on="entity_id", how="left").with_columns(pl.col("st").fill_null([])).select("entity_id", "country", "a_nums", "st")
    a = r.rename({"entity_id": "s1", "a_nums": "n1", "st": "st1"})
    b = r.drop("country").rename({"entity_id": "m", "a_nums": "n2", "st": "st2"})
    d = acc.join(a, on="s1").join(b, on="m")
    drop = d.filter((pl.col("country") == "France") & (pl.col("st1").list.len() > 0) & (pl.col("st2").list.len() > 0)
                    & (pl.col("st1").list.set_intersection("st2").list.len() == 0)
                    & (pl.col("n1").list.set_intersection("n2").list.len() > 0)).select("s1", "m")
    pred = acc.join(drop, on=["s1", "m"], how="anti")
    s1 = load_source("test", 1).select(pl.col("entity_id").alias("s1"))
    cand = pl.read_parquet(C.WORK / "cand" / "test.parquet", columns=["s1", "m"])

    def write(pairs, col, path):
        g = pairs.group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias(col))
        s1.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null("")).rename({"s1": "source1_entity_id"}) \
            .write_csv(path, separator="\t", quote_style="never")

    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    write(pred, "matched_entity_ids", C.OUTPUT / "matching_results.tsv")
    write(cand, "candidate_entity_ids", C.OUTPUT / "candidate_pairs.tsv")
    print(f"v6 accepted {acc.height:,}; dropped France street-disagreeing same-number pairs {drop.height:,}; written {pred.height:,}")
