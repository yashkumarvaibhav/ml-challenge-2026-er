"""France LLM veto (sub-2026-09-27-8): accepted France pairs with stage-2 p in (0.95, 0.999] are outside the stage-3
band; the Qwen2.5-7B pair classifier (ber.llm_ce, adapter qwen7b_r3) re-scores them and pairs it rejects outright
(score <= 0.02) are dropped. Label-free check on the 100,897 such pairs of #27-7: one-record share 6.0% for the 682
vetoed pairs vs 2.3% for the rest (public-confirmed false France cells: ~7%).

  BER_LLM_TAG=qwen7b_r3 BER_LLM_BASE=Qwen2.5-7B python -m ber.experiments.france_llm_veto --write output
"""
import argparse

import polars as pl

from .. import config as C

LO, HI, VETO = 0.95, 0.999, 0.02


def write(base: str):
    from ..llm_ce import _load, score
    arch = C.OUTPUT if base == "output" else C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    pairs = m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "").rename({c1: "s1", c2: "m"})
    fr = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
    p2 = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2.parquet")
    x = pairs.join(fr, on="s1").join(p2, on=["s1", "m"]).filter(pl.col("p").is_between(LO, HI, closed="right"))
    cache = C.WORK / "s2" / "france_llm_veto_scores.parquet"
    if cache.exists():
        s = pl.read_parquet(cache)
    else:
        tok, model = _load(False)
        s = x.select("s1", "m").with_columns(pl.Series("llm", score(x.select("s1", "m"), "test", name="france llm veto", model=model, tok=tok)))
        s.write_parquet(cache)
    d = x.join(s, on=["s1", "m"]).filter(pl.col("llm") <= VETO).select("s1", "m")
    kept = pairs.join(d, on=["s1", "m"], how="anti")
    g = kept.group_by("s1").agg(pl.col("m").sort().str.join(",").alias(c2)).rename({"s1": c1})
    out = m.select(c1).join(g, on=c1, how="left").with_columns(pl.col(c2).fill_null(""))
    assert out[c1].to_list() == m[c1].to_list()
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"{base}: France LLM veto {d.height:,} of {x.height:,} pairs with p2 in ({LO}, {HI}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", required=True)
    write(ap.parse_args().write)
