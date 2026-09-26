"""CE as a removal-only check on ACCEPTED pairs, including the confident region stage 2 never showed the CE.

Motivation (26 Sep): the France category-swap removal (+0.003063 public) confirmed that the earlier CE-judge cell
"accepted & CE < 0.1" (France 1.31% vs US/India 0.01%) was largely sibling businesses. Additions have lost on
public (sub-2026-09-26-10: -0.001779), removals have won. Here: score every accepted France pair (post-veto
decisions) and a sample of accepted US/India holdout pairs with the standard CE (BER_CE=e5base2), to size and
calibrate a CE veto.
  BER_CE=e5base2 python -m ber.experiments.ce_veto_confident
"""
import json

import polars as pl

from .. import config as C
from ..cross_encoder import score
from ..io import load_train_pairs
from ..model import decide

S2 = C.WORK / "s2"


def main():
    fr = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
    t = pl.read_parquet(S2 / "test_pred_e5base2_frveto.parquet").join(fr, on="s1")
    acc = decide(t, 0.4, 0.75).join(t, on=["s1", "m"])
    out = acc.with_columns(pl.Series("ce", score(acc.select("s1", "m"), "test", name="ce France accepted")))
    out.write_parquet(S2 / "ce_france_accepted_e5base2.parquet")
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
    hp = pl.read_parquet(S2 / "holdout_pred_w20_e5base2.parquet")
    hacc = decide(hp, 0.4, 0.75).join(hold, on="s1").join(hp, on=["s1", "m"]).sample(200_000, seed=C.SEED)
    hacc = hacc.with_columns(pl.Series("ce", score(hacc.select("s1", "m"), "train", name="ce holdout accepted")))
    truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
    hacc = hacc.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    hacc.write_parquet(S2 / "ce_holdout_accepted_e5base2.parquet")
    rep = {}
    for thr in (0.02, 0.05, 0.1, 0.2):
        f = out.filter(pl.col("ce") < thr)
        h = hacc.filter(pl.col("ce") < thr)
        rep[str(thr)] = {"france_share": round(f.height / out.height, 4), "france_pairs": f.height, "france_S1": f["s1"].n_unique(),
                         "holdout_share": round(h.height / hacc.height, 5), "holdout_true_rate": round(h["y"].mean(), 3) if h.height else None,
                         "holdout_n": h.height}
    print(json.dumps(rep, indent=1))
    (C.EXP_RUN / "ce_veto_confident.json").write_text(json.dumps(rep, indent=1))


def write(base: str, thr: float = 0.02):
    """output/ = archived `base` minus its France pairs whose standard-CE score < thr (removal only)."""
    import shutil
    arch = C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    drop = pl.read_parquet(S2 / "ce_france_accepted_e5base2.parquet").filter(pl.col("ce") < thr).select("s1", "m")
    pairs = (m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "")
             .rename({c1: "s1", c2: "m"}))
    kept = pairs.join(drop, on=["s1", "m"], how="anti")
    g = kept.group_by("s1").agg(pl.col("m").sort().str.join(",").alias(c2)).rename({"s1": c1})
    out = m.select(c1).join(g, on=c1, how="left").with_columns(pl.col(c2).fill_null(""))
    assert out[c1].to_list() == m[c1].to_list()
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    print(f"removed {pairs.height - kept.height:,} France pairs with CE < {thr} from {base}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--thr", type=float, default=0.02)
    a = ap.parse_args()
    write(a.write, a.thr) if a.write else main()
