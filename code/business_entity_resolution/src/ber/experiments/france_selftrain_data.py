"""France CE self-training data (26 Sep; FAQ allows self-training; reviewer 19:14 'safe negatives only').
Positives: France test pairs accepted by v6 with stage-2 p >= 0.99.
Negatives: (S1 X, record R) where R is in X's candidate list but R is confidently accepted (p >= 0.99) for a
           DIFFERENT France S1 -- safe under the one-S1-per-record rule (given that confident assignment).
Mixed 50/50 with the e5base2 US/India training pairs (to avoid forgetting). Written to cache/ce/train_pairs_frst.parquet."""
import polars as pl

from ber import config as C
from ber.cross_encoder import with_text
from ber.model import decide


def main(n_fr: int = 400_000):
    s1t = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    fr = s1t.filter(pl.col("country") == "France").select("s1")
    t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base.parquet").join(fr, on="s1")
    acc = decide(t, 0.4, 0.75).join(t, on=["s1", "m"])
    conf = acc.filter(pl.col("p") >= 0.99).select("s1", "m")
    owner = conf.rename({"s1": "owner"})
    cand = pl.read_parquet(C.CACHE / "v6" / "cand" / "test.parquet", columns=["s1", "m", "d_cos"]).join(fr, on="s1")
    neg = cand.join(owner, on="m").filter(pl.col("owner") != pl.col("s1")).sort(["s1", "d_cos"], descending=[False, True]) \
        .group_by("s1", maintain_order=True).head(2).select("s1", "m")
    pos = conf.sample(min(n_fr // 2, conf.height), seed=C.SEED)
    neg = neg.sample(min(n_fr // 2, neg.height), seed=C.SEED)
    frd = with_text(pl.concat([pos.with_columns(pl.lit(1, pl.Int8).alias("y")), neg.with_columns(pl.lit(0, pl.Int8).alias("y"))]), "test")
    us = pl.read_parquet(C.CACHE / "ce" / "train_pairs_e5base2.parquet").sample(frd.height, seed=C.SEED)
    out = pl.concat([frd.select("s1", "m", "y", "t1", "t2"), us.select("s1", "m", "y", "t1", "t2")]).sample(fraction=1.0, shuffle=True, seed=C.SEED)
    out.write_parquet(C.CACHE / "ce" / "train_pairs_frst.parquet")
    print(f"France confident positives available {conf.height:,}; safe negatives available {neg.height:,}; "
          f"written {out.height:,} pairs (France {frd.height:,}: pos {int(frd['y'].sum()):,}; US/India {us.height:,})")


if __name__ == "__main__":
    main()
