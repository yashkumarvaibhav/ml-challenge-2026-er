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


def main_v2(n_pos: int = 150_000, n_swap: int = 100_000, n_other: int = 50_000):
    """v2 (26 Sep, after the category-swap finding, public +0.003063): hard negatives that are false by the generator's
    own logic, not by our model -- France candidate pairs whose record name swaps one S1 category word
    (cache/v6/s2/france_veto_pairs_test.parquet, ber.experiments.france_veto_s2). Positives: confident accepted
    France pairs of the veto re-scored universe (p >= 0.99, not vetoed). Plus 'confident other owner' negatives
    and an equal number of US/India e5base2 training pairs."""
    s1t = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    fr = s1t.filter(pl.col("country") == "France").select("s1")
    t = pl.read_parquet(C.CACHE / "v6" / "s2" / "test_pred_e5base2_frveto.parquet").join(fr, on="s1")
    veto = pl.read_parquet(C.CACHE / "v6" / "s2" / "france_veto_pairs_test.parquet").select("s1", "m")
    acc = decide(t, 0.4, 0.75).join(t, on=["s1", "m"]).join(veto, on=["s1", "m"], how="anti")
    conf = acc.filter(pl.col("p") >= 0.99).select("s1", "m")
    owner = conf.rename({"s1": "owner"})
    cand = pl.read_parquet(C.CACHE / "v6" / "cand" / "test.parquet", columns=["s1", "m", "d_cos"]).join(fr, on="s1")
    other = cand.join(owner, on="m").filter(pl.col("owner") != pl.col("s1")).join(veto, on=["s1", "m"], how="anti") \
        .sort(["s1", "d_cos"], descending=[False, True]).group_by("s1", maintain_order=True).head(1).select("s1", "m")
    pos = conf.sample(min(n_pos, conf.height), seed=C.SEED)
    swap = veto.sample(min(n_swap, veto.height), seed=C.SEED)
    other = other.sample(min(n_other, other.height), seed=C.SEED)
    frd = with_text(pl.concat([pos.with_columns(pl.lit(1, pl.Int8).alias("y")),
                               swap.with_columns(pl.lit(0, pl.Int8).alias("y")),
                               other.with_columns(pl.lit(0, pl.Int8).alias("y"))]), "test")
    us = pl.read_parquet(C.CACHE / "ce" / "train_pairs_e5base2.parquet").sample(frd.height, seed=C.SEED)
    out = pl.concat([frd.select("s1", "m", "y", "t1", "t2"), us.select("s1", "m", "y", "t1", "t2")]).sample(fraction=1.0, shuffle=True, seed=C.SEED)
    out.write_parquet(C.CACHE / "ce" / "train_pairs_frst.parquet")
    print(f"France: pos {pos.height:,} / swap neg {swap.height:,} / other-owner neg {other.height:,}; US/India {us.height:,}; total {out.height:,}")


if __name__ == "__main__":
    import sys
    main_v2() if "--v2" in sys.argv else main()
