"""Sample true pairs missed by blocking (holdout, world20) with their normalized forms."""
import polars as pl

from ber import config as C
from ber.io import load_train_pairs

if __name__ == "__main__":
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
    truth = load_train_pairs().select("s1", "m").join(hold, on="s1")
    cand = pl.read_parquet(C.CACHE / "cand" / "train_w20.parquet", columns=["s1", "m"])
    miss = truth.join(cand, on=["s1", "m"], how="anti")
    n1 = pl.read_parquet(C.CACHE / "norm_final" / "train_s1.parquet", columns=["entity_id", "n_str", "a_str"]).rename({"entity_id": "s1", "n_str": "n1", "a_str": "a1"})
    n2 = pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"train_s{s}.parquet", columns=["entity_id", "n_str", "a_str", "a_null", "n_native"]) for s in (2, 3)]).rename({"entity_id": "m", "n_str": "n2", "a_str": "a2"})
    d = miss.join(n1, on="s1").join(n2, on="m")
    print("missed pairs:", d.height, d.group_by("country", "a_null").len().sort("len", descending=True).rows())
    # how ambiguous are address-null misses? count S1 in world with identical name string
    s1all = pl.read_parquet(C.CACHE / "norm_final" / "train_s1.parquet", columns=["entity_id", "country", "n_str"]).join(sp.filter(~pl.col("is_ghost")).select("entity_id"), on="entity_id")
    namecnt = s1all.group_by("country", "n_str").len().rename({"n_str": "n2", "len": "s1_same_name"})
    an = d.filter(pl.col("a_null")).join(namecnt, on=["country", "n2"], how="left").with_columns(pl.col("s1_same_name").fill_null(0))
    print("address-null misses: S1-name-exact-equal share", round((an["n1"] == an["n2"]).mean(), 3),
          " | #S1 in world sharing the m name: ", an.group_by(pl.col("s1_same_name").clip(upper_bound=5)).len().sort("s1_same_name").rows())
    for cty in ("US", "India"):
        print(f"\n--- {cty} address-null misses")
        for r in d.filter((pl.col("country") == cty) & pl.col("a_null")).sample(6, seed=3).select("n1", "a1", "n2").rows(): print("  ", r)
        print(f"--- {cty} misses with address")
        for r in d.filter((pl.col("country") == cty) & ~pl.col("a_null")).sample(10, seed=3).select("n1", "a1", "n2", "a2").rows(): print("  ", r)
