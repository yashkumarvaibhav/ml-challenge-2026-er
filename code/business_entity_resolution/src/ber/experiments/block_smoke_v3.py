import json, time
import polars as pl
from ber import config as C
from ber.blocking import block
from ber.io import load_train_pairs
from ber.metrics import blocking_report
from ber.run_blocking import mflags

sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost"))
q = pl.concat([g.sample(10000, seed=1) for _, g in hold.group_by("country")])
world = sp.filter(~pl.col("is_ghost")).select("entity_id")
t = time.time()
cand = block("train", q.select("entity_id"), world)
print("blocking time", round(time.time() - t), "s; pairs", cand.height)
rep = blocking_report(cand, load_train_pairs().select("s1", "m"), q.select(pl.col("entity_id").alias("s1"), "country"), mflags("train"))
print(json.dumps(rep, indent=2))
# channel contribution: recall if we only had each channel
truth = load_train_pairs().select("s1", "m").join(q.select(pl.col("entity_id").alias("s1")), on="s1")
lab = truth.join(cand, on=["s1", "m"], how="left")
for ch, expr in [("a", pl.col("r_a").is_not_null()), ("b", pl.col("r_b").is_not_null()), ("c", pl.col("r_c").is_not_null()),
                 ("d", pl.col("in_d").fill_null(False)), ("e", pl.col("r_e").is_not_null()), ("f", pl.col("r_f").is_not_null()), ("g", pl.col("r_g").is_not_null()), ("h", pl.col("r_h").is_not_null())]:
    only = lab.filter(expr)
    print(ch, "recall alone", round(only.height / truth.height, 4))
cand.write_parquet(C.CACHE / "cand" / "smoke_v3.parquet")
