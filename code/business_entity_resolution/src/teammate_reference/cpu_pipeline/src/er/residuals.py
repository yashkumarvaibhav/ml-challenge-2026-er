"""Where the holdout macro-F0.5 is lost: not retrieved / retrieved-but-rejected / false pairs, by record slice.

  python -m er.residuals
"""
import json

import polars as pl

from ber.metrics import per_s1_f05, summarize
from er.common import CACHE, EXP, hold_s1, truth_pairs
from er.model import decide


import os
RUN = os.environ.get("ER_RUN", "")
CANDTAG = "b2_" if os.environ.get("ER_BLOCK") == "2" else ""


def main():
    params = json.loads((CACHE.parent / f"models{RUN}" / "decision.json").read_text())
    sc = pl.read_parquet(CACHE / f"p2{RUN}_train_holdout.parquet")
    hold = hold_s1()
    truth = truth_pairs().join(hold.select("s1"), on="s1")
    pred = decide(sc, params["t_any"], params["t_add"], "p")
    cand = pl.scan_parquet(CACHE / "cand" / f"{CANDTAG}train_w20.parquet").select("s1", "m").join(hold.select("s1").lazy(), on="s1", how="semi").collect()
    ids = pl.concat([truth.select("m"), pred.select("m")]).unique().rename({"m": "entity_id"})
    flags = pl.concat([pl.scan_parquet(CACHE / "norm_final" / f"train_s{s}.parquet").select("entity_id", "a_null", "n_native", "n_domain")
                       for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect().rename({"entity_id": "m"})
    tp = truth.join(pred, on=["s1", "m"], how="semi")
    fn = truth.join(pred, on=["s1", "m"], how="anti").join(cand.with_columns(pl.lit(True).alias("retrieved")), on=["s1", "m"], how="left") \
        .with_columns(pl.col("retrieved").fill_null(False)).join(flags, on="m", how="left")
    fp = pred.join(truth, on=["s1", "m"], how="anti").join(flags, on="m", how="left")
    per = per_s1_f05(pred, truth, hold)
    rep = {"params": params, "f05": summarize(per),
           "true_pairs": truth.height, "tp": tp.height, "fn_not_retrieved": int((~fn["retrieved"]).sum()),
           "fn_rejected": int(fn["retrieved"].sum()), "fp": fp.height,
           "fn_rejected_addr_null": round(fn.filter("retrieved")["a_null"].mean(), 3),
           "fn_rejected_native": round(fn.filter("retrieved")["n_native"].mean(), 3),
           "fp_addr_null": round(fp["a_null"].mean(), 3) if fp.height else None,
           "fp_on_singletons": fp.join(truth.select("s1").unique(), on="s1", how="anti").height,
           "singleton_correct_empty": round(per.filter(pl.col("T") == 0)["f05"].mean(), 4),
           "loss_share_fn_only": round(per.filter((pl.col("TP") < pl.col("T")) & (pl.col("TP") == pl.col("P")))
                                       .select((1 - pl.col("f05")).sum()).item() / per.select((1 - pl.col("f05")).sum()).item(), 3)}
    print(json.dumps(rep, indent=1))
    (EXP / f"residuals{RUN}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
