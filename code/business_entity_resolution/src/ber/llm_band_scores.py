"""LLM pair-classifier scores on the stage-2 upper band (0.95 < p <= 0.995), used by stage 3 v4 (ber.llm_stage3.test_v4).
Holdout pairs = union of the world-0 and world-20 holdout bands; test pairs = all countries. Chunked and resumable.

  BER_LLM_TAG=qwen7b_r3  BER_LLM_BASE=Qwen2.5-7B   python -m ber.llm_band_scores
  BER_LLM_TAG=qwen15_r3  BER_LLM_BASE=Qwen2.5-1.5B python -m ber.llm_band_scores
"""
import os

import polars as pl

from . import config as C
from .llm_ce import _load, score
from .progress import step

LO, HI = 0.95, 0.995


def main():
    S = C.WORK / "s2"
    tag = os.environ["BER_LLM_TAG"]
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
    hp = pl.concat([pl.read_parquet(S / f"holdout_pred_w{w}_e5base2.parquet").join(hold, on="s1")
                    .filter(pl.col("p").is_between(LO, HI, closed="right")).select("s1", "m") for w in (0, 20)]).unique()
    tp = pl.read_parquet(S / "test_pred_e5base2.parquet").filter(pl.col("p").is_between(LO, HI, closed="right")).select("s1", "m")
    tok, model = _load(False)
    for name, pairs, split in (("holdout", hp, "train"), ("test", tp, "test")):
        path = S / f"llm_hiband_{name}_{tag}.parquet"
        done = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema={"s1": pl.Utf8, "m": pl.Utf8, "llm": pl.Float32})
        todo = pairs.join(done.select("s1", "m"), on=["s1", "m"], how="anti")
        for i in range(0, todo.height, 100_000):
            part = todo.slice(i, 100_000)
            done = pl.concat([done, part.with_columns(pl.Series("llm", score(part, split, name=f"{tag} upper band {name}", model=model, tok=tok)).cast(pl.Float32))])
            done.write_parquet(path)
        step("llm", f"upper band {name}: {done.height:,} pairs scored -> {path.name}")


if __name__ == "__main__":
    main()
