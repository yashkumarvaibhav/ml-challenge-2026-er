"""Re-score France's stage-1 band with a France-adapted CE (BER_CE=e5base2fr); US/India keep their e5base2 scores.

Writes cache/v6/s2/ce_test_e5base2_frce.parquet (s1, m, ce) for ber.experiments.france_veto_s2 --ce-file.
  BER_CE=e5base2fr python -m ber.experiments.france_ce_rescore
"""
import polars as pl

from .. import config as C
from ..cross_encoder import score

S2 = C.WORK / "s2"


def main():
    base = pl.read_parquet(S2 / "ce_test_e5base2.parquet")
    fr = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
    frb = base.join(fr, on="s1").select("s1", "m")
    new = frb.with_columns(pl.Series("ce", score(frb, "test", name="ce score France band (e5base2fr)")))
    out = pl.concat([base.join(fr, on="s1", how="anti"), new.with_columns(pl.col("ce").cast(base["ce"].dtype))])
    assert out.height == base.height
    out.write_parquet(S2 / "ce_test_e5base2_frce.parquet")
    old = base.join(fr, on="s1").rename({"ce": "ce_old"}).join(new, on=["s1", "m"])
    print(f"France band pairs {new.height:,}; mean CE old {old['ce_old'].mean():.4f} new {old['ce'].mean():.4f}; "
          f"corr {old.select(pl.corr('ce_old', 'ce')).item():.3f}; >0.5 old {(old['ce_old'] > 0.5).mean():.3f} new {(old['ce'] > 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
