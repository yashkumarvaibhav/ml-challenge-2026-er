"""France-only post-processing on any matching_results.tsv (ours or the team's); US/India rows keep the same ID sets.

1. House-number veto (er.frnum): drop France pairs whose house numbers (1-4 digits) are disjoint, unless our model
   scores the pair p >= 0.999 (--scores). Pairs our model never scored, or every pair when --scores is omitted,
   get no protection.
2. Filler rescue (--rescue, needs --scores): accept a not-yet-matched France record for its best S1 (p >= 0.05) when
   the names become identical only once the French generator filler words (fils, associes, france, groupe, ...) are
   removed besides legal forms / articles, both sides carry the same house-number set, and address coverage
   a_cov1 >= 0.7. Only pairs inside the base's candidate file are added, so an external base (e.g. the team's sub-8)
   keeps its matches within its own candidates.

  python -m er.france_post --base output/run2/matching_results.tsv --out output/run2_fr [--scores cache/p22_test.parquet]
      [--cand-file output/run2/candidate_pairs.tsv] [--feats feats2] [--rescue]
  python -m er.france_post --base ../team_files/matching_results.tsv --cand-file ../team_files/candidate_pairs.tsv \
      --scores cache/p2b_test.parquet --rescue --out output/sub8_fr
"""
import argparse
import os
import shutil
from pathlib import Path

import polars as pl

from er import frnum
from er.common import CACHE, ROOT, log

BASIC_STOP = ["inc", "llc", "ltd", "sarl", "sas", "sa", "sasu", "eurl", "ei", "snc", "sci", "the", "and", "of", "de", "du", "des",
              "la", "le", "les", "et", "cie", "l", "d"]
FILLER = ["fils", "freres", "associes", "developpement", "groupe", "distribution", "france", "services", "service"]
FR_STOP = BASIC_STOP + FILLER


def read_pairs(path, col=None) -> pl.DataFrame:
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    return m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "") \
        .rename({c1: "s1", c2: "m"}).select("s1", "m")


def rescue(acc: pl.DataFrame, scores: pl.DataFrame, feats_dir: str, cand: pl.DataFrame = None) -> pl.DataFrame:
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    sc = scores.join(cty.filter(pl.col("country") == "France"), on="s1").select("s1", "m", "p")
    if cand is not None:  # external base: only rescue pairs inside its own candidate set
        sc = sc.join(cand, on=["s1", "m"], how="semi")
    best = sc.filter(pl.col("p") == pl.col("p").max().over("m"))
    pool = best.join(acc, on=["s1", "m"], how="anti").join(acc.select("m").unique(), on="m", how="anti").filter(pl.col("p") >= 0.05)

    def core(sources, ids, tag):
        c = pl.concat([pl.scan_parquet(CACHE / "norm_final" / f"test_s{s}.parquet").select("entity_id", "n_tok") for s in sources]) \
            .join(ids.lazy(), on="entity_id", how="semi").collect().select(
                "entity_id",
                pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(FR_STOP))).list.unique().list.sort().alias(f"c{tag}"),
                pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(BASIC_STOP))).list.unique().list.sort().alias(f"b{tag}"))
        return c.join(frnum.house_numbers("test", ids, sources).rename({"hn": f"h{tag}"}), on="entity_id")

    c1 = core((1,), pool.select(pl.col("s1").alias("entity_id")).unique(), 1).rename({"entity_id": "s1"})
    c2 = core((2, 3), pool.select(pl.col("m").alias("entity_id")).unique(), 2).rename({"entity_id": "m"})
    f = pl.concat([pl.read_parquet(CACHE / feats_dir / "test" / f"part{k}.parquet", columns=["s1", "m", "a_cov1"]).join(pool.select("s1", "m"), on=["s1", "m"])
                   for k in range(16)])
    d = pool.join(c1, on="s1").join(c2, on="m").join(f, on=["s1", "m"])
    # names equal only once the French filler words are removed (a pair that is already equal without them was rejected by
    # the model for other reasons: on US/India holdout such rescues are 11-33% precise), identical house-number sets
    # (distractors copy the name and change one number component), and the S1's address mostly covered.
    d = d.filter((pl.col("c1") == pl.col("c2")) & (pl.col("c1").list.len() > 0) & (pl.col("b1") != pl.col("b2"))
                 & (pl.col("h1").list.len() > 0) & (pl.col("h1").list.sort() == pl.col("h2").list.sort()) & (pl.col("a_cov1") >= 0.7))
    log(f"France filler rescue: {d.height:,} pairs ({d['s1'].n_unique():,} S1s)")
    return d.select("s1", "m")


def main(base, out, scores_path=None, cand_file=None, feats_dir="feats2", do_rescue=False):
    import subprocess
    import sys
    from er.write import write_pairs
    base, out = Path(base), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    acc = read_pairs(base)
    fr = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France") \
        .select(pl.col("entity_id").alias("s1"))
    if scores_path:
        scores = pl.scan_parquet(scores_path).select("s1", "m", "p").join(fr.lazy(), on="s1", how="semi").collect()
    else:
        scores = acc.join(fr, on="s1", how="semi").with_columns(pl.lit(0.0).alias("p"))
    kept, dropped = frnum.veto(acc, scores)
    if do_rescue and scores_path:
        cand = read_pairs(cand_file).join(fr, on="s1", how="semi") if cand_file else None
        kept = pl.concat([kept, rescue(kept, scores, feats_dir, cand)]).unique()
    s1 = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    write_pairs(s1, kept, "matched_entity_ids", out / "matching_results.tsv")
    cand_file = Path(cand_file) if cand_file else base.parent / "candidate_pairs.tsv"
    dst = out / "candidate_pairs.tsv"
    if not (dst.exists() and dst.samefile(cand_file)):
        dst.unlink(missing_ok=True)
        try:
            os.link(cand_file, dst)  # identical file: a hard link saves ~450 MB of disk
        except OSError:
            shutil.copyfile(cand_file, dst)
    dropped.select("s1", "m").write_parquet(out / "france_numveto_dropped.parquet")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")], capture_output=True, text=True)
    log(f"france_post: {base} -> {out}: {acc.height:,} -> {kept.height:,} pairs; validator: {r.stdout.strip()[-120:]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores")
    ap.add_argument("--cand-file")
    ap.add_argument("--feats", default="feats2")
    ap.add_argument("--rescue", action="store_true")
    a = ap.parse_args()
    main(a.base, a.out, a.scores, a.cand_file, a.feats, a.rescue)
