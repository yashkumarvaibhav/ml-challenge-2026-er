"""Blend the team's v6 stage-2 scores (e5base2, sub-8's model) with ours: z = w*logit(p_team) + (1-w)*logit(p_ours) over the
union of both candidate universes (a pair missing from one universe gets P_MISS there).

  python -m er.teamblend eval                # two-way holdout: team alone vs blends (w grid) x decision
  python -m er.teamblend test --w 0.7 ...    # -> output/teamblend (France: category veto + house-number veto)
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import polars as pl

from ber.metrics import per_s1_f05, summarize
from er import frnum
from er.common import CACHE, ROOT, hold_s1, log, truth_pairs
from er.efdecide import ef_decide
from er.final import blend as our_blend
from er.model import decide
from er.post import category_veto

TEAM = Path(os.environ.get("ER_TEAM_PREDS", ROOT.parent / "e5base2_preds"))   # holdout_pred_w20_e5base2.parquet, test_pred_e5base2.parquet
P_MISS = 1e-3


def _logit(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


MODE = "team_universe"   # pairs outside our candidates keep the team score alone; ours-only pairs are ignored


def combine(team: pl.DataFrame, ours: pl.DataFrame, w: float, mode: str = None) -> pl.DataFrame:
    """Where both models scored a pair: w*logit(team) + (1-w)*logit(ours); where only the team did: the team score.
    mode "union" also keeps pairs only we scored, with our score alone."""
    mode = mode or MODE
    how = "full" if mode == "union" else "left"
    d = team.rename({"p": "pt"}).join(ours.rename({"p": "po"}), on=["s1", "m"], how=how, coalesce=True)
    d = d.with_columns(pl.coalesce("pt", "po").alias("pt"), pl.coalesce("po", "pt").alias("po"))
    z = w * _logit("pt") + (1 - w) * _logit("po")
    return d.select("s1", "m", (1 / (1 + (-z).exp())).alias("p"))


def _decide(sc, kind, prm):
    return decide(sc, *prm, "p") if kind == "thr" else ef_decide(sc, *prm)


GRIDS = {"thr": list(itertools.product([0.4, 0.5, 0.6, 0.7], [0.7, 0.75, 0.8]))}


def evaluate():
    hold = hold_s1()
    truth = truth_pairs().join(hold.select("s1"), on="s1")
    team = pl.read_parquet(TEAM / "holdout_pred_w20_e5base2.parquet").select("s1", "m", "p").join(hold.select("s1"), on="s1")
    ours = our_blend(["p22_train_holdout.parquet", "p23b_train_holdout.parquet"], hold.select("s1").lazy())
    log(f"teamblend eval: team holdout {team.height:,} pairs on {team['s1'].n_unique():,} S1 (ours {ours['s1'].n_unique():,} S1)")
    res = []
    for w, mode in ((0.8, "team_universe"), (0.75, "team_universe"), (0.65, "team_universe"), (0.7, "union")):
        sc = combine(team, ours, w, mode)
        for kind, grid in GRIDS.items():
            two = []
            for th, rh in (("A", "B"), ("B", "A")):
                ht, hr = hold.filter(pl.col("holdout_half") == th), hold.filter(pl.col("holdout_half") == rh)
                st, sr = sc.join(ht.select("s1"), on="s1"), sc.join(hr.select("s1"), on="s1")
                f, prm = max((summarize(per_s1_f05(_decide(st, kind, g), truth, ht))["macro_f05"], g) for g in grid)
                two.append((summarize(per_s1_f05(_decide(sr, kind, prm), truth, hr))["macro_f05"], prm))
            full = max((summarize(per_s1_f05(_decide(sc, kind, g), truth, hold))["macro_f05"], g) for g in grid)
            r = {"w": w, "mode": mode, "decision": kind, "two_way": two, "mean": round((two[0][0] + two[1][0]) / 2, 5), "full": full}
            res.append(r)
            log("teamblend eval " + json.dumps(r))
            print(json.dumps(r), flush=True)
    (CACHE.parent / "logs" / "teamblend_eval.json").write_text(json.dumps(res, indent=1))


def test(w, kind, prm, tag):
    from er.write import write_pairs
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    team_all = pl.scan_parquet(TEAM / "test_pred_e5base2.parquet").select("s1", "m", "p")
    preds, fr_sc = [], None
    for c in ("US", "India", "France"):
        ids = cty.filter(pl.col("country") == c).select("s1").lazy()
        team = team_all.join(ids, on="s1", how="semi").collect()
        files = [f"p22_test{'fr' if c == 'France' else ''}.parquet", f"p23b_test{'fr' if c == 'France' else ''}.parquet"]
        sc = combine(team, our_blend(files, ids), w)
        preds.append(_decide(sc, kind, prm))
        log(f"teamblend test {c}: {sc.height:,} pairs -> {preds[-1].height:,} accepted")
        if c == "France":
            fr_sc = sc
    pred = pl.concat(preds)
    kept, _, _ = category_veto(pred, "test")
    kept, dropped = frnum.veto(kept, fr_sc)
    out = CACHE.parent / "output" / tag
    out.mkdir(parents=True, exist_ok=True)
    s1 = cty.select("s1")
    write_pairs(s1, kept, "matched_entity_ids", out / "matching_results.tsv")
    (out / "meta.json").write_text(json.dumps({"w": w, "decision": kind, "params": prm, "numveto_dropped": dropped.height}))
    log(f"teamblend test -> {out}: {kept.height:,} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eval", "test"])
    ap.add_argument("--w", type=float, default=0.7)
    ap.add_argument("--decision", default="thr")
    ap.add_argument("--params", default="0.4,0.75")
    ap.add_argument("--tag", default="teamblend")
    a = ap.parse_args()
    evaluate() if a.step == "eval" else test(a.w, a.decision, tuple(float(x) for x in a.params.split(",")), a.tag)
