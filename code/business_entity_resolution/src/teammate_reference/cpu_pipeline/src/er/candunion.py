"""Candidate file for a submission that came without one: our candidate set plus every pair of its matching file that
our candidates miss (so matches stay a subset of candidates; streaming, low memory).

  python -m er.candunion --matching ../team_files/matching_results.tsv --out output/team_cand/candidate_pairs.tsv
"""
import argparse
from pathlib import Path

from er.common import CACHE, log

OURS = CACHE.parent / "output" / "final" / "candidate_pairs.tsv"


def main(matching, out, ours=OURS):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m = {}
    with open(matching) as f:
        next(f)
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            if v:
                m[k] = v.split(",")
    added = 0
    with open(ours) as f, open(out, "w") as g:
        g.write(next(f))
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            ids = v.split(",") if v else []
            have = set(ids)
            extra = [x for x in m.get(k, []) if x and x not in have]
            added += len(extra)
            g.write(k + "\t" + ",".join(ids + extra) + "\n")
    log(f"candunion: {out} = ours + {added:,} match pairs missing from our candidates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.matching, a.out)
