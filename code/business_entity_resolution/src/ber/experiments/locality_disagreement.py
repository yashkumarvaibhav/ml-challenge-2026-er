"""Diagnostic only: explicit conflicting localities learned from supplied S1s.

No geographic knowledge or downloaded locality lists. Canonical three-component
addresses whose first part is a numbered street supply a middle-component
vocabulary. Compare whole components, conservatively allowing close spellings.
Requires structured_disagreement's frozen accepted-pair caches.
"""
import json
import re
import unicodedata
from functools import lru_cache

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from ..io import load_train_pairs
from ..metrics import per_s1_f05
from ..progress import Progress, step
from .structured_disagreement import DEST


def clean(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def vocabulary(split):
    s1 = pl.read_parquet(C.CACHE / f"{split}_source1.parquet", columns=["country", "business_address"])
    vocab = {}
    for c, address in s1.iter_rows():
        p = [clean(x) for x in (address or "").split(",")]
        if len(p) == 3 and re.match(r"^\d+ .*[a-z]", p[0]) and len(p[1]) >= 3 and not re.search(r"\d", p[1]):
            vocab.setdefault(c, set()).add(p[1])
    return vocab


def add_flags(d, vocab, label):
    @lru_cache(maxsize=200_000)
    def localities(country, address):
        return set(clean(p) for p in (address or "").split(",")) & vocab.get(country, set())

    progress = Progress(f"locality {label}", d.height, unit="pairs")
    flags, counts = [], []
    for i, (c, a, b) in enumerate(d.select("country", "a1", "a2").iter_rows()):
        p, q = localities(c, a), localities(c, b)
        conflict = bool(p and q and not p & q and max(fuzz.ratio(x, y) for x in p for y in q) < 70)
        flags.append(conflict)
        counts.append(bool(p and q))
        if (i + 1) % 10_000 == 0:
            progress.update(10_000)
    progress.update(d.height % 10_000)
    progress.done()
    return d.with_columns(pl.Series("locality_conflict", flags), pl.Series("both_localities", counts))


def main():
    rep = {"geography_source": "only the supplied S1 address components", "france_labels": False}
    sp = pl.read_parquet(C.CACHE / "splits/train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    truth = load_train_pairs().select("s1", "m").join(hold.select("s1"), on="s1")
    vocabs = {}
    for label, split in (("w20", "train"), ("w0", "train"), ("france", "test")):
        step("locality", f"load {label}")
        if split not in vocabs:
            vocabs[split] = vocabulary(split)
        d = add_flags(pl.read_parquet(DEST / f"{label}.parquet"), vocabs[split], label)
        x = d.filter(pl.col("locality_conflict"))
        counts = d.group_by("s1").len().rename({"len": "P"})
        x.write_parquet(DEST / f"{label}_locality_conflicts.parquet")
        out = {"accepted": d.height, "conflicts": x.height, "both_localities": int(d["both_localities"].sum()),
               "vocabulary_sizes": {k: len(v) for k, v in vocabs[split].items()}}
        if split == "train":
            base = per_s1_f05(d.select("s1", "m"), truth, hold)
            new = per_s1_f05(d.filter(~pl.col("locality_conflict")).select("s1", "m"), truth, hold)
            out["true_removed"] = x.join(truth, on=["s1", "m"]).height
            out["delta"] = float(new["f05"].mean() - base["f05"].mean())
            out["by_country"] = {c: {"pairs": v.height, "true": v.join(truth, on=["s1", "m"]).height}
                                 for (c,), v in x.partition_by("country", as_dict=True).items()}
        else:
            hi = x.filter(pl.col("p") >= .95).join(counts, on="s1")
            out["k1_share_p_ge095"] = float((hi["P"] == 1).mean()) if hi.height else None
            z = x.group_by("s1").len().rename({"len": "D"}).join(counts, on="s1")
            out["optimistic_public_gain_bound"] = float((z["D"] / (1.25 * (z["P"] - z["D"]) + z["D"])).sum() * .15 / 259452)
        rep[label] = out
        (C.EXPERIMENTS / "v6/locality_disagreement.json").write_text(json.dumps(rep, indent=2))
    step("locality", "DONE; diagnostic only, no predictions changed")
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
