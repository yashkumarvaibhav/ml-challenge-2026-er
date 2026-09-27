"""Learn value mappings from **training-portion** positive pairs only (PLAN.md B1, V5).

Maps (all canonicalize toward the S1 form, since S1 is the clean reference source):
* native_name : native-script word key -> Latin word. Positional alignment of a native
                S2/S3 name with its S1 name when both have the same number of words.
* addr_comp   : per country, whole address component -> S1 component
                (e.g. "mh" -> "maharashtra", "ohio" -> "oh", native state -> English).
* addr_tok    : per country, address token -> S1 token (e.g. "ave" -> "avenue").
* name_tok    : per country, name token -> S1 token (e.g. "pvt" -> "private").

A mapping v -> c is kept only with repeated agreement: count(v,c) >= MIN_SUPPORT,
count(v,c) / n(v) >= MIN_SHARE (n(v) = pairs whose S2/S3 side contains v) and
count(v,c) >= DOMINANCE x the runner-up. Precision/coverage are then measured on
holdout pairs (non-ghost holdout S1s) and written to experiments/b1_maps.json.
"""
import json
import time
from collections import Counter, defaultdict

import polars as pl

from . import config as C
from .io import load_source, load_train_pairs
from .progress import Progress, step
from rapidfuzz.distance import JaroWinkler

from .text import DBA, INDIC, is_abbrev, word_key, words

MIN_SUPPORT, MIN_SHARE, DOMINANCE = 20, 0.5, 2.0
PAIR_SAMPLE = 1_500_000  # per country, for the token-level alias statistics


def _name_main(raw: str) -> str:
    s = (raw or "").split("|", 1)[0]
    parts = DBA.split(s, maxsplit=1)
    return parts[-1]


def plausible_token_map(v: str, c: str) -> bool:
    """Word-level maps must look like a spelling variant or an abbreviation of the target.

    Rejects collapses of a specific word into an unrelated/generic one (e.g. a city
    name -> "of"), which would discard information on both sides of every pair.
    """
    digits_v, digits_c = v.rstrip("abcdefghijklmnopqrstuvwxyz"), c.rstrip("abcdefghijklmnopqrstuvwxyz")
    if digits_v and digits_v.isdigit() and digits_v == digits_c:  # ordinal fixes: "55nd" -> "55th"
        return True
    return JaroWinkler.normalized_similarity(v, c) >= 0.75 or is_abbrev(v, c)


def _select(counts: dict, nv: dict, plausible=None) -> dict:
    """counts: {v: Counter(c -> n)}; returns accepted {v: c}."""
    out = {}
    for v, cnt in counts.items():
        top = cnt.most_common(2)
        c, n = top[0]
        second = top[1][1] if len(top) > 1 else 0
        if n >= MIN_SUPPORT and n / max(1, nv[v]) >= MIN_SHARE and n >= DOMINANCE * second and c != v \
                and (plausible is None or plausible(v, c)):
            out[v] = c
    # resolve chains, drop cycles
    final = {}
    for v, c in out.items():
        seen = {v}
        while c in out and c not in seen:
            seen.add(c)
            c = out[c]
        if c not in seen:
            final[v] = c
    return final


def _diff_counts(df: pl.DataFrame, a: str, b: str) -> tuple:
    """Per-country (v in b - a, c in a - b) co-occurrence counts and n(v)."""
    df = df.with_columns(
        pl.col(b).list.unique().alias("_b"),
        pl.col(b).list.set_difference(pl.col(a)).alias("_mo"),
        pl.col(a).list.set_difference(pl.col(b)).alias("_so"),
    )
    nv = df.select("country", pl.col("_b").alias("v")).explode("v").drop_nulls().group_by("country", "v").len()
    combos = (
        df.select("country", pl.col("_mo").alias("v"), pl.col("_so").alias("c"))
        .explode("v").explode("c").drop_nulls()
        .group_by("country", "v", "c").len()
        .filter(pl.col("len") >= MIN_SUPPORT)
    )
    return combos, nv


def _maps_from(combos: pl.DataFrame, nv: pl.DataFrame, exclude_digits: bool, plausible=None) -> dict:
    out = {}
    for (cty,), g in combos.group_by("country"):
        counts = defaultdict(Counter)
        for v, c, n in g.select("v", "c", "len").iter_rows():
            if exclude_digits and (v.isdigit() or c.isdigit()):
                continue
            counts[v][c] = n
        nvd = dict(nv.filter(pl.col("country") == cty).select("v", "len").iter_rows())
        out[cty] = _select(counts, nvd, plausible)
    return out


def _apply_comp(comps, cmap):
    return [cmap.get(k, k) for k in comps]


def learn():
    t0 = time.time()
    step("learn_maps", "START (training-portion pairs only)")
    split = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    pairs = load_train_pairs()
    train_s1 = split.filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1"))
    hold_s1 = split.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
    raw = {s: pl.read_parquet(C.CACHE / "norm_raw" / f"train_s{s}.parquet") for s in C.SOURCES}
    r1 = raw[1].select(pl.col("entity_id").alias("s1"), pl.col("a_comp").alias("c1"), pl.col("n_tok").alias("n1"))
    r23 = pl.concat([raw[2], raw[3]]).select(
        pl.col("entity_id").alias("m"), "country", pl.col("a_comp").alias("cm"), pl.col("n_tok").alias("nm"))
    src1 = load_source("train", 1).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("raw1"))
    src23 = pl.concat([load_source("train", 2), load_source("train", 3)]).select(
        pl.col("entity_id").alias("m"), pl.col("business_name").alias("rawm"))

    def with_data(p):
        return p.join(r1, on="s1").join(r23, on="m")

    tp = with_data(pairs.join(train_s1, on="s1"))
    hp = with_data(pairs.join(hold_s1, on="s1"))
    report = {"n_train_pairs": tp.height, "n_holdout_pairs": hp.height}

    # 1. native-script name dictionary (positional alignment, training portion)
    nat = (pairs.join(train_s1, on="s1").join(src1, on="s1").join(src23, on="m")
           .filter(pl.col("rawm").str.contains(INDIC.pattern)))
    counts, nv, aligned_pairs = defaultdict(Counter), Counter(), 0
    prog = Progress("learn_maps native alignment", nat.height, unit="pairs")
    for raw1, rawm in nat.select("raw1", "rawm").iter_rows():
        prog.update()
        mw = [(word_key(w), bool(INDIC.search(w))) for w in words(_name_main(rawm))]
        mw = [(k, n) for k, n in mw if k]
        sw = [k for k in (word_key(w) for w in words(_name_main(raw1))) if k]
        if len(mw) != len(sw):
            continue
        aligned_pairs += 1
        for (k, isn), c in zip(mw, sw):
            if isn:
                counts[k][c] += 1
                nv[k] += 1
    prog.done()
    native = _select(counts, nv)
    report["native"] = {"train_pairs_with_native": nat.height, "aligned_pairs": aligned_pairs,
                        "native_keys_seen": len(nv), "mapped_keys": len(native)}
    print(f"native dict: {len(native)}/{len(nv)} keys ({time.time() - t0:.0f}s)", flush=True)

    # 2. address component aliases (per country)
    step("learn_maps", "address component aliases ...")
    combos, nvc = _diff_counts(tp.select("country", "c1", "cm"), "c1", "cm")
    comp = _maps_from(combos, nvc, exclude_digits=False)
    print(f"comp aliases: { {k: len(v) for k, v in comp.items()} } ({time.time() - t0:.0f}s)", flush=True)

    # 3. token aliases (address after component mapping; names after native mapping)
    samp = pl.concat([g.sample(min(PAIR_SAMPLE, g.height), seed=C.SEED) for _, g in tp.group_by("country")])
    rows = []
    prog = Progress("learn_maps token-alias prep", samp.height, unit="pairs")
    for cty, c1, cm, n1, nm in samp.select("country", "c1", "cm", "n1", "nm").iter_rows():
        prog.update()
        cmap = comp.get(cty, {})
        t1 = [t for k in _apply_comp(c1, cmap) for t in k.split(" ")]
        tm = [t for k in _apply_comp(cm, cmap) for t in k.split(" ")]
        rows.append((cty, t1, tm, n1, [native.get(t, t) for t in nm]))
    prog.done()
    step("learn_maps", "token alias statistics ...")
    tokdf = pl.DataFrame(rows, schema=["country", "t1", "tm", "n1", "nm"], orient="row")
    combos, nvt = _diff_counts(tokdf.select("country", "t1", "tm"), "t1", "tm")
    addr_tok = _maps_from(combos, nvt, exclude_digits=True, plausible=plausible_token_map)
    combos, nvn = _diff_counts(tokdf.select("country", "n1", "nm"), "n1", "nm")
    name_tok = _maps_from(combos, nvn, exclude_digits=True, plausible=plausible_token_map)
    print(f"addr tok: { {k: len(v) for k, v in addr_tok.items()} }; name tok: { {k: len(v) for k, v in name_tok.items()} } ({time.time() - t0:.0f}s)", flush=True)

    maps = {"native_name": native, "addr_comp": comp, "addr_tok": addr_tok, "name_tok": name_tok}
    (C.CACHE / "maps").mkdir(parents=True, exist_ok=True)
    (C.CACHE / "maps" / "maps.json").write_text(json.dumps(maps, ensure_ascii=False))

    # 4. holdout evaluation (maps were learned without these entities)
    step("learn_maps", "holdout evaluation ...")
    hn = (pairs.join(hold_s1, on="s1").join(src1, on="s1").join(src23, on="m")
          .filter(pl.col("rawm").str.contains(INDIC.pattern)))
    mapped = hits = native_occ = 0
    before_hit = after_hit = m_tok_total = 0
    for raw1, rawm in hn.select("raw1", "rawm").iter_rows():
        s_set = {k for k in (word_key(w) for w in words(_name_main(raw1))) if k}
        for w in words(_name_main(rawm)):
            k = word_key(w)
            if not k:
                continue
            m_tok_total += 1
            before_hit += k in s_set
            if INDIC.search(w):
                native_occ += 1
                if k in native:
                    mapped += 1
                    hits += native[k] in s_set
                    k = native[k]
            after_hit += k in s_set
    report["holdout_native"] = {
        "pairs": hn.height,
        "native_token_occurrences": native_occ,
        "coverage": round(mapped / max(1, native_occ), 4),
        "precision_mapped_in_s1_name": round(hits / max(1, mapped), 4),
        "m_token_in_s1_name_before": round(before_hit / max(1, m_tok_total), 4),
        "m_token_in_s1_name_after": round(after_hit / max(1, m_tok_total), 4),
    }
    hs = hp.sample(min(200_000, hp.height), seed=C.SEED)
    jb = ja = 0.0
    for cty, c1, cm in hs.select("country", "c1", "cm").iter_rows():
        tb1 = {t for k in c1 for t in k.split(" ")}
        tbm = {t for k in cm for t in k.split(" ")}
        cmap, tmap = comp.get(cty, {}), addr_tok.get(cty, {})
        ta1 = {tmap.get(t, t) for k in _apply_comp(c1, cmap) for t in k.split(" ")}
        tam = {tmap.get(t, t) for k in _apply_comp(cm, cmap) for t in k.split(" ")}
        jb += len(tb1 & tbm) / max(1, len(tb1 | tbm))
        ja += len(ta1 & tam) / max(1, len(ta1 | tam))
    report["holdout_address_token_jaccard"] = {"before": round(jb / hs.height, 4), "after": round(ja / hs.height, 4),
                                               "pairs": hs.height}
    report["sizes"] = {"native_name": len(native), "addr_comp": {k: len(v) for k, v in comp.items()},
                       "addr_tok": {k: len(v) for k, v in addr_tok.items()},
                       "name_tok": {k: len(v) for k, v in name_tok.items()}}
    report["examples"] = {
        "addr_comp": {k: dict(list(v.items())[:40]) for k, v in comp.items()},
        "addr_tok": {k: dict(sorted(v.items())[:60]) for k, v in addr_tok.items()},
        "name_tok": {k: dict(sorted(v.items())[:60]) for k, v in name_tok.items()},
        "native_name": dict(list(native.items())[:40]),
    }
    (C.EXPERIMENTS / "b1_maps.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    step("learn_maps", f"DONE in {time.time() - t0:.0f}s -> cache/maps/maps.json, experiments/b1_maps.json")
    print(json.dumps({k: report[k] for k in ["native", "holdout_native", "holdout_address_token_jaccard", "sizes"]}, indent=2))


if __name__ == "__main__":
    learn()
