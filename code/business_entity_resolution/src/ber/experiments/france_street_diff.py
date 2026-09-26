"""Two France-only removal rules (26 Sep; subagent hunt report experiments/v6/subagent_france_hunt.md):

1. SAME NAME, COMPLETELY DIFFERENT STREET: an accepted France pair whose street words (numbers, street types,
   function words, saint/st removed; abbreviations expanded) share no fuzzy match (ratio >= 70) between the S1 and
   the record. France names are templated and repeat across a city; a same-name business on another street passes
   stage 2 on name + house number. The rule fires on 0.006% (US) / 0.031% (India) of test accepted pairs and on
   ~0.005%/0.04% of holdout TRUE pairs (those are address-parsing artefacts), i.e. the generator does not move a true
   record to another street; France: 0.71% of accepted pairs. Label-free check: among p >= 0.95 pairs, the share whose
   S1 has only that one accepted record is 5.8% here vs 2.1% for clean France pairs and 6.2% for the category swaps
   the public board confirmed false (+0.003063).
2. EXTENDED CATEGORY SWAP: the category-swap rule (france_category_swap) with dotted legal forms collapsed
   ("S.A.S.U." -> "sasu"), function words dropped (category-first templates "College de la X" -> "Comite de X"),
   and 'federation' added to the category words (substituted 62x, added 11x).

  python -m ber.experiments.france_street_diff              # size both rules on the post-veto France decisions
  python -m ber.experiments.france_street_diff --write BASE  # output/ = BASE minus both rule sets (France only)
"""
import argparse
import json
import re
import shutil
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from .. import config as C
from ..model import decide

AB = {'r': 'rue', 'av': 'avenue', 'ave': 'avenue', 'bd': 'boulevard', 'blvd': 'boulevard', 'all': 'allee', 'ch': 'chemin', 'che': 'chemin',
      'pl': 'place', 'imp': 'impasse', 'rte': 'route', 'crs': 'cours', 'sq': 'square', 'q': 'quai', 'psg': 'passage',
      'st': 'street', 'rd': 'road', 'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'hwy': 'highway', 'pkwy': 'parkway', 'n': 'north',
      's': 'south', 'e': 'east', 'w': 'west', 'cir': 'circle', 'ter': 'terrace', 'trl': 'trail', 'sr': 'state road'}
TYPES = {'rue', 'avenue', 'boulevard', 'allee', 'chemin', 'place', 'impasse', 'route', 'cours', 'square', 'quai', 'passage', 'street', 'road',
         'drive', 'lane', 'court', 'residence', 'res', 'cite', 'voie', 'sentier', 'promenade', 'esplanade', 'rond', 'point', 'parvis', 'hameau',
         'lieu', 'dit', 'boulenard', 'aevnue', 'avenu', 'bld', 'bvd', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w',
         'pass', 'psg', 'pas', 'chem', 'cour', 'lotissement', 'lot', 'zone', 'za', 'zi', 'parc'}
ORD = re.compile(r'^\d+(st|nd|rd|th)$')
ORDW = {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth',
        'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth'}
PRE = re.compile(r'^(n[o°º]?\.?|#|\(|\)|-|\s)+', re.I)
NUMST = re.compile(r'^(\d{1,4})\s*[a-z]?\d*\b\s*(bis|ter)?\s*(.*)$')
STOPW = {'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'of', 'the', 'bis', 'ter', 'b', 'saint', 'sainte', 'st', 'ste', 'residence', 'res',
         'appartement', 'appt', 'apt', 'bat', 'batiment', 'porte', 'entree', 'etage', 'unit', 'suite', 'fl', 'floor', 'pmb'}
STT = {'rue', 'r', 'avenue', 'av', 'ave', 'boulevard', 'bd', 'bld', 'allee', 'all', 'chemin', 'ch', 'chem', 'place', 'pl', 'impasse', 'imp',
       'route', 'rte', 'cours', 'crs', 'quai', 'passage', 'pass', 'psg', 'square', 'sq', 'street', 'road', 'rd', 'drive', 'dr', 'lane', 'ln',
       'way', 'court', 'ct'}


def strip_acc(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def _clean(p):
    s = strip_acc(p).lower().replace("’", " ").replace("'", " ").strip()
    s = PRE.sub('', s)
    return re.sub(r'[()#]', ' ', s).strip()


def _sw(rest):
    rest = re.sub(r'[^a-z0-9 ]', ' ', rest)
    w = [AB.get(t, t) for t in rest.split()]
    return [t for t in w if t not in TYPES and not ORD.match(t) and t not in ORDW and not any(c.isdigit() for c in t) and t not in STOPW and len(t) >= 3]


def _parts(a):
    out = []
    if a is None:
        return out
    for p in a.split(','):
        s = _clean(p)
        m = NUMST.match(s)
        if m and re.search(r'[a-z]{3,}', m.group(3)):
            w = _sw(m.group(3))
            if w:
                out.append((str(int(m.group(1))), w))
        else:
            tt = re.sub(r'[^a-z ]', ' ', s).split()
            if tt and tt[0] in STT:
                w = _sw(s)
                if w:
                    out.append((None, w))
    return out


def street_diff(a, b) -> bool:
    P, Q = _parts(a), _parts(b)
    if not P or not Q:
        return False
    same = any(fuzz.ratio(x, y) >= 70 for _, wp in P for _, wq in Q for x in wp for y in wq) or \
        any(fuzz.ratio(' '.join(wp), ' '.join(wq)) >= 70 for _, wp in P for _, wq in Q)
    return not same


LEG = set("inc llc ltd pvt private limited corp co corporation company pc pllc llp lp sarl sas sa sasu eurl ei snc sci".split())
FUNC = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "the", "of"}


def _toks2(s):
    if s is None:
        return []
    s = strip_acc(s).lower()
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b\.?)", "", s)  # S.A.S.U. -> sasu.
    s = s.replace(".", "")
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in LEG and t not in FUNC]


def swap2(n1, n2, cats) -> bool:
    t1, t2 = _toks2(n1), _toks2(n2)
    if not t1 or not t2:
        return False
    dr = [a for a in t1 if max(fuzz.ratio(a, b) for b in t2) < 70]
    ad = [b for b in t2 if max(fuzz.ratio(a, b) for a in t1) < 70]
    return len(dr) == 1 and len(ad) == 1 and len(dr[0]) >= 4 and ad[0] in cats


def _france_accepted():
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(
        pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{s}.parquet") for s in (2, 3)]).select(
        pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    return t1, rec


def flags(acc: pl.DataFrame) -> pl.DataFrame:
    cats = set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"]) | {"federation"}
    return acc.with_columns(pl.Series("street_diff", [street_diff(a, b) for a, b in zip(acc["a1"], acc["a2"])]),
                            pl.Series("swap2", [swap2(a, b, cats) for a, b in zip(acc["n1"], acc["n2"])]))


def main():
    t1, rec = _france_accepted()
    p = pl.read_parquet(C.WORK / "s2" / "test_pred_e5base2_frveto.parquet").join(t1.select("s1"), on="s1")
    acc = decide(p, 0.4, 0.75).join(t1, on="s1").join(rec, on="m")
    f = flags(acc)
    rep = {"accepted": f.height, "street_diff": int(f["street_diff"].sum()), "swap2": int(f["swap2"].sum()),
           "union": int((f["street_diff"] | f["swap2"]).sum())}
    print(json.dumps(rep))
    (C.EXP_RUN / "france_street_diff.json").write_text(json.dumps(rep, indent=1))


def write(base: str, only: str = "both"):
    arch = C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    t1, rec = _france_accepted()
    pairs = (m.with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "").rename({c1: "s1", c2: "m"}))
    fr = pairs.join(t1, on="s1").join(rec, on="m")
    f = flags(fr)
    cond = {"both": pl.col("street_diff") | pl.col("swap2"), "street": pl.col("street_diff"), "swap": pl.col("swap2")}[only]
    drop = f.filter(cond).select("s1", "m")
    kept = pairs.join(drop, on=["s1", "m"], how="anti")
    g = kept.group_by("s1").agg(pl.col("m").sort().str.join(",").alias(c2)).rename({"s1": c1})
    out = m.select(c1).join(g, on=c1, how="left").with_columns(pl.col(c2).fill_null(""))
    assert out[c1].to_list() == m[c1].to_list()
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    print(f"{base}: France street_diff {int(f['street_diff'].sum()):,}, swap2 {int(f['swap2'].sum()):,}, removed {drop.height:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--only", choices=["both", "street", "swap"], default="both")
    a = ap.parse_args()
    write(a.write, a.only) if a.write else main()
