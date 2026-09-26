"""Synthetic, labelled France world built from the REAL France test S1s (FAQ: synthetic pairs from provided records).

Why (26 Sep): France has no labels and its distractors follow a France-specific mechanism the US/India labels never
show (sibling businesses: same stem and address, another category word; public +0.003063 when removed). Here every
real France S1 becomes an entity; its records are produced with the noise the generator applies in US/India
(catalogued from true pairs and from real France records) and the France distractor types are added explicitly.
The result is a labelled split `synfr` that runs through the unchanged v6 pipeline:
  cache/synfr_source{1,2,3}.parquet, cache/synfr_truth.parquet (s1, m)

Entities: all 259,452 real France S1s; GHOST_FRAC of them are left out of the S1 file (their records stay).
Distractor entities (records never in the truth):
  sibling  : same address, one category word swapped            (SIB_FRAC of entities with a category word)
  namesake : same name, another address in the same city        (NAME_FRAC)
  stranger : templated stem + category + legal, address in city  (NEW_FRAC)
Per-entity record counts (n2, n3) are drawn from the train truth distribution.

  python -m ber.synth_france [--seed 7]
"""
import argparse
import json
import random
import re
import unicodedata

import polars as pl

from . import config as C

GHOST_FRAC, SIB_FRAC, NAME_FRAC, NEW_FRAC = 0.20, 0.25, 0.10, 0.15
LEGAL = {"sarl": ["SARL", "S.A.R.L.", "Sarl", "(SARL)", "[SARL]", "5ARL", "Sàrl", "S.A.R.L"],
         "sas": ["SAS", "S.A.S.", "Sas", "(SAS)", "5AS", "S.A.S"], "sasu": ["SASU", "S.A.S.U.", "Sasu", "5ASU"],
         "sa": ["SA", "S.A.", "Sa"], "eurl": ["EURL", "E.U.R.L.", "Eurl"], "ei": ["EI", "E.I.", "(EI)"],
         "sci": ["SCI", "S.C.I.", "5CI"], "snc": ["SNC", "S.N.C."]}
FILL_SUB = ["Services", "France", "Développement", "Groupe", "Fils", "Associés"]
FILL_ADD = ["& Fils", "Et Fils", "& Associés", "+ Associés", "France", "Développement", "Groupe", "Services",
            "Enterprises", "Trading", "(France)", "Et-Fils", "Cie", "& Cie"]
SYL = ["flux", "vera", "pyra", "rax", "gild", "brix", "quo", "novi", "jax", "lyra", "zeph", "umbra", "ira", "dova",
       "kor", "syn", "halo", "delta", "vio", "ecto", "arc", "orbi", "iri", "calo", "xylo", "onyx", "tavo", "cira",
       "nex", "lum", "drex", "evo", "zeta", "wex", "vantage", "mira", "belo", "fay", "kelo", "aria", "sol", "avi"]
WRAP = ["{p} doing business as {n}", "{p} a/k/a {n}", "{p} formerly: {n}", "{p} DBA: {n}", "{n} (formerly {p})"]
ACC = {"a": "àâ", "e": "éèêë", "i": "ïî", "o": "ôö", "u": "ûù", "c": "ç"}
LEET = {"o": "0", "l": "1", "s": "5", "i": "1", "e": "3"}
STREET_ABBR = {"rue": ["R", "R.", "RUE", "Rue"], "avenue": ["AV", "Av.", "AVE", "Av", "AV."], "boulevard": ["BD", "Bd.", "Blvd", "BD."],
               "allée": ["ALL", "All.", "ALLEE", "Allee"], "allee": ["ALL", "All."], "impasse": ["IMP", "Imp."], "place": ["PL", "Pl.", "PL."],
               "chemin": ["CH", "Ch.", "CHEM"], "route": ["RTE", "Rte", "Rte."], "quai": ["Q", "QU."], "cours": ["CRS", "Crs"], "square": ["SQ", "Sq."]}
NUM_FMT = ["N° {n}", "No. {n}", "No {n}", "Nº {n}", "#{n}", "# {n}", "({n})", "{n} -", "0{n}", "00{n}", "N°{n}", "NO. {n}"]
DEPT = {"lille": "Nord", "tourcoing": "Nord", "roubaix": "Nord", "dunkerque": "Nord", "calais": "Pas-de-Calais",
        "bordeaux": "Gironde", "pessac": "Gironde", "mérignac": "Gironde", "merignac": "Gironde", "la teste-de-buch": "Gironde",
        "lège-cap-ferret": "Gironde", "nantes": "Loire-Atlantique", "saint-nazaire": "Loire-Atlantique", "pornic": "Loire-Atlantique",
        "saint-herblain": "Loire-Atlantique", "la baule-escoublac": "Loire-Atlantique"}
REGIONS = {"hauts-de-france", "nouvelle-aquitaine", "pays de la loire"}


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def capwords(s):
    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" "))


class Gen:
    def __init__(self, seed, cats, cat_w):
        self.r = random.Random(seed)
        self.cats, self.cat_w = cats, cat_w

    def ch(self, p):
        return self.r.random() < p

    # ---------------- names ----------------
    def pseudo(self):
        w = "".join(self.r.choice(SYL) for _ in range(self.r.choice((2, 2, 3))))
        w = w.capitalize()
        return w + self.r.choice(["", "", "", " One", " Sys", "+"])

    def typo(self, w):
        if len(w) < 4:
            return w
        i = self.r.randrange(1, len(w) - 1)
        op = self.r.random()
        if op < 0.3:
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if op < 0.55:
            return w[:i] + w[i + 1:]
        if op < 0.8:
            return w[:i] + self.r.choice("aeioulnrst") + w[i:]
        return w[:i] + self.r.choice("aeioulnrst") + w[i + 1:]

    def name(self, n):
        toks = n.split()
        legal_idx = [i for i, t in enumerate(toks) if strip_acc(t).lower().strip("()[]").replace(".", "") in LEGAL]
        core = [t for i, t in enumerate(toks) if i not in legal_idx]
        if not core:
            core = toks[:]
        if self.ch(0.075):  # whole-name replacement
            x = self.r.random()
            if x < 0.33:
                return self.pseudo()
            if x < 0.58:
                ini = "".join(strip_acc(t)[0].upper() for t in core if t[0].isalnum())
                return ini if len(ini) >= 2 else self.pseudo()
            if x < 0.85:
                base = "".join(strip_acc(t).lower() for t in toks if t.isalnum() or t.replace(".", "").isalnum())
                if self.ch(0.3):
                    base = "".join(c for c in "".join(t.lower() for t in toks) if c.isascii() and c.isalnum())
                sty = self.r.random()
                return (base + ".com") if sty < 0.6 else ("#" + base[:18]) if sty < 0.8 else ("@" + base[:18].upper())
            return self.r.choice(WRAP).format(p=self.pseudo(), n=n)
        out = toks[:]
        if legal_idx:
            li = legal_idx[0]
            key = strip_acc(out[li]).lower().strip("()[]").replace(".", "")
            x = self.r.random()
            if x < 0.25:
                out.pop(li)
            elif x < 0.5:
                out[li] = self.r.choice(LEGAL[key])
            elif x < 0.62:
                t = out.pop(li)
                out.insert(0, t)
            elif x < 0.64:
                out[li] = self.r.choice(LEGAL[self.r.choice(list(LEGAL))])
        elif self.ch(0.05):
            out.append(self.r.choice(LEGAL[self.r.choice(["sarl", "sas", "sa", "eurl"])]))
        cidx = [i for i, t in enumerate(out) if strip_acc(t).lower().strip("()[]").replace(".", "") not in LEGAL]
        if len(cidx) >= 2 and self.ch(0.08):
            out.pop(self.r.choice(cidx))
        cidx = [i for i, t in enumerate(out) if strip_acc(t).lower().strip("()[]").replace(".", "") not in LEGAL]
        if len(cidx) >= 2 and self.ch(0.03):
            out[self.r.choice(cidx)] = self.r.choice(FILL_SUB)
        if self.ch(0.045):
            out.append(self.r.choice(FILL_ADD))
        if len(out) >= 2 and self.ch(0.03):
            i = self.r.randrange(len(out) - 1)
            out[i], out[i + 1] = out[i + 1], out[i]
        if self.ch(0.2):
            i = self.r.randrange(len(out))
            out[i] = self.typo(out[i])
        if self.ch(0.06):
            i = self.r.randrange(len(out))
            w = out[i]
            pos = [j for j, c in enumerate(w) if c.lower() in ACC]
            if pos:
                j = self.r.choice(pos)
                rep = self.r.choice(ACC[w[j].lower()])
                out[i] = w[:j] + (rep.upper() if w[j].isupper() else rep) + w[j + 1:]
        if self.ch(0.02):
            i = self.r.randrange(len(out))
            w = out[i]
            pos = [j for j, c in enumerate(w) if c.lower() in LEET]
            if pos:
                j = self.r.choice(pos)
                out[i] = w[:j] + LEET[w[j].lower()] + w[j + 1:]
        if len(out) >= 2 and self.ch(0.03):
            i = self.r.randrange(len(out) - 1)
            out[i:i + 2] = [out[i] + self.r.choice(["-", "-", "–"]) + out[i + 1]]
        if self.ch(0.02):
            i = self.r.randrange(len(out))
            out[i] = self.r.choice(["[{}]", "({})"]).format(out[i])
        s = (" " * 2 if self.ch(0.05) else " ").join(out)
        x = self.r.random()
        if x < 0.25:
            s = s.upper()
        elif x < 0.33:
            s = s.lower()
        if self.ch(0.015):
            s = self.r.choice(["-- ", "<< ", ">> ", "@"]) + s
        return s

    # ---------------- addresses ----------------
    def address(self, a, src, city_streets):
        if a is None or self.ch(0.03):
            return None
        comps = [c.strip() for c in a.split(",") if c.strip()]
        city = next((c for c in comps if c.lower() in DEPT), None)
        out = []
        for c in comps:
            lc = c.lower()
            if lc in REGIONS:
                x = self.r.random()
                if x < 0.2:
                    continue
                if x < 0.5 and city:
                    c = DEPT[city.lower()]
                out.append(("REG", c))
            elif c is city:
                if self.ch(0.03):
                    continue
                out.append(("CITY", c))
            elif re.search(r"\d", c) or any(w in lc.split() for w in STREET_ABBR):
                out.append(("ST", self.street(c, city, city_streets)))
            else:
                out.append(("OTHER", c))
        if not out:
            return None
        if self.ch(0.2):
            self.r.shuffle(out)
        up = self.r.random() < (0.7 if src == 2 else 0.1)
        res = []
        for kind, c in out:
            if kind == "REG":
                res.append(c)
            elif up:
                res.append(c.upper())
            else:
                res.append(capwords(c) if (src == 3 or self.ch(0.5)) else c)
        return ", ".join(res)

    def street(self, c, city, city_streets):
        if city and self.ch(0.006) and city_streets.get(city.lower()):
            nums = re.findall(r"^\s*(\d+\s*(?:bis|ter|[a-z])?)\b", c, flags=re.I)
            other = self.r.choice(city_streets[city.lower()])
            c = (nums[0] + " " if nums else "") + re.sub(r"^\s*\d+\s*(bis|ter|[a-z])?\b\s*", "", other, flags=re.I)
        toks = c.split()
        for i, t in enumerate(toks):
            k = strip_acc(t).lower()
            if k in STREET_ABBR and self.ch(0.5):
                toks[i] = self.r.choice(STREET_ABBR[k])
        m = re.match(r"^(\d+)(.*)$", toks[0]) if toks else None
        if m:
            n = int(m.group(1))
            x = self.r.random()
            if x < 0.06:
                toks = toks[1:] if len(toks) > 1 else toks
            else:
                if x < 0.065:
                    n = max(1, n + self.r.choice((-1, 1)))
                elif x < 0.067:
                    n = max(1, n + self.r.randint(2, 9))
                if self.ch(0.25):
                    toks[0] = self.r.choice(NUM_FMT).format(n=n) + m.group(2)
                else:
                    toks[0] = str(n) + m.group(2)
        if len(toks) > 1 and self.ch(0.18):
            i = self.r.randrange(1, len(toks))
            toks[i] = self.typo(toks[i])
        return " ".join(toks)


def main(seed: int = 7):
    rnd = random.Random(seed)
    s1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France")
    cats = json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"]
    # category-word surface forms as they appear in S1 names (accents/case), with frequencies
    surf = {}
    for n in s1["business_name"].to_list():
        for t in n.split():
            k = strip_acc(t).lower()
            if k in cats:
                surf.setdefault(k, {}).setdefault(t, 0)
                surf[k][t] += 1
    cat_forms = {k: max(v, key=v.get) for k, v in surf.items()}
    cat_keys = list(cat_forms)
    cat_w = [sum(surf[k].values()) for k in cat_keys]
    g = Gen(seed, cat_keys, cat_w)
    # streets per city (for namesake/stranger addresses and street replacement)
    city_addr = {}
    for a in s1["business_address"].drop_nulls().to_list():
        comps = [c.strip() for c in a.split(",")]
        city = next((c for c in comps if c.lower() in DEPT), None)
        if city:
            city_addr.setdefault(city.lower(), []).append(a)
    city_streets = {k: [c.strip() for a in v for c in a.split(",") if re.search(r"\d", c)] for k, v in city_addr.items()}
    gt = pl.read_parquet(C.CACHE / "train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
    counts = [(sum(1 for x in l if x.startswith("S2")), sum(1 for x in l if x.startswith("S3"))) for l in gt["matched_entity_ids"].to_list()]

    ents = []  # (entity_id, name, address, kind)
    rows = s1.select("entity_id", "business_name", "business_address").to_dicts()
    for i, r in enumerate(rows):
        eid = f"SF1-{i}"
        ents.append((eid, r["business_name"], r["business_address"], "s1"))
    n_s1 = len(ents)
    extra = []
    for eid, n, a, _ in ents:
        toks = n.split()
        ci = [j for j, t in enumerate(toks) if strip_acc(t).lower() in cat_forms]
        if ci and rnd.random() < SIB_FRAC:
            j = rnd.choice(ci)
            k = strip_acc(toks[j]).lower()
            new = rnd.choices(cat_keys, weights=cat_w)[0]
            while new == k:
                new = rnd.choices(cat_keys, weights=cat_w)[0]
            t2 = toks[:]
            t2[j] = cat_forms[new]
            extra.append((f"SFX-sib-{eid}", " ".join(t2), a, "sibling"))
        city = next((c.strip().lower() for c in (a or "").split(",") if c.strip().lower() in DEPT), None)
        if city and rnd.random() < NAME_FRAC:
            extra.append((f"SFX-nam-{eid}", n, rnd.choice(city_addr[city]), "namesake"))
        if city and rnd.random() < NEW_FRAC:
            stem = rnd.choice(rows)["business_name"].split()[0]
            leg = rnd.choice(["SARL", "SAS", "EURL", "SA", ""])
            extra.append((f"SFX-new-{eid}", " ".join(x for x in (stem, cat_forms[rnd.choices(cat_keys, weights=cat_w)[0]], leg) if x),
                          rnd.choice(city_addr[city]), "stranger"))
    ents += extra
    ghost = set(rnd.sample(range(n_s1), int(GHOST_FRAC * n_s1)))
    recs2, recs3, truth = [], [], []
    c2 = c3 = 0
    for idx, (eid, n, a, kind) in enumerate(ents):
        n2, n3 = rnd.choice(counts)
        for src, nn in ((2, n2), (3, n3)):
            for _ in range(nn):
                if src == 2:
                    mid = f"SF2-{c2}"
                    c2 += 1
                else:
                    mid = f"SF3-{c3}"
                    c3 += 1
                rec = {"entity_id": mid, "business_name": g.name(n), "business_address": g.address(a, src, city_streets), "country": "France"}
                (recs2 if src == 2 else recs3).append(rec)
                if kind == "s1" and idx not in ghost:
                    truth.append((eid, mid))
    s1_out = pl.DataFrame([{"entity_id": e, "business_name": n, "business_address": a, "country": "France"}
                           for i, (e, n, a, k) in enumerate(ents[:n_s1]) if i not in ghost])
    shuf = lambda rs: pl.DataFrame(rs).sample(fraction=1.0, shuffle=True, seed=seed)
    s1_out.sample(fraction=1.0, shuffle=True, seed=seed).write_parquet(C.CACHE / "synfr_source1.parquet")
    shuf(recs2).write_parquet(C.CACHE / "synfr_source2.parquet")
    shuf(recs3).write_parquet(C.CACHE / "synfr_source3.parquet")
    pl.DataFrame(truth, schema=["s1", "m"], orient="row").write_parquet(C.CACHE / "synfr_truth.parquet")
    kinds = {k: sum(1 for e in ents if e[3] == k) for k in ("s1", "sibling", "namesake", "stranger")}
    print(f"synfr: S1 {s1_out.height:,} (ghosts {len(ghost):,}); entities {kinds}; records S2 {len(recs2):,} S3 {len(recs3):,} "
          f"({(len(recs2) + len(recs3)) / s1_out.height:.2f}/S1); true pairs {len(truth):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args().seed)
