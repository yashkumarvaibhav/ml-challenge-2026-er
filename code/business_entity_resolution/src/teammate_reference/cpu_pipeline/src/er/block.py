"""Candidate generation by rare-feature inverted-index joins (CPU, polars).

Per country, every S1 and every S2/S3 record gets a bag of hashed features:
  n: name word keys            a: address word keys
  b: adjacent address bigrams  g: name character 4-grams (compact name; robust to typos / domains)
Only features shared by few S1s (df_s1 <= CAP_S1) generate pairs. A pair's score is the sum of
idf = log(N_s1 / df_s1) over its shared generating features (per kind weights). The candidate set is
the union of each record's top R S1s and each S1's top K records.
"""
import math
import os

import polars as pl

CAP_S1 = int(os.environ.get("ER_CAP_S1", 30))
R_TOP = int(os.environ.get("ER_R_TOP", 3))
K_TOP = int(os.environ.get("ER_K_TOP", 8))
KIND_W = {"n": 1.0, "a": 0.6, "b": 1.2, "g": 0.35, "p": 1.0, "k": 1.5, "x": 1.2, "q": 1.5, "y": 1.0}
LEGAL = ["inc", "llc", "ltd", "pvt", "private", "limited", "corp", "co", "corporation", "company", "pc", "pllc", "llp", "lp",
         "sarl", "sas", "sa", "sasu", "eurl", "ei", "snc", "sci", "the", "and", "of", "de", "du", "des", "la", "le", "les",
         "et", "cie", "l", "d",
         # filler / prefix words the generator adds to record names (train pairs: added >> present in S1 names)
         "center", "centre", "services", "service", "partners", "id", "formerly", "dba", "nee", "aka", "m", "s", "sri",
         "shri", "shrii", "smt", "mr", "dr", "ms", "corporation", "company", "corp",
         # France fillers (team analysis of accepted France pairs)
         "france", "fils", "associes", "developpement", "groupe", "distribution"]
GRAM, MAXLEN = 4, 32


KIND_CODE = {k: i for i, k in enumerate(KIND_W)}
KIND_WV = {i: KIND_W[k] for k, i in KIND_CODE.items()}


def _h(df: pl.DataFrame, kind: str) -> pl.DataFrame:
    """(i, t:str) -> (i, f:u64, kind:u8); hashing right away keeps memory small."""
    return df.select("i", (pl.lit(kind + ":") + pl.col("t")).hash(seed=7).alias("f"), pl.lit(KIND_CODE[kind], pl.UInt8).alias("kind"))


def features(df: pl.DataFrame) -> pl.DataFrame:
    """df: (i, n_tok, a_tok) -> long (i, f:u64, kind:u8) unique per (i, f)."""
    parts = [
        _h(df.select("i", pl.col("n_tok").alias("t")).explode("t").drop_nulls(), "n"),
        _h(df.select("i", pl.col("a_tok").alias("t")).explode("t").drop_nulls(), "a"),
        _h(df.select("i", pl.col("a_tok").list.eval(pl.element() + "_" + pl.element().shift(-1))
                     .list.slice(0, pl.max_horizontal(pl.col("a_tok").list.len() - 1, 0)).alias("t")).explode("t").drop_nulls(), "b"),
    ]
    core = df.select("i", pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(LEGAL) & (pl.element().str.len_chars() >= 2))).list.unique().list.sort().list.head(5).alias("c"),
                     pl.col("a_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.unique().list.head(3).alias("nums"))
    parts.append(_h(core.filter(pl.col("c").list.len() >= 1).select("i", pl.col("c").list.join("|").alias("t")), "k"))
    parts.append(_h(core.select("i", pl.col("c").alias("c1"), pl.col("c").alias("c2")).explode("c1").explode("c2")
                    .filter(pl.col("c1") < pl.col("c2")).select("i", (pl.col("c1") + "|" + pl.col("c2")).alias("t")), "p"))
    parts.append(_h(core.select("i", "c", "nums").explode("c").explode("nums").drop_nulls()
                    .select("i", (pl.col("c") + "|" + pl.col("nums")).alias("t")), "x"))
    # full core-name key combined with each address word (>= 3 chars) or number: rare even when the name is common
    parts.append(_h(core.filter(pl.col("c").list.len() >= 1).select("i", pl.col("c").list.join("|").alias("key"))
                    .join(df.select("i", pl.col("a_tok").list.unique().alias("w")), on="i").explode("w")
                    .filter((pl.col("w").str.len_chars() >= 3) | pl.col("w").str.contains(r"^\d+$"))
                    .select("i", (pl.col("key") + "#" + pl.col("w")).alias("t")), "q"))
    # address number x address word (>= 4 letters): partial / reordered addresses, records with made-up names
    parts.append(_h(core.select("i", pl.col("nums").list.head(2).alias("nm")).join(
        df.select("i", pl.col("a_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{4,}$"))).list.unique().alias("w")), on="i")
        .explode("nm").explode("w").drop_nulls().select("i", (pl.col("nm") + "@" + pl.col("w")).alias("t")), "y"))
    del core
    comp = df.select("i", pl.col("n_tok").list.join("").str.slice(0, MAXLEN).alias("c"))
    for lo in range(0, MAXLEN - GRAM + 1, 9):
        g = comp.select("i", pl.concat_list([pl.col("c").str.slice(j, GRAM) for j in range(lo, min(lo + 9, MAXLEN - GRAM + 1))]).alias("t")) \
            .explode("t").filter(pl.col("t").str.len_chars() == GRAM)
        parts.append(_h(g, "g"))
    return pl.concat(parts).unique(["i", "f"])


def block_country(s1: pl.DataFrame, rec: pl.DataFrame, chunk: int = 250_000, log=print) -> pl.DataFrame:
    """s1, rec: normalized frames with (entity_id, n_tok, a_tok). Returns (s1, m, bscore, r_rank, s_rank)."""
    s1 = s1.with_row_index("i")
    rec = rec.with_row_index("i")
    fs = features(s1)
    n_s1 = s1.height
    df = fs.group_by("f").agg(pl.len().alias("df"), pl.col("kind").first())
    df = df.filter(pl.col("df") <= CAP_S1).with_columns(
        (pl.col("df").cast(pl.Float64).pow(-1) * n_s1).log().alias("idf"))
    df = df.with_columns((pl.col("idf") * pl.col("kind").replace_strict(KIND_WV, return_dtype=pl.Float64)).alias("w"))
    fs = fs.join(df.select("f", pl.col("w").cast(pl.Float32)), on="f").select(pl.col("i").alias("si"), "f", "w")
    del df
    parts = []
    for lo in range(0, rec.height, chunk):
        fr = features(rec.slice(lo, chunk)).select(pl.col("i").alias("ri"), "f")
        pr = fr.join(fs, on="f").group_by("ri", "si").agg(pl.col("w").sum().cast(pl.Float32).alias("bscore"))
        # record-side top R
        pr = pr.with_columns(pl.col("bscore").rank("ordinal", descending=True).over("ri").alias("r_rank"))
        parts.append(pr.filter(pl.col("r_rank") <= 2 * R_TOP + 2).with_columns(pl.col("r_rank").cast(pl.Int32)))  # enough for S1-side top K
        log(f"  block chunk {lo // chunk + 1}/{math.ceil(rec.height / chunk)}: {pr.height:,} scored pairs")
    p = pl.concat(parts)
    p = p.with_columns(pl.col("bscore").rank("ordinal", descending=True).over("si").alias("s_rank"))
    p = p.filter((pl.col("r_rank") <= R_TOP) | (pl.col("s_rank") <= K_TOP))
    return p.join(s1.select(pl.col("i").alias("si"), pl.col("entity_id").alias("s1")), on="si") \
        .join(rec.select(pl.col("i").alias("ri"), pl.col("entity_id").alias("m")), on="ri") \
        .select("s1", "m", "bscore", pl.col("r_rank").cast(pl.Int32), pl.col("s_rank").cast(pl.Int32))
