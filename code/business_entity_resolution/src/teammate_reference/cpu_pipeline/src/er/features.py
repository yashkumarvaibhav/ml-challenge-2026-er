"""Pairwise features for candidate (S1, record) pairs, fully vectorized (polars + rapidfuzz cpdist).

Entity tables come from ber.normalize (n_tok, a_tok, a_nums, flags). IDF is computed per country from the
split's own S1 names/addresses (label-free).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

W = 8
LEGAL = {"inc", "llc", "ltd", "pvt", "private", "limited", "corp", "co", "corporation", "company", "pc", "pllc",
         "llp", "lp", "sarl", "sas", "sa", "sasu", "eurl", "ei", "snc", "sci", "the", "and", "of", "de", "du", "des",
         "la", "le", "les", "et", "cie", "l", "d"}

ENT_COLS = ["entity_id", "n_tok", "a_tok", "a_nums", "a_null", "n_native", "n_domain", "n_allcaps", "n_dba",
            "a_ncomma", "n_str", "a_str"]


NORM_COLS = ENT_COLS + ["country"]


def entity_table(df: pl.DataFrame) -> pl.DataFrame:
    """Adds compact name, initials, core tokens (legal/stop words removed)."""
    return df.select(ENT_COLS + ["country"]).with_columns(
        pl.col("n_tok").list.join("").alias("n_cmp"),
        pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL)))).alias("n_core"),
    ).with_columns(
        pl.col("n_core").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("n_ini"),
        pl.col("n_core").list.join(" ").alias("n_core_str"),
        pl.col("a_tok").list.unique().alias("a_set"),
        # address key ~ house number + longest non-numeric word (street name); null when no number
        pl.col("a_tok").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("_alpha"),
    ).with_columns(
        pl.concat_str([pl.col("a_nums").list.first(),
                       pl.col("_alpha").list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True)).list.first()],
                      separator="_").alias("akey"),
    ).drop("_alpha")


def add_crowd(E: pl.DataFrame, E1: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Ambiguity counts relative to the S1 population: S1s sharing this entity's core name, compact name, address key."""
    out = E
    for key, name in (("n_core_str", "core_cnt"), ("n_cmp", "cmp_cnt"), ("akey", "akey_cnt")):
        cnt = E1.filter(pl.col(key).is_not_null() & (pl.col(key) != "")).group_by("country", key).len().rename({"len": name + suffix})
        out = out.join(cnt, on=["country", key], how="left").with_columns(pl.col(name + suffix).fill_null(0).cast(pl.Int32))
    return out


def idf_tables(s1: pl.DataFrame) -> dict:
    """Per-country idf of name tokens and address tokens over S1 (log N/df)."""
    out = {}
    for kind, col in (("n", "n_tok"), ("a", "a_tok")):
        t = s1.select("country", pl.col("entity_id"), pl.col(col).list.unique().alias("t")).explode("t").drop_nulls()
        n = s1.group_by("country").len().rename({"len": "N"})
        out[kind] = t.group_by("country", "t").len().join(n, on="country").select(
            "country", "t", (pl.col("N") / pl.col("len")).log().cast(pl.Float32).alias("idf"))
    return out


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32, **kw)


def _overlap(pairs: pl.DataFrame, c1: str, c2: str, idf: pl.DataFrame, pre: str) -> pl.DataFrame:
    """IDF-weighted overlap features between token lists c1 (S1 side) and c2 (record side). pairs has pid, country."""
    a = pairs.select("pid", "country", pl.col(c1).list.unique().alias("t")).explode("t").drop_nulls()
    b = pairs.select("pid", "country", pl.col(c2).list.unique().alias("t")).explode("t").drop_nulls()
    a = a.join(idf, on=["country", "t"], how="left").with_columns(pl.col("idf").fill_null(12.0))
    b = b.join(idf, on=["country", "t"], how="left").with_columns(pl.col("idf").fill_null(12.0))
    a = a.join(b.select("pid", "t", pl.lit(True).alias("hit")), on=["pid", "t"], how="left").with_columns(pl.col("hit").fill_null(False))
    b = b.join(a.select("pid", "t", pl.lit(True).alias("hit")), on=["pid", "t"], how="left").with_columns(pl.col("hit").fill_null(False))
    ga = a.group_by("pid").agg(
        pl.col("idf").sum().alias(f"{pre}_idf1"),
        pl.col("idf").filter(pl.col("hit")).sum().alias(f"{pre}_shared"),
        pl.col("idf").filter(~pl.col("hit")).max().alias(f"{pre}_miss1_max"),
        (~pl.col("hit")).sum().alias(f"{pre}_nmiss1"),
        pl.len().alias(f"{pre}_n1"))
    gb = b.group_by("pid").agg(
        pl.col("idf").sum().alias(f"{pre}_idf2"),
        pl.col("idf").filter(~pl.col("hit")).max().alias(f"{pre}_miss2_max"),
        (~pl.col("hit")).sum().alias(f"{pre}_nmiss2"),
        pl.len().alias(f"{pre}_n2"))
    out = pairs.select("pid").join(ga, on="pid", how="left").join(gb, on="pid", how="left")
    return out.with_columns(
        (pl.col(f"{pre}_shared") / pl.col(f"{pre}_idf1")).alias(f"{pre}_cov1"),
        (pl.col(f"{pre}_shared") / pl.col(f"{pre}_idf2")).alias(f"{pre}_cov2"),
    )


def _soft_miss(pairs: pl.DataFrame, c1: str, c2: str, pre: str) -> pl.DataFrame:
    """For S1 tokens absent from the record: best Jaro-Winkler to any record token (typo vs real difference),
    and the reverse. Reported as the minimum over unmatched tokens (1.0 when nothing is unmatched)."""
    out = pairs.select("pid")
    for x, y, tag in ((c1, c2, "1"), (c2, c1, "2")):
        miss = pairs.select("pid", pl.col(x).list.set_difference(pl.col(y)).alias("u"), pl.col(y).alias("o")) \
            .explode("u").drop_nulls("u").explode("o").drop_nulls("o")
        if miss.height:
            miss = miss.with_columns(pl.Series("jw", _cp(miss["u"].to_list(), miss["o"].to_list(), JaroWinkler.normalized_similarity)))
            g = miss.group_by("pid", "u").agg(pl.col("jw").max()).group_by("pid").agg(pl.col("jw").min().alias(f"{pre}_soft{tag}"))
            out = out.join(g, on="pid", how="left")
        else:
            out = out.with_columns(pl.lit(None, pl.Float32).alias(f"{pre}_soft{tag}"))
        out = out.with_columns(pl.col(f"{pre}_soft{tag}").fill_null(1.0))
    return out


def build(pairs: pl.DataFrame, E1: pl.DataFrame, E2: pl.DataFrame, idf: dict, chunk: int = 1_500_000) -> pl.DataFrame:
    """pairs: (s1, m, [blocking cols]); E1/E2: entity tables (entity_table) for S1 / records."""
    outs = []
    for lo in range(0, pairs.height, chunk):
        outs.append(_build(pairs.slice(lo, chunk), E1, E2, idf))
    return pl.concat(outs)


def _build(pairs, E1, E2, idf):
    e1 = E1.rename({c: c + "1" for c in E1.columns if c not in ("entity_id", "country")}).rename({"entity_id": "s1"})
    e2 = E2.drop("country").rename({c: c + "2" for c in E2.columns if c not in ("entity_id", "country")}).rename({"entity_id": "m"})
    d = pairs.with_row_index("pid").join(e1, on="s1", how="left").join(e2, on="m", how="left").sort("pid")
    f = {}
    n1, n2 = d["n_str1"].to_list(), d["n_str2"].to_list()
    f["n_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["n_tsort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    f["n_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    f["n_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    f["n_jw"] = _cp(n1, n2, JaroWinkler.normalized_similarity)
    c1, c2 = d["n_core_str1"].to_list(), d["n_core_str2"].to_list()
    f["nc_ratio"] = _cp(c1, c2, fuzz.ratio)
    f["nc_tset"] = _cp(c1, c2, fuzz.token_set_ratio)
    k1, k2 = d["n_cmp1"].to_list(), d["n_cmp2"].to_list()
    f["cmp_ratio"] = _cp(k1, k2, fuzz.ratio)
    f["cmp_partial"] = _cp(k1, k2, fuzz.partial_ratio)
    f["cmp_lev"] = _cp(k1, k2, Levenshtein.distance)
    h1 = d["a_nums1"].list.first().fill_null("").to_list()
    h2 = d["a_nums2"].list.first().fill_null("").to_list()
    lev = _cp(h1, h2, Levenshtein.distance)
    miss = (d["a_nums1"].list.len() == 0) | (d["a_nums2"].list.len() == 0)
    f["num_lev"] = np.where(miss.to_numpy(), -1.0, lev).astype(np.float32)
    del h1, h2
    a1, a2 = d["a_str1"].to_list(), d["a_str2"].to_list()
    f["a_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["a_tsort"] = _cp(a1, a2, fuzz.token_sort_ratio)
    f["a_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["a_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    del n1, n2, c1, c2, k1, k2, a1, a2
    base = d.select(
        "pid", "s1", "m", *[c for c in pairs.columns if c not in ("s1", "m")],
        (pl.col("m").str.slice(0, 2) == "S3").cast(pl.Int8).alias("src3"),
        pl.col("a_null2").cast(pl.Int8).alias("a_null2"),
        pl.col("n_native2").cast(pl.Int8).alias("n_native2"),
        pl.col("n_domain2").cast(pl.Int8).alias("n_domain2"),
        pl.col("n_allcaps2").cast(pl.Int8).alias("n_allcaps2"),
        pl.col("n_dba2").cast(pl.Int8).alias("n_dba2"),
        pl.col("a_ncomma1"), pl.col("a_ncomma2"),
        pl.col("n_tok1").list.len().alias("n_len1"), pl.col("n_tok2").list.len().alias("n_len2"),
        pl.col("n_cmp1").str.len_chars().alias("cmp_len1"), pl.col("n_cmp2").str.len_chars().alias("cmp_len2"),
        (pl.col("n_ini1") == pl.col("n_cmp2")).cast(pl.Int8).alias("ini_eq"),
        (pl.col("n_cmp1") == pl.col("n_cmp2")).cast(pl.Int8).alias("cmp_eq"),
        pl.col("n_cmp1").str.contains(pl.col("n_cmp2"), literal=True).cast(pl.Int8).alias("cmp2_in_1"),
        pl.col("n_cmp2").str.contains(pl.col("n_cmp1"), literal=True).cast(pl.Int8).alias("cmp1_in_2"),
        pl.col("a_nums1").list.unique().list.len().alias("num_n1"),
        pl.col("a_nums2").list.unique().list.len().alias("num_n2"),
        pl.col("a_nums1").list.set_intersection("a_nums2").list.len().alias("num_shared"),
        (pl.col("a_nums1").list.first() == pl.col("a_nums2").list.first()).cast(pl.Int8).alias("num_first_eq"),
        pl.col("a_set1").list.set_intersection("a_set2").list.len().alias("a_shared_n"),
        pl.col("core_cnt1"), pl.col("akey_cnt1"), pl.col("core_cnt2"), pl.col("cmp_cnt2"), pl.col("akey_cnt2"),
        pl.col("rec_core_cnt1"), pl.col("rec_core_cnt2"),
        (pl.col("rec_core_cnt2") / pl.max_horizontal(pl.col("core_cnt2"), pl.lit(1))).cast(pl.Float32).alias("core_supply_ratio2"),
        (pl.col("akey1") == pl.col("akey2")).cast(pl.Int8).alias("akey_eq"),
        (pl.col("n_core_str1") == pl.col("n_core_str2")).cast(pl.Int8).alias("core_eq"),
        pl.col("n_core_str1").hash(seed=5).alias("_ch1"), pl.col("n_core_str2").hash(seed=5).alias("_ch2"),
        ((pl.col("a_nums1").list.first().cast(pl.Int64, strict=False) - pl.col("a_nums2").list.first().cast(pl.Int64, strict=False)).abs()
         / pl.max_horizontal(pl.col("a_nums1").list.first().cast(pl.Int64, strict=False), pl.col("a_nums2").list.first().cast(pl.Int64, strict=False), pl.lit(1)))
        .cast(pl.Float32).alias("num_reldiff"),
    ).with_columns(
        (pl.col("num_n1") - pl.col("num_shared")).alias("num_miss1"),
        (pl.col("num_n2") - pl.col("num_shared")).alias("num_miss2"),
    )
    base = base.with_columns([pl.Series(k, v) for k, v in f.items()])
    ov_n = _overlap(d, "n_tok1", "n_tok2", idf["n"], "n")
    ov_a = _overlap(d, "a_tok1", "a_tok2", idf["a"], "a")
    sm = _soft_miss(d, "n_core1", "n_core2", "n")
    return base.join(ov_n, on="pid", how="left").join(ov_a, on="pid", how="left").join(sm, on="pid", how="left").drop("pid")
