"""B2 v4(j) patch: exact FULL normalized name (sorted words, nothing removed) + address number key.

Motivation (label-free, plan-discussion 03:12 UTC): the (d)/(f)/(h) keys use the *core* name
(frequent words removed); French names are mostly frequent words, so their core is often empty.
(j): key = sorted full name + "#" + number (all-digit address token); keys shared by <= KJ_KEYCAP
pool records; per S1 rank by (#shared keys, IDF sum of shared non-numeric address words); top KJ.
New blocking feature r_j (model retrain required). Trialled alone, per reviewer 03:22 UTC.
"""
p = "code/business_entity_resolution/src/ber/blocking.py"
s = open(p).read()
s = s.replace('''  (h) name+house number   :''', '''  (j) full name+number    : full sorted name (no frequent-word removal) + one address number as
                            an exact key (keys shared by <= KJ_KEYCAP pool records), top KJ
  (h) name+house number   :''')
s = s.replace('''KH, KH_KEYCAP = 10, 50''', '''KH, KH_KEYCAP = 10, 50
KJ, KJ_KEYCAP = 10, 50''')
s = s.replace('''    cand = (cand.join(rev, on=["qrow", "prow"], how="full", coalesce=True)''', '''    # (j) exact full name (sorted words, nothing removed) + address number
    def fn_keys(d):
        return (d.select("row", pl.col("n_tok").list.unique().list.sort().list.join(" ").alias("fn"),
                         pl.col("a_tok").list.unique().alias("t"))
                .filter(pl.col("fn") != "").explode("t")
                .filter(pl.col("t").is_not_null() & pl.col("t").str.contains(r"^[0-9]+$"))
                .select("row", (pl.col("fn") + "#" + pl.col("t")).alias("k")))

    jk = fn_keys(pool)
    jk = jk.join(jk.group_by("k").len().filter(pl.col("len") <= KJ_KEYCAP).select("k"), on="k")
    jpairs = (fn_keys(q).rename({"row": "qrow"}).join(jk.rename({"row": "prow"}), on="k")
              .group_by("qrow", "prow").agg(pl.len().alias("n_j")))
    if jpairs.height:
        qa = q.select(pl.col("row").alias("qrow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        pa = pool.select(pl.col("row").alias("prow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        jshared = (jpairs.select("qrow", "prow").join(qa, on="qrow").join(pa, on=["prow", "t"])
                   .filter(~pl.col("t").str.contains(r"^[0-9]+$")).join(aidf, on="t", how="left")
                   .group_by("qrow", "prow").agg(pl.col("w").fill_null(0).sum().alias("w_j")))
        jch = (jpairs.join(jshared, on=["qrow", "prow"], how="left").with_columns(pl.col("w_j").fill_null(0))
               .sort(["qrow", "n_j", "w_j"], descending=[False, True, True])
               .with_columns(pl.int_range(1, pl.len() + 1).over("qrow").cast(pl.Int16).alias("r_j"))
               .filter(pl.col("r_j") <= KJ).select("qrow", "prow", "r_j"))
    else:
        jch = pl.DataFrame(schema={"qrow": pl.Int64, "prow": pl.Int64, "r_j": pl.Int16})

    cand = (cand.join(rev, on=["qrow", "prow"], how="full", coalesce=True)''')
s = s.replace('''            .join(hch, on=["qrow", "prow"], how="full", coalesce=True))''', '''            .join(hch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(jch, on=["qrow", "prow"], how="full", coalesce=True))''')
s = s.replace('''            pl.col("r_h").cast(pl.Int32).fill_null(big),
        ).alias("prio"),''', '''            pl.col("r_h").cast(pl.Int32).fill_null(big),
            pl.col("r_j").cast(pl.Int32).fill_null(big),
        ).alias("prio"),''')
s = s.replace('''"in_d", "r_f", "r_g", "r_h", "prio"))''', '''"in_d", "r_f", "r_g", "r_h", "r_j", "prio"))''')
open(p, "w").write(s)
p = "code/business_entity_resolution/src/ber/features.py"
s = open(p).read()
s = s.replace('''"r_f", "r_g", "r_h", "prio",
    # context''', '''"r_f", "r_g", "r_h", "r_j", "prio",
    # context''')
s = s.replace('''"r_f", "r_g", "r_h", "prio"]''', '''"r_f", "r_g", "r_h", "r_j", "prio"]''')
open(p, "w").write(s)
print("patched v4j")
