"""Apply B2 v2 to ber/blocking.py and ber/features.py (run AFTER sub-1 packaging).

(f) name + locality key: core name + one non-numeric address word (>= 3 chars), keys shared by
    <= KF_KEYCAP pool records; per S1 top KF by number of shared keys.
    Targets India misses: exact names with truncated, generic addresses.
(g) address-null reverse: pool records without address -> top KG world S1s by name-only IDF
    overlap with a higher df cap (DF_CAP_NREV). Targets address-null records with typo'd names.
New blocking features r_f, r_g (so the model must be retrained).
"""
p = "code/business_entity_resolution/src/ber/blocking.py"
s = open(p).read()

s = s.replace('''  (e) reverse  pool -> S1 : each S2/S3 record's top KE S1s by combined score
''', '''  (e) reverse  pool -> S1 : each S2/S3 record's top KE S1s by combined score
  (f) name+locality       : core name + one address word as an exact key (keys shared by
                            <= KF_KEYCAP pool records), top KF by shared-key count
  (g) null-addr reverse   : address-null pool records -> top KG S1s by name-only overlap
                            with a higher df cap (DF_CAP_NREV)
''')
s = s.replace('''KA, KB, KC, KD, KE = 15, 15, 30, 30, 3
CAP = 60
DF_CAP = 5000''', '''KA, KB, KC, KD, KE = 15, 15, 30, 30, 3
KF, KF_KEYCAP = 20, 50
KG, DF_CAP_NREV = 10, 30_000
CAP = 70
DF_CAP = 5000''')
s = s.replace('''def _rev_chunk(bounds):
    lo, hi = bounds
    S = (_G["RQ"][lo:hi] @ _G["S1T"]).tocsr()
    r, c, v, rk = _topk_rows(S, KE)''', '''def _rev_chunk(bounds):
    lo, hi = bounds
    S = (_G["RQ"][lo:hi] @ _G["S1T"]).tocsr()
    r, c, v, rk = _topk_rows(S, _G.get("K_REV", KE))''')
s = s.replace('''    pt = record_tokens(pool)
    df = pt.group_by("tok").agg(pl.len().alias("df"), pl.col("kind").first())
    df = df.filter(pl.col("df") <= DF_CAP).with_row_index("tid").with_columns(''', '''    pt = record_tokens(pool)
    df_all = pt.group_by("tok").agg(pl.len().alias("df"), pl.col("kind").first())
    df = df_all.filter(pl.col("df") <= DF_CAP).with_row_index("tid").with_columns(''')
# (g) after the reverse channel
s = s.replace('''    for k in ("RQ", "S1T"):
        _G.pop(k, None)

    # (d) core-name key equality''', '''    for k in ("RQ", "S1T"):
        _G.pop(k, None)

    # (g) address-null pool records -> top KG S1s by name-only IDF overlap (higher df cap)
    dfn = df_all.filter((pl.col("kind") == 0) & (pl.col("df") <= DF_CAP_NREV)).with_row_index("nid").with_columns(
        (np.log(pool.height) - pl.col("df").log()).cast(pl.Float32).alias("idf"))
    idfn = dfn["idf"].to_numpy()
    nullp = pool.filter(pl.col("a_tok").list.len() == 0).select("row", "n_tok").with_row_index("nrow") \\
        .with_columns(pl.col("nrow").cast(I64))
    gch = pl.DataFrame(schema={"prow": I64, "qrow": I64, "r_g": pl.Int16})
    if nullp.height:
        pnt = (nullp.select(pl.col("nrow").alias("row"), pl.col("n_tok").list.unique().alias("tok")).explode("tok")
               .with_columns(("n:" + pl.col("tok")).alias("tok")).join(dfn.select("tok", "nid"), on="tok")
               .rename({"nid": "tid"}))
        wnt = (world.select("row", pl.col("n_tok").list.unique().alias("tok")).explode("tok")
               .with_columns(("n:" + pl.col("tok")).alias("tok")).join(dfn.select("tok", "nid"), on="tok")
               .rename({"nid": "tid"}))
        _G["RQ"] = _csr(pnt, nullp.height, dfn.height, idfn[pnt["tid"].to_numpy()])
        _G["S1T"] = _csr(wnt, world.height, dfn.height).T.tocsr()
        _G["K_REV"] = KG
        gparts = _run_pool(_rev_chunk, nullp.height, 2000, f"block {cty} null-address reverse")
        for k in ("RQ", "S1T", "K_REV"):
            _G.pop(k, None)
        g = pl.concat([pl.DataFrame({"nrow": r.astype(np.int64), "wrow": c.astype(np.int64), "r_g": rk}) for r, c, v, rk in gparts])
        gch = (g.join(nullp.select("nrow", pl.col("row").alias("prow")), on="nrow")
               .join(q.select(pl.col("row").alias("qrow"), "wrow"), on="wrow").select("prow", "qrow", "r_g"))

    # (d) core-name key equality''')
# (f) after (d)
s = s.replace('''    cand = (cand.join(rev, on=["qrow", "prow"], how="full", coalesce=True)
            .join(dch, on=["qrow", "prow"], how="full", coalesce=True))''', '''    # (f) name + locality keys
    def nl_keys(d):
        return (core(d.select("row", "n_tok", "a_tok")).filter(pl.col("core") != "")
                .select("row", "core", pl.col("a_tok").list.unique().alias("t")).explode("t")
                .filter(pl.col("t").is_not_null() & (pl.col("t").str.len_chars() >= 3) & ~pl.col("t").str.contains(r"^[0-9]+$"))
                .select("row", (pl.col("core") + "|" + pl.col("t")).alias("k")))

    pk = nl_keys(pool)
    pk = pk.join(pk.group_by("k").len().filter(pl.col("len") <= KF_KEYCAP).select("k"), on="k")
    fch = (nl_keys(q).rename({"row": "qrow"}).join(pk.rename({"row": "prow"}), on="k")
           .group_by("qrow", "prow").agg(pl.len().alias("n_f"))
           .with_columns(pl.col("n_f").rank("ordinal", descending=True).over("qrow").cast(pl.Int16).alias("r_f"))
           .filter(pl.col("r_f") <= KF).drop("n_f"))

    cand = (cand.join(rev, on=["qrow", "prow"], how="full", coalesce=True)
            .join(dch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(fch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(gch, on=["qrow", "prow"], how="full", coalesce=True))''')
s = s.replace('''            pl.when(pl.col("in_d")).then(pl.lit(KD // 2)).otherwise(big),
        ).alias("prio"),''', '''            pl.when(pl.col("in_d")).then(pl.lit(KD // 2)).otherwise(big),
            pl.col("r_f").cast(pl.Int32).fill_null(big),
            pl.col("r_g").cast(pl.Int32).fill_null(big),
        ).alias("prio"),''')
s = s.replace('''           .select("s1", "m", "s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "prio"))''',
              '''           .select("s1", "m", "s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "prio"))''')
open(p, "w").write(s)

p = "code/business_entity_resolution/src/ber/features.py"
s = open(p).read()
s = s.replace('''    "s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "prio",
    # context''', '''    "s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "prio",
    # context''')
s = s.replace('''BLK_COLS = ["s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "prio"]''',
              '''BLK_COLS = ["s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "prio"]''')
open(p, "w").write(s)
print("patched")
