"""B2 v3 patch (apply only after the v2 package decision; see plan-discussion audit reply).

(h) exact core name + house number: key = core name + "#" + number (address tokens that are all
    digits). Keys shared by <= KH_KEYCAP pool records; per S1 rank by (#shared keys, IDF sum of
    shared non-numeric address words); top KH.
(f) re-ranked: IDF sum of the shared locality words instead of an ordinal rank of key counts.
Formatting (F1, reviewer-accepted part): "N°" / "Nº" -> "No" before tokenization.
New blocking feature r_h.
"""
p = "code/business_entity_resolution/src/ber/blocking.py"
s = open(p).read()
s = s.replace('''  (g) null-addr reverse   : address-null pool records -> top KG S1s by name-only overlap
                            with a higher df cap (DF_CAP_NREV)
''', '''  (g) null-addr reverse   : address-null pool records -> top KG S1s by name-only overlap
                            with a higher df cap (DF_CAP_NREV)
  (h) name+house number   : core name + one address number as an exact key (keys shared by
                            <= KH_KEYCAP pool records); ties ranked by IDF of shared address words
''')
s = s.replace('''KG, DF_CAP_NREV = 10, 30_000''', '''KG, DF_CAP_NREV = 10, 30_000
KH, KH_KEYCAP = 10, 50''')
# IDF of address words for tie-breaks
s = s.replace('''    # (f) name + locality keys
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
''', '''    # address-word IDF (uncapped, from the pool) for ranking the key channels
    aidf = (df_all.filter(pl.col("tok").str.starts_with("a:"))
            .select(pl.col("tok").str.slice(2).alias("t"),
                    (np.log(pool.height) - pl.col("df").log()).cast(pl.Float32).alias("w")))

    # (f) name + locality keys, ranked by the IDF of the shared locality words
    def nl_keys(d):
        return (core(d.select("row", "n_tok", "a_tok")).filter(pl.col("core") != "")
                .select("row", "core", pl.col("a_tok").list.unique().alias("t")).explode("t")
                .filter(pl.col("t").is_not_null() & (pl.col("t").str.len_chars() >= 3) & ~pl.col("t").str.contains(r"^[0-9]+$"))
                .select("row", "t", (pl.col("core") + "|" + pl.col("t")).alias("k")))

    pk = nl_keys(pool)
    pk = pk.join(pk.group_by("k").len().filter(pl.col("len") <= KF_KEYCAP).select("k"), on="k")
    fch = (nl_keys(q).rename({"row": "qrow"}).join(pk.rename({"row": "prow"}).drop("t"), on="k")
           .join(aidf, on="t", how="left")
           .group_by("qrow", "prow").agg(pl.col("w").fill_null(0).sum().alias("w_f"))
           .with_columns(pl.col("w_f").rank("ordinal", descending=True).over("qrow").cast(pl.Int16).alias("r_f"))
           .filter(pl.col("r_f") <= KF).drop("w_f"))

    # (h) exact core name + house number keys; ties ranked by IDF of shared address words
    def nn_keys(d):
        return (core(d.select("row", "n_tok", "a_tok")).filter(pl.col("core") != "")
                .select("row", "core", pl.col("a_tok").list.unique().alias("t")).explode("t")
                .filter(pl.col("t").is_not_null() & pl.col("t").str.contains(r"^[0-9]+$"))
                .select("row", (pl.col("core") + "#" + pl.col("t")).alias("k")))

    hk = nn_keys(pool)
    hk = hk.join(hk.group_by("k").len().filter(pl.col("len") <= KH_KEYCAP).select("k"), on="k")
    hpairs = (nn_keys(q).rename({"row": "qrow"}).join(hk.rename({"row": "prow"}), on="k")
              .group_by("qrow", "prow").agg(pl.len().alias("n_h")))
    if hpairs.height:
        qa = q.select(pl.col("row").alias("qrow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        pa = pool.select(pl.col("row").alias("prow"), pl.col("a_tok").list.unique().alias("t")).explode("t")
        shared = (hpairs.select("qrow", "prow").join(qa, on="qrow").join(pa, on=["prow", "t"])
                  .filter(~pl.col("t").str.contains(r"^[0-9]+$")).join(aidf, on="t", how="left")
                  .group_by("qrow", "prow").agg(pl.col("w").fill_null(0).sum().alias("w_h")))
        hch = (hpairs.join(shared, on=["qrow", "prow"], how="left").with_columns(pl.col("w_h").fill_null(0))
               .sort(["qrow", "n_h", "w_h"], descending=[False, True, True])
               .with_columns(pl.int_range(1, pl.len() + 1).over("qrow").cast(pl.Int16).alias("r_h"))
               .filter(pl.col("r_h") <= KH).select("qrow", "prow", "r_h"))
    else:
        hch = pl.DataFrame(schema={"qrow": pl.Int64, "prow": pl.Int64, "r_h": pl.Int16})
''')
s = s.replace('''            .join(fch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(gch, on=["qrow", "prow"], how="full", coalesce=True))''', '''            .join(fch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(gch, on=["qrow", "prow"], how="full", coalesce=True)
            .join(hch, on=["qrow", "prow"], how="full", coalesce=True))''')
s = s.replace('''            pl.col("r_g").cast(pl.Int32).fill_null(big),
        ).alias("prio"),''', '''            pl.col("r_g").cast(pl.Int32).fill_null(big),
            pl.col("r_h").cast(pl.Int32).fill_null(big),
        ).alias("prio"),''')
s = s.replace('''"in_d", "r_f", "r_g", "prio"))''', '''"in_d", "r_f", "r_g", "r_h", "prio"))''')
open(p, "w").write(s)

p = "code/business_entity_resolution/src/ber/features.py"
s = open(p).read()
s = s.replace('''"r_e", "in_d", "r_f", "r_g", "prio",
    # context''', '''"r_e", "in_d", "r_f", "r_g", "r_h", "prio",
    # context''')
s = s.replace('''BLK_COLS = ["s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "prio"]''',
              '''BLK_COLS = ["s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "in_d", "r_f", "r_g", "r_h", "prio"]''')
open(p, "w").write(s)

p = "code/business_entity_resolution/src/ber/text.py"
s = open(p).read()
s = s.replace('''* S3 placeholders "NULL"/"<NULL>" dropped.''', '''* S3 placeholders "NULL"/"<NULL>" dropped.
* "N°" / "Nº" number prefixes -> "No" (French test records write "N° 106"; India uses "No 106").''')
s = s.replace('''PLACEHOLDER = re.compile(r"<\\s*null\\s*>|\\bnull\\b", re.IGNORECASE)''', '''PLACEHOLDER = re.compile(r"<\\s*null\\s*>|\\bnull\\b", re.IGNORECASE)
NUMERO = re.compile(r"\\b[Nn]\\s*[°º]\\s*")''')
s = s.replace('''    s = PLACEHOLDER.sub(" ", unicodedata.normalize("NFKC", raw))''', '''    s = NUMERO.sub("No ", PLACEHOLDER.sub(" ", unicodedata.normalize("NFKC", raw)))''')
open(p, "w").write(s)
print("patched v3")
