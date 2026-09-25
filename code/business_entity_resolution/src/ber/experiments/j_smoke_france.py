"""v4(j) France test smoke: block France test S1s with (j), compare with v3 test candidates (label-free)."""
import json

import polars as pl

from ber import config as C
from ber.blocking import CAP, block
from ber.model import decide

if __name__ == "__main__":
    s1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select("entity_id", "country")
    fr = s1.filter(pl.col("country") == "France").select("entity_id")
    cand = block("test", fr, s1.select("entity_id"))
    cand.write_parquet(C.CACHE / "cand" / "test_france_v4j.parquet")
    v3 = pl.read_parquet(C.CACHE / "cand" / "test.parquet").join(fr.rename({"entity_id": "s1"}), on="s1")
    # proxy pairs: exact full name + shared address number
    n1 = pl.read_parquet(C.CACHE / "norm_final/test_s1.parquet", columns=["entity_id", "country", "n_tok", "a_tok"]).filter(pl.col("country") == "France").rename({"entity_id": "s1"})
    n2 = pl.concat([pl.read_parquet(C.CACHE / f"norm_final/test_s{s}.parquet", columns=["entity_id", "country", "n_tok", "a_tok"]) for s in (2, 3)]).filter(pl.col("country") == "France").rename({"entity_id": "m"})
    def keys(d, idc):
        return (d.select(idc, pl.col("n_tok").list.unique().list.sort().list.join(" ").alias("fn"), pl.col("a_tok").list.unique().alias("t"))
                .explode("t").filter(pl.col("t").str.contains(r"^[0-9]+$")).select(idc, (pl.col("fn") + "#" + pl.col("t")).alias("k")).unique())
    proxy = keys(n1, "s1").join(keys(n2, "m"), on="k").select("s1", "m").unique()
    def share(c):
        return proxy.join(c.select("s1", "m").with_columns(pl.lit(True).alias("x")), on=["s1", "m"], how="left")["x"].fill_null(False).mean()
    rep = {"proxy_pairs": proxy.height, "proxy_retrieved_v3": round(share(v3), 4), "proxy_retrieved_v4j": round(share(cand), 4),
           "cand_per_s1_v3": round(v3.height / fr.height, 2), "cand_per_s1_v4j": round(cand.height / fr.height, 2),
           "at_cap_v3": round((v3.group_by("s1").len()["len"] >= CAP).mean(), 4), "at_cap_v4j": round((cand.group_by("s1").len()["len"] >= CAP).mean(), 4)}
    # newly retrieved proxy pairs: do they share a non-numeric address word?
    new = proxy.join(cand.select("s1", "m"), on=["s1", "m"]).join(v3.select("s1", "m"), on=["s1", "m"], how="anti")
    toks = lambda d, idc: d.select(idc, pl.col("a_tok").list.eval(pl.element().filter(~pl.element().str.contains(r"^[0-9]+$"))).alias("w" + idc))
    nw = new.join(toks(n1, "s1"), on="s1").join(toks(n2, "m"), on="m").with_columns(pl.col("ws1").list.set_intersection(pl.col("wm")).list.len().alias("shared_words"))
    rep["newly_retrieved_proxy_pairs"] = new.height
    rep["new_proxy_pairs_sharing_street_or_city_word"] = round((nw["shared_words"] > 0).mean(), 4) if nw.height else None
    # overlap with the high-confidence v2 predictions dropped in v3
    v2p = pl.read_parquet(C.CACHE / "runs/v2/test_pred.parquet")
    prm2 = json.loads((C.EXPERIMENTS / "v2/decision_params.json").read_text())
    pred2 = decide(v2p, prm2["t_any"], prm2["t_add"]).join(v2p, on=["s1", "m"]).join(fr.rename({"entity_id": "s1"}), on="s1")
    dropped_hi = pred2.filter(pl.col("p") >= 0.99).join(v3.select("s1", "m"), on=["s1", "m"], how="anti")
    rep["v2_hi_conf_dropped_in_v3"] = dropped_hi.height
    rep["of_which_recovered_by_v4j"] = dropped_hi.join(cand.select("s1", "m"), on=["s1", "m"]).height
    # v3 high-confidence predictions displaced by v4j
    v3p = pl.read_parquet(C.CACHE / "pred/test.parquet").join(fr.rename({"entity_id": "s1"}), on="s1")
    prm3 = json.loads((C.EXPERIMENTS / "decision_params.json").read_text())
    pred3 = decide(v3p, prm3["t_any"], prm3["t_add"]).join(v3p, on=["s1", "m"])
    lost = pred3.join(cand.select("s1", "m"), on=["s1", "m"], how="anti")
    rep["v3_predicted_matches"] = pred3.height
    rep["v3_predicted_displaced_by_v4j"] = lost.height
    rep["v3_predicted_displaced_p>=0.99"] = lost.filter(pl.col("p") >= 0.99).height
    print(json.dumps(rep, indent=1))
    (C.EXPERIMENTS / "v4j_france_smoke.json").write_text(json.dumps(rep, indent=1))
