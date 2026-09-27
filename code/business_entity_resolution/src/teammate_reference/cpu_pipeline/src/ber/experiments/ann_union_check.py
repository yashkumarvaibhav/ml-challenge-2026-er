import sys, time, numpy as np, polars as pl
sys.path.insert(0, "code/business_entity_resolution/src")
from ber import config as C
from ber.ann import IVF, exact_search
from ber.io import load_train_pairs
DATA, EMB = C.CACHE / "dense", C.CACHE / "emb"
sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost"))
world = sp.filter(~pl.col("is_ghost")).select("entity_id")
s1ids = pl.read_parquet(DATA / "ids_train_s1.parquet").with_row_index("i")
pids = pl.read_parquet(DATA / "ids_train_pool.parquet").with_row_index("j")
E1, EP = np.load(EMB / "train_s1.npy", mmap_mode="r"), np.load(EMB / "train_pool.npy", mmap_mode="r")
truth = load_train_pairs().select("s1", "m")
for cty in ("US", "India"):
    w = s1ids.join(world, on="entity_id").filter(pl.col("country") == cty).with_row_index("wi")
    q = w.join(hold.select("entity_id"), on="entity_id").sample(20000, seed=C.SEED)
    pc = pids.filter(pl.col("country") == cty).with_row_index("pj")
    t = truth.join(q.select(pl.col("entity_id").alias("s1"), "wi").with_row_index("qi"), on="s1").join(pc.select(pl.col("entity_id").alias("m"), "pj", "j"), on="m")
    tq, tp, tw, tj = t["qi"].to_numpy(), t["pj"].to_numpy(), t["wi"].to_numpy(), t["j"].to_numpy()
    XQ, XP, XW = (np.ascontiguousarray(a) for a in (E1[q["i"].to_numpy()], EP[pc["j"].to_numpy()], E1[w["i"].to_numpy()]))
    XR = np.ascontiguousarray(EP[tj])  # the true records, used as reverse queries
    for npb, r in ((16, 3), (32, 2), (32, 3), (64, 3), (128, 3)):  # results: plan-discussion 2026-09-25 ~20:15 UTC
        fi, ri = IVF(XP, C.SEED, assign_r=r), IVF(XW, C.SEED, assign_r=r)
        _, If = fi.search(XQ, 20, nprobe=npb); _, Ir = ri.search(XR, 5, nprobe=npb)
        scan_f = npb * r / fi.nlist; scan_r = npb * r / ri.nlist
        for kf, kr in ((20, 0), (10, 2), (20, 2), (20, 5)):
            f = (If[tq, :kf] == tp[:, None]).any(1)
            rr = (Ir[:, :kr] == tw[:, None]).any(1) if kr else np.zeros(len(tq), bool)
            print(f"{cty} nprobe {npb} r {r} (scan fwd {scan_f:.2%} = {scan_f * len(XP):,.0f} vec/query, rev {scan_r:.2%}): fwd{kf} rev{kr} recall {(f | rr).mean():.4f}", flush=True)
        del fi, ri
