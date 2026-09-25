"""Learned dense retrieval for candidate generation (v6; plan-discussion 2026-09-25 16:44 UTC).

A bi-encoder (intfloat/multilingual-e5-small, MIT, 117.7M parameters, pinned revision 614241f6) is
fine-tuned on supplied training pairs with a symmetric InfoNCE loss (in-batch negatives + one hard
negative per anchor from the rule-based candidates). Every record is embedded once from its raw
"name | address" text; candidates are then retrieved per country with a faiss IVF inner-product
index (approximate nearest neighbours: sub-linear per query, so blocking scales):

  forward  S1 -> top-k S2/S3 records
  reverse  S2/S3 record -> top-k S1s   (finds address-less and heavily noised records)

Leakage-safe split: the bi-encoder trains only on training-portion S1s that are NOT in the stage-1
(v4) sample, the stage-2 fit sample, the CE sample (a subset of the v4 sample) or the holdout.

  python -m ber.dense --steps data,tokenize,train,encode,ann_check,recall_report
"""
import argparse
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from . import config as C
from .io import load_source, load_train_pairs
from .progress import Progress, step

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
BASE = C.ROOT / "models" / "hf" / "multilingual-e5-small"
OUT = C.ROOT / "models" / "dense_e5small"
DATA = C.CACHE / "dense"
EMB = C.CACHE / "emb"
MAX_LEN, BATCH, LR, TEMP = 64, 1024, 5e-5, 0.05
NLIST_PER_SQRT, NPROBE = 4, 32
K_FWD, K_REV = 50, 10


def device():
    for i in range(torch.cuda.device_count()):
        if "A6000" in torch.cuda.get_device_name(i):
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0")


class Encoder(torch.nn.Module):
    def __init__(self, base=BASE):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)

    def forward(self, ids, am):
        h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return torch.nn.functional.normalize(e.float(), dim=-1)


def _texts(split: str, which: str) -> pl.DataFrame:
    """which: 's1' or 'pool' (S2+S3). Raw text, e5 'query: ' prefix; name only when no address."""
    d = load_source(split, 1) if which == "s1" else pl.concat([load_source(split, 2), load_source(split, 3)])
    name, addr = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
    txt = pl.when(addr.str.strip_chars() == "").then("query: " + name).otherwise("query: " + name + " | " + addr)
    return d.select("entity_id", "country", txt.alias("txt"))


def tokenize():
    """Tokenize every record once (train + test, S1 + pool) into int32 [n, MAX_LEN] + lengths."""
    tok = AutoTokenizer.from_pretrained(BASE)
    DATA.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for which in ("s1", "pool"):
            path = DATA / f"tok_{split}_{which}.npz"
            if path.exists():
                continue
            t = _texts(split, which)
            txt = t["txt"].to_list()
            n = len(txt)
            ids = np.full((n, MAX_LEN), tok.pad_token_id, np.int32)
            lens = np.zeros(n, np.int16)
            prog = Progress(f"dense tokenize {split} {which}", n, unit="texts")
            for b in range(0, n, 200_000):
                enc = tok(txt[b:b + 200_000], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="np")
                ids[b:b + 200_000] = enc["input_ids"]
                lens[b:b + 200_000] = enc["attention_mask"].sum(1)
                prog.update(len(txt[b:b + 200_000]))
            prog.done()
            t.select("entity_id", "country").write_parquet(DATA / f"ids_{split}_{which}.parquet")
            np.savez(path, ids=ids, lens=lens)
            step("dense", f"tokenized {split} {which}: {n:,} texts, truncated {(lens >= MAX_LEN).mean():.2%}")


def _load_tok(split, which):
    z = np.load(DATA / f"tok_{split}_{which}.npz")
    return pl.read_parquet(DATA / f"ids_{split}_{which}.parquet"), z["ids"], z["lens"]


def data():
    """Training rows (s1, positive m, hard-negative m) on the disjoint bi-encoder S1 set."""
    DATA.mkdir(parents=True, exist_ok=True)
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    tr = sp.filter(pl.col("is_train"))
    v4 = tr.sample(400_000, seed=C.SEED).select(pl.col("entity_id").alias("s1"))                       # stage 1 (+ CE subset)
    s2 = tr.select(pl.col("entity_id").alias("s1")).sample(400_000, seed=C.SEED + 99)                   # stage-2 fit
    bi = tr.select(pl.col("entity_id").alias("s1"), "country").join(v4, on="s1", how="anti").join(s2, on="s1", how="anti")
    assert bi.join(sp.filter(pl.col("is_holdout")).select(pl.col("entity_id").alias("s1")), on="s1").height == 0
    truth = load_train_pairs().select("s1", "m")
    pos = truth.join(bi, on="s1").sample(fraction=1.0, shuffle=True, seed=C.SEED + 7) \
        .with_columns(pl.int_range(pl.len()).over("s1").alias("round"))
    cand = pl.read_parquet(C.CACHE / "cand" / "train_full_w20.parquet", columns=["s1", "m", "prio", "s_c"]).join(bi.select("s1"), on="s1")
    neg = (cand.join(truth, on=["s1", "m"], how="anti").sort(["s1", "prio", "s_c"], descending=[False, False, True], nulls_last=True)
           .with_columns(pl.int_range(pl.len()).over("s1").alias("round")).filter(pl.col("round") < 12)
           .select("s1", "round", pl.col("m").alias("neg")))
    rows = pos.join(neg, on=["s1", "round"], how="left")
    # S1s with fewer hard negatives than positives reuse their first one
    first = neg.filter(pl.col("round") == 0).select("s1", pl.col("neg").alias("neg0"))
    rows = rows.join(first, on="s1", how="left").with_columns(pl.coalesce("neg", "neg0").alias("neg")).drop("neg0")
    rows = rows.filter(pl.col("neg").is_not_null())
    rows.write_parquet(DATA / "train_rows.parquet")
    step("dense", f"data: {bi.height:,} bi-encoder S1s (disjoint from v4 {v4.height:,} / stage-2 {s2.height:,} samples), "
                  f"{rows.height:,} (anchor, positive, hard negative) rows")


def _gather(ids, lens, idx, dev):
    L = int(lens[idx].max())
    x = torch.from_numpy(ids[idx, :L].astype(np.int64))
    am = (torch.arange(L)[None, :] < torch.from_numpy(lens[idx].astype(np.int64))[:, None]).long()
    return x.to(dev, non_blocking=True), am.to(dev, non_blocking=True)


def train():
    dev = device()
    torch.manual_seed(C.SEED)
    s1ids, s1tok, s1len = _load_tok("train", "s1")
    pids, ptok, plen = _load_tok("train", "pool")
    rows = pl.read_parquet(DATA / "train_rows.parquet")
    s1map = s1ids.select(pl.col("entity_id").alias("s1")).with_row_index("a")
    pmap = pids.select(pl.col("entity_id").alias("m")).with_row_index("p")
    rows = (rows.join(s1map, on="s1").join(pmap, on="m")
            .join(pmap.rename({"m": "neg", "p": "n"}), on="neg")
            .with_columns(pl.int_range(pl.len()).shuffle(seed=C.SEED).alias("_r"))
            .sort(["round", "country", "_r"]))  # early rounds: one positive per S1, same-country batches
    a, p, n = (rows[c].to_numpy().astype(np.int64) for c in ("a", "p", "n"))
    nb = len(a) // BATCH
    order = np.random.default_rng(C.SEED + 1).permutation(nb)
    model = Encoder().to(dev)
    model.enc.gradient_checkpointing_enable()  # 3 x BATCH sequences per step do not fit in 48 GB otherwise
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, int(0.05 * nb))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (nb - s) / (nb - warm))))
    ce = torch.nn.CrossEntropyLoss()
    model.train()
    prog = Progress("dense train e5-small (A6000)", nb * BATCH, unit="pairs")
    run, n_same = 0.0, 0
    for i, b in enumerate(order):
        sl = slice(b * BATCH, (b + 1) * BATCH)
        xa, ma = _gather(s1tok, s1len, a[sl], dev)
        xp, mp_ = _gather(ptok, plen, np.concatenate([p[sl], n[sl]]), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ea = model(xa, ma)
            ep = model(xp, mp_)
        tgt = torch.arange(BATCH, device=dev)
        # other positives of the same S1 in this batch are not negatives (reviewer 16:50 UTC): mask them
        aid = torch.from_numpy(a[sl]).to(dev)
        same = (aid[:, None] == aid[None, :]) & ~torch.eye(BATCH, dtype=torch.bool, device=dev)
        n_same += int(same.sum().item())
        l_fwd = torch.cat([(ea @ ep[:BATCH].T / TEMP).masked_fill(same, -1e4), ea @ ep[BATCH:].T / TEMP], dim=1)
        l_rev = (ep[:BATCH] @ ea.T / TEMP).masked_fill(same, -1e4)
        loss = 0.5 * (ce(l_fwd, tgt) + ce(l_rev, tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if i else loss.item()
        prog.update(BATCH)
        if i % 200 == 0:
            step("dense", f"step {i}/{nb} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} masked same-S1 pairs so far {n_same}")
    prog.done()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "model.pt")
    step("dense", f"saved {OUT}")


@torch.no_grad()
def encode():
    """Embed every record (train + test) with the fine-tuned bi-encoder -> float16 [n, d]."""
    dev = device()
    model = Encoder().to(dev)
    model.load_state_dict(torch.load(OUT / "model.pt", map_location=dev))
    model.eval()
    EMB.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for which in ("s1", "pool"):
            path = EMB / f"{split}_{which}.npy"
            if path.exists():
                continue
            _, ids, lens = _load_tok(split, which)
            order = np.argsort(lens, kind="stable")  # length-sorted batches: little padding
            out = np.empty((len(lens), model.enc.config.hidden_size), np.float16)
            prog = Progress(f"dense encode {split} {which}", len(lens), unit="texts")
            for b in range(0, len(order), 8192):
                idx = order[b:b + 8192]
                x, m = _gather(ids, lens, idx, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out[idx] = model(x, m).half().cpu().numpy()
                prog.update(len(idx))
            prog.done()
            np.save(path, out)


def _ivf(x: np.ndarray):
    import faiss
    faiss.omp_set_num_threads(C.N_PROCS)
    d = x.shape[1]
    nlist = int(NLIST_PER_SQRT * math.sqrt(len(x)))
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(C.SEED)
    index.train(x[rng.choice(len(x), min(len(x), 100 * nlist), replace=False)])
    index.add(x)
    index.nprobe = NPROBE
    return index


def _search(index, q: np.ndarray, k: int, name: str):
    prog = Progress(name, len(q), unit="queries")
    D = np.empty((len(q), k), np.float32)
    I = np.empty((len(q), k), np.int64)
    for b in range(0, len(q), 200_000):
        D[b:b + 200_000], I[b:b + 200_000] = index.search(q[b:b + 200_000], k)
        prog.update(len(q[b:b + 200_000]))
    prog.done()
    return D, I


def retrieve(split: str, query_ids: pl.DataFrame, world_ids: pl.DataFrame, k_fwd=K_FWD, k_rev=K_REV) -> pl.DataFrame:
    """Dense candidates (s1, m, d_cos, r_df, r_dr) per country; forward for query S1s, reverse from
    every pool record into the S1 world (kept only for query S1s)."""
    s1ids = pl.read_parquet(DATA / f"ids_{split}_s1.parquet").with_row_index("i")
    pids = pl.read_parquet(DATA / f"ids_{split}_pool.parquet").with_row_index("j")
    E1 = np.load(EMB / f"{split}_s1.npy", mmap_mode="r")
    EP = np.load(EMB / f"{split}_pool.npy", mmap_mode="r")
    outs = []
    for cty in sorted(s1ids["country"].unique().to_list()):
        w = s1ids.filter(pl.col("country") == cty).join(world_ids, on="entity_id")
        qq = w.join(query_ids, on="entity_id")
        pc = pids.filter(pl.col("country") == cty)
        if qq.height == 0:
            continue
        t0 = time.time()
        XP = np.ascontiguousarray(EP[pc["j"].to_numpy()], dtype=np.float32)
        XW = np.ascontiguousarray(E1[w["i"].to_numpy()], dtype=np.float32)
        XQ = np.ascontiguousarray(E1[qq["i"].to_numpy()], dtype=np.float32)
        D, I = _search(_ivf(XP), XQ, k_fwd, f"dense {split} {cty} forward")
        fwd = pl.DataFrame({"s1": np.repeat(qq["entity_id"].to_numpy(), k_fwd), "pj": I.ravel(), "d_f": D.ravel(),
                            "r_df": np.tile(np.arange(1, k_fwd + 1, dtype=np.int16), len(qq))}).filter(pl.col("pj") >= 0)
        fwd = fwd.join(pc.select(pl.col("entity_id").alias("m")).with_row_index("pj").with_columns(pl.col("pj").cast(pl.Int64)), on="pj").drop("pj")
        D, I = _search(_ivf(XW), XP, k_rev, f"dense {split} {cty} reverse")
        rev = pl.DataFrame({"m": np.repeat(pc["entity_id"].to_numpy(), k_rev), "wi": I.ravel(), "d_r": D.ravel(),
                            "r_dr": np.tile(np.arange(1, k_rev + 1, dtype=np.int16), len(pc))}).filter(pl.col("wi") >= 0)
        rev = rev.join(w.select(pl.col("entity_id").alias("s1")).with_row_index("wi").with_columns(pl.col("wi").cast(pl.Int64)), on="wi").drop("wi") \
            .join(qq.select(pl.col("entity_id").alias("s1")), on="s1")
        c = fwd.join(rev, on=["s1", "m"], how="full", coalesce=True).with_columns(pl.coalesce("d_f", "d_r").alias("d_cos")).drop("d_f", "d_r")
        outs.append(c)
        step("dense", f"{split} {cty}: {c.height:,} pairs ({c.height / qq.height:.1f}/S1) in {time.time() - t0:.0f}s")
    return pl.concat(outs)


def ann_check(n_q: int = 4000, k: int = 50):
    """IVF vs exact cosine top-k from the SAME embeddings on a fixed holdout query sample (per country),
    for several nprobe values: separates embedding quality from approximate-index loss."""
    import faiss
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost"))
    s1ids = pl.read_parquet(DATA / "ids_train_s1.parquet").with_row_index("i")
    pids = pl.read_parquet(DATA / "ids_train_pool.parquet").with_row_index("j")
    E1, EP = np.load(EMB / "train_s1.npy", mmap_mode="r"), np.load(EMB / "train_pool.npy", mmap_mode="r")
    rep = {}
    for cty in ("US", "India"):
        q = s1ids.join(hold.select("entity_id"), on="entity_id").filter(pl.col("country") == cty).sample(n_q, seed=C.SEED)
        XQ = np.ascontiguousarray(E1[q["i"].to_numpy()], dtype=np.float32)
        XP = np.ascontiguousarray(EP[pids.filter(pl.col("country") == cty)["j"].to_numpy()], dtype=np.float32)
        flat = faiss.IndexFlatIP(XP.shape[1])  # exact inner-product search (brute force, multi-threaded, low memory)
        flat.add(XP)
        _, exact = flat.search(XQ, k)
        del flat
        index = _ivf(XP)
        for nprobe in (16, 32, 64, 128):
            index.nprobe = nprobe
            _, I = index.search(XQ, k)
            for kk in (10, 20, 50):
                ov = np.mean([len(set(exact[r, :kk]) & set(I[r, :kk])) / kk for r in range(len(XQ))])
                rep[f"{cty} nprobe{nprobe} overlap@{kk}"] = round(float(ov), 4)
    (C.EXPERIMENTS / "dense_ann_check.json").write_text(json.dumps(rep, indent=1))
    step("dense", "ann_check: " + json.dumps(rep))


def _union(ch, KF, KR, CAP, extra):
    """Production-style candidate set from the channel table `ch` (one row per pair, channel rank columns):
    dense forward <= KF, dense reverse <= KR, plus `extra` channels ('hj' = exact name+number keys h/j <= 10,
    'n' = address-null exact-name fan-out, 'cg' = address-null char-3-gram name retrieval top 3);
    ordered by the best channel rank, then bi-encoder cosine; final cap CAP per S1."""
    big = 10_000
    keep = (pl.col("r_df") <= KF) | (pl.col("r_dr") <= KR)
    ranks = [pl.when(pl.col("r_df") <= KF).then(pl.col("r_df").cast(pl.Int32)), pl.when(pl.col("r_dr") <= KR).then(pl.col("r_dr").cast(pl.Int32))]
    if "hj" in extra:
        keep = keep | pl.col("r_h").is_not_null() | pl.col("r_j").is_not_null()
        ranks += [pl.col("r_h").cast(pl.Int32), pl.col("r_j").cast(pl.Int32)]
    if "n" in extra:
        keep = keep | pl.col("in_fan")
        ranks += [pl.when(pl.col("in_fan")).then(pl.lit(1, pl.Int32))]
    if "cg" in extra:
        keep = keep | (pl.col("r_cg") <= 3)
        ranks += [pl.when(pl.col("r_cg") <= 3).then(pl.col("r_cg").cast(pl.Int32))]
    d = ch.filter(keep.fill_null(False)).with_columns(pl.min_horizontal([r.fill_null(big) for r in ranks]).alias("pri"))
    d = d.sort(["s1", "pri", "d_cos"], descending=[False, False, True], nulls_last=True) \
        .with_columns(pl.int_range(1, pl.len() + 1).over("s1").alias("pos")).filter(pl.col("pos") <= CAP)
    return d.select("s1", "m")


def _channels(dc, old, w, hold, sp):
    """All channel ranks for holdout S1s in world w: dense (dc), old h/j keys, fan-out, char-gram."""
    from .blocking_v6 import FAN, _cosine
    ch = dc.select("s1", "m", "r_df", "r_dr")
    ch = ch.join(old.filter(pl.col("r_h").is_not_null() | pl.col("r_j").is_not_null()).select("s1", "m", "r_h", "r_j"), on=["s1", "m"], how="full", coalesce=True)
    world = (sp.filter(~pl.col("is_ghost")) if w == 20 else sp).select(pl.col("entity_id").alias("s1"))
    r1 = pl.read_parquet(C.CACHE / "norm_final" / "train_s1.parquet", columns=["entity_id", "country", "n_str"]).rename({"entity_id": "s1", "n_str": "k"}).join(world, on="s1")
    pool = pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"train_s{s}.parquet", columns=["entity_id", "country", "n_str", "a_null"]) for s in (2, 3)])
    small = r1.filter(pl.col("k") != "").group_by("country", "k").len().filter(pl.col("len") <= FAN).select("country", "k")
    nullp = pool.filter(pl.col("a_null")).select(pl.col("entity_id").alias("m"), "country", pl.col("n_str").alias("k")).join(small, on=["country", "k"])
    fan = r1.join(hold.select("s1"), on="s1").join(nullp, on=["country", "k"]).select("s1", "m", pl.lit(True).alias("in_fan"))
    ch = ch.join(fan, on=["s1", "m"], how="full", coalesce=True).with_columns(pl.col("in_fan").fill_null(False))
    cgp = DATA / f"holdout_w{w}_chargram_null.parquet"
    if cgp.exists():
        ch = ch.join(pl.read_parquet(cgp).select("s1", "m", "r_cg"), on=["s1", "m"], how="full", coalesce=True).with_columns(pl.col("in_fan").fill_null(False))
    else:
        ch = ch.with_columns(pl.lit(None, pl.Int16).alias("r_cg"))
    return ch.with_columns(pl.Series("d_cos", _cosine(ch, "train")))


def recall_report(worlds=(20, 0)):
    """Reviewer 16:50 UTC: evaluate the EXACT final-capped union (dedup + per-S1 cap), mean/p95/max per S1,
    pair recall, oracle macro-F0.5 (all / half A / half B), country / address-null / native-script recall,
    true matches rescued vs displaced relative to the old rule-based set; both ghost worlds."""
    from .metrics import blocking_report
    from .run_blocking import mflags
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    s1c = hold.select("s1", "country")
    truth = load_train_pairs().select("s1", "m").join(hold.select("s1"), on="s1")
    flags = mflags("train")
    rep = {}

    def stats(c, tag, halves=False):
        cc = c.join(hold.select("s1"), on="s1")
        per = hold.select("s1").join(cc.group_by("s1").len(), on="s1", how="left").fill_null(0)["len"]
        r = blocking_report(cc, truth, s1c, flags)
        out = {"mean_per_s1": round(float(per.mean()), 2), "p95_per_s1": int(per.quantile(0.95)), "max_per_s1": int(per.max()),
               "pair_recall": r["pair_recall"], "oracle": r["oracle"]["macro_f05"], "by_country": r["by_country_recall"],
               "addr_null_recall": r["segment_address_null"]["recall"], "native_recall": r["segment_native_name"]["recall"]}
        if halves:
            for h in ("A", "B"):
                hs = hold.filter(pl.col("holdout_half") == h).select("s1", "country")
                out[f"oracle_half{h}"] = blocking_report(cc.join(hs.select("s1"), on="s1"), truth, hs, flags)["oracle"]["macro_f05"]
        return out

    for w in worlds:
        world = (sp.filter(~pl.col("is_ghost")) if w == 20 else sp).select("entity_id")
        path = DATA / f"holdout_w{w}_dense.parquet"
        if path.exists():
            dc = pl.read_parquet(path)
        else:
            dc = retrieve("train", hold.select(pl.col("s1").alias("entity_id")), world)
            dc.write_parquet(path)
        old = pl.read_parquet(C.CACHE / "cand" / f"train_w{w}.parquet").join(hold.select("s1"), on="s1")
        oldt = old.select("s1", "m").join(truth, on=["s1", "m"])
        rep[f"w{w} old rule-based (cap 70)"] = stats(old.select("s1", "m"), "old", halves=True)
        ch = _channels(dc, old, w, hold, sp)
        rows = []
        for KF in (10, 20, 30):
            for KR in (2, 5):
                for extra in ("", "hj", "hj+n+cg"):
                    for CAP in (15, 20, 25, 30, 40):
                        u = _union(ch, KF, KR, CAP, extra)
                        st = stats(u, "")
                        ut = u.join(truth, on=["s1", "m"])
                        st["rescued_vs_old"] = ut.join(oldt, on=["s1", "m"], how="anti").height
                        st["displaced_vs_old"] = oldt.join(ut, on=["s1", "m"], how="anti").height
                        key = f"w{w} dense fwd{KF} rev{KR}{' +' + extra if extra else ''} cap{CAP}"
                        rep[key] = st
                        rows.append((key, st))
                        step("dense", f"{key}: mean {st['mean_per_s1']}/S1 recall {st['pair_recall']} oracle {st['oracle']}")
        # halves for the best config at each cap (Pareto shortlist)
        for CAP in (15, 20, 25, 30, 40):
            key, _ = max([(k, s) for k, s in rows if k.endswith(f"cap{CAP}")], key=lambda x: x[1]["oracle"])
            KF, KR = int(key.split("fwd")[1].split()[0]), int(key.split("rev")[1].split()[0])
            extra = key.split(" +")[1].split(" ")[0] if " +" in key else ""
            rep[key].update({k: v for k, v in stats(_union(ch, KF, KR, CAP, extra), "", halves=True).items() if k.startswith("oracle_half")})
            rep[f"w{w} best at cap{CAP}"] = key
        (C.EXPERIMENTS / "dense_recall_report.json").write_text(json.dumps(rep, indent=1))
    step("dense", "recall_report: " + json.dumps({k: (v if isinstance(v, str) else {x: v.get(x) for x in ("mean_per_s1", "pair_recall", "oracle", "oracle_halfA", "oracle_halfB")})
                                                  for k, v in rep.items() if "old" in k or "best" in k or isinstance(v, str)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    t0 = time.time()
    for s in a.steps.split(","):
        step("dense", f"step {s} ...")
        {"data": data, "tokenize": tokenize, "train": train, "encode": encode, "ann_check": ann_check, "recall_report": recall_report}[s]()
    step("dense", f"steps {a.steps} DONE in {time.time() - t0:.0f}s")
