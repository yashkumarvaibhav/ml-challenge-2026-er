"""Pretrained multilingual cross-encoder (GPU) used as a stage-2 feature on the uncertain band.

Model: intfloat/multilingual-e5-small (MIT license, 117.7M parameters, revision 614241f6) by default;
BER_CE=e5base selects intfloat/multilingual-e5-base (MIT, 278.0M, revision d1287505). The organizer FAQ
allows MIT/Apache open-weight models <= 8B parameters, run offline, fine-tuned only on the provided data. Input: the RAW name + address of both records ("name | address"), so the multilingual
pretraining can read French text and Indian scripts; mean-pooled encoder output -> linear -> logit.

Leakage-safe data: trained on training-portion S1s from the v4 LightGBM training sample MINUS the
stage-2 fit sample (so stage-2 training rows get out-of-sample CE scores). Holdout / test never train it.

  python -m ber.cross_encoder --steps data,train,evalband
"""
import argparse
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from . import config as C
from .io import load_source, load_train_pairs
from .progress import Progress, step

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
CE_NAME = os.environ.get("BER_CE", "e5small")
BASES = {"e5small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
         "e5base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
         "e5large": ("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"),     # MIT, ~560M
         "bgerr": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")}          # Apache-2.0, ~568M
# a model pre-trained AS a cross-encoder keeps its own sequence-classification head (reviewer 18:10 UTC);
# plain encoders get mean pooling + a new linear head
SEQCLS = CE_NAME == "bgerr"
N_PAIRS = int(os.environ.get("BER_CE_NPAIRS", "0"))  # 0 = all CE training pairs; >0 = bounded pilot fit
BASE = C.ROOT / "models" / "hf" / BASES[CE_NAME][0].split("/")[1]
OUT = C.ROOT / "models" / f"ce_{CE_NAME}"
SUFFIX = "" if CE_NAME == "e5small" else f"_{CE_NAME}"  # cache/report names of non-default CEs
DATA = C.CACHE / "ce"
MAX_LEN, BATCH, LR, NEG_PER_POS = 128, 256, 3e-5, 2
BAND = (0.01, 0.995)


def device():
    for i in range(torch.cuda.device_count()):
        if "A6000" in torch.cuda.get_device_name(i):
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0")


class CrossEncoder(torch.nn.Module):
    def __init__(self, base=BASE):
        super().__init__()
        if SEQCLS:
            self.enc = AutoModelForSequenceClassification.from_pretrained(base)
            self.head = None
        else:
            self.enc = AutoModel.from_pretrained(base)
            self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        if SEQCLS:
            return self.enc(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1.0)).squeeze(-1)


def _texts(split: str) -> pl.DataFrame:
    """entity_id -> 'name | address' raw text for all sources of a split."""
    return pl.concat([load_source(split, s) for s in C.SOURCES]).select(
        "entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("txt"))


def with_text(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    t = _texts(split)
    return pairs.join(t.rename({"entity_id": "s1", "txt": "t1"}), on="s1").join(t.rename({"entity_id": "m", "txt": "t2"}), on="m")


def data():
    """Training pairs: CE S1s = v4 train sample minus stage-2 fit sample; positives + hardest negatives."""
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    v4_train = sp.filter(pl.col("is_train")).sample(400_000, seed=C.SEED).select(pl.col("entity_id").alias("s1"))
    s2_train = sp.filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1")).sample(400_000, seed=C.SEED + 99)
    ce_s1 = v4_train.join(s2_train, on="s1", how="anti")
    cand = pl.read_parquet(C.CACHE / "runs" / "v4_cand_train_w20.parquet" if (C.CACHE / "runs" / "v4_cand_train_w20.parquet").exists()
                           else C.CACHE / "cand" / "train_w20.parquet", columns=["s1", "m", "prio", "s_c"]).join(ce_s1, on="s1")
    truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1, pl.Int8).alias("y"))
    d = cand.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    pos = d.filter(pl.col("y") == 1)
    npos = pos.group_by("s1").len().rename({"len": "npos"})
    neg = (d.filter(pl.col("y") == 0).join(npos, on="s1", how="left").with_columns(pl.col("npos").fill_null(1))
           .sort(["s1", "prio", "s_c"], descending=[False, False, True])
           .with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
           .filter(pl.col("_r") < (pl.col("npos") * NEG_PER_POS).clip(lower_bound=2)).drop("_r", "npos"))
    tr = pl.concat([pos.select("s1", "m", "y"), neg.select("s1", "m", "y")]).sample(fraction=1.0, shuffle=True, seed=C.SEED)
    tr = with_text(tr, "train")
    DATA.mkdir(parents=True, exist_ok=True)
    tr.write_parquet(DATA / "train_pairs.parquet")
    step("ce", f"data: {ce_s1.height:,} S1s, {tr.height:,} pairs (pos {int(tr['y'].sum()):,})")


def pretokenize(t1, t2, tok, name):
    """Tokenize all pairs once into fixed-length int32 arrays (the GPU loop then only slices)."""
    n = len(t1)
    ids = np.zeros((n, MAX_LEN), np.int32)
    mask = np.zeros((n, MAX_LEN), np.int8)
    prog = Progress(name, n, unit="pairs")
    for b in range(0, n, 50_000):
        enc = tok(t1[b:b + 50_000], t2[b:b + 50_000], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="np")
        ids[b:b + 50_000], mask[b:b + 50_000] = enc["input_ids"], enc["attention_mask"]
        prog.update(len(t1[b:b + 50_000]))
    prog.done()
    return ids, mask


def train():
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = CrossEncoder().to(dev)
    d = pl.read_parquet(DATA / "train_pairs.parquet").sample(fraction=1.0, shuffle=True, seed=C.SEED + 5)  # joins may reorder
    if N_PAIRS:
        d = d.head(N_PAIRS)
    t1, t2, y = d["t1"].to_list(), d["t2"].to_list(), torch.tensor(d["y"].to_numpy(), dtype=torch.float32)
    ids_np, mask_np = pretokenize(t1, t2, tok, "ce tokenize train")
    n = len(t1)
    steps = math.ceil(n / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    prog = Progress(f"ce train {CE_NAME} (A6000)", n, unit="pairs")
    run = 0.0
    for b in range(steps):
        sl = slice(b * BATCH, (b + 1) * BATCH)
        m_np = mask_np[sl]
        L = int(m_np.sum(1).max())  # trim padding to the longest row in the batch
        ids = torch.from_numpy(ids_np[sl, :L].astype(np.int64)).to(dev, non_blocking=True)
        am = torch.from_numpy(m_np[:, :L].astype(np.int64)).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(ids, am)
        loss = lossf(logit.float(), y[sl].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if b else loss.item()
        prog.update(ids.shape[0])
        if b % 500 == 0:
            step("ce", f"step {b}/{steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e}")
    prog.done()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "model.pt")
    tok.save_pretrained(OUT)
    step("ce", f"saved {OUT}")


@torch.no_grad()
def score(pairs: pl.DataFrame, split: str, name: str = "ce score", batch: int = 1024) -> np.ndarray:
    """CE match probability for (s1, m) pairs (order preserved)."""
    dev = device()
    tok = AutoTokenizer.from_pretrained(OUT)
    model = CrossEncoder().to(dev)
    model.load_state_dict(torch.load(OUT / "model.pt", map_location=dev))
    model.eval()
    d = with_text(pairs.with_row_index("_i"), split).sort("_i")
    assert d.height == pairs.height, "text join lost rows"
    t1, t2 = d["t1"].to_list(), d["t2"].to_list()
    ids_np, mask_np = pretokenize(t1, t2, tok, name + " (tokenize)")
    out = np.empty(len(t1), np.float32)
    prog = Progress(name, len(t1), unit="pairs")
    for b in range(0, len(t1), batch):
        m_np = mask_np[b:b + batch]
        L = int(m_np.sum(1).max())
        ids = torch.from_numpy(ids_np[b:b + batch, :L].astype(np.int64)).to(dev)
        am = torch.from_numpy(m_np[:, :L].astype(np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[b:b + batch] = torch.sigmoid(model(ids, am).float()).cpu().numpy()
        prog.update(ids.shape[0])
    prog.done()
    return out


def bench(n_train_batches: int = 30, n_score: int = 50_000):
    """Throughput benchmark before committing a full fit: train step time and scoring pairs/s."""
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = CrossEncoder().to(dev)
    d = pl.read_parquet(DATA / "train_pairs.parquet").head(BATCH * n_train_batches + n_score)
    ids_np, mask_np = pretokenize(d["t1"].to_list(), d["t2"].to_list(), tok, f"ce bench {CE_NAME} tokenize")
    y = torch.tensor(d["y"].to_numpy(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    torch.cuda.synchronize(); t0 = time.time()
    for b in range(n_train_batches):
        sl = slice(b * BATCH, (b + 1) * BATCH)
        L = int(mask_np[sl].sum(1).max())
        ids = torch.from_numpy(ids_np[sl, :L].astype(np.int64)).to(dev)
        am = torch.from_numpy(mask_np[sl, :L].astype(np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(model(ids, am).float(), y[sl].to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize(); tr = (time.time() - t0) / (n_train_batches * BATCH)
    model.eval(); t0 = time.time()
    with torch.no_grad():
        for b in range(BATCH * n_train_batches, len(d), 1024):
            L = int(mask_np[b:b + 1024].sum(1).max())
            ids = torch.from_numpy(ids_np[b:b + 1024, :L].astype(np.int64)).to(dev)
            am = torch.from_numpy(mask_np[b:b + 1024, :L].astype(np.int64)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(ids, am)
    torch.cuda.synchronize(); sc = (time.time() - t0) / max(1, len(d) - BATCH * n_train_batches)
    rep = {"model": CE_NAME, "train_pairs_per_s": round(1 / tr), "score_pairs_per_s": round(1 / sc),
           "full_fit_3.06M_h": round(3.06e6 * tr / 3600, 2), "fit_1M_h": round(1e6 * tr / 3600, 2),
           "score_test_band_2M_min": round(2e6 * sc / 60, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 1)}
    (C.EXPERIMENTS / f"ce_bench{SUFFIX}.json").write_text(json.dumps(rep, indent=1))
    step("ce", "bench: " + json.dumps(rep))


def evalband():
    """Holdout uncertain-band check: CE vs stage-1 (v4) on pairs with v4 p in BAND."""
    from sklearn.metrics import log_loss, roc_auc_score
    h = pl.read_parquet(C.CACHE / "runs" / "v4" / "train_holdout_w20.parquet").filter(pl.col("p").is_between(*BAND))
    truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
    h = h.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    ce = score(h.select("s1", "m"), "train", name="ce score holdout band")
    y = h["y"].to_numpy()
    rep = {"band": BAND, "pairs": h.height, "pos_rate": float(y.mean()),
           "auc_stage1": float(roc_auc_score(y, h["p"].to_numpy())), "auc_ce": float(roc_auc_score(y, ce)),
           "logloss_stage1": float(log_loss(y, np.clip(h["p"].to_numpy(), 1e-6, 1 - 1e-6))),
           "logloss_ce": float(log_loss(y, np.clip(ce, 1e-6, 1 - 1e-6))),
           "auc_avg": float(roc_auc_score(y, (ce + h["p"].to_numpy()) / 2))}
    (C.EXPERIMENTS / f"ce_evalband{SUFFIX}.json").write_text(json.dumps(rep, indent=2))
    step("ce", "evalband: " + json.dumps(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    t0 = time.time()
    for s in a.steps.split(","):
        step("ce", f"step {s} ...")
        {"data": data, "train": train, "evalband": evalband, "bench": bench}[s]()
    step("ce", f"steps {a.steps} DONE in {time.time() - t0:.0f}s")
