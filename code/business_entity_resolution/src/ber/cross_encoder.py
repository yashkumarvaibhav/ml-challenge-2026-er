"""Pretrained multilingual cross-encoder (GPU) used as a stage-2 feature on the uncertain band.

Model: intfloat/multilingual-e5-small (MIT license, 117.7M parameters; README rule 5 allows MIT/Apache
models <= 8B). Input: the RAW name + address of both records ("name | address"), so the multilingual
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
from transformers import AutoModel, AutoTokenizer

from . import config as C
from .io import load_source, load_train_pairs
from .progress import Progress, step

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
BASE = C.ROOT / "models" / "hf" / "multilingual-e5-small"
OUT = C.ROOT / "models" / "ce_e5small"
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
        self.enc = AutoModel.from_pretrained(base)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
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


def train():
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = CrossEncoder().to(dev)
    d = pl.read_parquet(DATA / "train_pairs.parquet").sample(fraction=1.0, shuffle=True, seed=C.SEED + 5)  # joins may reorder
    t1, t2, y = d["t1"].to_list(), d["t2"].to_list(), torch.tensor(d["y"].to_numpy(), dtype=torch.float32)
    n = len(t1)
    steps = math.ceil(n / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    prog = Progress("ce train (A6000)", n, unit="pairs")
    run = 0.0
    for b in range(steps):
        sl = slice(b * BATCH, (b + 1) * BATCH)
        enc = tok(t1[sl], t2[sl], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(enc["input_ids"], enc["attention_mask"])
        loss = lossf(logit.float(), y[sl].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if b else loss.item()
        prog.update(len(t1[sl]))
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
    out = np.empty(len(t1), np.float32)
    prog = Progress(name, len(t1), unit="pairs")
    for b in range(0, len(t1), batch):
        enc = tok(t1[b:b + batch], t2[b:b + batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[b:b + batch] = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]).float()).cpu().numpy()
        prog.update(len(t1[b:b + batch]))
    prog.done()
    return out


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
    (C.EXPERIMENTS / "ce_evalband.json").write_text(json.dumps(rep, indent=2))
    step("ce", "evalband: " + json.dumps(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    t0 = time.time()
    for s in a.steps.split(","):
        step("ce", f"step {s} ...")
        {"data": data, "train": train, "evalband": evalband}[s]()
    step("ce", f"steps {a.steps} DONE in {time.time() - t0:.0f}s")
