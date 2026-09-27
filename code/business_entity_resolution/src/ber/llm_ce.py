"""LLM cross-encoder pilot (27 Sep): Qwen2.5-7B (Apache-2.0, 7.6B params, run offline) with a LoRA-tuned
sequence-classification head on the same leakage-safe CE training pairs as e5base2 (cache/ce/train_pairs_e5base2.parquet:
CE S1s = v4 train sample minus the stage-2 fit sample; never holdout).

Steps:
  train    : LoRA fine-tune on N pairs (BER_LLM_N, default 120k), 1 epoch, length-grouped batches
  evalband : score the fixed v6 world-20 holdout band (the evalband_v6 pairs) -> AUC / log loss vs e5base2, and a
             stacked check: LightGBM on the stage-2 features + CE, fit on holdout half A, macro F0.5 on half B (and B->A)
  score    : score an arbitrary (s1, m) parquet for a split (--pairs, --split, --out)
  BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 python -m ber.llm_ce --steps train,evalband
"""
import argparse
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch

from . import config as C
from .progress import step

BASE = C.ROOT / "models" / "hf" / os.environ.get("BER_LLM_BASE", "Qwen2.5-7B")
TAG = os.environ.get("BER_LLM_TAG", "qwen7b")
OUT = C.ROOT / "models" / f"llm_ce_{TAG}"
N_TRAIN = int(os.environ.get("BER_LLM_N", "120000"))
MAX_LEN = 160
BATCH, ACC, LR = 16, 4, 1.5e-4
SCORE_BATCH = 128
DEV = os.environ.get("BER_LLM_DEV", "cuda:0")


def _text(t1, t2):
    return [f"Entity: {a}\nRecord: {b}\nSame business?" for a, b in zip(t1, t2)]


INIT = os.environ.get("BER_LLM_INIT")          # continue training from this adapter tag (learning-curve runs)
SKIP_N = int(os.environ.get("BER_LLM_SKIP_N", "0"))  # exclude the first run's sample (same seed) from new training pairs


def _load(train: bool):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    if train and INIT:
        model = PeftModel.from_pretrained(model, C.ROOT / "models" / f"llm_ce_{INIT}", is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    elif train:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        model = PeftModel.from_pretrained(model, OUT)
        model = model.merge_and_unload()
    return tok, model.to(DEV)


def train():
    d = pl.read_parquet(C.CACHE / "ce" / "train_pairs_e5base2.parquet")
    if SKIP_N:
        d = d.join(d.sample(SKIP_N, seed=C.SEED + 7).select("s1", "m"), on=["s1", "m"], how="anti").sample(N_TRAIN, seed=C.SEED + int(os.environ.get("BER_LLM_SEED_OFF", "8")))
    else:
        d = d.sample(N_TRAIN, seed=C.SEED + 7)
    tok, model = _load(True)
    enc = tok(_text(d["t1"].to_list(), d["t2"].to_list()), truncation=True, max_length=MAX_LEN)
    ids = enc["input_ids"]
    y = d["y"].to_numpy().astype(np.float32)
    lens = np.array([len(x) for x in ids])
    step("llm", f"train: {len(ids):,} pairs, pos {y.mean():.3f}, mean len {lens.mean():.1f}, p99 {np.percentile(lens, 99):.0f}")
    # length-grouped batches (shuffled chunks of 50 batches sorted by length)
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(len(ids))
    chunks = [order[i:i + BATCH * 50] for i in range(0, len(order), BATCH * 50)]
    batches = []
    for ch in chunks:
        ch = ch[np.argsort(lens[ch])]
        batches += [ch[i:i + BATCH] for i in range(0, len(ch), BATCH)]
    rng.shuffle(batches)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps = math.ceil(len(batches) / ACC)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run = time.time(), []
    for bi, b in enumerate(batches):
        x = tok.pad({"input_ids": [ids[i] for i in b]}, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(input_ids=x["input_ids"].to(DEV), attention_mask=x["attention_mask"].to(DEV)).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[b]).to(DEV)) / ACC
        loss.backward()
        run.append(loss.item() * ACC)
        if (bi + 1) % ACC == 0 or bi == len(batches) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (bi + 1) % 200 == 0:
            el = time.time() - t0
            step("llm", f"train batch {bi + 1}/{len(batches)} loss {np.mean(run[-200:]):.4f} | {(bi + 1) * BATCH / el:.0f} pairs/s, eta {el / (bi + 1) * (len(batches) - bi - 1) / 60:.0f} min")
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    step("llm", f"train done in {(time.time() - t0) / 60:.0f} min -> {OUT}")


def score(pairs: pl.DataFrame, split: str, name: str = "llm score", model=None, tok=None) -> np.ndarray:
    from .cross_encoder import with_text
    if model is None:
        tok, model = _load(False)
    model.eval()
    d = with_text(pairs.select("s1", "m").with_row_index("_i"), split).sort("_i")
    assert d.height == pairs.height
    enc = tok(_text(d["t1"].to_list(), d["t2"].to_list()), truncation=True, max_length=MAX_LEN)["input_ids"]
    lens = np.array([len(x) for x in enc])
    order = np.argsort(lens)
    out = np.zeros(len(enc), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for k, i in enumerate(range(0, len(order), SCORE_BATCH)):
            b = order[i:i + SCORE_BATCH]
            x = tok.pad({"input_ids": [enc[j] for j in b]}, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=x["input_ids"].to(DEV), attention_mask=x["attention_mask"].to(DEV)).logits.squeeze(-1)
            out[b] = torch.sigmoid(lg.float()).cpu().numpy()
            if k % 500 == 0:
                el = time.time() - t0
                step("llm", f"{name}: {i:,}/{len(order):,} ({(i + len(b)) / max(el, 1e-6):.0f} pairs/s)")
    return out


def evalband():
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score
    from .io import load_train_pairs
    from .metrics import per_s1_f05
    from .model import decide
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    b = pl.read_parquet(C.CACHE / "v6" / "s2" / "ce_w20_e5base2.parquet").join(hold.select("s1", "holdout_half"), on="s1").rename({"ce": "ce_e5"})
    f = pl.read_parquet(C.CACHE / "v6" / "s2" / "s2feats_w20_e5base2.parquet").join(hold.select("s1"), on="s1")
    truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1).alias("y"))
    b = b.join(f.select("s1", "m", "m_a_null", "p1"), on=["s1", "m"], how="left").join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    cache = C.WORK / "s2" / f"llm_band_w20_{TAG}.parquet"
    if cache.exists():
        b = b.join(pl.read_parquet(cache), on=["s1", "m"], how="left")
    else:
        b = b.with_columns(pl.Series("ce_llm", score(b.select("s1", "m"), "train", name="llm band")))
        b.select("s1", "m", "ce_llm").write_parquet(cache)
    rep = {"pairs": b.height, "pos_rate": round(float(b["y"].mean()), 4), "n_train": N_TRAIN}

    def m(x, tag):
        yy = x["y"].to_numpy()
        for c in ("ce_e5", "ce_llm", "p1"):
            v = x[c].to_numpy()
            rep[f"{tag} auc_{c}"] = round(float(roc_auc_score(yy, v)), 4)
            if c != "p1":
                rep[f"{tag} logloss_{c}"] = round(float(log_loss(yy, np.clip(v, 1e-6, 1 - 1e-6))), 4)
        rep[f"{tag} auc_avg_e5_llm"] = round(float(roc_auc_score(yy, x["ce_e5"].to_numpy() + x["ce_llm"].to_numpy())), 4)

    m(b, "all")
    m(b.filter(pl.col("m_a_null") == 1), "addr_null")
    # stacked check on the full holdout candidate set: stage-2 features (+ e5 CE) vs (+ e5 CE + LLM CE)
    from .stage2 import _feat_list
    feats = _feat_list(f)
    d = f.join(b.select("s1", "m", "ce_llm"), on=["s1", "m"], how="left").join(hold, on="s1").join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    for name, fs in (("s2_e5", feats), ("s2_e5_llm", feats + ["ce_llm"])):
        for tr_h, te_h in (("A", "B"), ("B", "A")):
            tr, te = d.filter(pl.col("holdout_half") == tr_h), d.filter(pl.col("holdout_half") == te_h)
            bst = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "verbose": -1,
                             "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 7},
                            lgb.Dataset(tr.select(fs).to_numpy(), tr["y"].to_numpy()), 600)
            sc = te.select("s1", "m").with_columns(pl.Series("p", bst.predict(te.select(fs).to_numpy())))
            per = per_s1_f05(decide(sc, 0.4, 0.75), truth.select("s1", "m"), hold.filter(pl.col("holdout_half") == te_h))
            rep[f"stack {name} {tr_h}->{te_h}"] = round(float(per["f05"].mean()), 5)
    (C.EXP_RUN / f"llm_ce_{TAG}_evalband.json").write_text(json.dumps(rep, indent=1))
    step("llm", "evalband: " + json.dumps(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    for s in a.steps.split(","):
        step("llm", f"step {s} ...")
        {"train": train, "evalband": evalband}[s]()
