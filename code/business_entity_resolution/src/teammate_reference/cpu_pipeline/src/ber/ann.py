"""Inverted-file (IVF) approximate nearest-neighbour search on the GPU (torch), for dense candidate retrieval.

The standard IVF algorithm, as in faiss IndexIVFFlat: the database vectors are clustered (spherical k-means),
each vector is stored in the list of its nearest centroid, and a query scans only the lists of its `nprobe`
nearest centroids. The work per query is ~ nprobe * N / nlist vectors, not N, so it scales sub-linearly. It is
run on the GPU because the CPU faiss wheel on the shared server runs single-threaded for IVF (measured
2026-09-25: 1.0-1.3 cores for k-means, add and search).
"""
import math

import numpy as np
import torch

# nlist ~ 4 sqrt(N) lists, each vector stored in its ASSIGN_R nearest lists, NPROBE lists scanned per query:
# ~2% (S1 -> record) / ~5% (record -> S1) of the stored vectors per query on this data; per-query work grows
# ~sqrt(N). Chosen on 20k holdout S1s/country (plan-discussion 2026-09-25 20:07 UTC): forward top-20 U reverse
# top-2 true-match recall 0.9981 (US) / 0.9982 (India) vs exact 0.9985 / 0.9987.
NLIST_PER_SQRT, NPROBE, ASSIGN_R, KMEANS_ITERS = 4, 64, 3, 20


def _dev():
    for i in range(torch.cuda.device_count()):
        if "A6000" in torch.cuda.get_device_name(i):
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0")


def _nearest(X: torch.Tensor, C: torch.Tensor, k: int, bs: int = 262_144):
    """Top-k centroids (inner product) for each row of X, in batches."""
    D, I = [], []
    for b in range(0, len(X), bs):
        d, i = torch.topk(X[b:b + bs] @ C.T, k, dim=1)
        D.append(d.float())
        I.append(i)
    return torch.cat(D), torch.cat(I)


class IVF:
    """Spherical-k-means IVF over L2-normalized float16 vectors (inner product = cosine)."""

    def __init__(self, x: np.ndarray, seed: int, nlist: int = None, assign_r: int = ASSIGN_R):
        """assign_r > 1: each vector is stored in its assign_r nearest lists (multi-assignment IVF; better
        recall for vectors near cluster boundaries, at assign_r x storage). Search results are de-duplicated."""
        self.r = assign_r
        self.dev = _dev()
        g = torch.Generator(device=self.dev).manual_seed(seed)
        X = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float16)).to(self.dev)
        n = len(X)
        self.nlist = nlist or max(1, min(n, int(NLIST_PER_SQRT * math.sqrt(n))))
        S = X[torch.randperm(n, device=self.dev, generator=g)[:min(n, 64 * self.nlist)]]
        C = S[torch.randperm(len(S), device=self.dev, generator=g)[:self.nlist]].float()
        for _ in range(KMEANS_ITERS):
            a = _nearest(S, C.half(), 1)[1][:, 0]
            newC = torch.zeros_like(C).index_add_(0, a, S.float())
            cnt = torch.bincount(a, minlength=self.nlist)
            empty = cnt == 0
            if empty.any():  # re-seed empty clusters with random sample points
                newC[empty] = S[torch.randint(0, len(S), (int(empty.sum()),), device=self.dev, generator=g)].float()
            C = torch.nn.functional.normalize(newC, dim=1)
        self.C = C.half()
        a = _nearest(X, self.C, self.r)[1].reshape(-1)          # (n * r,) list of each stored copy
        pos = torch.arange(n, device=self.dev).repeat_interleave(self.r)
        order = torch.argsort(a)
        self.X = X[pos[order]]                  # database copies sorted by list
        self.ids = pos[order]                   # position in the original array
        cnt = torch.bincount(a, minlength=self.nlist)
        self.offs = torch.cat([torch.zeros(1, dtype=torch.long, device=self.dev), torch.cumsum(cnt, 0)]).cpu().numpy()
        self.n = n

    def scan_fraction(self, nprobe: int = NPROBE) -> float:
        """Expected share of the database's vectors scored per query (with multi-assignment copies)."""
        return min(1.0, nprobe * self.r / self.nlist)

    @torch.no_grad()
    def search(self, q: np.ndarray, k: int, nprobe: int = NPROBE, bs: int = 2_000_000):
        """(D, I) float32/int64 [len(q), k]; I = -1 where fewer than k vectors were scanned."""
        Dout, Iout = [], []
        for b in range(0, len(q), bs):
            D, I = self._search(q[b:b + bs], k, min(nprobe, self.nlist))
            Dout.append(D)
            Iout.append(I)
        return np.concatenate(Dout), np.concatenate(Iout)

    def _search(self, q, k_out, nprobe):
        k = k_out * self.r                      # room for duplicate copies of the same vector
        Q = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float16)).to(self.dev)
        m = len(Q)
        _, Iq = _nearest(Q, self.C, nprobe)                     # (m, nprobe) lists to scan
        pairs_l = Iq.reshape(-1)
        pairs_q = torch.arange(m, device=self.dev).repeat_interleave(nprobe)
        o = torch.argsort(pairs_l)
        pairs_l, pairs_q = pairs_l[o], pairs_q[o]
        bounds = torch.searchsorted(pairs_l, torch.arange(self.nlist + 1, device=self.dev)).cpu().numpy()
        bestD = torch.full((m, k), -2.0, device=self.dev)
        bestI = torch.full((m, k), -1, dtype=torch.long, device=self.dev)
        for l in range(self.nlist):
            a, b = bounds[l], bounds[l + 1]
            lo, hi = self.offs[l], self.offs[l + 1]
            if a == b or lo == hi:
                continue
            step_q = max(1, int(4e8 // (hi - lo)))           # bound the score block to ~1.6 GB for large lists
            for c0 in range(a, b, step_q):
                qi = pairs_q[c0:min(b, c0 + step_q)]
                S = (Q[qi] @ self.X[lo:hi].T).float()          # only this list's vectors
                d, j = torch.topk(S, min(k, hi - lo), dim=1)
                D2 = torch.cat([bestD[qi], d], 1)
                I2 = torch.cat([bestI[qi], self.ids[lo + j]], 1)
                d, j = torch.topk(D2, k, dim=1)
                bestD[qi] = d
                bestI[qi] = torch.gather(I2, 1, j)
        if self.r > 1:  # drop duplicate copies (same id, same score), keep the top k_out distinct ids
            srt, si = torch.sort(bestI, dim=1)
            dup = torch.zeros_like(bestI, dtype=torch.bool)
            dup.scatter_(1, si[:, 1:], (srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0))
            bestD = bestD.masked_fill(dup, -3.0)
            bestD, j = torch.topk(bestD, k_out, dim=1)
            bestI = torch.gather(bestI, 1, j)
            bestI = bestI.masked_fill(bestD <= -2.0, -1)
        return bestD.cpu().numpy(), bestI.cpu().numpy()


@torch.no_grad()
def exact_search(x: np.ndarray, q: np.ndarray, k: int):
    """Exact inner-product top-k (reference for the ANN check only; brute force on a query sample)."""
    dev = _dev()
    X = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float16)).to(dev)
    bs = max(1, int(5e8 // len(X)))  # score block <= ~1 GB fp16
    Dout, Iout = [], []
    for b in range(0, len(q), bs):
        Q = torch.from_numpy(np.ascontiguousarray(q[b:b + bs], dtype=np.float16)).to(dev)
        d, i = torch.topk((Q @ X.T).float(), k, dim=1)
        Dout.append(d.cpu().numpy())
        Iout.append(i.cpu().numpy())
    return np.concatenate(Dout), np.concatenate(Iout)
