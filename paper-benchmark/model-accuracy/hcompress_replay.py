#!/usr/bin/env python3
"""Offline replay of HCompress as the selector over NeuroPress v2's 45 settings
(no run), next to NeuroPress learning, the best single codec and the oracle.

    hcompress_replay.py DATASET W_CT,W_DT,W_IO BW_BYTES_PER_MS [--reads 10]
                        [--seed-dir DIR]

HCompress (the seed of train_hcompress_v2.py, Clio's format) predicts each
setting from its library key (algorithm, quantize, shuffle bit) and the chunk
size only; a chunk takes the setting with the lowest cost under the given
weights at the given bandwidth, as every option does. After each chunk the
executed setting's measured compress time and ratio are fed back by recursive
least squares (forget factor 1, every chunk), as HCompressCcpPredictor::Observe
does in Clio; no decompress time is measured on the write path, so that head
is not updated. Every measured value comes from the stored exhaustive search.
Prints the modelled runtime (1 write + READS reads at BW), the ratio and the
cost of each option against the best single codec.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay_learning as rl  # noqa: E402
import train_hcompress_v2 as th  # noqa: E402

MIN_SPEED, MIN_R, MAX_R = 1.0, 0.1, 1e5


def setting_key(name):
    """@return HCompress's library key for a v2 setting spec (e.g. 'ans shuffle=byte')."""
    parts = name.split()
    # as the corpus's shuffle column: only shuffle=byte / shuffle=bit; lz4's
    # bitshuffle=msb is a codec parameter (shuffle "none" in the corpus)
    shuffled = any(p in ("shuffle=byte", "shuffle=bit") for p in parts[1:])
    return f"{parts[0]}|q0|s{1 if shuffled else 0}"


class Hc:
    """HCompress's three RLS heads, seeded from the JSON, P = (A + C I)^-1 of the corpus."""

    def __init__(self, seed_dir, corpus, prior_only=False):
        m = json.load(open(os.path.join(seed_dir, "hcompress_ccp_seed.json")))
        self.libs = m["libraries"]
        self.w = [np.array(m[k], float) for _, k in th.HEADS]
        c = float(m["regularization"])
        d = th.load(corpus)
        x = th.encode(d, self.libs, [])
        y = th.targets(d, 0.0)
        self.p = []
        if prior_only:   # what Clio's Load() does: weights from the JSON, P = (1/C) I
            self.p = [np.eye(2 + len(self.libs)) / c for _ in range(3)]
            return
        for k in range(3):
            ok = np.isfinite(y[:, k])
            a = x[ok].T @ x[ok] + c * np.eye(x.shape[1])
            self.p.append(np.linalg.inv(a))

    def row(self, key, nbytes):
        """@return the regression row [1, log2(bytes), one-hot library]."""
        x = np.zeros(2 + len(self.libs))
        x[0], x[1] = 1.0, np.log2(nbytes)
        if key in self.libs:
            x[2 + self.libs.index(key)] = 1.0
        return x

    def predict(self, rows, nbytes):
        """@return predicted (compress ms, decompress ms, ratio) per row, as PredictFor()."""
        cs, ds, r = (rows @ w for w in self.w)
        return (nbytes / (np.maximum(MIN_SPEED, cs) * 1000.0),
                nbytes / (np.maximum(MIN_SPEED, ds) * 1000.0), np.clip(r, MIN_R, MAX_R))

    def update(self, k, x, y, lam=1.0):
        """One RLS step of head k with forget factor lam (CcpRls::Update)."""
        px = self.p[k] @ x
        g = px / (lam + x @ px)
        self.w[k] = self.w[k] + g * (y - x @ self.w[k])
        self.p[k] = (self.p[k] - np.outer(g, px)) / lam


def outcome(meas, nb, pick, store, w, bw, reads):
    """@return (modelled runtime s, ratio, cost s) of one pick per chunk."""
    i = np.arange(len(pick))
    ct, dt, r = meas[i, pick].T
    raw = (pick == store) | ~(r > 1)
    st = np.where(raw, nb, nb / np.where(raw, 1.0, r))
    ctz = np.where(pick == store, 0.0, ct)
    dtz = np.where(raw, 0.0, dt)
    run = ctz + reads * dtz + (1 + reads) * st / bw
    cost = w[0] * ctz + w[1] * dtz + w[2] * st / bw
    return run.sum() / 1e3, nb.sum() / st.sum(), cost.sum() / 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("w")
    ap.add_argument("bw", type=float)
    ap.add_argument("--reads", type=int, default=10)
    ap.add_argument("--seed-dir", default=os.path.join(
        HERE, "..", "..", "context-transport-primitives", "src", "compress", "model",
        "weights", "hcompress_v2"))
    ap.add_argument("--corpus", default=os.path.join(HERE, "nn-v2-training", "corpus_v2.csv.xz"))
    ap.add_argument("--forget", default="1", help="comma list of RLS forget factors to replay")
    ap.add_argument("--clio-load", action="store_true",
                    help="start feedback from P = (1/C) I, as Clio's Load() does")
    a = ap.parse_args()
    w = tuple(float(v) for v in a.w.split(","))
    names, store, order, meas, _ = rl.load(a.dataset)
    nb = order.bytes.to_numpy(float)
    n, ns = meas.shape[:2]
    cand = list(range(ns))   # as Clio: every v2 setting, store included
    keys = [setting_key(names[s]) for s in cand]

    # truth cost per chunk x setting (for the best single and the oracle)
    ct, dt, r = meas[..., 0], meas[..., 1], meas[..., 2]
    kept = r > 1
    st = np.where(kept, nb[:, None] / np.where(kept, r, 1.0), nb[:, None])
    cost = w[0] * ct + w[1] * np.where(kept, dt, 0.0) + w[2] * st / a.bw
    cost[:, store] = w[2] * nb / a.bw
    sel = np.where(np.isfinite(cost), cost, np.inf)
    best = int(np.argmin(rl.ev.for_selection(cost).sum(axis=0)))
    oracle = sel.argmin(axis=1)

    hc = Hc(a.seed_dir, a.corpus)
    runs = [("HCompress (seed only)", False, 1.0)] + [
        (f"HCompress (feedback, forget {f})", True, float(f)) for f in a.forget.split(",")]
    picks = {label: np.empty(n, int) for label, _, _ in runs}
    for label, learn, lam in runs:
        h = Hc(a.seed_dir, a.corpus, prior_only=a.clio_load) if learn else hc
        for i in range(n):
            rows = np.array([h.row(k, nb[i]) for k in keys])
            pc, pd, pr = h.predict(rows, nb[i])
            c = w[0] * pc + w[1] * pd + w[2] * nb[i] / (pr * a.bw)
            j = int(np.argmin(c))
            s = cand[j]
            picks[label][i] = s
            if learn:
                # as HCompressCcpPredictor::ApplyFeedback: each head learns only
                # from what was measured; storing raw is ratio 1 (a codec that
                # did not shrink the chunk was stored raw, so ratio 1 too)
                x = rows[j]
                ct_i, r_i = meas[i, s, 0], meas[i, s, 2]
                if s == store or not (r_i > 1):
                    r_i = 1.0
                if np.isfinite(ct_i) and ct_i > 0:
                    h.update(0, x, nb[i] / (ct_i * 1000.0), lam)   # compress speed
                h.update(2, x, r_i, lam)                            # ratio
    rec, _, _ = rl.replay(a.dataset, "nlms", 0.5, thr=rl.MAPE_THRESHOLD, data=rl.load(a.dataset),
                          w=w, bw=a.bw)
    opts = {"best single (" + names[best] + ")": np.full(n, best),
            "NeuroPress learning": rec.pick.to_numpy(), **picks, "oracle": oracle}
    tb, rb, cb = outcome(meas, nb, opts[next(iter(opts))], store, w, a.bw, a.reads)
    print(f"{a.dataset}: cost model {a.w} at {a.bw / 1e6:g} GB/s, 1 write + {a.reads} reads "
          f"(modelled, from the exhaustive search)")
    for label, p in opts.items():
        t, r_, c = outcome(meas, nb, p, store, w, a.bw, a.reads)
        print(f"  {label:38s} runtime {t:8.1f} s ({100 * (t / tb - 1):+6.1f}%)  ratio {r_:5.2f} "
              f"({100 * (r_ / rb - 1):+6.1f}%)  cost {100 * (c / cb - 1):+6.1f}%")
    for label, learn, _ in runs[1:]:
        top = np.bincount(picks[label], minlength=ns)
        print(f"  {label} picks:", ", ".join(f"{names[s]} {100 * top[s] / n:.0f}%"
                                             for s in np.argsort(-top)[:3] if top[s]))


if __name__ == "__main__":
    main()
