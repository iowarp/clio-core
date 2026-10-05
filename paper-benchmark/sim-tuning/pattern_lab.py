#!/usr/bin/env python3
"""Which kinds of 4 MiB float32 chunks make one codec bad and NeuroPress right?

    pattern_lab.py gen     write the candidate chunks (LAB/<pattern>_<seed>.f32)
    pattern_lab.py score   read the sweep (LAB/sweep.csv) and score

Each candidate is a 128 x 128 x 64 float32 block (x fastest, as a Nyx field
chunk), of a kind a Nyx initial condition can hold exactly (any density field
in pressure equilibrium stays as it is until a blast reaches it). `gen` writes
3 seeds of each; the GPU sweep used for training (corpus_sweep, the 45
lossless settings, shuffle and un-shuffle timed) measures them; `score`
computes, per pattern and tier (12 / 1 / 0.5 / 0.25 GB/s), the balanced cost
of every setting, the cheapest setting, and NeuroPress v2's prediction (the
trained model as shipped, its four inputs computed as Clio does) and pick;
then, for every pair of patterns mixed half and half over the 1:3:3:3 tier
mix, the opportunity (per-chunk best vs best single) and NeuroPress-as-trained
vs the best single codec.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model-accuracy"))
LAB = "/mnt/nvme0/tune/lab"
SHAPE = (64, 128, 128)            # z, y, x: x fastest
TIERS = [(12e6, 0.1), (1e6, 0.3), (0.5e6, 0.3), (0.25e6, 0.3)]   # bytes/ms, share


def coords():
    """Cell-centre coordinates in [0, 1) of the block."""
    z, y, x = np.meshgrid(*[(np.arange(n) + 0.5) / n for n in SHAPE], indexing="ij")
    return x, y, z


def sss(k, x, y, z):
    """Product of sines: a smooth field in [-1, 1]."""
    w = 2 * np.pi * k
    return np.sin(w * x) * np.sin(w * y + 0.7) * np.sin(w * z + 1.3)


def patterns(seed):
    """name -> float32 array, one candidate chunk per pattern."""
    rng = np.random.default_rng(seed)
    x, y, z = coords()
    x, y, z = x + 0.13 * seed, y + 0.29 * seed, z + 0.07 * seed
    n = int(np.prod(SHAPE))
    p = {}
    for lv in (4, 8, 16, 64):
        p[f"clumpy_pow2_{lv}"] = np.ldexp(1.0, rng.integers(0, lv, n) - lv // 2)
    geo = 100.0 ** (np.arange(8) / 7 - 0.5)
    p["clumpy_geo_8"] = geo[rng.integers(0, 8, n)]
    blk = rng.integers(0, 8, (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2))
    p["clumpy_pow2_8_blk2"] = np.ldexp(1.0, np.kron(blk, np.ones((2, 2, 2), int)).ravel() - 4)
    p["smooth_exp1_k1"] = np.exp(1.0 * sss(1, x, y, z))
    p["smooth_exp3_k2"] = np.exp(3.0 * sss(2, x, y, z))
    p["smooth_exp10_k2"] = np.exp(10.0 * sss(2, x, y, z))
    p["smooth_wave_k2"] = 1.0 + 0.5 * sss(2, x, y, z)
    p["smooth_exp3_k8"] = np.exp(3.0 * sss(8, x, y, z))
    p["quantized_smooth"] = np.floor(100 * np.exp(3.0 * sss(2, x, y, z))) / 100
    s = np.exp(3.0 * sss(2, x, y, z))
    p["sparse_smooth"] = np.where(rng.random(SHAPE) < 0.9, 0.0, s)
    p["noise_uniform"] = 1.0 + rng.random(SHAPE)
    p["smooth_plus_1e4_noise"] = s * (1.0 + 1e-4 * rng.standard_normal(SHAPE))
    p["int_counts_0_255"] = rng.integers(0, 256, n).astype(float)
    # VPIC frozen vacuum fields: ex of E = -grad(phi) (x differences), phi on
    # nodes with periodic wrap, unit 2^-7 (LAYERED SLABS in the VPIC deck).
    u = 2.0 ** -7
    def ex_of(phi):
        return -(np.roll(phi, -1, axis=2) - phi)
    lv = rng.integers(0, 8, SHAPE).astype(float) * u
    p["vpic_clumpy_ex"] = ex_of(lv)
    wave = sss(1, x, y, z)
    for a in (12, 16, 20):
        p[f"vpic_smooth_exact_A2^{a}"] = ex_of(np.round(2.0 ** a * wave) * u)
    p["vpic_smooth_float"] = ex_of(0.05 / (2 * np.pi / 128) * wave)
    return {k: np.ascontiguousarray(v, dtype=np.float32).reshape(-1) for k, v in p.items()}


def features(a):
    """Clio's four inputs (model_v2.json): log2 bytes, byte entropy,
    log10(mad / range + 1e-12), log10(mean |second difference| / range + 1e-12)."""
    b = np.frombuffer(a.tobytes(), np.uint8)
    h = np.bincount(b, minlength=256) / b.size
    ent = -np.sum(h[h > 0] * np.log2(h[h > 0]))
    v = a.astype(np.float64)
    rng_ = v.max() - v.min()
    rng_ = rng_ if rng_ > 0 else 1.0
    mad = np.mean(np.abs(v - v.mean()))
    d2 = np.mean(np.abs(v[2:] - 2 * v[1:-1] + v[:-2]))
    return [np.log2(a.nbytes), ent, np.log10(mad / rng_ + 1e-12), np.log10(d2 / rng_ + 1e-12)]


def gen():
    os.makedirs(LAB, exist_ok=True)
    rows = []
    with open(os.path.join(LAB, "list.txt"), "w") as fl:
        for seed in (1, 2, 3):
            for name, a in patterns(seed).items():
                f = f"{name}_{seed}.f32"
                a.tofile(os.path.join(LAB, f))
                fl.write(f"{f} 0\n")
                rows.append([f, name, seed] + features(a))
    pd.DataFrame(rows, columns=["file", "pattern", "seed", "x0", "x1", "x2", "x3"]).to_csv(
        os.path.join(LAB, "features.csv"), index=False)
    print(f"wrote {len(rows)} chunks to {LAB}")


def score():
    import eval_v2_workloads as ev
    import replay_learning as rl
    names, store = ev.settings_list()
    ft = pd.read_csv(os.path.join(LAB, "features.csv"))
    sw = pd.read_csv(os.path.join(LAB, "sweep.csv"))
    sw = sw[sw.ok == 1]
    sw["file"] = sw.file.str.replace("#0", "", regex=False)
    sw["spec"] = sw.apply(canon, axis=1)
    sw["ratio"] = sw.bytes / sw.comp_bytes
    sw["ct"], sw["dt"] = sw.comp_ms, sw.decomp_ms
    missing = set(names) - set(sw.spec) - {"store"}
    if missing:
        print("settings not matched in the sweep:", sorted(missing))
    bytes_ = 4 << 20
    net = rl.Net()
    out = []
    for f, g in sw.groupby("file"):
        g = g.drop_duplicates("spec").set_index("spec")
        fr = ft[ft.file == os.path.basename(f)].iloc[0]
        a = net.forward(fr[["x0", "x1", "x2", "x3"]].to_numpy(float))
        lp = net.log_pred(a[-1])
        for bw, w in TIERS:
            io = bytes_ / bw
            truth = np.full(len(names), np.nan)
            for k, s in enumerate(names):
                if s == "store":
                    truth[k] = io
                elif s in g.index:
                    r = g.loc[s]
                    truth[k] = (r["ct"] + r["dt"] + io / r["ratio"]) if r["ratio"] > 1 else r["ct"] + io
            pc = np.exp(lp[:, 0]) + np.exp(lp[:, 1]) + bytes_ / (np.exp(lp[:, 2]) * bw)
            pc[store] = io
            out.append({"pattern": fr.pattern, "seed": fr.seed, "bw": bw, "w": w,
                        "truth": truth, "np_pick": int(np.nanargmin(pc))})
    np.save(os.path.join(LAB, "scored.npy"), out, allow_pickle=True)
    report(out, names)


def canon(r):
    """The sweep row's setting as the model's spec string: the codec, then
    its key=value settings in alphabetical order."""
    toks = [] if pd.isna(r["settings"]) else str(r["settings"]).split()
    return " ".join([r["algorithm"]] + sorted(toks))


def report(out, names):
    pats = sorted(set(o["pattern"] for o in out))
    T = {p: [o for o in out if o["pattern"] == p] for p in pats}
    print(f"{'pattern':24s} {'cheapest (by tier mix)':30s} {'NP pick':30s} NP/best")
    for p in pats:
        tot = sum(o["w"] * o["truth"] for o in T[p])
        b = int(np.nanargmin(tot))
        npc = sum(o["w"] * o["truth"][o["np_pick"]] for o in T[p])
        picks = pd.Series([names[o["np_pick"]] for o in T[p]]).value_counts().index[0]
        print(f"{p:24s} {names[b]:30s} {picks:30s} {npc / tot[b]:6.2f}")
    rows = []
    for p, q in itertools.combinations(pats, 2):
        tot = sum(o["w"] * o["truth"] for o in T[p] + T[q])
        b = int(np.nanargmin(tot))
        orc = sum(o["w"] * np.nanmin(o["truth"]) for o in T[p] + T[q])
        npc = sum(o["w"] * o["truth"][o["np_pick"]] for o in T[p] + T[q])
        rows.append({"pair": f"{p} + {q}", "best_single": names[b],
                     "opportunity_pct": 100 * (tot[b] - orc) / tot[b],
                     "np_vs_single_pct": 100 * (npc / tot[b] - 1)})
    t = pd.DataFrame(rows).sort_values("np_vs_single_pct")
    pd.set_option("display.width", 200)
    print(t.head(15).round(1).to_string(index=False))


if __name__ == "__main__":
    {"gen": gen, "score": score}[sys.argv[1]]()
