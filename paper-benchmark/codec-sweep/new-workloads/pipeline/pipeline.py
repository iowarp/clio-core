#!/usr/bin/env python3
"""End-to-end producer-consumer pipeline with real storage, standalone (no
clio): a producer compresses its output on the GPU and writes it; a consumer
reads it back, decompresses, runs its analysis on the GPU and writes its own
results compressed. Every phase is timed, for three codec policies, REPS
times each.

    pipeline.py --workload vpic|lammps|nanoaod|fem --store DIR --out DIR
                [--reps 3] [--frames N] [--policies none,best,perchunk]
                [--candidates FILE] [--bw-write GBPS --bw-read GBPS]

Workloads (producer data -> consumer analysis, see consumers.py):
  vpic     VPIC Weibel fields, 153 dumps x 16 float32 vars
           -> k-means, connected structures, 2-D PDF
  lammps   LAMMPS LJ melt positions / velocities / forces, 201 frames float64
           -> coordination numbers, RDF, k-means, cell order
  nanoaod  CMS NanoAOD columns, 8 event batches -> Z -> ee selection, mass,
           PDFs
  fem      SuiteSparse FEM matrices (CSR), 3 -> 100 CG iterations with
           checkpoints of x and the residual

Stages
  1 profile  (once, cached in --out): every producer chunk and every output
             chunk (4 MiB) through every candidate setting -- compressed
             bytes, GPU compress / decompress ms, bit-exact check. The
             consumer runs once on the original data to produce the outputs.
  2 plan     cost of a chunk under a setting =
               producer chunk: compress + bytes / bw_write
                               + reads x (bytes / bw_read + decompress)
               output chunk:   compress + bytes / bw_write
             (a setting that cannot take a chunk stores it raw).
             none     = store everything uncompressed
             best     = the one setting with the lowest total cost
             perchunk = every chunk its own cheapest setting (an oracle over
                        the candidates, from the profile)
  3 run      per policy and rep: clear the store; producer (data staged on
             the GPU untimed, as a simulation would hold it) compresses,
             copies to the host and writes each frame, fsync per frame;
             drop the page cache; consumer reads, copies to the GPU,
             decompresses, runs the analysis, compresses and writes its
             outputs, fsync per frame. Rep 1 also checks every decompressed
             frame against the source, untimed.
Writes <out>/profile.csv, <out>/plan.json, <out>/runs.csv.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consumers  # noqa: E402
from gpucodecs import STORE, GpuCodec  # noqa: E402

CHUNK = 4 << 20
DATA = os.path.expanduser("~/np-data")


# ----------------------------------------------------------------- workloads
class Workload:
    """Producer frames (name -> host uint8 array) and the consumer."""
    name = ""

    def frames(self):
        raise NotImplementedError

    def load(self, fr):
        raise NotImplementedError

    def consume(self, fr, arrays):
        """@param arrays name -> uint8 CUDA tensor (the producer data)"""
        raise NotImplementedError


def typed(t, dtype):
    return t.view(dtype)


class Vpic(Workload):
    name = "vpic"

    def __init__(self):
        self.dirs = sorted(glob.glob(f"{DATA}/vpic-126-2000/fields/plt*"))

    def frames(self):
        return list(range(len(self.dirs)))

    def load(self, fr):
        out = {}
        for p in sorted(glob.glob(f"{self.dirs[fr]}/*.f32")):
            out[re.match(r"fab\d+_comp\d+_(.+)\.f32$", os.path.basename(p)).group(1)] = \
                np.fromfile(p, dtype=np.uint8)
        return out

    def consume(self, fr, arrays):
        return consumers.vpic({k: typed(v, torch.float32) for k, v in arrays.items()})


class Lammps(Workload):
    name = "lammps"
    L = 70 * (4 / 0.8442) ** (1 / 3)

    def __init__(self):
        d = f"{DATA}/lammps-b70-2000/fields"
        self.d = d
        self.steps = sorted({int(m.group(1)) for f in os.listdir(d)
                             if (m := re.match(r"position_step_(\d+)_chunk_0\.bin$", f))})

    def frames(self):
        return list(range(len(self.steps)))

    def load(self, fr):
        s = self.steps[fr]
        return {k: np.fromfile(f"{self.d}/{k}_step_{s}_chunk_0.bin", dtype=np.uint8)
                for k in ("position", "velocity", "force")}

    def consume(self, fr, arrays):
        return consumers.lammps({k: typed(v, torch.float64) for k, v in arrays.items()}, self.L)


class NanoAOD(Workload):
    name = "nanoaod"
    NEED = ("nElectron", "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass",
            "Electron_charge")

    def __init__(self):
        self.d = f"{DATA}/new/hep-nanoaod"
        self.batches = sorted(int(m.group(1)) for f in os.listdir(self.d)
                              if (m := re.match(r"index_b(\d+)\.json$", f)))

    def frames(self):
        return list(range(len(self.batches)))

    def load(self, fr):
        b = self.batches[fr]
        return {os.path.basename(p)[:-8]: np.fromfile(p, dtype=np.uint8)
                for p in sorted(glob.glob(f"{self.d}/*_b{b:02d}.bin"))}

    def consume(self, fr, arrays):
        ix = json.load(open(f"{self.d}/index_b{self.batches[fr]:02d}.json"))
        cols = {}
        for n in self.NEED:
            e = ix[n]
            kind = e["file"][:-8]
            # copy: branches are packed back to back, so a column can start
            # at an offset its element size does not divide
            raw = arrays[kind][e["offset"]:e["offset"] + e["nbytes"]].clone()
            cols[n] = raw.view({"<f4": torch.float32, "<u4": torch.int32, "<i4": torch.int32}[e["dtype"]])
        return consumers.nanoaod(cols)


class Fem(Workload):
    name = "fem"

    def __init__(self):
        self.d = f"{DATA}/new/sparse-fem"
        self.mats = sorted(os.path.basename(p)[:-12] for p in glob.glob(f"{self.d}/*_row_ptr.i64"))

    def frames(self):
        return list(range(len(self.mats)))

    def load(self, fr):
        m = self.mats[fr]
        return {k: np.fromfile(f"{self.d}/{m}_{k}.{e}", dtype=np.uint8)
                for k, e in (("row_ptr", "i64"), ("col_idx", "i32"), ("values", "f64"))}

    def consume(self, fr, arrays):
        return consumers.fem({"row_ptr": typed(arrays["row_ptr"], torch.int64),
                              "col_idx": typed(arrays["col_idx"], torch.int32),
                              "values": typed(arrays["values"], torch.float64)})


WORKLOADS = {"vpic": Vpic, "lammps": Lammps, "nanoaod": NanoAOD, "fem": Fem}


# ----------------------------------------------------------------- helpers
def chunks(name, t):
    """Split a flat uint8 CUDA tensor into CHUNK pieces: [(key, tensor)]."""
    return [(f"{name}#{i}", t[a:a + CHUNK]) for i, a in enumerate(range(0, t.numel(), CHUNK))]


def as_bytes(t):
    """A CUDA tensor of any dtype as a flat uint8 view."""
    return t.contiguous().reshape(-1).view(torch.uint8)


def sync():
    torch.cuda.synchronize()


def drop_caches():
    subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True)


def probe_bandwidth(store, gb=1):
    """Direct (O_DIRECT, fdatasync) write and read GB/s of the store device."""
    f = os.path.join(store, "probe.bin")
    w = subprocess.run(["dd", "if=/dev/zero", f"of={f}", "bs=4M", f"count={256 * gb}",
                        "oflag=direct", "conv=fdatasync"], capture_output=True, text=True).stderr
    drop_caches()
    r = subprocess.run(["dd", f"if={f}", "of=/dev/null", "bs=4M", "iflag=direct"],
                       capture_output=True, text=True).stderr
    os.remove(f)

    def gbps(s):
        m = re.search(r"copied, ([\d.]+) s", s)
        return (256 * gb * CHUNK) / float(m.group(1)) / 1e9
    return gbps(w), gbps(r)


def candidates_from_sweep(datasets, bws=(12, 1, 0.5, 0.25), min_share=0.02, cap=24):
    """Settings worth profiling, from the sweep CSVs: at each bandwidth (GB/s)
    the best single setting of each dataset (cost comp + decomp + bytes / bw)
    and every setting that is the cheapest on at least min_share of its
    sampled chunks; plus "store". Best singles first, then by chunk share."""
    score = {}
    for ds in datasets:
        parts = [pd.read_csv(p) for p in glob.glob(os.path.expanduser(f"~/np-newsweep/{ds}/configs*.csv"))]
        if not parts:
            print(f"warning: no sweep results for {ds}", file=sys.stderr)
            continue
        d = pd.concat(parts)
        d = d[d.ok == 1].copy()
        d["s"] = (d.algorithm + " " + d.settings.fillna("")).str.strip()
        for bw in bws:
            d["cost"] = d.comp_ms + d.decomp_ms + d.comp_bytes / (bw * 1e6)
            win = d.loc[d.groupby("file").cost.idxmin(), "s"].value_counts(normalize=True)
            for s, sh in win[win >= min_share].items():
                score[s] = max(score.get(s, 0), sh)
            best = d.groupby("s").cost.sum().idxmin()
            score[best] = 2.0
    keep = [s for s, _ in sorted(score.items(), key=lambda kv: -kv[1]) if s != STORE][:cap]
    return [STORE] + keep


SWEEP_SETS = {"vpic": ["ref-vpic-126-2000", "consumer-vpic-full"],
              "lammps": ["ref-lammps-b70-2000", "consumer-lammps-full"],
              "nanoaod": ["hep-nanoaod"], "fem": ["sparse-fem"]}


# ----------------------------------------------------------------- profile
def profile(wl, frames, cands, out):
    """Every producer and output chunk through every candidate. Cached."""
    path = os.path.join(out, "profile.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    codecs = {s: GpuCodec(s) for s in cands}
    rows = []
    for fr in frames:
        src = {k: torch.from_numpy(v).cuda() for k, v in wl.load(fr).items()}
        outs = {k: as_bytes(v) for k, v in wl.consume(fr, src).items()}
        sync()
        for kind, arrays in (("producer", src), ("output", outs)):
            for name, t in arrays.items():
                for key, x in chunks(name, t):
                    for s, c in codecs.items():
                        y, cms = c.compress(x)
                        if y is None:
                            rows.append((fr, kind, key, x.numel(), s, x.numel(), 0.0, 0.0, 0))
                            continue
                        z, dms = c.decompress(y, x.numel())
                        rows.append((fr, kind, key, x.numel(), s, y.numel(), cms, dms,
                                     int(torch.equal(z, x))))
        print(f"profile {wl.name} frame {fr}: {len(rows)} rows", flush=True)
    p = pd.DataFrame(rows, columns=["frame", "kind", "key", "bytes", "setting", "comp_bytes",
                                    "comp_ms", "decomp_ms", "ok"])
    p.to_csv(path, index=False)
    return p


def plan(p, bw_w, bw_r, reads, out):
    """none / best / perchunk settings per chunk, from the profile."""
    p = p.copy()
    store = p[p.setting == STORE].set_index(["frame", "kind", "key"])
    bad = p.ok == 0  # cannot take the chunk (or not bit-exact): stored raw
    k = pd.MultiIndex.from_frame(p.loc[bad, ["frame", "kind", "key"]])
    p.loc[bad, "comp_bytes"] = p.loc[bad, "bytes"]
    p.loc[bad, "comp_ms"] = store.reindex(k)["comp_ms"].to_numpy()
    p.loc[bad, "decomp_ms"] = store.reindex(k)["decomp_ms"].to_numpy()
    io_w = p.comp_bytes / (bw_w * 1e6)
    io_r = p.comp_bytes / (bw_r * 1e6)
    prod = p.kind == "producer"
    p["cost"] = p.comp_ms + io_w + prod * reads * (io_r + p.decomp_ms)
    best = p.groupby("setting").cost.sum().idxmin()
    pick = p.loc[p.groupby(["frame", "kind", "key"]).cost.idxmin()]
    keyf = lambda r: f"{r.frame}|{r.kind}|{r.key}"  # noqa: E731
    plans = {"none": {}, "best": {}, "perchunk": {keyf(r): r.setting for r in pick.itertuples()}}
    for r in pick.itertuples():
        plans["none"][keyf(r)] = STORE
        plans["best"][keyf(r)] = best
    cost = p.set_index(["frame", "kind", "key", "setting"]).cost
    summary = {}
    for pol, pl in plans.items():
        idx = [(int(f), kd, kk, st) for x, st in pl.items() for f, kd, kk in [x.split("|", 2)]]
        summary[pol] = float(cost.loc[idx].sum())
    meta = {"best_setting": best, "bw_write_gbs": bw_w, "bw_read_gbs": bw_r, "reads": reads,
            "modelled_cost_ms": summary,
            "perchunk_mix": pick.setting.value_counts(normalize=True).round(4).to_dict()}
    json.dump({"meta": meta, "plans": plans}, open(os.path.join(out, "plan.json"), "w"), indent=1)
    return plans, meta


# ----------------------------------------------------------------- run
def run(wl, frames, plans, policy, rep, store, verify):
    """One timed producer -> storage -> consumer pass. @return metrics dict"""
    for f in glob.glob(os.path.join(store, "*.bin")):
        os.remove(f)
    pl = plans[policy]
    codecs = {s: GpuCodec(s) for s in set(pl.values())}
    m = dict.fromkeys(["prod_compress_s", "prod_d2h_s", "prod_write_s", "cons_read_s",
                       "cons_h2d_s", "cons_decompress_s", "cons_analysis_s", "cons_out_compress_s",
                       "cons_out_write_s", "prod_gpu_ms", "cons_gpu_ms", "out_gpu_ms",
                       "prod_in", "prod_stored", "out_in", "out_stored"], 0.0)
    index = {}
    t_prod = 0.0
    # ---- producer
    for fr in frames:
        src = {k: torch.from_numpy(v).cuda() for k, v in wl.load(fr).items()}
        sync()
        t0 = time.perf_counter()
        fn = os.path.join(store, f"frame_{fr:05d}.bin")
        ent = []
        with open(fn, "wb", buffering=0) as fh:
            pos = 0
            for name, t in src.items():
                for key, x in chunks(name, t):
                    s = pl[f"{fr}|producer|{key}"]
                    a = time.perf_counter()
                    y, ms = codecs[s].compress(x)
                    if y is None:
                        y, ms = codecs.setdefault(STORE, GpuCodec(STORE)).compress(x)
                        s = STORE
                    sync()
                    b = time.perf_counter()
                    h = y.cpu().numpy()
                    c = time.perf_counter()
                    fh.write(h.tobytes())
                    d = time.perf_counter()
                    m["prod_compress_s"] += b - a
                    m["prod_d2h_s"] += c - b
                    m["prod_write_s"] += d - c
                    m["prod_gpu_ms"] += ms
                    m["prod_in"] += x.numel()
                    m["prod_stored"] += y.numel()
                    ent.append((name, key, pos, y.numel(), x.numel(), s))
                    pos += y.numel()
            a = time.perf_counter()
            os.fsync(fh.fileno())
            m["prod_write_s"] += time.perf_counter() - a
        index[fr] = ent
        t_prod += time.perf_counter() - t0
        del src
    drop_caches()
    # ---- consumer
    t_cons = 0.0
    for fr in frames:
        t0 = time.perf_counter()
        a = time.perf_counter()
        with open(os.path.join(store, f"frame_{fr:05d}.bin"), "rb", buffering=0) as fh:
            blob = np.frombuffer(fh.read(), dtype=np.uint8)
        b = time.perf_counter()
        dev = torch.from_numpy(blob).cuda()
        sync()
        c = time.perf_counter()
        m["cons_read_s"] += b - a
        m["cons_h2d_s"] += c - b
        parts = {}
        for name, key, pos, n, orig, s in index[fr]:
            # own (aligned) buffer: chunks sit at arbitrary offsets in the
            # file, and the decoders read their headers as aligned words
            z, ms = codecs.setdefault(s, GpuCodec(s)).decompress(dev[pos:pos + n].clone(), orig)
            parts.setdefault(name, []).append(z)
            m["cons_gpu_ms"] += ms
        arrays = {k: torch.cat(v) for k, v in parts.items()}
        sync()
        d = time.perf_counter()
        m["cons_decompress_s"] += d - c
        outs = {k: as_bytes(v) for k, v in wl.consume(fr, arrays).items()}
        sync()
        e = time.perf_counter()
        m["cons_analysis_s"] += e - d
        with open(os.path.join(store, f"out_{fr:05d}.bin"), "wb", buffering=0) as fh:
            for name, t in outs.items():
                for key, x in chunks(name, t):
                    s = pl.get(f"{fr}|output|{key}", STORE)
                    a2 = time.perf_counter()
                    y, ms = codecs.setdefault(s, GpuCodec(s)).compress(x)
                    if y is None:
                        y, ms = codecs.setdefault(STORE, GpuCodec(STORE)).compress(x)
                    sync()
                    b2 = time.perf_counter()
                    fh.write(y.cpu().numpy().tobytes())
                    c2 = time.perf_counter()
                    m["cons_out_compress_s"] += b2 - a2
                    m["cons_out_write_s"] += c2 - b2
                    m["out_gpu_ms"] += ms
                    m["out_in"] += x.numel()
                    m["out_stored"] += y.numel()
            a2 = time.perf_counter()
            os.fsync(fh.fileno())
            m["cons_out_write_s"] += time.perf_counter() - a2
        t_cons += time.perf_counter() - t0
        if verify:  # untimed: every decompressed array against the source
            src = wl.load(fr)
            for k, v in arrays.items():
                if not torch.equal(v, torch.from_numpy(src[k]).cuda()):
                    raise RuntimeError(f"{wl.name} frame {fr} {k}: decompressed data differs")
        del dev, arrays, outs
    m.update(prod_total_s=t_prod, cons_total_s=t_cons, end_to_end_s=t_prod + t_cons)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=list(WORKLOADS))
    ap.add_argument("--store", required=True, help="directory on the storage device under test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--frames", type=int, default=0, help="first N frames only (0 = all)")
    ap.add_argument("--policies", default="none,best,perchunk")
    ap.add_argument("--candidates", default="", help="file of setting strings; default: from the sweep")
    ap.add_argument("--reads", type=int, default=1, help="consumer reads per producer chunk (cost model)")
    ap.add_argument("--bw-write", type=float, default=0.0)
    ap.add_argument("--bw-read", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.store, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    wl = WORKLOADS[a.workload]()
    frames = wl.frames()[:a.frames] if a.frames else wl.frames()
    bw_w, bw_r = (a.bw_write, a.bw_read) if a.bw_write and a.bw_read else probe_bandwidth(a.store)
    cands = ([l.strip() for l in open(a.candidates) if l.strip() and not l.startswith("#")]
             if a.candidates else
             candidates_from_sweep(SWEEP_SETS[a.workload], bws=(12, 1, 0.5, 0.25, bw_w)))
    open(os.path.join(a.out, "candidates.txt"), "w").write("\n".join(cands) + "\n")
    print(f"{wl.name}: {len(frames)} frames, {len(cands)} candidates, store {a.store} "
          f"write {bw_w:.2f} GB/s read {bw_r:.2f} GB/s", flush=True)
    prof = profile(wl, frames, cands, a.out)
    plans, meta = plan(prof, bw_w, bw_r, a.reads, a.out)
    print("plan:", json.dumps({k: meta[k] for k in ("best_setting", "modelled_cost_ms")}), flush=True)
    rows = []
    for rep in range(1, a.reps + 1):
        for pol in a.policies.split(","):
            m = run(wl, frames, plans, pol, rep, a.store, verify=(rep == 1))
            m.update(workload=wl.name, policy=pol, rep=rep, frames=len(frames))
            rows.append(m)
            print(f"rep {rep} {pol:9s} end-to-end {m['end_to_end_s']:.2f} s  "
                  f"producer {m['prod_total_s']:.2f} s  consumer {m['cons_total_s']:.2f} s  "
                  f"stored {m['prod_stored'] / 1e9:.2f}+{m['out_stored'] / 1e9:.2f} GB", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "runs.csv"), index=False)
    for f in glob.glob(os.path.join(a.store, "*.bin")):
        os.remove(f)


if __name__ == "__main__":
    main()
