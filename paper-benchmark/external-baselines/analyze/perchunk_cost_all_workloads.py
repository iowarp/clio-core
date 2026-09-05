#!/usr/bin/env python3
"""CC1/CC2-style tables on Clio/A100 data.

COST MODEL: mean per-chunk cost = compression time + transfer, exactly as the
rebuttal's CC2 defines it (and as upstream's trace campaign computes it):

    cost_ms = compress_ms + stored_bytes / BW

This is NOT the ratio-only model Phase 1/1b used (W_CT=0 W_DT=0 W_IO=1). Under
a time-inclusive cost a slow high-ratio codec is penalised for its kernel; under
a bytes-only cost it is not. cuSZ's standing depends entirely on which is used,
so BW is reported at several values rather than one.

The per-chunk optimum is taken over ALL measured strategies -- the 32 nvcomp
actions from the Phase 1 sweep PLUS the external codecs from Phase 3 -- so it
is a true cross-codec oracle, not an in-space one.
"""
import csv, os, sys, collections

EPS = 1e-12
H = os.environ.get("WORK", os.path.join(os.getcwd(), "np-baselines"))
EXT = ("cusz", "cuszp", "ndzip")

def nvcomp_per_chunk(wl):
    """{blob: {action: (ct_ms, stored_bytes)}} for the 32 nvcomp actions."""
    p = f"{H}/phase1/{wl}/oracle/explore.csv"
    if not os.path.exists(p):
        return None
    per = collections.OrderedDict()
    for r in csv.DictReader(open(p)):
        if r.get("quality_measured", "NA") == "0":
            continue                      # violated the bound: not a candidate
        cb = float(r["chunk_bytes"])
        by = cb / max(float(r["ratio"]), EPS)
        a = f"{r['lib_name']}|q{r['quantize']}|s{r['shuffle']}"
        d = per.setdefault(r["blob"], {})
        # one entry per executed config, cheapest representative
        if a not in d or by < d[a][1]:
            d[a] = (float(r["ct_ms"]), by)
    return per

def blobs_per_chunk(d):
    """{blob: (compress_ms, stored_bytes)} from a per-blob ledger."""
    p = f"{d}/blobs.csv"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        if r.get("rc", "0") != "0":
            continue
        out[r["blob"]] = (float(r["compress_ms"]), float(r["stored"]))
    return out or None

def table(wl, bw):
    nv = nvcomp_per_chunk(wl)
    if not nv:
        return None
    ext = {c: blobs_per_chunk(f"{H}/phase3/{wl}/{c}") for c in EXT}
    # Prefer the arm run under the SAME cost weights this table scores. Runs
    # made with W_CT=0 optimised bytes only and would be graded on an objective
    # they were never given.
    npz = (blobs_per_chunk(f"{H}/phase1b/{wl}-cc/adaptive")
           or blobs_per_chunk(f"{H}/phase1b/{wl}/adaptive"))

    # Score only chunks every arm actually produced, or the means are not
    # comparable across rows.
    blobs = set(nv)
    for m in list(ext.values()) + [npz]:
        if m:
            blobs &= set(m)
    if not blobs:
        return None
    n = len(blobs)
    cost = lambda ct, by: ct + by / bw

    rows = {}
    # per-chunk optimum over EVERY measured strategy
    opt = 0.0
    for b in blobs:
        best = min(cost(ct, by) for ct, by in nv[b].values())
        for c in EXT:
            if ext[c] and b in ext[c]:
                best = min(best, cost(*ext[c][b]))
        opt += best
    rows["per-chunk optimum"] = opt / n
    if npz:
        rows["NeuroPress (online)"] = sum(cost(*npz[b]) for b in blobs) / n
    # best FIXED nvcomp action: one action, every chunk, lowest total
    acts = collections.defaultdict(float); cnt = collections.Counter()
    for b in blobs:
        for a, (ct, by) in nv[b].items():
            acts[a] += cost(ct, by); cnt[a] += 1
    full = {a: v for a, v in acts.items() if cnt[a] == n}
    if full:
        ba = min(full, key=full.get)
        rows[f"best fixed nvCOMP"] = full[ba] / n
        rows["_bestfixed_name"] = ba
    for c in EXT:
        if ext[c]:
            rows[f"{c} only"] = sum(cost(*ext[c][b]) for b in blobs) / n
    rows["_n"] = n
    return rows

def main(bws):
    wls = [w for w in ("nyx", "vpic", "warpx", "lammps")
           if os.path.exists(f"{H}/phase1/{w}/oracle/explore.csv")]
    for bw in bws:
        print(f"\n{'='*78}")
        print(f"CC2-style: mean per-chunk cost = compress_ms + bytes/BW   "
              f"(BW = {bw/1e6:.1f} GB/s)")
        print(f"{'='*78}")
        tabs = {w: table(w, bw) for w in wls}
        tabs = {w: t for w, t in tabs.items() if t}
        if not tabs:
            print("  no data"); continue
        order = ["per-chunk optimum", "NeuroPress (online)", "best fixed nvCOMP",
                 "ndzip only", "cuszp only", "cusz only"]
        hdr = "".join(f"{w.upper():>18s}" for w in tabs)
        print(f"{'strategy':24s}{hdr}")
        print(f"{'':24s}" + "".join(f"{'n=%d' % tabs[w]['_n']:>18s}" for w in tabs))
        for k in order:
            key = k.replace("cuszp only", "cuszp only").replace("cusz only", "cusz only")
            cells = ""
            for w in tabs:
                t = tabs[w]
                if key not in t:
                    cells += f"{'--':>18s}"; continue
                base = t["per-chunk optimum"]
                cells += f"{t[key]:>10.2f} ({t[key]/max(base,EPS):4.2f}x)"
            print(f"{k:24s}{cells}")
        print(f"{'':24s}" + "".join(
            f"{tabs[w].get('_bestfixed_name','?').replace('nvcomp-',''):>18s}" for w in tabs))

if __name__ == "__main__":
    bws = [float(x) for x in sys.argv[1:]] or [1.2e6, 5e6]
    main(bws)
