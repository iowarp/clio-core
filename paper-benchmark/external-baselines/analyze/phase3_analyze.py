#!/usr/bin/env python3
"""Phase 3: the external codecs against what the action space can reach.

cusz/cuszp/ndzip are OUTSIDE NeuroPress's 32-action space by construction
(kNeuroPressTrainedGpuBaseIds), so these are not selector errors -- they
measure what the space itself is missing.

Every number is bytes under W_IO=1. Ratios are global (sum in / sum out).
A codec whose fidelity was not verified is reported as such rather than
being quietly ranked: the campaign's first cuSZ result claimed 9.387x with
its bound unchecked, and it had exceeded eb on 118 of 120 chunks.
"""
import csv, json, os, re, sys, collections

EPS = 1e-12
H = os.environ.get("WORK", os.path.join(os.getcwd(), "np-baselines"))

def arm(d, keep=None):
    p = f"{d}/blobs.csv"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    rows = [r for r in csv.DictReader(open(p)) if r.get("rc", "0") == "0"]
    if keep is not None:
        rows = [r for r in rows if r["blob"] in keep]
    if not rows:
        return None
    tin = sum(float(r["bytes"]) for r in rows)
    tout = sum(float(r["stored"]) for r in rows)
    census = collections.Counter(r["codec"] or r["lib"] for r in rows)
    return {"n": len(rows), "in": tin, "out": tout, "census": census}

def fidelity(d):
    """What the run actually proved about fidelity, from its own stdout."""
    p = f"{d}/stdout.log"
    if not os.path.exists(p):
        return "no stdout"
    t = open(p, errors="replace").read()
    if "BOUND FAILED" in t:
        m = re.search(r"BOUND FAILED:.*", t)
        return "VIOLATED: " + (m.group(0)[:70] if m else "bound exceeded")
    if "BOUND OK" in t:
        return "bound OK"
    if "VERIFIED" in t:
        return "bit-exact"
    return "NOT VERIFIED"

def in_space(wl):
    """Oracle and best fixed action from the Phase 1 sweep, in raw bytes."""
    p = f"{H}/phase1/{wl}/oracle/explore.csv"
    if not os.path.exists(p):
        return None
    per = collections.OrderedDict()
    for r in csv.DictReader(open(p)):
        per.setdefault(r["blob"], []).append(r)
    tin = b_or = b_np = 0.0
    fixed = collections.defaultdict(list); blobs = set(); n = 0
    for blob, cs in per.items():
        vs = [c for c in cs if c.get("quality_measured", "NA") != "0"]
        pri = [c for c in cs if c["role"] == "primary"]
        if not vs or len(pri) != 1:
            continue
        n += 1; blobs.add(blob)
        cb = float(cs[0]["chunk_bytes"]); tin += cb
        b_or += min(cb / max(float(c["ratio"]), EPS) for c in vs)
        b_np += cb / max(float(pri[0]["ratio"]), EPS)
        pc = {}
        for c in vs:
            a = f"{c['lib_name']}|q{c['quantize']}|s{c['shuffle']}"
            by = cb / max(float(c["ratio"]), EPS)
            if a not in pc or by < pc[a]:
                pc[a] = by
        for a, by in pc.items():
            fixed[a].append(by)
    full = {a: v for a, v in fixed.items() if len(v) == n}
    best = min(full.items(), key=lambda kv: sum(kv[1])) if full else None
    return {"n": n, "in": tin, "oracle": b_or, "frozen": b_np, "blobs": blobs,
            "best_fixed": (best[0], sum(best[1])) if best else None}

def main(root, wl):
    ref = in_space(wl)
    keep = ref["blobs"] if ref else None
    print(f"\n{'='*74}\n{wl.upper()}  --  Phase 3: state-of-the-art baselines\n{'='*74}")

    rows = []
    for c in ("cusz", "cuszp", "ndzip"):
        t = arm(f"{root}/{c}", keep)
        if not t:
            print(f"  {c:22s} no output")
            continue
        fid = fidelity(f"{root}/{c}")
        # WireIdForName falls back to zstd for a name this build cannot make,
        # so a wrong census means the arm is mislabelled, not merely poor.
        got = t["census"].most_common(1)[0][0] if t["census"] else "?"
        ok = c in got.lower()
        rows.append((c, t, fid, ok, got))
        flag = "" if ok else f"   !! census says '{got}', NOT {c}"
        print(f"  {c:22s} n={t['n']:4d}  {t['in']/max(t['out'],EPS):7.2f}x"
              f"   {t['out']/2**20:9.2f} MiB   [{fid}]{flag}")

    if not ref:
        print("\n  no Phase 1 sweep for this workload -- nothing to compare against")
        return
    print(f"\n  -- inside NeuroPress's 32-action space, same {ref['n']} chunks --")
    print(f"  {'in-space oracle':22s} {ref['in']/max(ref['oracle'],EPS):7.2f}x")
    if ref["best_fixed"]:
        a, by = ref["best_fixed"]
        print(f"  {'best fixed action':22s} {ref['in']/max(by,EPS):7.2f}x   {a}")
    print(f"  {'NeuroPress (frozen)':22s} {ref['in']/max(ref['frozen'],EPS):7.2f}x")
    ad = arm(f"{H}/phase1b/{wl}/adaptive", keep)
    if ad:
        print(f"  {'NeuroPress (adaptive)':22s} {ad['in']/max(ad['out'],EPS):7.2f}x")

    # The question Phase 3 exists to answer.
    print(f"\n  -- what the action space gives up --")
    for c, t, fid, ok, _ in rows:
        if not ok:
            print(f"  {c:8s} SKIPPED -- mislabelled arm, not a {c} result")
            continue
        vs_or = t["out"] / max(ref["oracle"], EPS)
        vs_np = t["out"] / max(ad["out"] if ad else ref["frozen"], EPS)
        who = "adaptive" if ad else "frozen"
        verdict = ("BEATS the whole action space" if vs_or < 1.0 else
                   "inside the space is better")
        note = "" if fid in ("bound OK", "bit-exact") else "   [fidelity unproven]"
        print(f"  {c:8s} vs oracle {vs_or:5.2f}x   vs NeuroPress({who}) {vs_np:5.2f}x"
              f"   {verdict}{note}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: phase3_analyze.py <phase3/WL dir> <workload>")
    main(sys.argv[1], sys.argv[2])
