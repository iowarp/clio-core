#!/usr/bin/env python3
"""Phase 1b: does online SGD change what Phase 1 measured on frozen weights?

Compares two no-sweep arms (adaptive / frozen) against the two reference
points Phase 1 established on the SAME chunks: the in-action-space oracle and
the best deployable fixed action.

Cost is W_IO=1 only, so cost is proportional to stored bytes and every number
here is a byte count. Ratios are global (sum of inputs / sum of outputs),
never a mean of per-chunk ratios.
"""
import csv, os, sys, collections

EPS = 1e-12

def action(r):
    return f"{r['lib_name']}|q{r['quantize']}|s{r['shuffle']}"

def valid(r):
    # 'NA' is lossless (no bound to violate); only an explicit 0 is a failure.
    return r.get("quality_measured", "NA") != "0"

def arm_totals(d, keep=None):
    """Input and output bytes for one arm, from the per-blob ledger.

    `keep` restricts to the chunks the Phase 1 sweep actually covers. LAMMPS
    dies of CUDA OOM partway through, and a no-sweep arm gets further than the
    32x sweep did, so without this the arm's total would be compared against a
    fixed-action baseline built from FEWER chunks -- a guaranteed wrong answer.
    """
    p = f"{d}/blobs.csv"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    rows = [r for r in csv.DictReader(open(p)) if r.get("rc", "0") == "0"]
    if keep is not None:
        rows = [r for r in rows if r["blob"] in keep]
    if not rows:
        return None
    tin = sum(float(r["bytes"]) for r in rows)
    # codec bytes (comparable with Phase 1, which works from `ratio`) and
    # physically stored bytes (includes the frame header) side by side.
    tcodec = sum(float(r["bytes"]) / max(float(r["ratio"]), EPS) for r in rows)
    tstored = sum(float(r["stored"]) for r in rows)
    census = collections.Counter(r["codec"] or r["lib"] for r in rows)
    return {"n": len(rows), "in": tin, "codec_bytes": tcodec,
            "stored": tstored, "census": census}

def phase1_reference(d):
    """Oracle and best-fixed-action byte totals from the Phase 1 sweep."""
    p = f"{d}/explore.csv"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    per = collections.OrderedDict()
    for r in csv.DictReader(open(p)):
        per.setdefault(r["blob"], []).append(r)

    tin = 0.0
    b_or = 0.0
    b_np = 0.0
    fixed = collections.defaultdict(list)
    blobs = set()
    n = 0
    for blob, cands in per.items():
        vs = [c for c in cands if valid(c)]
        if not vs:
            continue
        pri = [c for c in cands if c["role"] == "primary"]
        if len(pri) != 1:
            continue
        n += 1
        blobs.add(blob)
        cb = float(cands[0]["chunk_bytes"])
        tin += cb
        orc = min(vs, key=lambda c: float(c["cost"]))
        b_or += cb / max(float(orc["ratio"]), EPS)
        b_np += cb / max(float(pri[0]["ratio"]), EPS)
        # One entry per (chunk, executed config): on WarpX a refused q1
        # candidate EXECUTES as its q0 twin, so two rows are one config.
        pc = {}
        for c in vs:
            a, by = action(c), cb / max(float(c["ratio"]), EPS)
            if a not in pc or by < pc[a]:
                pc[a] = by
        for a, by in pc.items():
            fixed[a].append(by)

    # Candidates whose measured ratio clears the cost model's 100x ceiling
    # (CLIO_NEUROPRESS_RATIO_CAP, neuropress_bridge.cc:77). Their `cost` column
    # scores them as exactly 100x, so the selector cannot tell a 227x action
    # from a 100x one. Everything above is computed from RAW ratio -- physical
    # bytes -- which is why these totals can differ from a cost-column ranking.
    clipped = sum(1 for cs in per.values() for c in cs
                  if float(c["ratio"]) > 100.0)
    total_c = sum(len(cs) for cs in per.values())

    # A fixed action is only a baseline if it is usable on EVERY chunk.
    full = {a: v for a, v in fixed.items() if len(v) == n}
    best = min(full.items(), key=lambda kv: sum(kv[1])) if full else None
    return {"n": n, "in": tin, "oracle": b_or, "frozen_np": b_np,
            "clipped": clipped, "total_c": total_c, "blobs": blobs,
            "best_fixed": (best[0], sum(best[1])) if best else None}

def main(root, wl):
    ref = phase1_reference(os.path.join(H, "phase1", wl, "oracle"))
    keep = ref["blobs"] if ref else None
    arms = {a: arm_totals(f"{root}/{a}", keep) for a in ("adaptive", "frozen")}
    if ref:
        for a in ("adaptive", "frozen"):
            raw = arm_totals(f"{root}/{a}")
            if raw and arms[a] and raw["n"] != arms[a]["n"]:
                print(f"  NOTE {a}: {raw['n']} chunks written, scored on the "
                      f"{arms[a]['n']} the Phase 1 sweep also covers")

    print(f"\n{'='*70}\n{wl.upper()}  --  Phase 1b, no sweep, ratio cost\n{'='*70}")
    for a in ("adaptive", "frozen"):
        t = arms[a]
        if not t:
            print(f"  {a:9s} no blobs.csv -- arm did not produce output")
            continue
        print(f"  {a:9s} n={t['n']:4d}  global ratio {t['in']/max(t['codec_bytes'],EPS):7.2f}x"
              f"   stored {t['stored']/2**20:9.2f} MiB"
              f"   (on-tier ratio {t['in']/max(t['stored'],EPS):6.2f}x)")
        top = ", ".join(f"{k.replace('nvcomp-','')} {v}"
                        for k, v in t["census"].most_common(4))
        print(f"            codecs: {top}")

    ad, fz = arms["adaptive"], arms["frozen"]
    if ad and fz:
        d = (ad["codec_bytes"] - fz["codec_bytes"]) / max(fz["codec_bytes"], EPS)
        verdict = "SMALLER" if d < 0 else "LARGER"
        print(f"\n  adaptive vs frozen: {d*100:+.2f}% bytes  ({verdict} with SGD on)")

    if not ref:
        print("\n  no Phase 1 sweep for this workload -- no oracle reference")
        return
    print(f"\n  -- Phase 1 reference, same {ref['n']} chunks (raw bytes) --")
    if ref["clipped"]:
        print(f"  NOTE {ref['clipped']}/{ref['total_c']} candidates exceed the "
              f"cost model's 100x cap and are scored as 100x")
    print(f"  oracle (in-space)        {ref['in']/max(ref['oracle'],EPS):7.2f}x")
    print(f"  frozen model (Phase 1)   {ref['in']/max(ref['frozen_np'],EPS):7.2f}x")
    if ref["best_fixed"]:
        a, by = ref["best_fixed"]
        print(f"  best fixed action        {ref['in']/max(by,EPS):7.2f}x   {a}")
        if ad:
            # The claim under test: does adaptation beat the best single
            # action you could have hardcoded and never touched again?
            r = ad["codec_bytes"] / max(by, EPS)
            print(f"\n  ADAPTIVE / BEST-FIXED    {r:.3f}x  "
                  f"({'adaptive wins' if r < 1.0 else 'fixed action still wins'})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: phase1b_analyze.py <phase1b/WL dir> <workload>")
    main(sys.argv[1], sys.argv[2])
