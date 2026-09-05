#!/usr/bin/env python3
"""Phase 1: NeuroPress's per-chunk selection against the in-action-space oracle.

THE ORACLE IS MEASURED, NOT PREDICTED. best_mode compresses every chunk with all
32 actions and records each outcome, so the oracle is argmin(cost) over measured
candidates -- verified to coincide with the `adopted` row on every chunk.

NO LEAKAGE, structurally: best_mode suppresses both SGD phases, so nothing
measured here reaches the weights.

WHAT THIS ORACLE EXCLUDES: cuSZ, ndzip and cuSZp are filtered out of NeuroPress's
candidate set by kNeuroPressTrainedGpuBaseIds, so they are not in the action
space and not in this oracle. Whether the SPACE is adequate is a separate
question (Phase 3) from whether the SELECTOR is good, which is this one.
"""
import csv, json, os, sys, collections, statistics as st

EPS = 1e-12

def action(r):
    """A fixed strategy is a codec AND its preprocessing, not a codec alone:
    zstd+quantize+shuffle4 and bare zstd are different strategies."""
    return f"{r['lib_name']}|q{r['quantize']}|s{r['shuffle']}"

def valid(r):
    # quality_measured is 'NA' on lossless actions (no error bound to violate)
    # and '1'/'0' on quantized ones. Only an explicit 0 is a violation.
    return r.get("quality_measured", "NA") != "0"

def load(d):
    ex = [r for r in csv.DictReader(open(f"{d}/explore.csv"))]
    per = collections.OrderedDict()
    for r in ex: per.setdefault(r["blob"], []).append(r)
    return per

def position(r):
    """Where this candidate sat in NeuroPress's own ranking. 0 = its pick."""
    return 0 if r["role"] == "primary" else int(r["rank"]) + 1

def analyze(wl, d):
    per = load(d)
    rows = []
    for blob, cands in per.items():
        vs = [c for c in cands if valid(c)]
        if not vs:
            rows.append({"blob": blob, "no_valid": True}); continue
        oracle = min(vs, key=lambda c: float(c["cost"]))
        prim = next(c for c in cands if c["role"] == "primary")
        jo, jp = float(oracle["cost"]), float(prim["cost"])
        rows.append({
            "blob": blob, "no_valid": False,
            "np_action": action(prim), "oracle_action": action(oracle),
            "match": action(prim) == action(oracle),
            "np_valid": valid(prim),
            "oracle_pos": position(oracle),          # for Top-k
            "J_np": jp, "J_or": jo,
            "regret": jp - jo,
            "rel_regret": (jp - jo) / (abs(jo) + EPS),
            "bytes_np": float(prim["chunk_bytes"]) / max(float(prim["ratio"]), EPS),
            "bytes_or": float(prim["chunk_bytes"]) / max(float(oracle["ratio"]), EPS),
            "ratio_np": float(prim["ratio"]), "ratio_or": float(oracle["ratio"]),
            "ct_np": float(prim["ct_ms"]), "ct_or": float(oracle["ct_ms"]),
            "cands": cands,
        })
    good = [r for r in rows if not r["no_valid"]]
    n = len(good)
    if not n:
        print(f"  {wl}: no valid chunks"); return None

    exact = sum(r["match"] for r in good) / n
    topk = {k: sum(1 for r in good if r["oracle_pos"] < k) / n for k in (1, 2, 3, 5)}
    near = {d_: sum(1 for r in good if r["rel_regret"] <= d_) / n
            for d_ in (0.01, 0.02, 0.05, 0.10)}
    rr = sorted(r["rel_regret"] for r in good)

    print(f"\n{'='*66}\n{wl.upper()}  --  {n} chunks, {len(good[0]['cands'])} candidates each")
    print(f"{'='*66}")
    print(f"  EXACT oracle match      {exact*100:6.1f}%")
    print(f"  Top-1 / 2 / 3 / 5       " + " / ".join(f"{topk[k]*100:.1f}%" for k in (1,2,3,5)))
    print(f"  within  1% of oracle    {near[0.01]*100:6.1f}%")
    print(f"  within  2%              {near[0.02]*100:6.1f}%")
    print(f"  within  5%              {near[0.05]*100:6.1f}%")
    print(f"  within 10%              {near[0.10]*100:6.1f}%")
    print(f"  relative cost regret    mean {st.fmean(rr)*100:.2f}%   median {st.median(rr)*100:.2f}%"
          f"   p95 {rr[int(.95*len(rr))]*100:.2f}%   worst {rr[-1]*100:.2f}%")
    qv = sum(1 for r in good if not r["np_valid"])
    print(f"  NeuroPress quality violations  {qv}")

    # aggregate bytes: global ratio from totals, never a mean of ratios
    tot_in = sum(float(r["cands"][0]["chunk_bytes"]) for r in good)
    print(f"  global ratio  NeuroPress {tot_in/sum(r['bytes_np'] for r in good):7.2f}x"
          f"   oracle {tot_in/sum(r['bytes_or'] for r in good):7.2f}x"
          f"   byte overhead {(sum(r['bytes_np'] for r in good)/sum(r['bytes_or'] for r in good)-1)*100:+.2f}%")

    # mismatch anatomy
    mm = [r for r in good if not r["match"]]
    if mm:
        cat = collections.Counter(
            "near-tie (<1%)" if r["rel_regret"] <= .01 else
            "low regret (1-5%)" if r["rel_regret"] <= .05 else
            "moderate (5-20%)" if r["rel_regret"] <= .20 else "high (>20%)"
            for r in mm)
        print(f"  mismatches {len(mm)}: " + ", ".join(f"{k} {v}" for k, v in cat.most_common()))
        second = sum(1 for r in mm if r["oracle_pos"] == 1)
        print(f"    of those, oracle was NeuroPress's 2nd choice: {second}")

    # which actions the oracle wants vs which NeuroPress picks
    print("  top oracle actions      " + ", ".join(
        f"{a.split('|')[0].replace('nvcomp-','')}{a.split('|')[1]}{a.split('|')[2]} {c}"
        for a, c in collections.Counter(r["oracle_action"] for r in good).most_common(4)))
    print("  top NeuroPress actions  " + ", ".join(
        f"{a.split('|')[0].replace('nvcomp-','')}{a.split('|')[1]}{a.split('|')[2]} {c}"
        for a, c in collections.Counter(r["np_action"] for r in good).most_common(4)))

    # every fixed action as a whole-workload strategy
    # COLLAPSE DUPLICATE LABELS WITHIN A CHUNK. On WarpX the quantizer refuses
    # 2944/9600 candidates (index_exceeds_int32: eb 1e-3 needs a finer grid than
    # an int32 can address over that range), and a refused q1 candidate EXECUTES
    # as its q0 twin -- two rows, one configuration. Counting both would credit a
    # fixed strategy with 484 of 300 chunks. One entry per (chunk, executed
    # config), cheapest representative.
    fixed = collections.defaultdict(list)
    for r in good:
        per_chunk = {}
        for c in r["cands"]:
            if not valid(c): continue
            a, j = action(c), float(c["cost"])
            if a not in per_chunk or j < per_chunk[a]: per_chunk[a] = j
        for a, j in per_chunk.items(): fixed[a].append(j)
    # A fixed strategy is only a strategy if it is usable on EVERY chunk. An
    # action that fails the error bound on some chunks is not a baseline you
    # could actually deploy, so it is reported separately rather than being
    # silently credited with the chunks it happened to survive.
    J_np = sum(r["J_np"] for r in good)
    full = {a: v for a, v in fixed.items() if len(v) == n}
    partial = len(fixed) - len(full)
    if full:
        ranked = sorted(full.items(), key=lambda kv: sum(kv[1]))
        beat = sum(1 for a, v in full.items() if sum(v) < J_np)
        print(f"  BEST fixed action       {ranked[0][0]}  total cost {sum(ranked[0][1]):.3f}"
              f"  vs NeuroPress {J_np:.3f}  ({sum(ranked[0][1])/J_np:.2f}x)")
        print(f"  fixed actions beating NeuroPress in aggregate: {beat}/{len(full)}"
              f"   ({partial}/32 actions unusable -- quality-invalid on some chunk)")
    else:
        print(f"  NO fixed action is quality-valid on all {n} chunks "
              f"(all 32 fail the bound somewhere) -- NeuroPress total cost {J_np:.3f}")
        cover = sorted(((len(v), a, st.fmean(v)) for a, v in fixed.items()), reverse=True)[:3]
        print("    widest coverage: " + ", ".join(f"{a} {c}/{n} (mean {m:.4f})"
                                                  for c, a, m in cover))
        print(f"    NeuroPress mean cost {J_np/n:.4f} on all {n}")
    return {"wl": wl, "n": n, "exact": exact, "topk": topk, "near": near,
            "rows": good}

if __name__ == "__main__":
    H = os.path.join(os.environ.get("WORK",
        os.path.join(os.getcwd(), "np-baselines")), "phase1")
    out = []
    for wl in ("nyx", "vpic", "warpx", "lammps"):
        d = f"{H}/{wl}/oracle"
        if os.path.exists(f"{d}/explore.csv") and os.path.getsize(f"{d}/explore.csv") > 0:
            r = analyze(wl, d)
            if r: out.append(r)
        else:
            print(f"\n{wl.upper()}: not available yet")
    if out:
        print(f"\n{'='*66}\nSUMMARY\n{'='*66}")
        print(f"{'workload':10s} {'n':>6s} {'exact':>8s} {'top-2':>8s} {'top-3':>8s} {'<=1%':>8s} {'<=5%':>8s}")
        for r in out:
            print(f"{r['wl']:10s} {r['n']:6d} {r['exact']*100:7.1f}% {r['topk'][2]*100:7.1f}% "
                  f"{r['topk'][3]*100:7.1f}% {r['near'][0.01]*100:7.1f}% {r['near'][0.05]*100:7.1f}%")
