#!/usr/bin/env python3
"""CC2 tables from one run directory produced by compare_perchunk_oracle.sh.

  Table 1  mean per-chunk cost = compress_ms + stored_bytes / BW
  Table 2  the oracle's per-chunk winner mix, by data content
  Table 3  amortization crossover: the bandwidth at which each fixed codec
           stops losing to the best fixed nvCOMP action

Compression time is MEASURED; transfer is MODELLED at BW. Every row is scored
on the same chunks -- the intersection of what all arms produced -- because a
mean over different chunk sets is not a comparison.

"Best fixed nvCOMP codec" comes from the SWEEP, which measured all 32 actions
on every chunk, not from a separate timed run. Naming anything other than the
cheapest of those 32 understates the fixed baseline and inflates every ratio
reported against it.
"""
import csv, os, re, sys, collections

EPS = 1e-12
FRAME = re.compile(r'^(plt\d+|step\d+|step_\d+|frame\d+)$')
EXT = (("cusz", "cuSZ only"), ("cuszp", "cuSZp3 only"), ("ndzip", "ndzip only"))

def field_of(blob):
    p = blob.split("/")
    if len(p) < 2:
        return p[0]
    name = p[0] if FRAME.match(p[1]) else p[1]
    return re.sub(r'^fab\d+_comp\d+_', '', name)

def action(r):
    return f"{r['lib_name'].replace('nvcomp-','')}|q{r['quantize']}|s{r['shuffle']}"

def _find(d, name):
    """run_config.sh nests a run under <results>/<tag>/; campaign runners do not."""
    for p in (os.path.join(d, os.path.basename(d), name), os.path.join(d, name)):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None

def sweep(d, bw):
    """blob -> {action: (cost_ms, ct_ms, bytes)} for all 32 nvcomp actions."""
    p = _find(f"{d}/best", "explore.csv")
    if not p:
        return None
    per = collections.defaultdict(dict)
    for r in csv.DictReader(open(p)):
        if r.get("quality_measured", "NA") == "0":
            continue                                  # bound violated: not usable
        cb = float(r["chunk_bytes"]); by = cb / max(float(r["ratio"]), EPS)
        ct = float(r["ct_ms"]); a = action(r)
        cur = per[r["blob"]].get(a)
        c = ct + by / bw
        if cur is None or c < cur[0]:
            per[r["blob"]][a] = (c, ct, by)
    return per

def ledger(d, bw):
    """blob -> (cost_ms, ct_ms, bytes) from a single-codec or NeuroPress arm."""
    p = _find(d, "blobs.csv")
    if not p:
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        if r.get("rc", "0") != "0":
            continue
        ct = float(r["compress_ms"]); by = float(r["stored"])
        out[r["blob"]] = (ct + by / bw, ct, by)
    return out or None

def fidelity(path):
    if not os.path.exists(path):
        return "?"
    t = open(path, errors="replace").read()
    if "BOUND FAILED" in t:
        m = re.search(r"BOUND FAILED: .*?(\d+) exceeded", t)
        return f"BOUND FAILED ({m.group(1)})" if m else "BOUND FAILED"
    if "BOUND OK" in t: return "bound OK"
    if "VERIFIED" in t: return "bit-exact"
    return "unchecked"

def main(root, wl, bw):
    sw = sweep(root, bw)
    if not sw:
        print(f"no sweep at {root}/best -- run without --analyze-only first"); return
    arms = {"NeuroPress (lossy low)": ledger(f"{root}/learn", bw)}
    for key, label in EXT:
        arms[label] = ledger(f"{root}/static-{key}", bw)

    blobs = set(sw)
    for m in arms.values():
        if m: blobs &= set(m)
    if not blobs:
        print("no chunks common to every arm"); return
    n = len(blobs)

    # per-chunk optimum over EVERY measured candidate, nvcomp and external
    opt = 0.0
    winner = {}
    for b in blobs:
        cands = dict(sw[b])
        for key, label in EXT:
            m = arms[label]
            if m and b in m: cands[key] = m[b]
        w = min(cands, key=lambda a: cands[a][0])
        winner[b] = w
        opt += cands[w][0]

    # best FIXED nvcomp action: one action on every chunk, from the sweep
    tot = collections.defaultdict(float); cnt = collections.Counter()
    for b in blobs:
        for a, (c, _, _) in sw[b].items():
            tot[a] += c; cnt[a] += 1
    full = {a: v for a, v in tot.items() if cnt[a] == n}
    bfa = min(full, key=full.get) if full else None

    print(f"\n{'='*76}")
    print(f"{wl.upper()}  --  CC2 table 1: mean per-chunk cost   (n={n}, BW={bw/1e6:.2f} GB/s)")
    print(f"{'='*76}")
    rows = [("Per-chunk optimum", opt / n, "")]
    if arms["NeuroPress (lossy low)"]:
        m = arms["NeuroPress (lossy low)"]
        rows.append(("NeuroPress (lossy low)", sum(m[b][0] for b in blobs)/n,
                     fidelity(f"{root}/learn.console")))
    if bfa:
        rows.append((f"Best fixed nvCOMP codec", full[bfa]/n, bfa))
    for key, label in EXT:
        m = arms[label]
        if m:
            rows.append((label, sum(m[b][0] for b in blobs)/n,
                         fidelity(f"{root}/static-{key}.console")))
    base = rows[0][1]
    print(f"{'Strategy':28s}{'ms':>9s}{'vs opt':>9s}   note")
    for name, c, note in rows:
        print(f"{name:28s}{c:9.2f}{c/max(base,EPS):8.2f}x   {note}")

    print(f"\n{'='*76}")
    print(f"CC2 table 2: oracle per-chunk winner mix, by data content")
    print(f"{'='*76}")
    per = collections.defaultdict(list)
    for b in blobs:
        per[field_of(b)].append(winner[b])
    print(f"{'content':22s}{'chunks':>7s}  winner mix")
    for f in sorted(per):
        c = collections.Counter(per[f]); t = len(per[f])
        print(f"{f:22s}{t:7d}  " + ", ".join(f"{a} {100*k/t:.0f}%" for a, k in c.most_common(3)))

    # Amortization: a slow codec only wins where transfer dominates its kernel.
    # Solve ct_x + B_x/bw == ct_ref + B_ref/bw for bw.
    if bfa:
        print(f"\n{'='*76}")
        print(f"CC2 table 3: amortization crossover vs best fixed ({bfa})")
        print(f"{'='*76}")
        rct = sum(sw[b][bfa][1] for b in blobs)/n
        rby = sum(sw[b][bfa][2] for b in blobs)/n
        print(f"{'codec':16s}{'kernel ms':>11s}{'stored MiB':>12s}   crossover bandwidth")
        print(f"{bfa:16s}{rct:11.2f}{rby/2**20:12.3f}   (reference)")
        for key, label in EXT:
            m = arms[label]
            if not m: continue
            ct = sum(m[b][1] for b in blobs)/n; by = sum(m[b][2] for b in blobs)/n
            dct, dby = ct - rct, by - rby
            # cost_x - cost_ref = dct + dby/bw. Setting it to zero gives the
            # crossover bw = -dby/dct. Which SIDE of it wins depends on which
            # term is the penalty: a slower-but-smaller codec is amortized only
            # at LOW bandwidth (transfer dominates), a faster-but-bigger one
            # only at HIGH bandwidth (kernel dominates).
            if dct <= 0 and dby <= 0:   verdict = "wins at every bandwidth"
            elif dct >= 0 and dby >= 0: verdict = "loses at every bandwidth"
            else:
                bwx = -dby / dct                      # bytes/ms, always > 0 here
                side = "below" if dct > 0 else "above"
                verdict = (f"wins {side} {bwx/1e6:.3f} GB/s "
                           f"({bwx*1000/2**20:.0f} MiB/s)")
            print(f"{key:16s}{ct:11.2f}{by/2**20:12.3f}   {verdict}")
        print("\n  a slow codec pays off only when transfer dominates its kernel;")
        print("  above the crossover the kernel cost is never amortized.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: perchunk_oracle_tables.py <run dir> <workload> [bw_bytes_per_ms]")
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.2e6)
