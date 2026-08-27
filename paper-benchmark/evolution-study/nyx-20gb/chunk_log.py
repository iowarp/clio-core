#!/usr/bin/env python3
"""Everything this run recorded about one chunk, or about all 4,800 of them.

    ./chunk_log.py                                   # index: what is joinable
    ./chunk_log.py --blob xmom/step_00799/fab0000/chunk_5
    ./chunk_log.py --field xmom --step 799 --chunk 5  # same, spelled out
    ./chunk_log.py --csv /tmp/chunks.csv              # all 4,800, one row each
    ./chunk_log.py --csv - | head                     # ...to stdout

Four files record the same run from four angles and NONE of them is a
superset of the others:

    insitu.selection.csv.gz  the model's INPUTS (entropy, mad, second_deriv)
                             and its PREDICTION, one row per chunk
    insitu.explore.csv.gz    every candidate MEASURED, 4 rows per chunk
    insitu.blobs.csv.gz      what actually reached the tier, one row per chunk
    blocks.csv.gz            how much the DATA IN that chunk moved since the
                             previous frame -- from the dump route, not this run

THE FOURTH ONE NEEDS TWO CONVERSIONS, which is the reason this script exists
rather than a join in awk:

  STEP. The in-situ hook fires after the step completes and labels the frame
  with the step index, so `step_00039` is the frame at simulation step 40.
  blocks.csv labels the same frame `step_to = 40`. dump_step = insitu_step + 1.

  BLOCK SIZE. A compressor chunk is 4 MiB; an evolution block is 1 MiB. Chunk c
  covers blocks 4c..4c+3, so the evolution figures here are aggregates over
  four rows, not a single lookup.

Both runs are the same simulation at the same settings (see README.md), so the
join is meaningful -- but they ARE two executions, and the evolution numbers
describe the dump route's bytes.
"""
import argparse, csv, gzip, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK_BYTES = 4 << 20
BLOCK_BYTES = 1 << 20
PER_CHUNK = CHUNK_BYTES // BLOCK_BYTES      # evolution blocks per compressor chunk


def load(name):
    with gzip.open(os.path.join(HERE, name), "rt") as fh:
        return list(csv.DictReader(fh))


def parse_blob(b):
    """'xmom/step_00799/fab0000/chunk_5' -> ('xmom', 799, 5)"""
    field, step, _fab, chunk = b.split("/")
    return field, int(step.removeprefix("step_")), int(chunk.removeprefix("chunk_"))


def read_all():
    sel = {r["blob"]: r for r in load("insitu.selection.csv.gz")}
    blobs = {r["blob"]: r for r in load("insitu.blobs.csv.gz")}
    exp = {}
    for r in load("insitu.explore.csv.gz"):
        exp.setdefault(r["blob"], []).append(r)

    # Evolution, keyed by (field, dump step_to, 1 MiB block index).
    ev = {}
    for r in load("blocks.csv.gz"):
        ev[(r["field"], int(r["step_to"]), int(r["block"]))] = r
    return sel, blobs, exp, ev


def ev_rows(ev, field, insitu_step, chunk):
    """The PER_CHUNK evolution blocks under one compressor chunk."""
    dump_step = insitu_step + 1
    out = []
    for b in range(chunk * PER_CHUNK, (chunk + 1) * PER_CHUNK):
        r = ev.get((field, dump_step, b))
        if r:
            out.append(r)
    return dump_step, out


def report(blob, sel, blobs, exp, ev):
    if blob not in sel and blob not in blobs:
        sys.exit(f"no such chunk: {blob}\n(try --csv - | head to see the names)")
    field, step, chunk = parse_blob(blob)
    s, b = sel.get(blob), blobs.get(blob)
    print(f"== {blob}")

    if s:
        print("\n-- what the model saw and predicted")
        print(f"   features   entropy {float(s['entropy']):.4f}   mad {float(s['mad']):.6g}"
              f"   second_deriv {float(s['second_deriv']):.6g}")
        print(f"   picked     {s['lib_name']} (algo {s['algo_idx']}, preset {s['preset']}, "
              f"quantize {s['quantize']}, shuffle {s['shuffle']})")
        print(f"   predicted  ratio {float(s['pred_ratio']):.2f}   "
              f"ct {float(s['pred_ct_ms']):.3f} ms   dt {float(s['pred_dt_ms']):.3f} ms")

    trials = sorted(exp.get(blob, []), key=lambda r: int(r["rank"]))
    if trials:
        print("\n-- every candidate, measured")
        print(f"   {'role':8} {'rank':>4}  {'lib':16} {'pred':>8} {'ratio':>8} "
              f"{'ct_ms':>8} {'dt_ms':>8} {'cost':>9}  adopted")
        for t in trials:
            print(f"   {t['role']:8} {t['rank']:>4}  {t['lib_name']:16} "
                  f"{float(t['pred_ratio']):8.2f} {float(t['ratio']):8.3f} "
                  f"{float(t['ct_ms']):8.3f} {float(t['dt_ms']):8.3f} "
                  f"{float(t['cost']):9.4f}  {'<-- YES' if t['adopted']=='1' else ''}")
        prim = next((t for t in trials if t["role"] == "primary"), None)
        adop = next((t for t in trials if t["adopted"] == "1"), None)
        if prim and adop and prim is not adop:
            print(f"   exploring overrode the model: cost {float(prim['cost']):.4f}"
                  f" -> {float(adop['cost']):.4f}, ratio {float(prim['ratio']):.3f}"
                  f" -> {float(adop['ratio']):.3f}")
    else:
        print("\n-- not explored (selection came from cache, not a fresh trial)")

    if b:
        print("\n-- what reached the tier")
        print(f"   {b['codec']}   {int(b['bytes']):,} B -> {int(b['stored']):,} B"
              f"   ratio {float(b['ratio']):.3f}   stored_ratio {float(b['stored_ratio']):.3f}")
        print(f"   compress {float(b['compress_ms']):.3f} ms   "
              f"decompress {float(b['decompress_ms']):.3f} ms   rc {b['rc']}   "
              f"fnv1a64 {b['fnv1a64']}")

    dump_step, rows = ev_rows(ev, field, step, chunk)
    print(f"\n-- how far the DATA moved, interval ending at dump step {dump_step}")
    if not rows:
        print("   no evolution rows (dump route has no frame here)")
        return
    print(f"   {'1MiB block':>10} {'evolution':>10} {'%cells same':>12}")
    for r in rows:
        print(f"   {r['block']:>10} {float(r['evolution']):10.6f} "
              f"{float(r['pct_cells_same']):12.4f}")
    es = [float(r["evolution"]) for r in rows]
    print(f"   {'mean':>10} {st.mean(es):10.6f} "
          f"{st.mean(float(r['pct_cells_same']) for r in rows):12.4f}")


def joined(sel, blobs, exp, ev, out):
    cols = ["blob", "field", "insitu_step", "dump_step", "chunk", "chunk_bytes",
            "entropy", "mad", "second_deriv",
            "pred_lib", "pred_ratio", "pred_ct_ms", "pred_dt_ms",
            "n_trials", "primary_lib", "primary_ratio", "primary_cost",
            "adopted_role", "adopted_rank", "adopted_lib", "adopted_cost",
            "cost_gain", "overrode_model",
            "stored_codec", "stored_ratio", "stored_bytes",
            "compress_ms", "decompress_ms", "rc",
            "ev_blocks", "ev_mean", "ev_max", "ev_pct_cells_same"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for blob in sorted(set(sel) | set(blobs)):
        field, step, chunk = parse_blob(blob)
        s, b = sel.get(blob, {}), blobs.get(blob, {})
        trials = exp.get(blob, [])
        prim = next((t for t in trials if t["role"] == "primary"), {})
        adop = next((t for t in trials if t["adopted"] == "1"), {})
        dump_step, rows = ev_rows(ev, field, step, chunk)
        es = [float(r["evolution"]) for r in rows]
        row = {
            "blob": blob, "field": field, "insitu_step": step,
            "dump_step": dump_step, "chunk": chunk,
            "chunk_bytes": s.get("chunk_bytes") or b.get("bytes", ""),
            "entropy": s.get("entropy", ""), "mad": s.get("mad", ""),
            "second_deriv": s.get("second_deriv", ""),
            "pred_lib": s.get("lib_name", ""), "pred_ratio": s.get("pred_ratio", ""),
            "pred_ct_ms": s.get("pred_ct_ms", ""), "pred_dt_ms": s.get("pred_dt_ms", ""),
            "n_trials": len(trials),
            "primary_lib": prim.get("lib_name", ""),
            "primary_ratio": prim.get("ratio", ""),
            "primary_cost": prim.get("cost", ""),
            "adopted_role": adop.get("role", ""), "adopted_rank": adop.get("rank", ""),
            "adopted_lib": adop.get("lib_name", ""), "adopted_cost": adop.get("cost", ""),
            "cost_gain": (f"{float(prim['cost']) - float(adop['cost']):.6f}"
                          if prim and adop else ""),
            "overrode_model": (1 if adop and prim and adop is not prim else 0) if trials else "",
            "stored_codec": b.get("codec", ""), "stored_ratio": b.get("stored_ratio", ""),
            "stored_bytes": b.get("stored", ""),
            "compress_ms": b.get("compress_ms", ""),
            "decompress_ms": b.get("decompress_ms", ""), "rc": b.get("rc", ""),
            "ev_blocks": len(rows),
            "ev_mean": f"{st.mean(es):.6f}" if es else "",
            "ev_max": f"{max(es):.6f}" if es else "",
            "ev_pct_cells_same": (f"{st.mean(float(r['pct_cells_same']) for r in rows):.4f}"
                                  if rows else ""),
        }
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", help="field/step_NNNNN/fabNNNN/chunk_N")
    ap.add_argument("--field")
    ap.add_argument("--step", type=int, help="in-situ step, e.g. 799")
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--csv", help="write the joined per-chunk table here ('-' for stdout)")
    a = ap.parse_args()

    sel, blobs, exp, ev = read_all()

    if a.csv:
        if a.csv == "-":
            joined(sel, blobs, exp, ev, sys.stdout)
        else:
            with open(a.csv, "w", newline="") as fh:
                joined(sel, blobs, exp, ev, fh)
            print(f"{len(set(sel) | set(blobs))} chunks -> {a.csv}")
        return

    blob = a.blob
    if not blob and a.field is not None and a.step is not None and a.chunk is not None:
        blob = f"{a.field}/step_{a.step:05d}/fab0000/chunk_{a.chunk}"
    if blob:
        report(blob, sel, blobs, exp, ev)
        return

    fields = sorted({parse_blob(b)[0] for b in blobs})
    steps = sorted({parse_blob(b)[1] for b in blobs})
    chunks = sorted({parse_blob(b)[2] for b in blobs})
    print(f"{len(blobs)} chunks: {len(fields)} fields x {len(steps)} frames x "
          f"{len(chunks)} chunks/field")
    print(f"   fields  {' '.join(fields)}")
    print(f"   steps   {steps[0]} .. {steps[-1]} (every {steps[1]-steps[0]}), "
          f"in-situ numbering; dump step is +1")
    print(f"   chunks  {chunks[0]} .. {chunks[-1]}  ({CHUNK_BYTES>>20} MiB each, "
          f"{PER_CHUNK} evolution blocks apiece)")
    print(f"   explored {len(exp)} of {len(blobs)}; the rest came from cache")
    print("\ntry:  ./chunk_log.py --blob %s" % sorted(blobs)[len(blobs)//2])
    print("      ./chunk_log.py --csv /tmp/chunks.csv")


if __name__ == "__main__":
    main()
