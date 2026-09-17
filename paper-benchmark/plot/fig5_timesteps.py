#!/usr/bin/env python3
"""Per-dump timing table for Figure 5, from the runtime's per-chunk phase log.

  fig5_timesteps.py <phase.csv> --workload VPIC --out benchmark_vpic_timesteps.csv
                    [--warmup-step N]

Sums CLIO_NEUROPRESS_PHASE_LOG rows per dump (from the blob name: step_NNNNN,
step_N, plt00010 or step00430) into the table plot_fig5.py --timesteps reads.
--warmup-step N drops dumps at step <= N. Throughput is bytes over the summed
chunk wall time, a lower bound.
"""
import argparse, collections, csv, re, sys

DUMP_RE = re.compile(r"^(?:step_?|plt|diag)(\d+)$")
WRITE = ["stats_ms", "nn_ms", "choice_ms", "factory_ms", "compress_ms", "io_ms"]
READ = {"factory_ms": "read_factory_ms", "decompress_ms": "decompress_ms",
        "io_ms": "read_io_ms"}
COLUMNS = (["workload", "timestep", "dump", "chunks_write", "chunks_read",
            "orig_mib", "stored_mib", "ratio"] + WRITE + list(READ.values()) +
           ["preproc_ms", "h2d_ms", "write_other_ms", "read_other_ms",
            "write_wall_ms", "read_wall_ms", "write_mib_s", "read_mib_s",
            "explorations", "explored_alternatives", "sgd_updates",
            "reused_chunks", "explore_ms", "sgd_ms"])


def num(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def dump_of(chunk_id):
    for seg in chunk_id.split("/"):
        m = DUMP_RE.match(seg)
        if m:
            return seg, int(m.group(1))
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase_log")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-step", type=int, default=-1)
    a = ap.parse_args()

    dumps = collections.OrderedDict()
    unnamed = missing = warm = 0
    with open(a.phase_log, newline="") as fh:
        for r in csv.DictReader(fh):
            name, step = dump_of(r["chunk_id"])
            if name is None:
                unnamed += 1
                continue
            if step <= a.warmup_step:
                warm += 1
                continue
            d = dumps.setdefault((step, name), collections.defaultdict(float))
            path = r["path"]
            cols = WRITE if path == "write" else list(READ)
            if any(num(r.get(c)) is None for c in cols):
                missing += 1
                continue
            if path == "write":
                d["chunks_write"] += 1
                for c in WRITE:
                    d[c] += num(r[c])
                d["orig_b"] += num(r["chunk_bytes"]) or 0
                d["stored_b"] += num(r.get("stored_bytes")) or 0
                d["write_wall_ms"] += num(r["wall_ms"]) or 0
                d["write_other_ms"] += num(r["other_ms"]) or 0
                d["preproc_ms"] += num(r.get("preproc_ms")) or 0
                d["h2d_ms"] += num(r.get("h2d_ms")) or 0
                d["explore_ms"] += num(r.get("explore_ms")) or 0
                d["sgd_ms"] += num(r.get("sgd_ms")) or 0
                n_alt = num(r.get("explored")) or 0
                d["explored_alternatives"] += n_alt
                d["explorations"] += 1 if n_alt > 0 else 0
                d["sgd_updates"] += num(r.get("sgd_updates")) or 0
                d["reused_chunks"] += 1 if r.get("reused") == "1" else 0
            elif path == "read":
                d["chunks_read"] += 1
                for c, out in READ.items():
                    d[out] += num(r[c])
                d["read_b"] += num(r["chunk_bytes"]) or 0
                d["read_wall_ms"] += num(r["wall_ms"]) or 0
                d["read_other_ms"] += num(r["other_ms"]) or 0

    rows = []
    for ts, ((step, name), d) in enumerate(sorted(dumps.items())):
        mib = d["orig_b"] / 2**20
        rows.append({
            "workload": a.workload, "timestep": ts, "dump": name,
            "chunks_write": int(d["chunks_write"]), "chunks_read": int(d["chunks_read"]),
            "orig_mib": f"{mib:.3f}", "stored_mib": f"{d['stored_b'] / 2**20:.3f}",
            "ratio": f"{d['orig_b'] / d['stored_b']:.3f}" if d["stored_b"] else "",
            **{c: f"{d[c]:.4f}" for c in WRITE + list(READ.values())},
            **{c: f"{d[c]:.4f}" for c in ("preproc_ms", "h2d_ms", "write_other_ms",
                                          "read_other_ms", "write_wall_ms",
                                          "read_wall_ms", "explore_ms", "sgd_ms")},
            "write_mib_s": f"{mib / (d['write_wall_ms'] / 1e3):.2f}" if d["write_wall_ms"] else "",
            "read_mib_s": (f"{d['read_b'] / 2**20 / (d['read_wall_ms'] / 1e3):.2f}"
                           if d["read_wall_ms"] else ""),
            **{c: int(d[c]) for c in ("explorations", "explored_alternatives",
                                      "sgd_updates", "reused_chunks")},
        })

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{a.workload}: {len(rows)} dump(s) -> {a.out}"
          f"   (excluded: {warm} warmup chunk row(s), {missing} with a missing stage,"
          f" {unnamed} without a dump in the name)")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
