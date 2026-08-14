#!/usr/bin/env python3
"""Attribute every cudaMemcpy in an nsys report to the NVTX phase that caused it.

`nsys stats` gives totals per direction, which is not enough to decide whether a
transfer is redundant -- for that you need to know WHICH phase made it and
whether its size scales with the chunk. This joins CUPTI_ACTIVITY_KIND_MEMCPY
against NVTX_EVENTS on the timestamp and groups by (phase, direction, size).

Usage:
    nsys profile --trace=cuda,nvtx -o rep ./bin/neuropress_transfer_audit
    python3 analyze_nsys.py rep.sqlite        # nsys creates the .sqlite on
                                              # first `nsys stats` run, or pass
                                              # the .nsys-rep and run stats once

CAVEAT, and it matters for reading the output: NVTX ranges are stamped in CPU
time while memcpy start/end are GPU time. An async copy enqueued at the end of
one phase can execute after that phase's range has closed and be attributed to
the next one. Totals are exact; per-phase attribution is approximate at the
boundaries. Check a suspicious attribution against the global count before
concluding a phase made a copy it did not.
"""

import collections
import sqlite3
import sys


def main(path: str) -> int:
    db = sqlite3.connect(path)
    cur = db.cursor()

    cur.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")
    oper = dict(cur.fetchall())

    cur.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND end IS NOT NULL ORDER BY start"
    )
    ranges = cur.fetchall()

    def phase_of(ts):
        # Innermost enclosing range wins, so harness-verify beats
        # readback-to-device rather than the other way round.
        best = None
        for text, start, end in ranges:
            if start <= ts <= end:
                if best is None or (end - start) < (best[2] - best[1]):
                    best = (text, start, end)
        return best[0] if best else "(outside any phase)"

    cur.execute(
        "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "ORDER BY start"
    )
    rows = cur.fetchall()

    agg = collections.OrderedDict()
    for start, end, nbytes, kind in rows:
        key = (phase_of(start), oper.get(kind, str(kind)), nbytes)
        entry = agg.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += end - start

    print(f"{'phase':<26}{'direction':<18}{'bytes':>14}{'count':>7}{'us':>10}")
    print("-" * 75)
    for (phase, kind, nbytes), (count, nanos) in agg.items():
        print(f"{phase:<26}{kind:<18}{nbytes:>14,}{count:>7}{nanos/1000:>10.1f}")
    print("-" * 75)

    total_bytes = sum(r[2] for r in rows)
    total_ns = sum(r[1] - r[0] for r in rows)
    small = [r for r in rows if r[2] <= 1024]
    print(f"{len(rows)} copies, {total_bytes/2**20:.3f} MiB, {total_ns/1e6:.3f} ms")
    if small:
        avg = sum(r[1] - r[0] for r in small) / len(small)
        print(
            f"{len(small)} of them are <=1 KiB: {avg:.0f} ns each on average, "
            f"{avg*len(small)/1000:.1f} us total -- these are latency-bound, so "
            f"their COUNT matters and their size does not."
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
