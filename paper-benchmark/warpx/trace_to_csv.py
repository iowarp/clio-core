#!/usr/bin/env python3
"""Turn a WarpX+VOL path trace into the per-chunk CSV the aggregator reads.

The other workloads run a Clio driver that records every chunk itself. This one
does not: the application is a stock WarpX and the staging happens inside the
VOL, so the only per-chunk record is the compile-time path trace
(-DCLIO_NEUROPRESS_PATH_TRACE=ON).

WHICH LINE SAYS WHAT -- this is the part that is easy to get wrong, and an
earlier version of this script did:

  DynamicSchedule blob='<name>' bytes=<n>     one per chunk, carries the name
  codec ran lib=<L> in=<i> out=<o> kept=<k>   the PRIMARY candidate only, no name
  neuropress FINAL blob='<name>' lib=<L> ...  the DECISION, and it says whether
                                              it was "(primary kept)" or
                                              "(EXPLORATION OVERRODE THE PRIMARY)"

`codec ran` reports the model's FIRST pick. Under exploration or best mode the
stored bytes may come from a different candidate entirely, and that candidate's
size appears only in the exploration log. Reading `codec ran` as if it were the
outcome silently reports the primary's codec and size for exactly the chunks
where selection did something -- which understated an exploration run by 5x
(12.13x reported against 16.86x actual) while the line counts matched 1:1, so
no consistency check caught it.

So: FINAL is the source of truth for the codec, and for an overridden chunk the
stored size comes from the adopted row of the exploration log.

`kept=0` means the codec ran and did not shrink the chunk, so the ORIGINAL
bytes were stored -- stored size is `in`, and the library is recorded as 0 the
way the compressor records it.

No digest column: verification for this workload is a read-back through the VOL
compared against a native read, not a digest the writer computed.
"""
import csv
import os
import re
import sys

BLOB = re.compile(r"DynamicSchedule blob='([^']*)' bytes=(\d+)")
CODEC = re.compile(r"codec ran lib=(\d+) in=(\d+) out=(\d+) kept=([01])")
FINAL = re.compile(r"neuropress FINAL blob='([^']*)' lib=(\d+) \(([^)]*)\)"
                   r".*?\((primary kept|EXPLORATION OVERRODE THE PRIMARY)\)")
OPMD = re.compile(r"/data/(\d+)/fields/([^/]+)(?:/([^/]+))?/chunk_(\d+)$")


def normalise(name):
    """openPMD path -> "<field>/step_<n>/chunk_<i>".

    The aggregator groups by the first path segment; passing /data/<step>/...
    through would file every chunk under "data" and produce no per-field
    breakdown.
    """
    m = OPMD.match(name)
    if not m:
        return name.lstrip("/").replace("/", "_")
    step, fld, comp, chunk = m.groups()
    return f"{fld}{comp or ''}/step_{step}/chunk_{chunk}"


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: trace_to_csv.py <runtime.log> <out.csv> "
                 "[--fields-only] [--explore-log <explore.csv>]")
    log, out = sys.argv[1], sys.argv[2]
    fields_only = "--fields-only" in sys.argv
    explore_log = None
    if "--explore-log" in sys.argv:
        explore_log = sys.argv[sys.argv.index("--explore-log") + 1]

    blobs, primaries, finals = [], [], {}
    with open(log, errors="replace") as f:
        for line in f:
            m = BLOB.search(line)
            if m:
                blobs.append((m.group(1), int(m.group(2))))
                continue
            m = CODEC.search(line)
            if m:
                primaries.append(tuple(int(x) for x in m.groups()))
                continue
            m = FINAL.search(line)
            if m:
                finals[m.group(1)] = (int(m.group(2)), m.group(3),
                                      m.group(4).startswith("EXPLORATION"))

    # The adopted candidate's measured ratio, per blob, when exploration ran.
    adopted = {}
    if explore_log and os.path.isfile(explore_log):
        for r in csv.DictReader(open(explore_log)):
            if r.get("adopted") == "1":
                try:
                    adopted[r["blob"]] = (float(r["ratio"]),
                                          int(r["chunk_bytes"]))
                except (ValueError, KeyError):
                    pass

    n = min(len(blobs), len(primaries))
    if len(blobs) != len(primaries):
        print(f"warning: {len(blobs)} blob lines vs {len(primaries)} primary "
              f"codec lines; pairing the first {n}", file=sys.stderr)

    overridden = sum(1 for v in finals.values() if v[2])
    if overridden and not adopted:
        print(f"warning: {overridden} chunk(s) had the primary overridden by "
              f"selection, but no exploration log was given -- their stored "
              f"sizes would be the PRIMARY's, not what was stored. Pass "
              f"--explore-log.", file=sys.stderr)

    rows, corrected = [], 0
    for (name, nbytes), (lib, cin, cout, kept) in zip(blobs[:n], primaries[:n]):
        if fields_only and "/fields/" not in name:
            continue
        fin = finals.get(name)
        if fin and fin[2] and name in adopted:          # selection overrode
            ratio, cbytes = adopted[name]
            stored = int(cbytes / ratio) if ratio > 0 else cin
            lib_out, codec_out = fin[0], fin[1]
            corrected += 1
        elif kept:                                       # primary, kept
            stored, lib_out = cout, (fin[0] if fin else lib)
            codec_out = fin[1] if fin else f"lib{lib}"
        else:                                            # nothing shrank it
            stored, lib_out, codec_out = cin, 0, "raw"
        rows.append({
            "blob": normalise(name), "bytes": nbytes, "fnv1a64": "0",
            "lib": lib_out, "codec": codec_out,
            "ratio": round(nbytes / stored, 6) if stored else 0.0,
            "stored": stored, "compress_ms": 0.0, "rc": 0,
        })

    cols = ["blob", "bytes", "fnv1a64", "lib", "codec", "ratio", "stored",
            "compress_ms", "rc"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    tin = sum(r["bytes"] for r in rows)
    tst = sum(r["stored"] for r in rows)
    note = f", {corrected} from the adopted candidate" if corrected else ""
    print(f"{len(rows)} chunks{note}, {tin} B -> {tst} B "
          + (f"({tin / tst:.3f}x)" if tst else ""))


if __name__ == "__main__":
    main()
