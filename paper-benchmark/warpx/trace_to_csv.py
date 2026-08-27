#!/usr/bin/env python3
"""Turn a WarpX+VOL path trace into the per-chunk CSV the aggregator reads.

The other workloads run a Clio driver that records every chunk itself. This one
does not: the application is a stock WarpX and the staging happens inside the
VOL, so the only per-chunk record is the compile-time path trace
(CLIO_NEUROPRESS_PATH_TRACE=1 in the environment -- run_config.sh sets it
unconditionally for this reason: unlike the other three workloads there is no
Clio driver recording chunks, so this trace IS this workload's results, and
without it the run completes, reports rc=0, and produces an EMPTY blobs.csv).

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
# CompressionHeader, plus QuantHeaderExtension when the chunk was quantized.
# The trace's `out=` already includes these; a size derived from a ratio does
# not, and must add them back to describe the same thing.
HDR, QHDR = 24, 32

CODEC = re.compile(r"codec ran lib=(\d+) in=(\d+) out=(\d+) kept=([01])")
FINAL = re.compile(r"neuropress FINAL blob='([^']*)' lib=(\d+) \(([^)]*)\)"
                   r".*?\((primary kept|EXPLORATION OVERRODE THE PRIMARY)\)")
OPMD = re.compile(r"/data/(\d+)/fields/([^/]+)(?:/([^/]+))?/chunk_(\d+)$")
# Gray-Scott writes one dataset per snapshot, named "<field>_<step>", so its
# blobs arrive as "/V_00025/chunk_3". Same treatment as openPMD's: the field
# has to end up first or the aggregator files every chunk under its own name.
GS = re.compile(r"/([A-Za-z]+)_(\d+)/chunk_(\d+)$")


def normalise(name):
    """VOL blob path -> "<field>/step_<n>/chunk_<i>".

    The aggregator groups by the first path segment; passing /data/<step>/...
    through would file every chunk under "data" and produce no per-field
    breakdown.
    """
    m = OPMD.match(name)
    if m:
        step, fld, comp, chunk = m.groups()
        return f"{fld}{comp or ''}/step_{step}/chunk_{chunk}"
    m = GS.match(name)
    if m:
        fld, step, chunk = m.groups()
        return f"{fld}/step_{step}/chunk_{chunk}"
    return name.lstrip("/").replace("/", "_")


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
            # finditer, and no `continue` between the three: worker threads log
            # concurrently and two records land on ONE line often enough to
            # matter. Matching once per line dropped 3 of 128 codec records on
            # a Gray-Scott run, which then mis-paired every chunk after the
            # first collision -- silently, because the counts only differed by
            # three.
            for m in BLOB.finditer(line):
                blobs.append((m.group(1), int(m.group(2))))
            for m in CODEC.finditer(line):
                primaries.append(tuple(int(x) for x in m.groups()))
            for m in FINAL.finditer(line):
                finals[m.group(1)] = (int(m.group(2)), m.group(3),
                                      m.group(4).startswith("EXPLORATION"))

    # The adopted candidate's measured ratio, per blob, when exploration ran.
    #
    # The same row is also the only per-chunk TIMING this workload has. The
    # application is a stock WarpX and the staging happens inside the VOL, so
    # unlike the other three there is no Clio driver recording a compress time
    # -- the path trace carries none, and compress_ms was reported as 0.0 for
    # every chunk. The exploration log's adopted row (primary or alternative,
    # whichever was stored) carries ct_ms for the codec that actually ran, and
    # dt_ms when CLIO_NEUROPRESS_EXPLORE_MEASURE_DT asked for a decompression
    # measurement. Both are CUDA-event brackets around the codec call alone,
    # so they are the same clock the other workloads' columns use.
    adopted, times = {}, {}
    if explore_log and os.path.isfile(explore_log):
        for r in csv.DictReader(open(explore_log)):
            if r.get("adopted") == "1":
                try:
                    adopted[r["blob"]] = (float(r["ratio"]),
                                          int(r["chunk_bytes"]),
                                          r.get("quantize") == "1")
                except (ValueError, KeyError):
                    pass
                try:
                    times[r["blob"]] = (float(r.get("ct_ms", -1) or -1),
                                        float(r.get("dt_ms", -1) or -1))
                except ValueError:
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
            ratio, cbytes, quant = adopted[name]
            # + the CTEC header. explore.csv's `ratio` is bytes/CODEC PAYLOAD
            # (upstream's definition, header excluded), so cbytes/ratio is the
            # payload alone -- while the `kept` branch below takes `out` from
            # the trace, which is total_stored_size and DOES include it. Two
            # definitions in one column: every overridden chunk was understated
            # by 24 bytes, and the run's total with it. Caught by comparing
            # against what fs_bdev actually wrote (44 of 160 chunks here, 1056
            # of a 1154-byte shortfall; the rest is metadata blobs and the
            # truncation below).
            stored = (int(cbytes / ratio) + HDR + (QHDR if quant else 0)
                      if ratio > 0 else cin)
            lib_out, codec_out = fin[0], fin[1]
            corrected += 1
        elif kept:                                       # primary, kept
            stored, lib_out = cout, (fin[0] if fin else lib)
            codec_out = fin[1] if fin else f"lib{lib}"
        else:                                            # nothing shrank it
            stored, lib_out, codec_out = cin, 0, "raw"
        ct_ms, dt_ms = times.get(name, (0.0, -1.0))
        rows.append({
            # THE PRETTY NAME IS NOT THE NAME THE RUNTIME LOGGED. normalise()
            # rewrites the VOL's openPMD path so the aggregator can group by
            # field, but selection.csv records the ORIGINAL blob name -- so the
            # two files could not be joined at all:
            #     blobs.csv     Bx/step_0/chunk_0
            #     selection.csv /data/0/fields/B/x/chunk_0
            # A join on `blob` silently matched nothing for this workload and
            # left every model-feature column empty, 240 of 240 rows, with no
            # error. Carry the runtime's own key alongside so the join is exact
            # rather than reconstructed by regex on the far side.
            "blob": normalise(name), "runtime_blob": name,
            "bytes": nbytes, "fnv1a64": "0",
            "lib": lib_out, "codec": codec_out,
            "ratio": round(nbytes / stored, 6) if stored else 0.0,
            "stored": stored, "compress_ms": ct_ms,
            "decompress_ms": dt_ms, "rc": 0,
        })

    cols = ["blob", "runtime_blob", "bytes", "fnv1a64", "lib", "codec",
            "ratio", "stored", "compress_ms", "decompress_ms", "rc"]
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
