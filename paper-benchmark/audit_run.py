#!/usr/bin/env python3
"""Cross-check one run directory: does what the tier holds match what the log says?

    ./audit_run.py <run-dir> [<run-dir> ...]

Four sources are compared, and they are independent -- a mistake in any single
one shows up as a disagreement rather than a consistent wrong answer:

    <log>          the adapter's own summary, per field and in total
    blobs.csv      one row per chunk, as the adapter recorded it
    explore.csv    the candidate the sweep marked `adopted`
    runtime.log    the bytes fs_bdev actually wrote, when the run was traced
                   with CLIO_NEUROPRESS_PATH_TRACE=1 (skipped otherwise)

The check that matters is the last hop: an ADOPTED candidate's bytes are the
ones stored, not the model's own pick, so a run where exploration overrode the
primary is the one that would expose a mix-up. Reported per field.

Exits non-zero if anything disagrees, so it can gate a campaign.

NOTE on the two ratios, which are NOT interchangeable and must not be compared
directly (they differ by the CTEC header, and that is by design):
    explore.csv `ratio`   = bytes / CODEC PAYLOAD  -- upstream's definition,
                            the NN's training label
    blobs.csv `stored_ratio`, and the log's per-field and total ratios
                          = bytes / STORED         -- what the tier holds
So this reconciles them by adding the header back: payload + 24, or + 56 when
the adopted candidate quantized (24-byte CompressionHeader + 32-byte
QuantHeaderExtension).
"""
import csv, os, re, sys
from collections import Counter

HDR, QHDR = 24, 32


def read_log(d):
    """Per-field and total figures from whichever log the adapter wrote."""
    per, tot, codecs, seen = {}, None, Counter(), []
    for name in ("nyx.log", "vpic.log", "stdout.log", "convert.log", "run.log"):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        for line in open(p, errors="replace"):
            m = re.match(r"\s*(\w+)\s+(\d+) -> (\d+)\s+\(([\d.]+)x\)", line)
            if m:
                per[m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
            m = re.search(r"(\d+) B in -> (\d+) B on the tier\s+\((?:stored )?ratio ([\d.]+)\)", line)
            if m:
                tot = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
            m = re.search(r"codec\s+([\w-]+)\s*:\s*(\d+) chunk", line)
            if m:
                codecs[m.group(1)] = int(m.group(2))
            # WarpX has no adapter: trace_to_csv.py reconstructs the run from
            # the runtime log and writes convert.log, whose summary reads
            # "N chunks, M from the adopted candidate, X B -> Y B (Zx)".
            # No per-field table and no codec tally, so those checks simply do
            # not run for it.
            m = re.search(r"(\d+) chunks?, \d+ from the adopted candidate, "
                          r"(\d+) B -> (\d+) B \(([\d.]+)x\)", line)
            if m:
                tot = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
        if per or tot:
            seen.append(name)
    # Every candidate is read and MERGED, not just the first that matched.
    # A VPIC run directory has both stdout.log (which the wrapper echoes the
    # total into) and vpic.log (which carries the per-field table and the codec
    # tally); returning at the first hit silently dropped the per-field checks.
    return per, tot, codecs, ",".join(seen) or None


def field(blob):
    return blob.split("/")[0]


def audit(d):
    print(f"=== {d}")
    per, tot, log_codecs, logname = read_log(d)
    if tot is None:
        print("   no adapter summary found (looked for nyx.log/stdout.log/vpic.log/run.log)")
        return 1

    blobs = list(csv.DictReader(open(os.path.join(d, "blobs.csv"))))
    ex_p = os.path.join(d, "explore.csv")
    # Keyed by FULL blob name, not field: a run with more than one frame has
    # several chunks per field, and keying on the field kept only the last of
    # them while comparing it against the SUM of all of them.
    adopted = {}
    if os.path.isfile(ex_p):
        for r in csv.DictReader(open(ex_p)):
            if r["adopted"] == "1":
                adopted[r["blob"]] = r

    # fs_bdev writes, when the run was traced. Sizes are matched as a multiset:
    # the trace prints byte counts without blob names.
    writes = Counter()
    rl = os.path.join(d, "runtime.log")
    if os.path.isfile(rl):
        for line in open(rl, errors="replace"):
            if "fs_bdev Write" in line:
                writes[int(line.split()[4])] += 1

    bad = 0
    by_field = Counter()
    for r in blobs:
        by_field[field(r["blob"])] += int(r["stored"])

    print(f"   {'field':<10}{'log stored':>11}{'blobs.csv':>11}  {'codec(s)':<24}"
          f"{'checked':>9}   {'verdict'}")
    for f, (b_in, b_st, b_ra) in sorted(per.items()):
        got = by_field.get(f)
        note, ok = "", got == b_st
        # Every CHUNK of this field that explored: its adopted candidate's
        # payload plus the CTEC header must be the bytes recorded for it.
        n_ck = 0
        for r in blobs:
            a = adopted.get(r["blob"])
            if field(r["blob"]) != f or a is None:
                continue
            if r.get("lib") == "0":
                # STORED RAW: the codec was not beneficial, so the ORIGINAL
                # bytes went to the tier verbatim -- no CTEC header, and the
                # adopted row's ratio describes a payload that was discarded.
                # stored must equal bytes exactly.
                n_ck += 1
                if int(r["stored"]) != int(r["bytes"]):
                    ok, note = False, note + f" raw chunk {r['blob'].split('/')[-1]}" \
                                             f" stored {r['stored']} != {r['bytes']}"
                continue
            h = HDR + (QHDR if a["quantize"] == "1" else 0)
            implied = round(int(a["chunk_bytes"]) / float(a["ratio"])) + h
            n_ck += 1
            if implied != int(r["stored"]):
                ok, note = False, note + f" chunk {r['blob'].split('/')[-1]}" \
                                         f" implies {implied} not {r['stored']}"
        codecs_here = {r["codec"] for r in blobs if field(r["blob"]) == f}
        a = {"lib_name": "/".join(sorted(codecs_here))} if codecs_here else None
        implied = n_ck or None
        # The log's own per-field ratio must be bytes/stored. ABSOLUTE
        # tolerance, not relative: the adapter prints these with a FIXED three
        # decimals, so the error is bounded by half the last digit regardless
        # of magnitude. A relative test passes on Nyx (ratios 60-80, six
        # significant figures) and fails on every VPIC field (ratios ~1.1, four
        # figures) -- a false alarm on the workload that compresses least.
        if abs(b_in / b_st - b_ra) > 5.1e-4:
            ok, note = False, note + " ratio!"
        bad += not ok
        print(f"   {f:<10}{b_st:>11}{(got if got is not None else -1):>11}"
              f"  {(a['lib_name'] if a else '-'):<24}{(n_ck if n_ck else 0):>9}"
              f"   {'OK' if ok else 'MISMATCH' + note}")

    st_sum = sum(int(r["stored"]) for r in blobs)
    in_sum = sum(int(r["bytes"]) for r in blobs)
    checks = [
        ("blobs.csv total in  == log", in_sum == tot[0], f"{in_sum} vs {tot[0]}"),
        ("blobs.csv total out == log", st_sum == tot[1], f"{st_sum} vs {tot[1]}"),
        ("log total ratio     == in/out", abs(tot[0] / tot[1] - tot[2]) <= 5.1e-4,
         f"{tot[0]/tot[1]:.4f} vs {tot[2]}"),
    ]
    if writes:
        w_sum = sum(k * v for k, v in writes.items())
        n_w = sum(writes.values())
        if n_w == len(blobs):
            # One PutBlob per chunk, one bdev write per PutBlob: the in-situ
            # adapters and the LAMMPS driver. Exact equality is meaningful.
            every = all(writes.get(int(r["stored"]), 0) > 0 for r in blobs)
            checks += [("fs_bdev bytes written == log", w_sum == tot[1],
                        f"{w_sum} vs {tot[1]}"),
                       ("every blob's size was written", every, "")]
        else:
            # WarpX: blobs.csv is written --fields-only, while the tier also
            # holds openPMD metadata blobs, so the writes are a SUPERSET and
            # exact equality would be wrong. The surplus is still bounded and
            # is reported, because a large one means field bytes are missing.
            checks.append(("fs_bdev bytes >= log (superset)", w_sum >= tot[1],
                           f"{w_sum} vs {tot[1]}, surplus {w_sum - tot[1]} B "
                           f"in {n_w - len(blobs)} non-field write(s)"))
    if log_codecs:
        # From blobs.csv, which has EVERY chunk. The adopted set only covers
        # chunks that actually explored, so tallying it against the log's
        # all-chunk count fails on any run where the gate held some back.
        # lib 0 is "stored raw, no codec kept". blobs.csv still renders it
        # through NameForWireId(0), which answers "brotli", so a raw chunk
        # looks like a compressed one here; the adapters' own tallies count
        # only chunks a codec was kept for. Excluded on the lib id, never on
        # the name -- brotli is a real codec that can legitimately be chosen.
        got = Counter(r["codec"] for r in blobs if r.get("lib") != "0")
        checks.append(("codec tally == log tally", got == log_codecs,
                       f"{dict(got)} vs {dict(log_codecs)}"))
    if adopted:
        ov = sum(1 for a in adopted.values() if a["role"] == "alt")
        print(f"   {len(adopted)} of {len(blobs)} chunk(s) explored; exploration "
              f"overrode the model on {ov}"
              f"{'  <- the case that would expose a mix-up' if ov else ''}")

    print()
    for name, ok, detail in checks:
        bad += not ok
        print(f"   [{'ok ' if ok else 'FAIL'}] {name:<32}{detail}")
    if not writes:
        print("   [skip] fs_bdev writes -- rerun with CLIO_NEUROPRESS_PATH_TRACE=1 to include them")
    print()
    return bad


if __name__ == "__main__":
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    rc = sum(audit(d) for d in dirs)
    print("ALL CHECKS PASSED" if rc == 0 else f"{rc} DISAGREEMENT(S)")
    sys.exit(1 if rc else 0)
