#!/usr/bin/env python3
"""Measure what the VOL's write-side admission policy costs and buys.

Plan item 5 (§2 capacity) says to measure admission BEFORE building eviction,
on the reasoning that skipping work whose product is never used is cheaper than
machinery for choosing what to throw away after paying for it. This is that
measurement.

WHAT IS COMPARED
    CLIO_VOL_ADMIT=write          (default) stage on write and on read-miss
    CLIO_VOL_ADMIT=read-miss                stage only on read-miss
    CLIO_VOL_ADMIT=second-access            stage only once a dataset has been
                                            read twice -- decide on evidence
                                            instead of on a guess

across workloads that differ in whether the data is ever read back:

    write_only        write N datasets, close. Never read.
    write_then_read   write, close, reopen, read all.
    write_read_read   write, close, then TWO read sessions -- the case where a
                      cache should look best, and the one that separates "the
                      first read paid for the staging" from "later reads did".
    write_read_x3     THREE read sessions. Needed to score second-access at all:
                      that policy stages on the second read, so with only two it
                      pays the staging and never serves from it. The third read
                      is the first one it can answer, and a policy that defers
                      admission cannot be judged on a workload that stops before
                      its first payoff.

METHOD, AND WHY THE NUMBERS ARE WHAT THEY ARE
    Cost is `bytes_staged`: bytes written a second time, into the tier. It is
    counted where the puts are submitted, not derived from transfer size --
    application write volume is NOT the cost of admission, because most of it
    may never have been eligible for the tier at all. Using it as the
    denominator is the error that made an earlier attempt at this measurement
    unquotable.

    Benefit is `read_bytes_from_cache`: bytes a read was actually served from
    the tier instead of from the file.

    Both come from the connector's own v2 telemetry, aggregated over every
    session of a run (a reopen produces its own summary, so a single-summary
    read would miss the reads entirely).

    Each combination uses a FRESH file path so its tag cannot inherit staging
    from a previous combination. Without that the second run of a pair scores
    the first run's work.

REPORTED RATIO
    benefit / cost = bytes served from the tier per byte staged into it.
    Below 1.0 the tier gave back less than it was given.

    This is a TRAFFIC measure, not a speed measure, and the distinction
    matters for how far the result can be pushed. It says how many bytes the
    tier answered for per byte it was handed; it does not say those answers
    were faster. On a local filesystem they may well not be -- the scoping
    doc's conclusion is that a read tier is net-negative there and pays off
    only against slow or remote authoritative storage, or across clients.
    Timing is deliberately not folded in: on this host the run-to-run spread
    is comparable to the effect (see the VFD trace header, where the same
    problem made a +5% overhead figure unquotable), so a latency column here
    would carry more authority than the measurement can support. Footprint is
    what admission controls and footprint is what this reports.

Usage:  admission_measure.py --bin <build>/bin [--datasets N] [--mib M]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# The compat suite already owns runtime bring-up (its own compose config, shm
# wipe, readiness wait and failure diagnostics). Reuse it rather than grow a
# second, subtly-different copy that drifts.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hdf5_compat_suite as suite  # noqa: E402

WORKLOADS = ("write_only", "write_then_read", "write_read_read",
             "write_read_x3")
POLICIES = ("write", "read-miss", "second-access")


def _env(bin_dir, policy, trace_dir):
    """A worker environment with the connector selected and tracing on.

    HDF5 reads the connector variable once at library init, so every run is a
    fresh process -- the policy cannot be switched inside a live one either,
    since it is latched in a function-local static by design (re-reading it per
    access would let a workload change policy halfway through a measurement).

    CLIO_SERVER_CONF points the client at the SAME config the runtime was
    started with. Despite the name it is not server-only: ConfigManager's
    lookup order is CLIO_SERVER_CONF first, then $HOME/.clio/clio.yaml, and a
    client that resolves a different config does not fail loudly -- it fails to
    connect, the connector correctly degrades to pass-through, and every
    measurement comes back a clean zero. That is indistinguishable from
    "admission stages nothing", which is a real possible result.

    An earlier version of this set HOME instead. That worked, but only as a
    blunt proxy for "find the right config" -- it moves every other
    HOME-relative lookup as a side effect, and it obscures what is actually
    required. Naming the config directly is both narrower and true.
    """
    e = dict(os.environ)
    e["HDF5_PLUGIN_PATH"] = bin_dir
    e["HDF5_VOL_CONNECTOR"] = "clio"
    e["CLIO_SERVER_CONF"] = suite.SUITE_CONF
    e["LD_LIBRARY_PATH"] = bin_dir + ":/usr/local/lib:" + e.get("LD_LIBRARY_PATH", "")
    e["CLIO_VOL_ADMIT"] = policy
    e["CLIO_VOL_TRACE"] = trace_dir
    return e


# The worker runs in its own process (see _env). Kept as source text rather than
# a separate file so the measurement is one artifact.
WORKER = r"""
import sys, numpy as np, h5py
path, workload, ndsets, nbytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
data = np.arange(nbytes // 8, dtype="i8")

with h5py.File(path, "w") as f:
    for i in range(ndsets):
        f.create_dataset("d%d" % i, data=data)

reads = {"write_only": 0, "write_then_read": 1,
         "write_read_read": 2, "write_read_x3": 3}[workload]
for _pass in range(reads):
    with h5py.File(path, "r") as f:
        for i in range(ndsets):
            _ = f["d%d" % i][()]
print("OK")
"""


def _totals(trace_dir):
    """Sum the v2 totals over every session summary produced by one run."""
    agg = dict(bytes_written=0, bytes_staged=0, bytes_staged_discarded=0,
               bytes_staged_resident=0, read_bytes_from_cache=0,
               read_bytes_from_native=0, reads=0, writes=0)
    versions = set()
    for p in glob.glob(os.path.join(trace_dir, "*.access.json")):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        versions.add(s.get("v"))
        t = s.get("totals", {})
        for k in agg:
            agg[k] += t.get(k, 0)
    # A v1 summary has no staged fields at all and would silently read as zero
    # cost -- the exact confusion the schema version exists to prevent.
    if versions and versions != {2}:
        raise SystemExit(f"expected only schema v2 summaries, saw {versions}; "
                         "rebuild the connector")
    return agg


def run(bin_dir, workload, policy, ndsets, mib, tmp_root):
    # A fresh runtime PER COMBINATION, not once per invocation. The tier has no
    # eviction yet (plan W11), so combinations sharing a runtime accumulate:
    # by the sixth, the tier is full, puts fail, the partial image is
    # invalidated, and the next read re-stages it. That showed up as a 32 MiB
    # corpus reporting 50 MiB staged with 4 MiB resident -- a policy scored on
    # its predecessors' footprint rather than its own. Restarting isolates the
    # comparison; the pile-up itself is a real property of an unbounded tier
    # and belongs in the capacity discussion, not in these numbers.
    if not suite.restart_runtime():
        print(f"  !! {workload}/{policy}: clio_run did not become ready")
        return None
    trace_dir = tempfile.mkdtemp(prefix="adm-trace-", dir=tmp_root)
    # Fresh path per combination: the tag is keyed by path, so reusing one lets
    # a run be served by its predecessor's staging and score work it did not do.
    path = os.path.join(tmp_root, f"{workload}-{policy}.h5".replace("-", "_"))
    for f in glob.glob(path + "*"):
        os.remove(f)
    r = subprocess.run([sys.executable, "-c", WORKER, path, workload,
                        str(ndsets), str(mib * 1024 * 1024)],
                       capture_output=True, text=True,
                       env=_env(bin_dir, policy, trace_dir), timeout=600)
    if r.returncode != 0 or "OK" not in r.stdout:
        print(f"  !! {workload}/{policy} worker failed rc={r.returncode}")
        for line in (r.stderr or "").strip().splitlines()[-15:]:
            print(f"     | {line}")
        return None
    return _totals(trace_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="build bin/ holding libclio_vol.so")
    ap.add_argument("--datasets", type=int, default=8)
    ap.add_argument("--mib", type=int, default=4, help="MiB per dataset")
    a = ap.parse_args()

    suite.BIN = a.bin   # run() restarts the runtime per combination
    tmp_root = tempfile.mkdtemp(prefix="clio-admission-")
    print(f"corpus: {a.datasets} datasets x {a.mib} MiB "
          f"= {a.datasets * a.mib} MiB written per run\n")
    hdr = (f"{'workload':<17}{'policy':<11}{'staged MiB':>11}"
           f"{'resident':>10}{'served MiB':>12}{'served/staged':>15}")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    try:
        for w in WORKLOADS:
            for p in POLICIES:
                t = run(a.bin, w, p, a.datasets, a.mib, tmp_root)
                if t is None:
                    continue
                rows[(w, p)] = t
                mib = lambda n: n / (1024 * 1024)
                ratio = (t["read_bytes_from_cache"] / t["bytes_staged"]
                         if t["bytes_staged"] else float("nan"))
                print(f"{w:<17}{p:<11}{mib(t['bytes_staged']):>11.1f}"
                      f"{mib(t['bytes_staged_resident']):>10.1f}"
                      f"{mib(t['read_bytes_from_cache']):>12.1f}"
                      f"{ratio:>15.2f}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # A zero here is not a finding, it is a broken measurement. The default
    # policy stages every whole write by definition, so if it staged nothing
    # the connector was not engaged -- a runtime it could not reach, a wrong
    # HOME, a plugin path that did not resolve. All of those degrade to
    # pass-through SILENTLY and by design, which is correct behaviour for a
    # cache and useless behaviour for a measurement. Say so rather than print a
    # table of clean zeros that reads like a result.
    if any(rows.get((w, "write"), {}).get("bytes_staged", 0) == 0
           for w in WORKLOADS if (w, "write") in rows):
        print("\nMEASUREMENT INVALID: CLIO_VOL_ADMIT=write staged 0 bytes.\n"
              "  The connector was loaded but never cached -- almost always a\n"
              "  runtime it could not reach. Check the worker stderr for\n"
              "  'running as a pure pass-through'.")
        return 1

    print()
    mib = lambda n: n / (1024 * 1024)
    # Each alternative is scored against the DEFAULT, not against each other:
    # the question a policy has to answer is "is this better than what ships".
    for w in WORKLOADS:
        on = rows.get((w, "write"))
        if not on or not on["bytes_staged"]:
            continue
        for p in POLICIES:
            if p == "write":
                continue
            off = rows.get((w, p))
            if not off:
                continue
            saved = on["bytes_staged"] - off["bytes_staged"]
            forgone = on["read_bytes_from_cache"] - off["read_bytes_from_cache"]
            print(f"{w:<17} {p:<14}: {mib(saved):.1f} MiB less staged "
                  f"({100.0 * saved / on['bytes_staged']:.0f}% of the default's "
                  f"footprint), {mib(forgone):.1f} MiB fewer served")
    return 0


if __name__ == "__main__":
    sys.exit(main())
