#!/usr/bin/env python3
"""Download N objects from S3 with K concurrent cae_s3_tool processes.

The wire-speed FLOOR for the CLIO-vs-Zarr S3 READ benchmark -- the read-side
counterpart of s3_raw_put.py. Counterpart drivers:
``context-assimilation-engine/benchmark/clio_s3_read_bench.cc`` (CLIO) and
``jarvis_clio_core/zarr_s3_bench/scripts/zarr_s3_read.py`` (Zarr).

Why it matters: after the CAE read path moved in-process (persistent Poco
connection, no fork+exec, no temp file), a CLIO read number still has to be
attributed against SOMETHING. This floor is the crudest possible read -- one
`cae_s3_tool get` process per in-flight object, straight to a temp file, no CTE
and no chunking -- so its throughput bounds what any read stack could achieve
on this host/bucket/link. Report ratio-to-floor, not absolute MB/s.

It also stands in for the OLD (fork+exec) cost profile: each GET here pays a
process spawn, an Aws::InitAPI, a fresh TCP+TLS handshake, and a whole-object
temp file -- exactly what the in-process assimilator no longer pays. NOTE the
same trap the write floor hit: this is NOT a floor at K=1 (serialized spawn +
temp file per object only pipelines at K>=8); it floors SUSTAINED throughput.

It GETs the SAME keys the CLIO read row reads (``<prefix>/obj_%06d.bin``), so
the two are measuring the same objects. Reports in the exact
``clio_bench::PrintResults`` wording so one parser serves all stacks.

Usage:
    s3_raw_get.py --bucket B --key-prefix <prefix> --num-objects 64 \\
        --object-size 33554432 --concurrency 8 [--label Rawget]

Requires: cae_s3_tool on PATH (or --s3-tool). Credentials come from the
environment / the standard AWS chain, resolved by cae_s3_tool itself.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import timeit


def parse_args(argv):
    """Parse command-line arguments.

    :param argv: Argument list excluding the program name.
    :return: Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True)
    p.add_argument("--key-prefix", required=True,
                   help="keys are <prefix>/obj_%%06d.bin (matches the CLIO row)")
    p.add_argument("--num-objects", type=int, required=True)
    p.add_argument("--object-size", type=int, required=True,
                   help="bytes per object; used only for the reported math")
    p.add_argument("--concurrency", type=int, default=8,
                   help="concurrent cae_s3_tool processes (K)")
    p.add_argument("--label", default="Rawget",
                   help="results namespace (becomes the CSV 'operation')")
    p.add_argument("--s3-tool", default=None,
                   help="path to cae_s3_tool; also read from CAE_S3_TOOL")
    p.add_argument("--tmpdir", default=None,
                   help="where downloaded objects land when --keep-downloads "
                        "is set; peak usage is concurrency * object_size")
    p.add_argument("--keep-downloads", action="store_true",
                   help="write each GET to a real file under --tmpdir instead "
                        "of /dev/null. Off by default: this is a WIRE-speed "
                        "floor, and the bytes are never read back")
    return p.parse_args(argv)


def resolve_tool(explicit):
    """Locate the cae_s3_tool helper.

    :param explicit: Value of --s3-tool, or None.
    :return: Path or bare name to exec.
    """
    return (explicit or os.environ.get("CAE_S3_TOOL")
            or shutil.which("cae_s3_tool") or "cae_s3_tool")


def run_gets(args, tool, tmpdir):
    """Run the timed download: K concurrent cae_s3_tool get processes.

    By default every slot writes to /dev/null. The downloaded bytes are never
    read back -- this measures wire speed -- and discarding them keeps the
    floor comparable to the engines it is a floor FOR: CLIO lands bytes in CTE
    (shm) and Zarr in numpy arrays (RAM), so charging only this path for a disk
    write would understate it. It also removes the hard scaling limit that
    real files impose: one file per slot is concurrency * object_size of peak
    space, which is 1 GiB at K=32 x 32 MiB and 8 GiB at K=32 x 256 MiB. Ares
    node-local /tmp could not absorb the 1 GiB case -- 62 of 64 GETs died with
    `cae_s3_tool get: write failed` (job 23921) and the row was voided.

    cae_s3_tool opens the destination with std::ofstream(trunc) and streams the
    SDK body into it, so /dev/null is a valid target and all slots can share
    it. The SDK completes the download before that write, so the timing still
    covers the full transfer.

    Pass --keep-downloads for real files under --tmpdir.

    :param args: Parsed arguments.
    :param tool: Path to cae_s3_tool.
    :param tmpdir: Directory for per-slot files; None when discarding.
    :return: Tuple of (elapsed microseconds, failure count).
    """
    k = min(args.concurrency, args.num_objects)
    if tmpdir is None:
        dests = [os.devnull] * k
    else:
        dests = [os.path.join(tmpdir, f"dst_{slot}.bin") for slot in range(k)]
    running = {}   # Popen -> slot index
    failures = 0
    next_idx = 0

    def launch(slot, idx):
        key = f"{args.key_prefix}/obj_{idx:06d}.bin"
        return subprocess.Popen(
            [tool, "get", args.bucket, key, dests[slot]],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    t0 = timeit.default_timer()
    for slot in range(k):
        running[launch(slot, next_idx)] = slot
        next_idx += 1

    while running:
        # Block on any one process, then immediately refill its slot so K
        # downloads stay in flight rather than draining between batches.
        for proc in list(running):
            rc = proc.wait()
            slot = running.pop(proc)
            if rc != 0:
                failures += 1
                err = proc.stderr.read().decode(errors="replace").strip()
                print(f"WARNING: cae_s3_tool get failed (rc={rc}): {err}")
            if next_idx < args.num_objects:
                running[launch(slot, next_idx)] = slot
                next_idx += 1
            break

    elapsed_us = (timeit.default_timer() - t0) * 1e6
    return elapsed_us, failures


def print_results(label, elapsed_us, n_ops, io_size, k):
    """Emit the throughput block in the exact bench_common.h wording.

    :param label: Results namespace.
    :param elapsed_us: Wall time of the download in microseconds.
    :param n_ops: Number of objects downloaded.
    :param io_size: Bytes per object.
    :param k: Effective concurrency, used as the "thread" count.
    """
    total_bytes = n_ops * io_size
    seconds = elapsed_us / 1e6
    bw = (total_bytes / (1024.0 * 1024.0)) / seconds if seconds > 0 else 0.0
    ops_s = (n_ops / seconds) if seconds > 0 else 0.0
    per_thread_bw = bw / k if k else bw
    print("")
    print(f"=== {label} Benchmark Results ===")
    print(f"Time (min): {elapsed_us} us ({elapsed_us / 1000.0} ms)")
    print(f"Time (max): {elapsed_us} us ({elapsed_us / 1000.0} ms)")
    print(f"Time (avg): {elapsed_us} us ({elapsed_us / 1000.0} ms)")
    print(f"Bandwidth per thread (min): {per_thread_bw} MB/s")
    print(f"Bandwidth per thread (max): {per_thread_bw} MB/s")
    print(f"Bandwidth per thread (avg): {per_thread_bw} MB/s")
    print(f"Aggregate bandwidth: {bw} MB/s")
    print(f"Aggregate IOPS: {ops_s}")
    print(f"IOPS per thread (avg): {ops_s / k if k else ops_s}")
    print(f"Avg latency per op: {elapsed_us / n_ops if n_ops else 0.0} us")
    print("Latency stddev: 0.0 us")
    print(f"Total data: {total_bytes / (1024.0 * 1024.0)} MB")
    print(f"Total ops: {n_ops}")
    print("===========================")


def print_fairness(label, args, elapsed_us, n_ops, k):
    """Emit the equivalence-caveat block.

    :param label: Results namespace.
    :param args: Parsed arguments.
    :param elapsed_us: Wall time of the download in microseconds.
    :param n_ops: Number of objects downloaded.
    :param k: Effective concurrency.
    """
    total_bytes = n_ops * args.object_size
    seconds = elapsed_us / 1e6
    wire_bw = ((total_bytes / (1024.0 * 1024.0)) / seconds
               if seconds > 0 else 0.0)
    print("")
    print(f"=== {label} Fairness ===")
    print(f"Objects read: {n_ops}")
    print(f"Bytes moved: {total_bytes}")
    print(f"Logical bytes: {total_bytes}")
    print(f"GET count: {n_ops}")
    print("Compression: none")
    print("Decode step: no")
    print(f"Requested concurrency: {args.concurrency}")
    print(f"Effective concurrency: {k}")
    print("Runtime worker threads: 0")
    print(f"Wall time us: {elapsed_us} ({elapsed_us / 1000.0} ms)")
    print(f"Wire bandwidth: {wire_bw} MB/s")
    # One process per GET is this floor's defining cost -- process spawn +
    # Aws::InitAPI + fresh handshake, exactly what the in-process assimilator no
    # longer pays. That is why it is a floor, not a competitor: CLIO's read path
    # pays none of it.
    print(f"Subprocess spawns: {n_ops}")
    # Temp staging is only charged when --keep-downloads writes real files. By
    # default every GET streams to /dev/null, so the honest number is 0 --
    # reporting k * object_size here would claim disk traffic that never
    # happened. This makes the floor DELIBERATELY GENEROUS: it is credited with
    # none of the whole-object staging an out-of-process path would really do,
    # and CLIO's margin is measured against that best case. The per-GET process
    # cost above is still charged, and remains the floor's defining handicap.
    print(f"Temp file bytes: "
          f"{k * args.object_size if args.keep_downloads else 0}")
    print(f"Transport chunk bytes: {args.object_size}")
    print("===================")


def main(argv):
    """Run the timed download and report both blocks.

    :param argv: Argument list excluding the program name.
    :return: Process exit code.
    """
    args = parse_args(argv)
    if args.num_objects <= 0 or args.object_size <= 0:
        print("ERROR: --num-objects and --object-size must be > 0")
        return 1
    if args.concurrency <= 0:
        print("ERROR: --concurrency must be > 0")
        return 1

    tool = resolve_tool(args.s3_tool)
    k = min(args.concurrency, args.num_objects)
    tmpdir = (tempfile.mkdtemp(prefix="s3_raw_get_", dir=args.tmpdir)
              if args.keep_downloads else None)
    try:
        dest_desc = tmpdir if tmpdir else os.devnull
        print(f"s3_raw_get: s3://{args.bucket}/{args.key_prefix} x "
              f"{args.num_objects} objects of {args.object_size} B, K={k}, "
              f"tool={tool}, dest={dest_desc}")
        elapsed_us, failures = run_gets(args, tool, tmpdir)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print_results(args.label, elapsed_us, args.num_objects, args.object_size,
                  k)
    print_fairness(args.label, args, elapsed_us, args.num_objects, k)

    if failures:
        # A partial download timed fewer bytes than it claims: fail the row
        # rather than publish a flattering number.
        print(f"ERROR: {failures} of {args.num_objects} GETs failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
