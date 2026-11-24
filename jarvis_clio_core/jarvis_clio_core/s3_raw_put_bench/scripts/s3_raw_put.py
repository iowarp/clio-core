#!/usr/bin/env python3
"""Upload N objects to S3 with K concurrent cae_s3_tool processes.

The wire-speed FLOOR for the CLIO-vs-Zarr S3 write benchmark. Counterpart
drivers: ``context-transfer-engine/benchmark/clio_s3_write_bench.cc`` (CLIO)
and ``jarvis_clio_core/zarr_s3_bench/scripts/zarr_s3_write.py`` (Zarr).

This is what separates "S3 is slow" from "CLIO is slow". Without it a poor
CLIO row is uninterpretable: there is no way to tell whether the bottleneck is
CLIO's block layer or simply what this host can push to this bucket. It does
the least possible work -- no CTE, no chunking layer, no compression, just
concurrent PUTs of pre-staged files -- so its throughput bounds what any
stack in the comparison could achieve.

``cae_s3_tool put`` is reused rather than reimplementing SigV4 here: it
already exists under CAE_ENABLE_S3, is the same helper the read benchmark's
floor would use, and keeps this script free of an AWS dependency. One process
per in-flight PUT is the point, not an accident -- it is the crudest possible
concurrency and therefore the fairest floor.

Reports in exactly the wording ``clio_bench::PrintResults`` uses, so one
parser (``jarvis_clio_core/bench_parse.py``) serves all three stacks.

Credentials come from the environment / the standard AWS chain, as resolved by
cae_s3_tool itself -- never passed on the command line.

Usage:
    s3_raw_put.py --bucket B --key-prefix <prefix> --num-objects 64 \\
        --object-size 4194304 --concurrency 8 [--label Rawput]

Requires: cae_s3_tool on PATH (or --s3-tool).
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
                   help="keys are <prefix>/raw_%%06d.bin")
    p.add_argument("--num-objects", type=int, required=True)
    p.add_argument("--object-size", type=int, required=True,
                   help="bytes per object; match the CLIO row's object_size")
    p.add_argument("--concurrency", type=int, default=8,
                   help="concurrent cae_s3_tool processes (K)")
    p.add_argument("--label", default="Rawput",
                   help="results namespace (becomes the CSV 'operation')")
    p.add_argument("--s3-tool", default=None,
                   help="path to cae_s3_tool; also read from CAE_S3_TOOL")
    p.add_argument("--tmpdir", default=None,
                   help="where the staged source files live; peak usage is "
                        "concurrency * object_size")
    return p.parse_args(argv)


def resolve_tool(explicit):
    """Locate the cae_s3_tool helper.

    :param explicit: Value of --s3-tool, or None.
    :return: Path or bare name to exec.
    """
    return (explicit or os.environ.get("CAE_S3_TOOL")
            or shutil.which("cae_s3_tool") or "cae_s3_tool")


def stage_sources(tmpdir, concurrency, object_size):
    """Write the source files the PUTs will upload.

    One file PER SLOT rather than per object: the bytes are irrelevant to a
    wire-speed measurement, and staging num_objects files would make setup
    cost more than the benchmark. Staging happens BEFORE timing starts so
    local disk write speed never enters the result.

    :param tmpdir: Directory to stage into.
    :param concurrency: Number of slot files to create.
    :param object_size: Bytes per file.
    :return: List of staged file paths.
    """
    paths = []
    payload = os.urandom(min(object_size, 1 << 20))
    for slot in range(concurrency):
        path = os.path.join(tmpdir, f"src_{slot}.bin")
        with open(path, "wb") as f:
            written = 0
            while written < object_size:
                n = min(len(payload), object_size - written)
                f.write(payload[:n])
                written += n
        paths.append(path)
    return paths


def run_puts(args, tool, sources):
    """Run the timed upload: K concurrent cae_s3_tool put processes.

    :param args: Parsed arguments.
    :param tool: Path to cae_s3_tool.
    :param sources: Staged source files, one per slot.
    :return: Tuple of (elapsed microseconds, failure count).
    """
    k = min(args.concurrency, args.num_objects)
    running = {}   # Popen -> slot index
    failures = 0
    next_idx = 0

    def launch(slot, idx):
        key = f"{args.key_prefix}/raw_{idx:06d}.bin"
        return subprocess.Popen(
            [tool, "put", args.bucket, key, sources[slot]],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    t0 = timeit.default_timer()
    for slot in range(k):
        running[launch(slot, next_idx)] = slot
        next_idx += 1

    done = 0
    while running:
        # Block on any one process, then immediately refill its slot: this
        # keeps K uploads in flight rather than draining to zero between
        # batches, which is what a naive wait-for-all loop would do.
        for proc in list(running):
            rc = proc.wait()
            slot = running.pop(proc)
            done += 1
            if rc != 0:
                failures += 1
                err = proc.stderr.read().decode(errors="replace").strip()
                print(f"WARNING: cae_s3_tool put failed (rc={rc}): {err}")
            if next_idx < args.num_objects:
                running[launch(slot, next_idx)] = slot
                next_idx += 1
            break

    elapsed_us = (timeit.default_timer() - t0) * 1e6
    return elapsed_us, failures


def print_results(label, elapsed_us, n_ops, io_size, k):
    """Emit the throughput block in the exact bench_common.h wording.

    :param label: Results namespace.
    :param elapsed_us: Wall time of the upload in microseconds.
    :param n_ops: Number of objects uploaded.
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
    :param elapsed_us: Wall time of the upload in microseconds.
    :param n_ops: Number of objects uploaded.
    :param k: Effective concurrency.
    """
    total_bytes = n_ops * args.object_size
    seconds = elapsed_us / 1e6
    wire_bw = ((total_bytes / (1024.0 * 1024.0)) / seconds
               if seconds > 0 else 0.0)
    print("")
    print(f"=== {label} Fairness ===")
    print(f"Objects written: {n_ops}")
    print(f"Bytes moved: {total_bytes}")
    print(f"Logical bytes: {total_bytes}")
    print(f"PUT count: {n_ops}")
    print("Compression: none")
    print("Decode step: no")
    print(f"Requested concurrency: {args.concurrency}")
    print(f"Effective concurrency: {k}")
    print("Runtime worker threads: 0")
    print(f"Wall time us: {elapsed_us} ({elapsed_us / 1000.0} ms)")
    print(f"Wire bandwidth: {wire_bw} MB/s")
    # One process per PUT is this floor's defining cost -- and the reason it
    # is a floor rather than a ceiling: CLIO pays no spawn at all. The staged
    # sources are written before timing starts, but they are real bytes on
    # local disk, so they are reported rather than claimed as zero.
    print(f"Subprocess spawns: {n_ops}")
    print(f"Temp file bytes: {k * args.object_size}")
    print(f"Transport chunk bytes: {args.object_size}")
    print("===================")


def main(argv):
    """Stage the sources, run the timed upload, and report both blocks.

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
    tmpdir = tempfile.mkdtemp(prefix="s3_raw_put_", dir=args.tmpdir)
    try:
        print(f"s3_raw_put: s3://{args.bucket}/{args.key_prefix} x "
              f"{args.num_objects} objects of {args.object_size} B, K={k}, "
              f"tool={tool}")
        sources = stage_sources(tmpdir, k, args.object_size)
        elapsed_us, failures = run_puts(args, tool, sources)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print_results(args.label, elapsed_us, args.num_objects, args.object_size,
                  k)
    print_fairness(args.label, args, elapsed_us, args.num_objects, k)

    if failures:
        # A partial upload makes the throughput number meaningless: it timed
        # fewer bytes than it claims. Fail the row rather than publish it.
        print(f"ERROR: {failures} of {args.num_objects} PUTs failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
