#!/usr/bin/env python3
"""Read a Zarr v3 array from S3 and report throughput + equivalence caveats.

The Zarr half of the CLIO-vs-Zarr S3 read benchmark. Counterpart driver:
``context-assimilation-engine/benchmark/clio_s3_read_bench.cc``.

This runs zarr the way cloud users actually deploy it -- zarr-python over s3fs,
chunked, optionally compressed, reading the array in one shot -- and reports in
exactly the wording ``clio_bench::PrintResults`` uses, so one parser
(``jarvis_clio_core/bench_parse.py``) serves both stacks.

Two reporting decisions make the comparison honest rather than flattering:

  * ``Total data`` / ``Aggregate bandwidth`` use LOGICAL (uncompressed) bytes on
    both sides, so they compare application-level throughput.
  * ``Bytes moved`` / ``Wire bandwidth`` report what actually crossed the
    network, counted at the fsspec layer. For a zstd store these differ by the
    compression ratio, which is the single biggest confound in the comparison.

Credentials come from the standard botocore chain (~/.aws/credentials +
AWS_PROFILE, or AWS_* env vars) -- never passed on the command line.

Usage:
    zarr_s3_read.py --bucket B --store-key <prefix>/zarr/bench_c256_none.zarr \\
        [--label Read] [--async-concurrency 32] [--region us-east-1] [--anon]

Requires: zarr>=3, s3fs, numpy.
"""

import argparse
import os
import sys
import timeit

import numpy as np


def parse_args(argv):
    """Parse command-line arguments.

    :param argv: Argument list excluding the program name.
    :return: Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True)
    p.add_argument("--store-key", required=True,
                   help="key of the .zarr store within the bucket")
    p.add_argument("--label", default="Read",
                   help="results namespace (becomes the CSV 'operation')")
    p.add_argument("--async-concurrency", type=int, default=32,
                   help="zarr async.concurrency -- the Zarr-side concurrency "
                        "knob; the reference suite's default of 10 is far too "
                        "low for WAN S3")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--endpoint-url", default=None,
                   help="S3-compatible endpoint; also read from S3_ENDPOINT / "
                        "AWS_ENDPOINT_URL. Leave unset for real AWS.")
    p.add_argument("--anon", action="store_true",
                   help="skip credential resolution (public buckets only)")
    return p.parse_args(argv)


def resolve_endpoint(explicit):
    """Pick the S3 endpoint override, if any.

    :param explicit: Value of --endpoint-url, or None.
    :return: Endpoint URL string, or None for real AWS.
    """
    return (explicit or os.environ.get("S3_ENDPOINT")
            or os.environ.get("AWS_ENDPOINT_URL") or None)


def open_array(args):
    """Open the S3-backed Zarr v3 array and return it with its filesystem.

    Uses ``FsspecStore.from_url`` rather than building the filesystem with
    ``fsspec.url_to_fs`` and wrapping it: a filesystem constructed in a plain
    sync context ends up bound to a different event loop than the one zarr
    drives it from, which fails at read time with "attached to a different
    loop". ``from_url`` constructs it in the right context. (The reference
    suite's store branch is no help here -- it only ever matched ``http``,
    never ``s3://``.)

    :param args: Parsed arguments (bucket, store key, region, concurrency).
    :return: Tuple of (zarr array, fsspec filesystem).
    """
    import zarr
    from zarr.storage import FsspecStore

    # MUST precede array construction: zarr snapshots config at open time.
    zarr.config.set({"async.concurrency": args.async_concurrency})

    client_kwargs = {"region_name": args.region}
    endpoint = resolve_endpoint(args.endpoint_url)
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint

    storage_options = {
        "anon": args.anon,
        "client_kwargs": client_kwargs,
        "config_kwargs": {
            # Don't let the connection pool become the real concurrency cap.
            "max_pool_connections": max(args.async_concurrency * 2, 32),
            "retries": {"max_attempts": 5, "mode": "standard"},
        },
        # No client-side caching: every run must actually cross the network.
        "default_cache_type": "none",
        "skip_instance_cache": True,
    }
    url = f"s3://{args.bucket}/{args.store_key}"
    store = FsspecStore.from_url(url, storage_options=storage_options,
                                 read_only=True)
    return zarr.open(store=store, mode="r"), store.fs


def install_counters(fs):
    """Wrap the filesystem's read methods to count GETs and bytes on the wire.

    Instrumenting fsspec rather than the zarr Store is deliberate: zarr Store
    subclasses may reject attribute shadowing, while fsspec filesystem objects
    accept it. ``_cat_file`` covers whole-chunk reads and ``_cat_ranges`` the
    sharded/partial-read path.

    :param fs: The fsspec filesystem to instrument, modified in place.
    :return: A counters dict with 'gets' and 'wire_bytes' keys.
    """
    counters = {"gets": 0, "wire_bytes": 0}

    orig_cat_file = fs._cat_file

    async def counting_cat_file(path, start=None, end=None, **kwargs):
        data = await orig_cat_file(path, start=start, end=end, **kwargs)
        counters["gets"] += 1
        counters["wire_bytes"] += len(data)
        return data

    fs._cat_file = counting_cat_file

    orig_cat_ranges = getattr(fs, "_cat_ranges", None)
    if orig_cat_ranges is not None:
        async def counting_cat_ranges(paths, starts, ends, **kwargs):
            out = await orig_cat_ranges(paths, starts, ends, **kwargs)
            counters["gets"] += len(out)
            counters["wire_bytes"] += sum(len(b) for b in out)
            return out

        fs._cat_ranges = counting_cat_ranges

    return counters


def print_results(label, elapsed_us, n_ops, io_size):
    """Emit the throughput block in the exact bench_common.h wording.

    Reimplements ``clio_bench::PrintResults`` for a single-threaded reader
    (min == max == avg, stddev 0), so the shared regex table applies unchanged.

    :param label: Results namespace.
    :param elapsed_us: Wall time of the read in microseconds.
    :param n_ops: Number of chunks read.
    :param io_size: Logical bytes per chunk.
    """
    total_bytes = n_ops * io_size
    seconds = elapsed_us / 1e6
    bw = (total_bytes / (1024.0 * 1024.0)) / seconds if seconds > 0 else 0.0
    ops_s = (n_ops / seconds) if seconds > 0 else 0.0
    print("")
    print(f"=== {label} Benchmark Results ===")
    print(f"Time (min): {elapsed_us} us ({elapsed_us / 1000.0} ms)")
    print(f"Time (max): {elapsed_us} us ({elapsed_us / 1000.0} ms)")
    print(f"Time (avg): {elapsed_us} us ({elapsed_us / 1000.0} ms)")
    print(f"Bandwidth per thread (min): {bw} MB/s")
    print(f"Bandwidth per thread (max): {bw} MB/s")
    print(f"Bandwidth per thread (avg): {bw} MB/s")
    print(f"Aggregate bandwidth: {bw} MB/s")
    print(f"Aggregate IOPS: {ops_s}")
    print(f"IOPS per thread (avg): {ops_s}")
    print(f"Avg latency per op: {elapsed_us / n_ops if n_ops else 0.0} us")
    print("Latency stddev: 0.0 us")
    print(f"Total data: {total_bytes / (1024.0 * 1024.0)} MB")
    print(f"Total ops: {n_ops}")
    print("===========================")


def print_fairness(label, args, arr, counters, elapsed_us, n_chunks, checksum):
    """Emit the equivalence-caveat block.

    :param label: Results namespace.
    :param args: Parsed arguments (for the requested concurrency).
    :param arr: The zarr array that was read.
    :param counters: Wire-byte / GET counters from install_counters.
    :param elapsed_us: Wall time of the read in microseconds.
    :param n_chunks: Number of chunks in the array.
    :param checksum: Strided sum forcing materialization of the result.
    """
    # zarr returns a tuple of codec instances (empty when uncompressed).
    # ZstdCodec -> "zstd" reads better as a CSV value than "zstdcodec".
    codecs = [c for c in (getattr(arr, "compressors", None) or [])]
    if codecs:
        name = codecs[0].__class__.__name__.lower()
        compression = name[:-5] if name.endswith("codec") else name
    else:
        compression = "none"
    seconds = elapsed_us / 1e6
    wire_bw = ((counters["wire_bytes"] / (1024.0 * 1024.0)) / seconds
               if seconds > 0 else 0.0)
    print("")
    print(f"=== {label} Fairness ===")
    print(f"Objects read: {n_chunks}")
    print(f"Bytes moved: {counters['wire_bytes']}")
    print(f"Logical bytes: {arr.nbytes}")
    print(f"GET count: {counters['gets']}")
    print(f"Compression: {compression}")
    print(f"Decode step: {'yes' if codecs else 'no'}")
    print(f"Requested concurrency: {args.async_concurrency}")
    print(f"Effective concurrency: {min(args.async_concurrency, n_chunks)}")
    print("Runtime worker threads: 0")
    print(f"Wall time us: {elapsed_us} ({elapsed_us / 1000.0} ms)")
    print(f"Wire bandwidth: {wire_bw} MB/s")
    # Zarr reads in-process: no helper subprocess, no temp-file staging.
    print("Subprocess spawns: 0")
    print("Temp file bytes: 0")
    print(f"Transport chunk bytes: {int(np.prod(arr.chunks)) * arr.dtype.itemsize}")
    print(f"Checksum: {checksum}")
    print("===================")


def main(argv):
    """Open the store, read the whole array, and report both blocks.

    :param argv: Argument list excluding the program name.
    :return: Process exit code.
    """
    args = parse_args(argv)
    arr, fs = open_array(args)
    n_chunks = int(np.prod([-(-s // c) for s, c in zip(arr.shape, arr.chunks)]))
    chunk_bytes = int(np.prod(arr.chunks)) * arr.dtype.itemsize
    print(f"zarr_s3_read: s3://{args.bucket}/{args.store_key} "
          f"shape={arr.shape} chunks={arr.chunks} dtype={arr.dtype} "
          f"chunks_total={n_chunks} concurrency={args.async_concurrency}")

    counters = install_counters(fs)
    t0 = timeit.default_timer()
    data = arr[:]
    elapsed_us = (timeit.default_timer() - t0) * 1e6

    # Touching the result forces materialization and doubles as a cross-run
    # correctness signal: the same store must yield the same checksum at every
    # concurrency setting.
    checksum = int(data[::97, ::89, ::83].sum())

    expected_gets = n_chunks + 1  # chunks + the zarr.json metadata object
    if abs(counters["gets"] - expected_gets) > 2:
        # A warning, never a failure: a miscounted GET must not lose the row.
        print(f"WARNING: GET count {counters['gets']} differs from expected "
              f"{expected_gets}")

    print_results(args.label, elapsed_us, n_chunks, chunk_bytes)
    print_fairness(args.label, args, arr, counters, elapsed_us, n_chunks,
                   checksum)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
