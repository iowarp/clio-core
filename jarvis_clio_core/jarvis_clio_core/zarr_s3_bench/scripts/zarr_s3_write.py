#!/usr/bin/env python3
"""Write a Zarr v3 array to S3 and report throughput + equivalence caveats.

The Zarr half of the CLIO-vs-Zarr S3 WRITE benchmark. Counterpart drivers:
``context-transfer-engine/benchmark/clio_s3_write_bench.cc`` (CLIO) and
``jarvis_clio_core/s3_raw_put_bench/scripts/s3_raw_put.py`` (wire floor).

This runs zarr the way cloud users actually deploy it -- zarr-python over s3fs,
chunked, optionally compressed, writing the array in one shot -- and reports in
exactly the wording ``clio_bench::PrintResults`` uses, so one parser
(``jarvis_clio_core/bench_parse.py``) serves all three stacks.

Two reporting decisions make the comparison honest rather than flattering:

  * ``Total data`` / ``Aggregate bandwidth`` use LOGICAL (uncompressed) bytes on
    both sides, so they compare application-level throughput.
  * ``Bytes moved`` / ``Wire bandwidth`` report what actually crossed the
    network, counted at the fsspec layer. For a zstd store these differ by the
    compression ratio, which is the single biggest confound in the comparison
    -- and it cuts the OTHER way from the read benchmark: compression makes
    zarr look faster on writes because it sends fewer bytes, while CLIO's bdev
    sends every byte. Compare ``Wire bandwidth``, not just the headline. The
    ratio is controlled by ``--compressibility``, so it is a stated input to
    the run rather than an artifact of whatever test data happened to be used.

Credentials come from the standard botocore chain (~/.aws/credentials +
AWS_PROFILE, or AWS_* env vars) -- never passed on the command line.

Usage:
    zarr_s3_write.py --bucket B --store-key <prefix>/zarr/bench.zarr \\
        --total-bytes 268435456 --chunk-bytes 4194304 \\
        [--label Write] [--compressor none|zstd] [--async-concurrency 32]

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
                   help="key of the .zarr store within the bucket; it is "
                        "OVERWRITTEN on every run")
    p.add_argument("--total-bytes", type=int, required=True,
                   help="logical size of the array, matching the CLIO row's "
                        "num_blobs * blob_size")
    p.add_argument("--chunk-bytes", type=int, required=True,
                   help="bytes per zarr chunk, matching the CLIO row's "
                        "blob_size so both stacks move the same unit")
    p.add_argument("--label", default="Write",
                   help="results namespace (becomes the CSV 'operation')")
    p.add_argument("--compressor", default="none", choices=["none", "zstd"],
                   help="zstd sends fewer bytes than CLIO does; see the "
                        "module docstring before comparing headlines")
    p.add_argument("--compressibility", type=float, default=0.5,
                   help="0.0 = incompressible random bytes, 1.0 = a single "
                        "repeated byte. Controls the source data's entropy, "
                        "which is what decides whether the zstd variant sends "
                        "fewer bytes than CLIO's uncompressed bdev. Compare "
                        "'Bytes moved' against 'Logical bytes' to see the "
                        "ratio actually achieved.")
    p.add_argument("--async-concurrency", type=int, default=32,
                   help="zarr async.concurrency -- the Zarr-side concurrency "
                        "knob; the reference suite's default of 10 is far too "
                        "low for WAN S3")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--endpoint-url", default=None,
                   help="S3-compatible endpoint; also read from S3_ENDPOINT / "
                        "AWS_ENDPOINT_URL. Leave unset for real AWS.")
    return p.parse_args(argv)


def resolve_endpoint(explicit):
    """Pick the S3 endpoint override, if any.

    :param explicit: Value of --endpoint-url, or None.
    :return: Endpoint URL string, or None for real AWS.
    """
    return (explicit or os.environ.get("S3_ENDPOINT")
            or os.environ.get("AWS_ENDPOINT_URL") or None)


def open_store(args):
    """Open a writable S3-backed Zarr store and return it with its filesystem.

    Uses ``FsspecStore.from_url`` rather than building the filesystem with
    ``fsspec.url_to_fs`` and wrapping it: a filesystem constructed in a plain
    sync context ends up bound to a different event loop than the one zarr
    drives it from, which fails at I/O time with "attached to a different
    loop". ``from_url`` constructs it in the right context.

    :param args: Parsed arguments (bucket, store key, region, concurrency).
    :return: Tuple of (zarr store, fsspec filesystem).
    """
    import zarr
    from zarr.storage import FsspecStore

    # MUST precede store/array construction: zarr snapshots config at open
    # time, so setting it afterwards silently has no effect.
    zarr.config.set({"async.concurrency": args.async_concurrency})

    client_kwargs = {"region_name": args.region}
    endpoint = resolve_endpoint(args.endpoint_url)
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint

    storage_options = {
        "anon": False,
        "client_kwargs": client_kwargs,
        "config_kwargs": {
            # Don't let the connection pool become the real concurrency cap.
            "max_pool_connections": max(args.async_concurrency * 2, 32),
            "retries": {"max_attempts": 5, "mode": "standard"},
        },
        "default_cache_type": "none",
        "skip_instance_cache": True,
    }
    url = f"s3://{args.bucket}/{args.store_key}"
    store = FsspecStore.from_url(url, storage_options=storage_options,
                                 read_only=False)
    return store, store.fs


def install_counters(fs):
    """Wrap the filesystem's write methods to count PUTs and bytes on the wire.

    The write-side counterpart of the read bench's ``_cat_file`` patch.
    Instrumenting fsspec rather than the zarr Store is deliberate: zarr Store
    subclasses may reject attribute shadowing, while fsspec filesystem objects
    accept it. ``_pipe_file`` is the single-object write path zarr uses for
    every chunk; ``_pipe`` covers any batched variant.

    :param fs: The fsspec filesystem to instrument, modified in place.
    :return: A counters dict with 'puts' and 'wire_bytes' keys.
    """
    counters = {"puts": 0, "wire_bytes": 0}

    orig_pipe_file = fs._pipe_file

    async def counting_pipe_file(path, value, **kwargs):
        counters["puts"] += 1
        counters["wire_bytes"] += len(value)
        return await orig_pipe_file(path, value, **kwargs)

    fs._pipe_file = counting_pipe_file

    return counters


def make_array(args, store, shape, chunks, dtype):
    """Create the destination Zarr array, overwriting any previous run.

    :param args: Parsed arguments (compressor choice).
    :param store: Writable zarr store.
    :param shape: Array shape.
    :param chunks: Chunk shape.
    :param dtype: numpy dtype of the array.
    :return: The created zarr array.
    """
    import zarr

    if args.compressor == "zstd":
        from zarr.codecs import ZstdCodec
        compressors = [ZstdCodec()]
    else:
        compressors = []

    return zarr.create_array(store=store, shape=shape, chunks=chunks,
                             dtype=dtype, compressors=compressors,
                             overwrite=True)


def print_results(label, elapsed_us, n_ops, io_size):
    """Emit the throughput block in the exact bench_common.h wording.

    Reimplements ``clio_bench::PrintResults`` for a single-threaded writer
    (min == max == avg, stddev 0), so the shared regex table applies unchanged.

    :param label: Results namespace.
    :param elapsed_us: Wall time of the write in microseconds.
    :param n_ops: Number of chunks written.
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
    :param arr: The zarr array that was written.
    :param counters: Wire-byte / PUT counters from install_counters.
    :param elapsed_us: Wall time of the write in microseconds.
    :param n_chunks: Number of chunks in the array.
    :param checksum: Strided sum of the source data, matching the read bench's.
    """
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
    print(f"Objects written: {counters['puts']}")
    print(f"Bytes moved: {counters['wire_bytes']}")
    print(f"Logical bytes: {arr.nbytes}")
    print(f"PUT count: {counters['puts']}")
    print(f"Compression: {compression}")
    # "Decode step" is named for the read side; on writes it is the ENCODE
    # pass. Kept under the read-side name so one parser key serves both.
    print(f"Decode step: {'yes' if codecs else 'no'}")
    print(f"Requested concurrency: {args.async_concurrency}")
    print(f"Effective concurrency: {min(args.async_concurrency, n_chunks)}")
    print("Runtime worker threads: 0")
    print(f"Wall time us: {elapsed_us} ({elapsed_us / 1000.0} ms)")
    print(f"Wire bandwidth: {wire_bw} MB/s")
    # Zarr writes in-process: no helper subprocess, no temp-file staging.
    print("Subprocess spawns: 0")
    print("Temp file bytes: 0")
    print(f"Transport chunk bytes: {int(np.prod(arr.chunks)) * arr.dtype.itemsize}")
    print(f"Checksum: {checksum}")
    print("===================")


def main(argv):
    """Create the array, write it whole, and report both blocks.

    :param argv: Argument list excluding the program name.
    :return: Process exit code.
    """
    args = parse_args(argv)

    # 1-D uint8 array: one byte per element makes the byte math exact and
    # keeps chunk_bytes directly comparable to the CLIO row's blob_size.
    dtype = np.dtype("uint8")
    if args.chunk_bytes <= 0 or args.total_bytes <= 0:
        print("ERROR: --total-bytes and --chunk-bytes must be > 0")
        return 1
    n_chunks = -(-args.total_bytes // args.chunk_bytes)
    shape = (n_chunks * args.chunk_bytes,)
    chunks = (args.chunk_bytes,)

    # Source entropy is a benchmark INPUT, not an accident. Two degenerate
    # choices were both rejected: all-zeros makes zstd look arbitrarily good
    # (it compresses to nothing, so the zstd row measures no real transfer),
    # while uniform random makes it look arbitrarily bad (nothing compresses,
    # so the row measures pure encode overhead and zstd can even expand).
    # Neither resembles the scientific arrays zarr is actually used for.
    # --compressibility picks the point on that axis by shrinking the symbol
    # alphabet: fewer distinct byte values -> lower entropy -> better ratio.
    c = min(max(args.compressibility, 0.0), 1.0)
    n_symbols = max(1, int(round(256 ** (1.0 - c))))
    rng = np.random.default_rng(20260825)
    data = rng.integers(0, n_symbols, size=shape, dtype=dtype)
    checksum = int(data[::97].sum())

    store, fs = open_store(args)
    counters = install_counters(fs)
    arr = make_array(args, store, shape, chunks, dtype)

    print(f"zarr_s3_write: s3://{args.bucket}/{args.store_key} "
          f"shape={shape} chunks={chunks} dtype={dtype} "
          f"chunks_total={n_chunks} compressor={args.compressor} "
          f"compressibility={c} (alphabet={n_symbols}) "
          f"concurrency={args.async_concurrency}")

    # Counters are installed BEFORE make_array, so the store's metadata PUTs
    # are already counted. Reset here so the timed block reports only the
    # data written inside it.
    counters["puts"] = 0
    counters["wire_bytes"] = 0

    t0 = timeit.default_timer()
    arr[:] = data
    elapsed_us = (timeit.default_timer() - t0) * 1e6

    if counters["puts"] < n_chunks:
        # A warning, never a failure: a miscounted PUT must not lose the row.
        print(f"WARNING: PUT count {counters['puts']} is below the {n_chunks} "
              f"chunks written -- the fsspec write path may have changed")

    print_results(args.label, elapsed_us, n_chunks, args.chunk_bytes)
    print_fairness(args.label, args, arr, counters, elapsed_us, n_chunks,
                   checksum)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
