#!/usr/bin/env python3
"""Stage the CLIO-vs-Zarr S3 read benchmark dataset into an S3 bucket.

This is an OPERATOR TOOL, run once by hand from a host with S3 egress. It is not
invoked by the sweep -- the benchmark only ever reads what this script wrote.

It publishes one logical dataset (by default a 1024^3 uint16 array, 2 GiB,
matching the zarr_benchmarks reference suite) in two shapes:

  1. Zarr v3 stores at several chunk edges, uncompressed and zstd-compressed --
     what the Zarr driver reads.
  2. The exact same bytes re-split into flat objects at matching sizes -- what
     the CLIO CAE assimilator reads.

Because every raw object set is the same buffer re-split, total bytes moved is
constant across the granularity axis and only the REQUEST COUNT varies. That is
what makes the granularity sweep a controlled comparison rather than four
unrelated measurements.

Layout under s3://<bucket>/<prefix>/:

    manifest.json
    zarr/bench_c<edge>_<none|zstd>.zarr
    raw/<object_size_bytes>/obj_%06d.bin

Chunk edge c yields c^3 * 2 bytes per chunk/object; at the default array edge:
    64 -> 512 KiB, 128 -> 4 MiB, 256 -> 32 MiB, 512 -> 256 MiB

Idempotent: manifest.json is written LAST and records what completed, so a
re-run skips finished work and redoes only partial uploads.

Usage:
    python3 stage_s3_read_bench_data.py --bucket my-bucket \\
        --prefix clio-s3-read-bench [--region us-east-1]
        [--only zarr|raw] [--only-granularity 256] [--force] [--dry-run]

Credentials come from the standard botocore chain (~/.aws/credentials +
AWS_PROFILE, or AWS_* environment variables) -- never passed on the CLI.

--endpoint-url (or S3_ENDPOINT / AWS_ENDPOINT_URL) targets an S3-compatible
store instead of real AWS; leave unset for AWS.

Requires: zarr>=3, s3fs, numpy.
"""

import argparse
import concurrent.futures
import json
import os
import sys

import numpy as np

# Logical array defaults: 1024^3 uint16 == 2 GiB exactly (zarr_benchmarks
# convention). --array-edge shrinks this for smoke tests.
DEFAULT_ARRAY_EDGE = 1024
DTYPE = np.dtype("uint16")

# Chunk edges swept by the benchmark. Each must divide the array edge so the
# array tiles exactly with no partial edge chunks.
DEFAULT_CHUNK_EDGES = (64, 128, 256, 512)
VARIANTS = ("none", "zstd")

# Streaming granularity for raw uploads: bounds peak RSS regardless of how big
# the object being written is.
UPLOAD_SLICE_ELEMS = (8 * 1024 * 1024) // DTYPE.itemsize


def chunk_nbytes(edge):
    """Bytes in one chunk/object at the given cubic chunk edge.

    :param edge: Cubic chunk edge length in elements.
    :return: Number of bytes occupied by one uncompressed chunk.
    """
    return (edge ** 3) * DTYPE.itemsize


def total_bytes(array_edge):
    """Total logical size of the array in bytes.

    :param array_edge: Cubic array edge length in elements.
    :return: Byte count of the whole uncompressed array.
    """
    return (array_edge ** 3) * DTYPE.itemsize


def gen_elems(start_elem, n_elems, pattern, seed):
    """Generate array elements [start_elem, start_elem + n_elems).

    Deterministic and position-addressable: any slice of the logical array can
    be regenerated without materializing its neighbours, which is what keeps
    peak memory flat while writing gigabytes.

    :param start_elem: Index of the first element to produce.
    :param n_elems: How many elements to produce.
    :param pattern: 'smooth' (compressible) or 'random' (incompressible).
    :param seed: Base seed; mixed with position for the random pattern.
    :return: A uint16 numpy array of length n_elems.
    """
    if pattern == "random":
        # Position-mixed seed keeps this reproducible per slice.
        rng = np.random.default_rng((int(seed) << 32) ^ int(start_elem))
        return rng.integers(0, 65536, size=n_elems, dtype=np.uint16)
    # 'smooth': a quantized sum of sinusoids. The periods are deliberately long
    # relative to the amplitude so consecutive uint16 samples repeat, which is
    # what makes the array compressible the way a real scientific field is.
    # Measured ~20x with zstd level 0, close to the zarr_benchmarks reference
    # dataset's 24x, and stable across array sizes (the ratio depends on the
    # local derivative, not on the element count). Shortening these periods
    # collapses the ratio toward 1x and turns the compression axis into a
    # measurement of zstd's CPU cost instead of its bandwidth saving.
    i = np.arange(start_elem, start_elem + n_elems, dtype=np.int64)
    vals = (np.sin(i / 400000.0) + np.sin(i / 100000.0)) * 8000.0 + 16000.0
    return vals.astype(np.uint16)


def resolve_endpoint(explicit):
    """Pick the S3 endpoint override, if any.

    :param explicit: Value of --endpoint-url, or None.
    :return: Endpoint URL string, or None for real AWS.
    """
    return (explicit or os.environ.get("S3_ENDPOINT")
            or os.environ.get("AWS_ENDPOINT_URL") or None)


def make_fs(region, endpoint_url=None, anon=False):
    """Build the s3fs filesystem used for every upload.

    :param region: AWS region name.
    :param endpoint_url: S3-compatible endpoint, or None for real AWS.
    :param anon: Whether to skip credential resolution entirely.
    :return: A configured s3fs.S3FileSystem.
    """
    import s3fs

    client_kwargs = {"region_name": region}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    return s3fs.S3FileSystem(
        anon=anon,
        client_kwargs=client_kwargs,
        config_kwargs={"max_pool_connections": 32,
                       "retries": {"max_attempts": 5, "mode": "standard"}},
        skip_instance_cache=True,
    )


def create_bucket(bucket, region, endpoint_url):
    """Create the target bucket, tolerating the us-east-1 special case.

    CreateBucket must OMIT CreateBucketConfiguration for us-east-1 -- passing
    a LocationConstraint of us-east-1 is rejected as invalid, which is exactly
    what s3fs.mkdir does. Going through botocore directly avoids that.

    :param bucket: Bucket name to create.
    :param region: AWS region name.
    :param endpoint_url: S3-compatible endpoint, or None for real AWS.
    """
    import boto3

    client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
    kwargs = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        client.create_bucket(**kwargs)
        print(f"created bucket {bucket}")
    except client.exceptions.BucketAlreadyOwnedByYou:
        print(f"bucket {bucket} already exists")
    except Exception as e:
        print(f"bucket {bucket}: {e}")


def store_stored_bytes(fs, url):
    """Total stored (post-compression) bytes of a written store.

    :param fs: s3fs filesystem.
    :param url: '<bucket>/<key>' of the store.
    :return: Stored byte count, or 0 when it cannot be determined.
    """
    try:
        info = fs.du(url, total=True)
        if isinstance(info, dict):
            return int(sum(info.values()))
        return int(info)
    except Exception:
        return 0


def zarr_storage_options(region, endpoint_url):
    """fsspec options for a zarr store, matching what the reader uses.

    :param region: AWS region name.
    :param endpoint_url: S3-compatible endpoint, or None for real AWS.
    :return: Dict of storage options for FsspecStore.from_url.
    """
    client_kwargs = {"region_name": region}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    return {"client_kwargs": client_kwargs, "skip_instance_cache": True}


def write_zarr_store(fs, bucket, prefix, array_edge, edge, variant, pattern,
                     seed, dry_run, storage_options):
    """Write one Zarr v3 store, one chunk at a time.

    Peak memory is one chunk, so the largest store costs one chunk of RSS
    rather than the whole array.

    :param fs: s3fs filesystem.
    :param bucket: Target bucket.
    :param prefix: Key prefix under the bucket.
    :param array_edge: Cubic array edge.
    :param edge: Cubic chunk edge.
    :param variant: 'none' or 'zstd'.
    :param pattern: Data pattern passed to gen_elems.
    :param seed: Data seed passed to gen_elems.
    :param dry_run: When true, report and skip all writes.
    :param storage_options: fsspec options for FsspecStore.from_url.
    :return: Dict describing the store, or None when skipped.
    """
    import zarr
    from zarr.storage import FsspecStore

    name = f"bench_c{edge}_{variant}.zarr"
    url = f"{bucket}/{prefix}/zarr/{name}"
    if dry_run:
        print(f"  [dry-run] would write zarr store s3://{url}")
        return None

    compressors = None if variant == "none" else zarr.codecs.ZstdCodec(level=0)
    # from_url (not FsspecStore(fs, ...)): a filesystem built in a sync context
    # binds to a different event loop than the one zarr drives it from, which
    # fails with "attached to a different loop".
    store = FsspecStore.from_url(f"s3://{url}",
                                 storage_options=storage_options)
    arr = zarr.create_array(
        store, name="/", shape=(array_edge,) * 3,
        chunks=(edge,) * 3, dtype=DTYPE, compressors=compressors,
        overwrite=True,
    )

    n_per_axis = array_edge // edge
    n_chunks = n_per_axis ** 3
    written = 0
    for zi in range(n_per_axis):
        for yi in range(n_per_axis):
            for xi in range(n_per_axis):
                # Chunks are filled from a contiguous element range so the
                # bytes match the raw-object layout for the same logical data.
                start = ((zi * n_per_axis + yi) * n_per_axis + xi) * (edge ** 3)
                block = gen_elems(start, edge ** 3, pattern, seed)
                arr[zi * edge:(zi + 1) * edge,
                    yi * edge:(yi + 1) * edge,
                    xi * edge:(xi + 1) * edge] = block.reshape((edge,) * 3)
                written += 1
        print(f"    {name}: {written}/{n_chunks} chunks", flush=True)

    stored = store_stored_bytes(fs, url)
    ratio = (total_bytes(array_edge) / stored) if stored else 0.0
    print(f"  wrote s3://{url}  chunks={n_chunks} stored={stored} "
          f"ratio={ratio:.2f}x")
    return {"name": name, "chunk_edge": edge, "variant": variant,
            "n_chunks": n_chunks, "stored_bytes": int(stored),
            "ratio": round(ratio, 3)}


def write_one_object(fs, url, obj_index, obj_elems, pattern, seed):
    """Stream a single raw object to S3 in bounded slices.

    :param fs: s3fs filesystem.
    :param url: Full '<bucket>/<key>' target.
    :param obj_index: Index of this object within its granularity set.
    :param obj_elems: Elements this object carries.
    :param pattern: Data pattern passed to gen_elems.
    :param seed: Data seed passed to gen_elems.
    """
    base = obj_index * obj_elems
    with fs.open(url, "wb", block_size=8 * 1024 * 1024) as f:
        done = 0
        while done < obj_elems:
            n = min(UPLOAD_SLICE_ELEMS, obj_elems - done)
            f.write(gen_elems(base + done, n, pattern, seed).tobytes())
            done += n


def write_raw_set(fs, bucket, prefix, array_edge, obj_size, pattern, seed,
                  dry_run, workers=16):
    """Write one granularity's worth of flat objects covering the whole array.

    :param fs: s3fs filesystem.
    :param bucket: Target bucket.
    :param prefix: Key prefix under the bucket.
    :param array_edge: Cubic array edge.
    :param obj_size: Bytes per object.
    :param pattern: Data pattern passed to gen_elems.
    :param seed: Data seed passed to gen_elems.
    :param dry_run: When true, report and skip all writes.
    :param workers: Parallel upload threads.
    :return: Dict describing the object set, or None when skipped.
    """
    obj_elems = obj_size // DTYPE.itemsize
    n_objects = total_bytes(array_edge) // obj_size
    base_key = f"{prefix}/raw/{obj_size}"
    if dry_run:
        print(f"  [dry-run] would write {n_objects} x {obj_size}B objects "
              f"to s3://{bucket}/{base_key}/")
        return None

    def task(i):
        write_one_object(fs, f"{bucket}/{base_key}/obj_{i:06d}.bin", i,
                         obj_elems, pattern, seed)
        return i

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(task, range(n_objects)):
            done += 1
            if done % 256 == 0 or done == n_objects:
                print(f"    raw/{obj_size}: {done}/{n_objects} objects",
                      flush=True)
    print(f"  wrote s3://{bucket}/{base_key}/ ({n_objects} objects)")
    return {"object_size": int(obj_size), "n_objects": int(n_objects),
            "key_prefix": base_key}


def load_manifest(fs, bucket, prefix):
    """Read the existing manifest, or an empty skeleton if absent.

    :param fs: s3fs filesystem.
    :param bucket: Target bucket.
    :param prefix: Key prefix under the bucket.
    :return: Manifest dict.
    """
    try:
        with fs.open(f"{bucket}/{prefix}/manifest.json", "rb") as f:
            return json.loads(f.read().decode())
    except Exception:
        return {"zarr_stores": {}, "raw_sets": {}}


def raw_set_complete(fs, bucket, entry):
    """Check a recorded raw set still has all its objects in the bucket.

    :param fs: s3fs filesystem.
    :param bucket: Target bucket.
    :param entry: Manifest entry for the set.
    :return: True when the object count matches what was recorded.
    """
    try:
        listing = fs.ls(f"{bucket}/{entry['key_prefix']}", detail=False)
    except Exception:
        return False
    return len(listing) >= entry["n_objects"]


def parse_args(argv):
    """Parse command-line arguments.

    :param argv: Argument list excluding the program name.
    :return: Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="clio-s3-read-bench")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--endpoint-url", default=None,
                   help="S3-compatible endpoint; also read from S3_ENDPOINT / "
                        "AWS_ENDPOINT_URL. Leave unset for real AWS.")
    p.add_argument("--array-edge", type=int, default=DEFAULT_ARRAY_EDGE,
                   help="cubic array edge (default 1024 -> 2 GiB uint16)")
    p.add_argument("--chunk-edges", type=int, nargs="+",
                   default=list(DEFAULT_CHUNK_EDGES),
                   help="cubic chunk edges to stage; each must divide "
                        "--array-edge")
    p.add_argument("--pattern", choices=("smooth", "random"), default="smooth",
                   help="'smooth' compresses realistically; 'random' is "
                        "incompressible and makes the zstd variant a no-op")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--only", choices=("zarr", "raw"), default=None)
    p.add_argument("--only-granularity", type=int, default=None,
                   help="restrict to one chunk edge")
    p.add_argument("--force", action="store_true",
                   help="re-upload even when the manifest says it is done")
    p.add_argument("--create-bucket", action="store_true",
                   help="create the bucket first (S3-compatible test stores)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args(argv)


def main(argv):
    """Stage every requested store and object set, then write the manifest.

    :param argv: Argument list excluding the program name.
    :return: Process exit code.
    """
    args = parse_args(argv)
    edges = ([args.only_granularity] if args.only_granularity
             else list(args.chunk_edges))
    for e in edges:
        if e <= 0 or args.array_edge % e != 0:
            print(f"ERROR: chunk edge {e} must be > 0 and divide "
                  f"--array-edge {args.array_edge}")
            return 2

    endpoint = resolve_endpoint(args.endpoint_url)
    fs = make_fs(args.region, endpoint)
    zarr_opts = zarr_storage_options(args.region, endpoint)
    if args.create_bucket and not args.dry_run:
        create_bucket(args.bucket, args.region, endpoint)

    manifest = load_manifest(fs, args.bucket, args.prefix)
    manifest.setdefault("zarr_stores", {})
    manifest.setdefault("raw_sets", {})

    print(f"Staging into s3://{args.bucket}/{args.prefix}/  "
          f"array={args.array_edge}^3 {DTYPE.name} "
          f"({total_bytes(args.array_edge)} bytes) pattern={args.pattern}")

    if args.only != "raw":
        for edge in edges:
            for variant in VARIANTS:
                key = f"c{edge}_{variant}"
                if key in manifest["zarr_stores"] and not args.force:
                    print(f"  skip zarr {key} (already in manifest)")
                    continue
                entry = write_zarr_store(fs, args.bucket, args.prefix,
                                         args.array_edge, edge, variant,
                                         args.pattern, args.seed,
                                         args.dry_run, zarr_opts)
                if entry:
                    manifest["zarr_stores"][key] = entry

    if args.only != "zarr":
        for edge in edges:
            size = chunk_nbytes(edge)
            key = str(size)
            existing = manifest["raw_sets"].get(key)
            if existing and not args.force and raw_set_complete(
                    fs, args.bucket, existing):
                print(f"  skip raw/{size} (already complete)")
                continue
            entry = write_raw_set(fs, args.bucket, args.prefix,
                                  args.array_edge, size, args.pattern,
                                  args.seed, args.dry_run, args.workers)
            if entry:
                manifest["raw_sets"][key] = entry

    if args.dry_run:
        print("dry-run: manifest not written")
        return 0

    # Written LAST so a crash mid-upload leaves no entry claiming completion.
    manifest.update({
        "array_shape": [args.array_edge] * 3, "dtype": DTYPE.name,
        "total_bytes": int(total_bytes(args.array_edge)),
        "pattern": args.pattern, "seed": args.seed,
        "chunk_edges": list(edges),
    })
    with fs.open(f"{args.bucket}/{args.prefix}/manifest.json", "wb") as f:
        f.write(json.dumps(manifest, indent=2).encode())
    print(f"wrote s3://{args.bucket}/{args.prefix}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
