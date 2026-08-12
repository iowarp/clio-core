#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# Stream one array out of an OGB .zip (or a bare .npz) to stdout as raw
# little-endian bytes, converting dtype on the way, WITHOUT unpacking anything
# to disk.
#
# Why: gnn_prep.py extracts the archive, then unpacks each .npz member to a
# .npy so it can be mmapped, then converts that to a flat .f32. For
# ogbn-papers100M each of those steps is ~50 GiB and they coexist. Piped into
# gnn_ingest, this writes zero intermediate bytes -- the pages land straight in
# the CTE, tiered across RAM and NVMe.
#
#   python3 gnn_stream_npz.py --zip papers100M-bin.zip --member node_feat \
#       --dtype float32 | gnn_ingest --tag papers100M_feat --page-bytes 1048576
#
# Print the shape without streaming with --info.
#
# NOTE on nesting: the OGB download is a .zip containing raw/data.npz, and an
# .npz is itself a zip of .npy members. Both layers are opened as streams. A
# .npy inside an .npz is usually STORED (not deflated), so this reads at disk
# speed; if it is deflated, Python inflates on the fly and it still never lands
# on disk.

import argparse
import io
import sys
import zipfile

import numpy as np


def log(msg):
    print(f"[stream] {msg}", file=sys.stderr, flush=True)


def find_npz(zf, want):
    """Locate a .npz member inside the outer zip."""
    cands = [n for n in zf.namelist() if n.endswith(".npz")]
    if want:
        cands = [n for n in cands if want in n]
    if not cands:
        raise SystemExit(f"no .npz member found in archive (looked for {want})")
    return sorted(cands, key=len)[0]


def open_member(path, npz_hint, member):
    """Return (file_object, close_stack) positioned at the .npy member."""
    stack = []
    if path.endswith(".zip"):
        outer = zipfile.ZipFile(path)
        stack.append(outer)
        inner_name = find_npz(outer, npz_hint)
        log(f"outer zip -> {inner_name}")
        # ZipFile needs a seekable object; the nested npz must be materialised
        # in memory only if it is small. OGB's data.npz is huge, so instead we
        # rely on the outer member being STORED and wrap it with a seekable
        # view over the outer file handle.
        info = outer.getinfo(inner_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise SystemExit(
                f"inner {inner_name} is compressed in the outer zip "
                "(compress_type=%d); cannot stream it seekably. Extract the "
                "outer zip first, then point --zip at the .npz." %
                info.compress_type)
        base = open(path, "rb")
        stack.append(base)
        start = info.header_offset
        base.seek(start)
        # Skip the local file header to reach the raw member bytes.
        import struct
        sig, ver, flags, comp, t, d, crc, csz, usz, nlen, elen = struct.unpack(
            "<IHHHHHIIIHH", base.read(30))
        if sig != 0x04034b50:
            raise SystemExit("bad local file header in outer zip")
        base.seek(start + 30 + nlen + elen)
        view = SubFile(base, base.tell(), info.file_size)
        inner = zipfile.ZipFile(view)
        stack.append(inner)
        return inner, stack
    inner = zipfile.ZipFile(path)
    stack.append(inner)
    return inner, stack


class SubFile(io.RawIOBase):
    """Seekable read-only window onto a slice of another file."""

    def __init__(self, base, offset, length):
        self._b, self._o, self._n, self._p = base, offset, length, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, pos, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._p = pos
        elif whence == io.SEEK_CUR:
            self._p += pos
        else:
            self._p = self._n + pos
        self._p = max(0, min(self._p, self._n))
        return self._p

    def tell(self):
        return self._p

    def readinto(self, b):
        want = min(len(b), self._n - self._p)
        if want <= 0:
            return 0
        self._b.seek(self._o + self._p)
        data = self._b.read(want)
        b[:len(data)] = data
        self._p += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help=".zip (OGB download) or .npz")
    ap.add_argument("--npz", default="data.npz",
                    help="which .npz inside a .zip (substring match)")
    ap.add_argument("--member", required=True,
                    help="array name, e.g. node_feat / edge_index / node_label")
    ap.add_argument("--dtype", default=None,
                    help="cast to this dtype (e.g. float32); default = as-is")
    ap.add_argument("--rows-per-chunk", type=int, default=1 << 16)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="emit only the first N rows (0 = all). Lets a machine "
                         "that cannot hold the whole matrix still run on real "
                         "data rather than falling back to synthetic.")
    ap.add_argument("--info", action="store_true",
                    help="print shape/dtype and exit without streaming")
    args = ap.parse_args()

    zf, stack = open_member(args.zip, args.npz, args.member)
    names = zf.namelist()
    target = None
    for n in names:
        if n[:-4] == args.member or n == args.member + ".npy":
            target = n
            break
    if target is None:
        raise SystemExit(f"member '{args.member}' not in {sorted(names)}")

    with zf.open(target) as fh:
        version = np.lib.format.read_magic(fh)
        if version[0] == 1:
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        if fortran:
            raise SystemExit("fortran-order arrays not supported")
        out_dtype = np.dtype(args.dtype) if args.dtype else dtype
        rows = shape[0]
        if args.max_rows > 0:
            rows = min(rows, args.max_rows)
        cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        total_out = rows * cols * out_dtype.itemsize
        log(f"member={target} shape={shape} dtype={dtype} -> {out_dtype}"
            + (f"  (limited to first {rows} rows)" if args.max_rows else ""))
        log(f"streaming {total_out} bytes ({total_out / 2**30:.2f} GiB)")
        if args.info:
            print(f"{rows} {cols} {dtype} {out_dtype} {total_out}")
            return

        row_bytes = cols * dtype.itemsize
        chunk = max(1, args.rows_per_chunk)
        done = 0
        out = sys.stdout.buffer
        while done < rows:
            take = min(chunk, rows - done)
            raw = fh.read(row_bytes * take)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=dtype)
            if out_dtype != dtype:
                arr = arr.astype(out_dtype, copy=False)
            out.write(arr.tobytes())
            done += take
            if (done // chunk) % 64 == 0:
                log(f"  {done}/{rows} rows ({100.0 * done / rows:.1f}%)")
        out.flush()
        log(f"done: {done}/{rows} rows")
        if done != rows:
            raise SystemExit(f"short stream: {done} of {rows} rows")

    for s in reversed(stack):
        try:
            s.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
