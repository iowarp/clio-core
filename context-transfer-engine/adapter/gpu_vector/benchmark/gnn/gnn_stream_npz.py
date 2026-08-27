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
# .npz is itself a zip of .npy members -- two zip layers deep.
#
# The obvious approach, handing the inner member to zipfile.ZipFile, needs a
# SEEKABLE view of it, because zipfile reads the central directory at the end.
# That works only if the inner .npz is STORED in the outer zip. In the real
# papers100M archive it is DEFLATED, so there is no seekable view without
# inflating all ~56 GiB to disk first -- which is the flat copy this whole
# pipeline exists to avoid.
#
# So the inner zip is parsed SEQUENTIALLY instead: walk local file headers from
# the front, and for each .npy member read its numpy header to learn the exact
# payload length (the local header's size fields are unreliable when the writer
# used a data descriptor). Nothing seeks backwards, so the outer member can be
# a plain inflating stream.

import argparse
import io
import struct
import sys
import zipfile
import zlib

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


def _npy_payload_len(fh):
    """Read a .npy header from `fh` and return (shape, dtype, payload bytes)."""
    version = np.lib.format.read_magic(fh)
    if version[0] == 1:
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    if fortran:
        raise SystemExit("fortran-order arrays not supported")
    n = 1
    for d in shape:
        n *= int(d)
    return shape, dtype, n * dtype.itemsize


def _read_exact(fh, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _skip(fh, n, blk=1 << 22):
    while n > 0:
        got = fh.read(min(blk, n))
        if not got:
            raise SystemExit("unexpected EOF while skipping a member")
        n -= len(got)


class Pushback:
    """Forward-only stream with an unread() buffer.

    Inflating a zip member always over-reads: zlib consumes past the end of the
    deflate stream and hands the surplus back as unused_data. Those bytes are
    the next member's header, so they have to go back into the stream.
    """

    def __init__(self, fh):
        self._fh, self._buf = fh, b""

    def read(self, n):
        if not self._buf:
            return self._fh.read(n)
        out = self._buf[:n]
        self._buf = self._buf[len(out):]
        if len(out) < n:
            out += self._fh.read(n - len(out)) or b""
        return out

    def unread(self, b):
        if b:
            self._buf = b + self._buf


class MemberStream:
    """Decompressed bytes of one zip member, read forward only."""

    def __init__(self, src, method):
        self._src = src
        self._d = zlib.decompressobj(-15) if method == 8 else None
        self._buf = b""
        self._eof = False

    def _pump(self):
        if self._eof:
            return
        raw = self._src.read(1 << 20)
        if not raw:
            self._eof = True
            return
        if self._d is None:
            self._buf += raw
            return
        self._buf += self._d.decompress(raw)
        if self._d.eof:
            # Give the surplus back; it belongs to the next local header.
            self._src.unread(self._d.unused_data)
            self._eof = True

    def read(self, n=-1):
        if n is None or n < 0:
            while not self._eof:
                self._pump()
            out, self._buf = self._buf, b""
            return out
        while len(self._buf) < n and not self._eof:
            self._pump()
        out = self._buf[:n]
        self._buf = self._buf[len(out):]
        return out

    def release(self):
        """Hand back bytes read past the end of a STORED member.

        _pump reads the source a megabyte at a time, so a stored member always
        over-reads into _buf. Those are raw source bytes belonging to the next
        local header; without returning them the walk desynchronises and the
        following member is never found.
        """
        if self._d is None and self._buf:
            self._src.unread(self._buf)
            self._buf = b""

    def finish(self):
        """Consume whatever remains of a DEFLATED member so unused_data is
        pushed back. A stored member has no end marker, so its length must be
        known by the caller -- draining one would swallow the rest of the
        archive, which is exactly how skipping node_feat lost the stream."""
        if self._d is None:
            return
        while not self._eof:
            self._buf = b""
            self._pump()
        self._buf = b""


def open_inner_sequential(path, npz_hint, member):
    """Walk the inner .npz forward and stop at `member`'s payload.

    Returns (stream, stack, shape, dtype) with the stream positioned at the
    first payload byte. Never seeks, so it works when the inner .npz -- and its
    members -- are deflated, which is what the real papers100M archive is.
    """
    stack = []
    if path.endswith(".zip"):
        outer = zipfile.ZipFile(path)
        stack.append(outer)
        inner_name = find_npz(outer, npz_hint)
        info = outer.getinfo(inner_name)
        log(f"outer zip -> {inner_name} "
            f"({'stored' if info.compress_type == 0 else 'deflated'}, "
            f"{info.file_size / 2**30:.1f} GiB), parsing sequentially")
        src = Pushback(outer.open(inner_name))
    else:
        fh = open(path, "rb")
        stack.append(fh)
        src = Pushback(fh)

    while True:
        hdr = src.read(30)
        if len(hdr) < 30:
            raise SystemExit(f"member '{member}' not found in inner npz")
        sig, ver, flags, comp, t, d, crc, csz, usz, nlen, elen = struct.unpack(
            "<IHHHHHIIIHH", hdr)
        if sig == 0x02014b50:      # central directory reached
            raise SystemExit(f"member '{member}' not found in inner npz")
        if sig != 0x04034b50:
            raise SystemExit(f"bad local header signature {sig:#x}")
        name = src.read(nlen).decode()
        if elen:
            src.read(elen)

        ms = MemberStream(src, comp)
        key = name[:-4] if name.endswith(".npy") else name
        if key == member:
            shape, dtype, _ = _npy_payload_len(ms)
            return ms, stack, shape, dtype

        log(f"  skipping member {name}")
        if name.endswith(".npy"):
            # Read the numpy header for the exact payload length, then consume
            # precisely that much. Never drain a stored member.
            _, _, payload = _npy_payload_len(ms)
            left = payload
            while left > 0:
                chunk = ms.read(min(1 << 22, left))
                if not chunk:
                    raise SystemExit(f"EOF while skipping {name}")
                left -= len(chunk)
            ms.finish()
            ms.release()
        elif csz:
            _skip(src, csz)
        else:
            raise SystemExit(f"cannot skip non-npy member {name} of unknown size")
        # If the writer used a data descriptor (general-purpose bit 3), the
        # payload is followed by crc/csize/usize -- 12 bytes, or 16 when the
        # optional 0x08074b50 signature is present. Both forms must be consumed
        # in full; putting back only the 4 peeked bytes leaves 8 stray bytes
        # that make the next local header unrecognisable.
        if flags & 0x08:  # only when the writer actually wrote a descriptor
            tail = src.read(4)
            if tail and struct.unpack("<I", tail)[0] == 0x08074b50:
                src.read(12)      # signature form: 4 + 12
            else:
                src.read(8)       # bare form: the 4 peeked + 8 more


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

    fh, stack, shape, dtype = open_inner_sequential(args.zip, args.npz,
                                                    args.member)
    out_dtype = np.dtype(args.dtype) if args.dtype else dtype
    rows = shape[0]
    if args.max_rows > 0:
        rows = min(rows, args.max_rows)
    cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    total_out = rows * cols * out_dtype.itemsize
    log(f"member={args.member} shape={shape} dtype={dtype} -> {out_dtype}"
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
        raw = _read_exact(fh, row_bytes * take)
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
