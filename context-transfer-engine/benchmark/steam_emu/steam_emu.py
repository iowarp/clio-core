"""steam_emu.py -- replay a Steam game install against a directory (Windows).

Reproduces the file-system calls Steam's content system makes when it
installs (or restores from a backup) a game, measured from a WinFsp trace of
a real Age of Empires IV restore (see README.md). It makes the same Win32
calls Steam does, with the same access masks, so the kernel (cache manager
read-ins, parent-directory opens for a rename) produces the same requests a
file system sees. Three phases, as Steam runs them:

  prealloc  one thread. For every file: open the name in downloading/ (it
            fails), and for each large file create it, set its size
            (EndOfFile, then Allocation), read its head and tail, close.
  staging   --threads threads. For every file: open-if, set basic info
            (no change), write it in 1 MiB chunks plus a short tail, query
            its information, read its head and tail, close.

The head/tail reads before each close are in Steam's trace (64 KiB, aligned,
in the closing thread) but are not Steam's own calls: Steam's handles have
no read access, and files written by cmd, PowerShell or Python on the same
volume get none. They look like a filter driver scanning each file Steam
closes, and they appear on the WinFsp volume only. --scan-reads cached (or
unbuffered) emulates them, which is what a WinFsp file system serves under
Steam. The default is off: on NTFS the emulator's extra read-only open is
itself scanned (about 7 ms a file), which Steam's own install on NTFS does
not pay (it preallocates 8,271 files in 3-4 s there).
They are issued through a second,
read-only handle opened just before the file's write handle closes: adding
read access to the write handle itself makes the kernel read pages in after
the open, which Steam's trace does not show. The write handle therefore
shares read access (Steam's shares nothing), and each file costs one more
open and close than Steam's.
  commit    one thread. For every file: three opens of the name under
            common/ (they fail), open the staged file with DELETE, rename it
            over into common/, close. Then remove downloading/.

Usage:
  python steam_emu.py ROOT [--manifest M.json.gz] [--scale S] [--files N]
                      [--threads T] [--phases prealloc,staging,commit]
                      [--verify] [--json OUT]

ROOT is the library directory (Steam's SteamLibrary). --scale multiplies
every file size (sizes keep at least one byte) so the metadata pattern runs
in full with less data; --files keeps the first N files. Prints wall time
per phase and per-call latency; --verify reads every installed file back.
"""
import argparse
import collections
import ctypes
import ctypes.wintypes as w
import gzip
import hashlib
import json
import os
import queue
import sys
import threading
import time

APP_ID = "1466860"
GAME_DIR = "Game"
CHUNK = 1 << 20

# Access masks Steam passes to CreateFile (from the trace).
ACC_PROBE = 0x00100080          # SYNCHRONIZE | FILE_READ_ATTRIBUTES
ACC_STAGE = 0x00120196          # write data/append/EA/attributes, READ_CONTROL
ACC_SCAN = 0x00120089           # GENERIC_READ: the scan-read handle
ACC_COMMIT_PROBES = (0x0113008A, 0x0013008A, 0x00130088)
ACC_RENAME_SRC = 0x00110080     # DELETE | SYNCHRONIZE | FILE_READ_ATTRIBUTES
SHARE_ALL = 0x7
SHARE_READ = 0x1                # lets the scan handle open (Steam shares none)
SCAN_READ = 64 << 10
OPEN_EXISTING, OPEN_ALWAYS = 3, 4
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_NO_BUFFERING = 0x20000000  # --scan-reads unbuffered
FILE_FLAG_RANDOM_ACCESS = 0x10000000  # no read-ahead on the scan handle
FileBasicInfo, FileRenameInfo, FileEndOfFileInfo, FileAllocationInfo = 0, 3, 6, 5
INVALID_HANDLE = w.HANDLE(-1).value

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = w.HANDLE
k32.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p,
                            w.DWORD, w.DWORD, w.HANDLE]
k32.CloseHandle.argtypes = [w.HANDLE]
k32.WriteFile.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD,
                          ctypes.POINTER(w.DWORD), ctypes.c_void_p]
k32.SetFileInformationByHandle.argtypes = [w.HANDLE, ctypes.c_int,
                                           ctypes.c_void_p, w.DWORD]
k32.GetFileInformationByHandle.argtypes = [w.HANDLE, ctypes.c_void_p]
k32.ReadFile.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD,
                         ctypes.POINTER(w.DWORD), ctypes.c_void_p]
k32.SetFilePointerEx.argtypes = [w.HANDLE, ctypes.c_longlong,
                                 ctypes.c_void_p, w.DWORD]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD, w.DWORD]
# Page-aligned buffer per thread for the unbuffered scan reads.
_scan_buf = threading.local()


class Stats:
  """Per-call latency samples, safe to record from several threads."""

  def __init__(self):
    self.lock = threading.Lock()
    self.samples = collections.defaultdict(list)

  def timed(self, name, fn, *args):
    """Run fn(*args), record its latency under name, return its result."""
    t0 = time.perf_counter()
    r = fn(*args)
    dt = time.perf_counter() - t0
    with self.lock:
      self.samples[name].append(dt)
    return r

  def table(self):
    """Rows of (call, count, total_s, mean_us, p99_us), costliest first."""
    rows = []
    for name, v in self.samples.items():
      v = sorted(v)
      rows.append((name, len(v), sum(v), sum(v) / len(v) * 1e6,
                   v[min(len(v) - 1, int(len(v) * 0.99))] * 1e6))
    return sorted(rows, key=lambda r: -r[2])


def open_file(path, access, disposition, share=SHARE_ALL,
              flags=FILE_ATTRIBUTE_NORMAL):
  """CreateFileW; returns a handle or None."""
  h = k32.CreateFileW(path, access, share, None, disposition, flags, None)
  return None if h in (None, INVALID_HANDLE) else h


def set_size(h, size):
  """SetEndOfFile through FILE_END_OF_FILE_INFO."""
  v = ctypes.c_longlong(size)
  return k32.SetFileInformationByHandle(h, FileEndOfFileInfo, ctypes.byref(v), 8)


def set_allocation(h, size):
  """Reserve the file's allocation (FILE_ALLOCATION_INFO), 4 KiB rounded."""
  v = ctypes.c_longlong((size + 4095) & ~4095)
  return k32.SetFileInformationByHandle(h, FileAllocationInfo, ctypes.byref(v), 8)


def set_basic_nochange(h):
  """FILE_BASIC_INFO of zeros: Steam's 'touch nothing' basic-info call."""
  buf = (ctypes.c_byte * 40)()
  return k32.SetFileInformationByHandle(h, FileBasicInfo, buf, 40)


def query_info(h):
  """GetFileInformationByHandle (Steam's QueryInformation)."""
  buf = (ctypes.c_byte * 52)()
  return k32.GetFileInformationByHandle(h, buf)


def rename_over(h, target):
  """Rename an open file to target, replacing it (FILE_RENAME_INFO)."""
  name = target.encode("utf-16-le")

  class Info(ctypes.Structure):
    _fields_ = [("replace", ctypes.c_ubyte), ("root", w.HANDLE),
                ("len", w.DWORD), ("name", ctypes.c_byte * (len(name) + 2))]
  info = Info(1, None, len(name))
  ctypes.memmove(info.name, name, len(name))
  return k32.SetFileInformationByHandle(h, FileRenameInfo, ctypes.byref(info),
                                        ctypes.sizeof(info))


def scan_reads(path, size, mode):
  """Read 64 KiB at the head and the last 64 KiB through a new read-only
  handle, at the offsets seen before each close of a file Steam wrote: 0
  and (size rounded up to 4 KiB) - 64 KiB. mode: cached or unbuffered."""
  flag = FILE_FLAG_NO_BUFFERING if mode == "unbuffered" else FILE_FLAG_RANDOM_ACCESS
  h = open_file(path, ACC_SCAN, OPEN_EXISTING, SHARE_ALL,
                FILE_ATTRIBUTE_NORMAL | flag)
  if h is None:
    raise OSError(ctypes.get_last_error(), "scan open " + path)
  buf = getattr(_scan_buf, "p", None)
  if buf is None:
    buf = _scan_buf.p = k32.VirtualAlloc(None, SCAN_READ, 0x3000, 0x04)
  got = w.DWORD(0)
  tail = max(0, ((size + 4095) & ~4095) - SCAN_READ)
  for off in sorted({0, tail}):
    k32.SetFilePointerEx(h, off, None, 0)
    k32.ReadFile(h, buf, SCAN_READ, ctypes.byref(got), None)
  k32.CloseHandle(h)


def file_bytes(index, size):
  """Deterministic contents of file `index`: a 32-byte tag, then a pattern."""
  head = hashlib.sha256(b"steam_emu %d" % index).digest()
  body = (head * (CHUNK // len(head) + 1))[:CHUNK]
  return head, body


def write_file(stats, h, index, size):
  """Write `size` bytes in 1 MiB chunks plus a short tail, as Steam does."""
  head, body = file_bytes(index, size)
  done = ctypes.c_ulong(0)
  off = 0
  while off < size:
    n = min(CHUNK, size - off)
    data = (head + body[len(head):])[:n] if off == 0 else body[:n]
    ok = stats.timed("WriteFile %s" % ("1MiB" if n == CHUNK else "tail"),
                     k32.WriteFile, h, data, n, ctypes.byref(done), None)
    if not ok:
      raise OSError(ctypes.get_last_error(), "WriteFile failed at %d" % off)
    off += n


class Library:
  """Paths of one app install under a Steam library root."""

  def __init__(self, root):
    self.root = os.path.abspath(root)
    self.dl = os.path.join(self.root, "steamapps", "downloading", APP_ID)
    self.common = os.path.join(self.root, "steamapps", "common", GAME_DIR)

  def staged(self, rel):
    return os.path.join(self.dl, rel)

  def installed(self, rel):
    return os.path.join(self.common, rel)


def make_dirs(paths):
  """Create the parent directory of every path (once each)."""
  for d in sorted({os.path.dirname(p) for p in paths}):
    os.makedirs(d, exist_ok=True)


def phase_prealloc(lib, files, stats, scan):
  """Probe every staged name; create and size the large files."""
  make_dirs([lib.staged(f[0]) for f in files])
  for rel, size, prealloc in files:
    path = lib.staged(rel)
    h = stats.timed("CreateFile probe (missing)", open_file, path, ACC_PROBE,
                    OPEN_EXISTING)
    if h is not None:
      k32.CloseHandle(h)
    if not prealloc:
      continue
    h = stats.timed("CreateFile open-if (new)", open_file, path, ACC_STAGE,
                    OPEN_ALWAYS, SHARE_READ)
    if h is None:
      raise OSError(ctypes.get_last_error(), "create " + path)
    stats.timed("SetEndOfFile", set_size, h, size)
    stats.timed("SetAllocation", set_allocation, h, size)
    if scan != "off":
      stats.timed("scan reads", scan_reads, path, size, scan)
    stats.timed("CloseHandle", k32.CloseHandle, h)


def stage_one(lib, index, rel, size, stats, scan):
  """Open-if, no-op basic info, write the data, query, close."""
  path = lib.staged(rel)
  h = stats.timed("CreateFile open-if (stage)", open_file, path, ACC_STAGE,
                  OPEN_ALWAYS, SHARE_READ)
  if h is None:
    raise OSError(ctypes.get_last_error(), "open " + path)
  try:
    stats.timed("SetBasicInfo", set_basic_nochange, h)
    write_file(stats, h, index, size)
    stats.timed("GetFileInformation", query_info, h)
    if scan != "off":
      stats.timed("scan reads", scan_reads, path, size, scan)
  finally:
    stats.timed("CloseHandle", k32.CloseHandle, h)


def phase_staging(lib, files, stats, threads, scan):
  """Stage every file from `threads` threads (Steam keeps ~9 in flight)."""
  make_dirs([lib.staged(f[0]) for f in files])
  work = queue.Queue()
  for i, f in enumerate(files):
    work.put((i, f))
  errors = []

  def worker():
    while True:
      try:
        i, (rel, size, _) = work.get_nowait()
      except queue.Empty:
        return
      try:
        stage_one(lib, i, rel, size, stats, scan)
      except OSError as e:
        errors.append(e)
        return
  ts = [threading.Thread(target=worker) for _ in range(threads)]
  for t in ts: t.start()
  for t in ts: t.join()
  if errors:
    raise errors[0]


def phase_commit(lib, files, stats):
  """Probe each install name three times, then rename the staged file in."""
  make_dirs([lib.installed(f[0]) for f in files])
  for rel, _, _ in files:
    dst = lib.installed(rel)
    for acc in ACC_COMMIT_PROBES:
      h = stats.timed("CreateFile probe (missing)", open_file, dst, acc,
                      OPEN_EXISTING)
      if h is not None:
        k32.CloseHandle(h)
    h = stats.timed("CreateFile open (rename src)", open_file,
                    lib.staged(rel), ACC_RENAME_SRC, OPEN_EXISTING)
    if h is None:
      raise OSError(ctypes.get_last_error(), "open " + lib.staged(rel))
    ok = stats.timed("Rename", rename_over, h, dst)
    stats.timed("CloseHandle", k32.CloseHandle, h)
    if not ok:
      raise OSError(ctypes.get_last_error(), "rename to " + dst)
  t0 = time.perf_counter()
  for d, subdirs, names in os.walk(lib.dl, topdown=False):
    os.rmdir(d)
  stats.samples["remove downloading/"].append(time.perf_counter() - t0)


def verify(lib, files):
  """Read every installed file back; return a list of problems."""
  bad = []
  for i, (rel, size, _) in enumerate(files):
    path = lib.installed(rel)
    try:
      got = os.path.getsize(path)
      if got != size:
        bad.append("%s: size %d, expected %d" % (rel, got, size))
        continue
      head, body = file_bytes(i, size)
      with open(path, "rb") as f:
        first = f.read(min(size, CHUNK))
      want = (head + body[len(head):])[:len(first)]
      if first != want:
        bad.append("%s: first chunk differs" % rel)
    except OSError as e:
      bad.append("%s: %s" % (rel, e))
  return bad


def load_manifest(path, scale, limit):
  """[(relpath, size, preallocated)], sizes scaled (at least one byte)."""
  opener = gzip.open if path.endswith(".gz") else open
  with opener(path, "rt") as f:
    m = json.load(f)["manifest"]
  if limit:
    m = m[:limit]
  return [(rel.replace("/", os.sep), max(1, int(size * scale)), bool(pre))
          for rel, size, pre in m]


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("root")
  ap.add_argument("--manifest", default=os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "aoe4_manifest.json.gz"))
  ap.add_argument("--scale", type=float, default=1.0)
  ap.add_argument("--files", type=int, default=0)
  ap.add_argument("--threads", type=int, default=9)
  ap.add_argument("--phases", default="prealloc,staging,commit")
  ap.add_argument("--verify", action="store_true")
  ap.add_argument("--scan-reads", dest="scan", default="off",
                  choices=["off", "cached", "unbuffered"])
  ap.add_argument("--json")
  a = ap.parse_args()
  files = load_manifest(a.manifest, a.scale, a.files)
  lib = Library(a.root)
  print("%d files, %.2f GB, %d preallocated, root %s" % (
      len(files), sum(f[1] for f in files) / 1e9,
      sum(1 for f in files if f[2]), lib.root))
  report = {"files": len(files), "bytes": sum(f[1] for f in files), "phases": {}}
  for ph in a.phases.split(","):
    stats = Stats()
    t0 = time.perf_counter()
    if ph == "prealloc": phase_prealloc(lib, files, stats, a.scan)
    elif ph == "staging": phase_staging(lib, files, stats, a.threads, a.scan)
    elif ph == "commit": phase_commit(lib, files, stats)
    else: sys.exit("unknown phase " + ph)
    secs = time.perf_counter() - t0
    print("\n== %s: %.2f s (ended at %.3f unix s)" % (ph, secs, time.time()))
    print("  %-30s %8s %9s %9s %9s" % ("call", "count", "total_s", "mean_us", "p99_us"))
    rows = stats.table()
    for r in rows:
      print("  %-30s %8d %9.2f %9.0f %9.0f" % r)
    report["phases"][ph] = {"seconds": secs, "calls": [list(r) for r in rows]}
  if a.verify:
    bad = verify(lib, files)
    print("\nverify: %d of %d files bad" % (len(bad), len(files)))
    for b in bad[:10]: print("  " + b)
    report["verify_bad"] = len(bad)
  if a.json:
    with open(a.json, "w") as f: json.dump(report, f, indent=1)


if __name__ == "__main__":
  main()
