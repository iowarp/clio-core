#!/usr/bin/env python3
"""Per-node POSIX operation agent for the clio-fs scale/reliability suite.

The orchestrator (suite.py) starts one agent per node over ssh and drives it
with newline-delimited JSON requests on stdin; each request gets exactly one
JSON reply on stdout.  Keeping a live agent per node (instead of one ssh per
operation) lets a test hold a file descriptor open on node A while node B
unlinks / renames / rewrites the file, which is where distributed filesystem
bugs live.

Every operation runs on a worker thread with a deadline.  A filesystem call
that does not return inside the deadline is reported as a HANG rather than
blocking the agent: that is the failure mode that matters most for a FUSE
filesystem (a stuck request parks the caller in uninterruptible sleep, and
on a batch cluster an unkillable process drains the node).  The agent can
abort the FUSE connection behind a mount (``abort_mount``) so the stuck
callers return ENOTCONN and the node stays healthy.

Request:  {"id": 1, "op": "write_file", "args": {...}, "timeout": 60}
Reply:    {"id": 1, "ok": true, "ret": ..., "ms": 1.2}
          {"id": 1, "ok": false, "errno": 2, "err": "ENOENT ...", "ms": ..}
          {"id": 1, "ok": false, "hang": true, "ms": ...}
"""

import ctypes
import errno as errno_mod
import fcntl
import hashlib
import json
import os
import random
import shutil
import stat as stat_mod
import subprocess
import sys
import threading
import time

import stress_records as sr

BLOCK = 65536  # Granularity of the deterministic data pattern.

_libc = ctypes.CDLL(None, use_errno=True)
_SYS_RENAMEAT2 = 316  # x86_64
_AT_FDCWD = -100
RENAME_NOREPLACE = 1
RENAME_EXCHANGE = 2
FALLOC_FL_KEEP_SIZE = 1
FALLOC_FL_PUNCH_HOLE = 2


# ---------------------------------------------------------------------------
# Deterministic data pattern.  Content of a file written with seed S is the
# concatenation of pattern blocks P(S, 0), P(S, 1), ...  Any node can then
# regenerate and verify any byte range without the data crossing the wire.
# ---------------------------------------------------------------------------
_pat_cache = {}


def pattern_block(seed, idx):
  """Return the idx-th BLOCK-sized pattern block for the given seed."""
  key = (seed, idx)
  blk = _pat_cache.get(key)
  if blk is None:
    blk = random.Random(f'{seed}:{idx}').randbytes(BLOCK)
    if len(_pat_cache) > 4096:
      _pat_cache.clear()
    _pat_cache[key] = blk
  return blk


def pattern(seed, off, length):
  """Return pattern bytes [off, off+length) for seed."""
  out = bytearray()
  pos = off
  end = off + length
  while pos < end:
    idx = pos // BLOCK
    boff = pos % BLOCK
    take = min(BLOCK - boff, end - pos)
    out += pattern_block(seed, idx)[boff:boff + take]
    pos += take
  return bytes(out)


def _err(e):
  """Convert an OSError into a reply fragment."""
  code = getattr(e, 'errno', None)
  name = errno_mod.errorcode.get(code, str(code)) if code else type(e).__name__
  return {'ok': False, 'errno': code, 'err': f'{name}: {e}'}


def _st(s):
  """Serialize an os.stat_result."""
  return {'mode': s.st_mode, 'ino': s.st_ino, 'nlink': s.st_nlink,
          'uid': s.st_uid, 'gid': s.st_gid, 'size': s.st_size,
          'blocks': s.st_blocks, 'atime_ns': s.st_atime_ns,
          'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
          'isdir': stat_mod.S_ISDIR(s.st_mode),
          'isreg': stat_mod.S_ISREG(s.st_mode),
          'islnk': stat_mod.S_ISLNK(s.st_mode),
          'perm': stat_mod.S_IMODE(s.st_mode)}


class Agent:
  """Executes filesystem operations; holds open descriptors by handle."""

  def __init__(self):
    self.fds = {}
    self.next_h = 1

  # -- descriptor management --------------------------------------------
  def op_open(self, path, flags='r', mode=0o644):
    """Open path and return a handle; flags is a string of r/w/c/x/t/a/d."""
    fl = 0
    if 'w' in flags and 'r' in flags:
      fl |= os.O_RDWR
    elif 'w' in flags:
      fl |= os.O_WRONLY
    else:
      fl |= os.O_RDONLY
    if 'c' in flags:
      fl |= os.O_CREAT
    if 'x' in flags:
      fl |= os.O_EXCL
    if 't' in flags:
      fl |= os.O_TRUNC
    if 'a' in flags:
      fl |= os.O_APPEND
    if 'd' in flags:
      fl |= os.O_DIRECTORY
    if 's' in flags:
      fl |= os.O_SYNC
    fd = os.open(path, fl, mode)
    h = self.next_h
    self.next_h += 1
    self.fds[h] = fd
    return h

  def op_close(self, h):
    os.close(self.fds.pop(h))
    return True

  def op_fpwrite(self, h, off, length, seed):
    """pwrite pattern(seed, off, length) at off on handle h."""
    data = pattern(seed, off, length)
    return os.pwrite(self.fds[h], data, off)

  def op_fwrite(self, h, data_hex=None, length=0, seed=0, off=0):
    """write() (not pwrite) either literal hex data or pattern bytes."""
    data = bytes.fromhex(data_hex) if data_hex is not None else \
        pattern(seed, off, length)
    return os.write(self.fds[h], data)

  def op_fpread_verify(self, h, off, length, seed):
    """pread [off,off+len) and compare with the pattern; return mismatch."""
    got = os.pread(self.fds[h], length, off)
    return _compare(got, pattern(seed, off, length), off)

  def op_fpread_hex(self, h, off, length):
    return os.pread(self.fds[h], length, off).hex()

  def op_fstat(self, h):
    return _st(os.fstat(self.fds[h]))

  def op_fsync(self, h):
    os.fsync(self.fds[h])
    return True

  def op_fdatasync(self, h):
    os.fdatasync(self.fds[h])
    return True

  def op_ftruncate(self, h, size):
    os.ftruncate(self.fds[h], size)
    return True

  def op_lseek(self, h, off, whence):
    return os.lseek(self.fds[h], off, {'set': 0, 'cur': 1, 'end': 2,
                                       'data': 3, 'hole': 4}[whence])

  def op_flock(self, h, kind):
    """kind: ex, sh, un, ex_nb, sh_nb."""
    m = {'ex': fcntl.LOCK_EX, 'sh': fcntl.LOCK_SH, 'un': fcntl.LOCK_UN,
         'ex_nb': fcntl.LOCK_EX | fcntl.LOCK_NB,
         'sh_nb': fcntl.LOCK_SH | fcntl.LOCK_NB}[kind]
    fcntl.flock(self.fds[h], m)
    return True

  def op_lockf(self, h, kind, off=0, length=0):
    m = {'ex': fcntl.LOCK_EX, 'un': fcntl.LOCK_UN,
         'ex_nb': fcntl.LOCK_EX | fcntl.LOCK_NB}[kind]
    fcntl.lockf(self.fds[h], m, length, off)
    return True

  def op_fallocate(self, h, off, length, flags=0):
    fd = self.fds[h]
    if flags == 0:
      os.posix_fallocate(fd, off, length)
      return True
    r = _libc.fallocate(ctypes.c_int(fd), ctypes.c_int(flags),
                        ctypes.c_long(off), ctypes.c_long(length))
    if r != 0:
      e = ctypes.get_errno()
      raise OSError(e, os.strerror(e))
    return True

  def op_mmap_rw(self, h, off, length, seed):
    """Write pattern through a shared mapping, msync, then read back."""
    import mmap
    fd = self.fds[h]
    mm = mmap.mmap(fd, off + length, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    data = pattern(seed, off, length)
    mm[off:off + length] = data
    mm.flush()
    back = bytes(mm[off:off + length])
    mm.close()
    return _compare(back, data, off)

  def op_close_all(self):
    n = 0
    for h in list(self.fds):
      try:
        os.close(self.fds.pop(h))
        n += 1
      except OSError:
        pass
    return n

  # -- whole-file helpers ------------------------------------------------
  def op_write_file(self, path, size, seed, chunk=1 << 20, fsync=False,
                    flags='wct', mode=0o644):
    """Create/overwrite path with pattern(seed); return sha256 of content."""
    fl = os.O_WRONLY
    if 'c' in flags:
      fl |= os.O_CREAT
    if 't' in flags:
      fl |= os.O_TRUNC
    if 'x' in flags:
      fl |= os.O_EXCL
    fd = os.open(path, fl, mode)
    h = hashlib.sha256()
    try:
      off = 0
      while off < size:
        n = min(chunk, size - off)
        d = pattern(seed, off, n)
        w = os.write(fd, d)
        if w != n:
          raise OSError(errno_mod.EIO, f'short write {w}/{n} at {off}')
        h.update(d)
        off += n
      if fsync:
        os.fsync(fd)
    finally:
      os.close(fd)
    return h.hexdigest()

  def op_append_records(self, path, writer, count, reclen=64, start=0,
                        close=True, max_secs=0.0, fsync_every=0):
    """O_APPEND `count` self-identifying records (W<writer>R<seq>, padded to
    reclen bytes incl. the newline) with one write(2) each. Stops early after
    max_secs (0 = never). Returns the acknowledged seqs as [first, last]
    runs, the failed writes, and the close() result."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    acked, fails, t0 = [], [], time.time()
    synced = []  # [last seq covered, wall time] per successful fsync
    fsync_fails = []  # [last seq covered, error] per failed fsync
    try:
      for k in range(start, start + count):
        if max_secs and time.time() - t0 > max_secs:
          break
        rec = f'W{writer:03d}R{k:07d}'.ljust(reclen - 1, '.') + '\n'
        try:
          if os.write(fd, rec.encode()) == reclen:
            if acked and acked[-1][1] == k - 1:
              acked[-1][1] = k
            else:
              acked.append([k, k])
          else:
            fails.append([k, 'short'])
        except OSError as e:
          fails.append([k, _err(e)['err']])
          if len(fails) > 200:
            break
        if fsync_every and (k + 1) % fsync_every == 0:
          try:
            os.fsync(fd)
            synced.append([k, time.time()])
          except OSError as e:
            fsync_fails.append([k, _err(e)['err']])
    finally:
      close_err = None
      if close:
        try:
          os.close(fd)
        except OSError as e:
          close_err = _err(e)['err']
    return {'acked': acked, 'fails': fails[:20], 'nfail': len(fails),
            'close_err': close_err, 'synced': synced,
            'fsync_fails': fsync_fails[:50]}

  def op_scan_records(self, path, reclen=64):
    """Parse a file of append_records records. Returns the byte size, each
    writer's seqs in file order, and the offsets of unparseable records."""
    with open(path, 'rb') as f:
      data = f.read()
    by_writer, bad, bad_sample, nonzero_bad = {}, [], '', 0
    for off in range(0, len(data) - len(data) % reclen, reclen):
      rec = data[off:off + reclen]
      try:
        txt = rec.decode()
        w, k = int(txt[1:4]), int(txt[5:12])
        ok = (txt[0] == 'W' and txt[4] == 'R' and txt.endswith('\n') and
              txt == f'W{w:03d}R{k:07d}'.ljust(reclen - 1, '.') + '\n')
      except (UnicodeDecodeError, ValueError):
        ok = False
      if not ok:
        if rec != bytes(reclen):
          nonzero_bad += 1
        if not bad:
          bad_sample = rec.hex()
        if len(bad) < 20:
          bad.append(off)
        continue
      by_writer.setdefault(str(w), []).append(k)
    zero_runs, start = [], None
    for off in range(0, len(data) - len(data) % reclen, reclen):
      z = data[off:off + reclen] == bytes(reclen)
      if z and start is None:
        start = off
      if not z and start is not None:
        zero_runs.append([start, off])
        start = None
    if start is not None:
      zero_runs.append([start, len(data)])
    return {'size': len(data), 'by_writer': by_writer, 'bad': bad,
            'tail': len(data) % reclen, 'bad_sample': bad_sample,
            'zero_runs': zero_runs[:10], 'nonzero_bad': nonzero_bad}

  def op_verify_file(self, path, size, seed, chunk=1 << 20):
    """Read path; check size and content against pattern(seed)."""
    st = os.stat(path)
    res = {'size': st.st_size, 'size_ok': st.st_size == size,
           'mismatch': None}
    fd = os.open(path, os.O_RDONLY)
    try:
      off = 0
      while off < size:
        n = min(chunk, size - off)
        got = os.pread(fd, n, off)
        mm = _compare(got, pattern(seed, off, n), off)
        if mm is not None:
          res['mismatch'] = mm
          break
        off += n
      tail = os.pread(fd, 16, size)
      if tail:
        res['extra_bytes'] = len(tail)
    finally:
      os.close(fd)
    res['ok'] = res['size_ok'] and res['mismatch'] is None and \
        'extra_bytes' not in res
    return res

  # -- self-verifying records (stress_records.py) ------------------------
  def op_rec_write(self, path, name, runs, writer, gen, fsync=False,
                   blk=sr.BLK):
    """Write records of (writer, gen) over block runs of file `name`."""
    return sr.write_runs(path, sr.file_id_of(name), runs, writer, gen,
                         fsync, blk)

  def op_rec_scan(self, path, name, nblocks, blk=sr.BLK):
    """Classify every block of a record file (see stress_records.scan)."""
    return sr.scan(path, sr.file_id_of(name), nblocks, blk)

  def op_rec_shared_stress(self, path, name, nblocks, writer_base, writers,
                           readers, secs, seed, blk=sr.BLK):
    """Concurrent overwriters + readers on one shared record file."""
    return sr.SharedFileStress(path, sr.file_id_of(name), nblocks,
                               writer_base, writers, readers, secs, seed,
                               blk).run()

  def op_rec_fileset(self, dirpath, writer, nfiles, blocks, secs, seed,
                     log_path=None, blk=sr.BLK):
    """Rewrite a cycling set of record files, logging each fsynced one."""
    return sr.FileSetWriter(dirpath, writer, nfiles, blocks, secs, seed,
                            blk, log_path).run()

  def op_sha256(self, path, chunk=1 << 20):
    h = hashlib.sha256()
    n = 0
    with open(path, 'rb') as f:
      while True:
        b = f.read(chunk)
        if not b:
          break
        h.update(b)
        n += len(b)
    return {'sha256': h.hexdigest(), 'size': n}

  def op_read_hex(self, path, off=0, length=-1):
    with open(path, 'rb') as f:
      f.seek(off)
      return f.read(length).hex()

  def op_write_hex(self, path, data_hex, flags='wct'):
    fl = os.O_WRONLY | (os.O_CREAT if 'c' in flags else 0) | \
        (os.O_TRUNC if 't' in flags else 0) | \
        (os.O_APPEND if 'a' in flags else 0)
    fd = os.open(path, fl, 0o644)
    try:
      return os.write(fd, bytes.fromhex(data_hex))
    finally:
      os.close(fd)

  # -- namespace -----------------------------------------------------------
  def op_stat(self, path):
    return _st(os.stat(path))

  def op_lstat(self, path):
    return _st(os.lstat(path))

  def op_exists(self, path):
    return os.path.lexists(path)

  def op_listdir(self, path):
    return sorted(os.listdir(path))

  def op_scandir_count(self, path):
    return sum(1 for _ in os.scandir(path))

  def op_mkdir(self, path, mode=0o755):
    os.mkdir(path, mode)
    return True

  def op_makedirs(self, path):
    os.makedirs(path, exist_ok=True)
    return True

  def op_rmdir(self, path):
    os.rmdir(path)
    return True

  def op_unlink(self, path):
    os.unlink(path)
    return True

  def op_rmtree(self, path):
    shutil.rmtree(path)
    return True

  def op_rename(self, src, dst, flags=0):
    if flags == 0:
      os.rename(src, dst)
      return True
    r = _libc.syscall(_SYS_RENAMEAT2, _AT_FDCWD, src.encode(), _AT_FDCWD,
                      dst.encode(), ctypes.c_uint(flags))
    if r != 0:
      e = ctypes.get_errno()
      raise OSError(e, os.strerror(e))
    return True

  def op_link(self, src, dst):
    os.link(src, dst)
    return True

  def op_symlink(self, target, path):
    os.symlink(target, path)
    return True

  def op_readlink(self, path):
    return os.readlink(path)

  def op_chmod(self, path, mode):
    os.chmod(path, mode)
    return True

  def op_chown_self(self, path):
    os.chown(path, os.getuid(), os.getgid())
    return True

  def op_utime(self, path, atime_ns, mtime_ns):
    os.utime(path, ns=(atime_ns, mtime_ns))
    return True

  def op_truncate(self, path, size):
    os.truncate(path, size)
    return True

  def op_access(self, path, mode):
    return os.access(path, mode)

  def op_setxattr(self, path, name, value_hex, flags=0):
    os.setxattr(path, name, bytes.fromhex(value_hex), flags)
    return True

  def op_getxattr(self, path, name):
    return os.getxattr(path, name).hex()

  def op_listxattr(self, path):
    return sorted(os.listxattr(path))

  def op_removexattr(self, path, name):
    os.removexattr(path, name)
    return True

  def op_statvfs(self, path):
    s = os.statvfs(path)
    return {'bsize': s.f_bsize, 'blocks': s.f_blocks, 'bfree': s.f_bfree,
            'bavail': s.f_bavail, 'files': s.f_files, 'namemax': s.f_namemax}

  # -- bulk metadata -------------------------------------------------------
  def op_create_many(self, dirpath, prefix, count, size=0, seed=0):
    """Create count files (optionally with size bytes); return failures."""
    fails = []
    for i in range(count):
      p = os.path.join(dirpath, f'{prefix}{i}')
      try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        if size:
          os.write(fd, pattern(seed + i, 0, size))
        os.close(fd)
      except OSError as e:
        fails.append([p, _err(e)['err']])
        if len(fails) > 20:
          break
    return {'created': count - len(fails), 'fails': fails}

  def op_stat_many(self, dirpath, prefix, count, size=None):
    bad = []
    for i in range(count):
      p = os.path.join(dirpath, f'{prefix}{i}')
      try:
        s = os.stat(p)
        if size is not None and s.st_size != size:
          bad.append([p, f'size {s.st_size} != {size}'])
      except OSError as e:
        bad.append([p, _err(e)['err']])
      if len(bad) > 20:
        break
    return {'bad': bad}

  def op_verify_many(self, dirpath, prefix, count, size, seed=0):
    bad = []
    for i in range(count):
      p = os.path.join(dirpath, f'{prefix}{i}')
      try:
        with open(p, 'rb') as f:
          got = f.read()
        if got != pattern(seed + i, 0, size):
          bad.append([p, f'content mismatch len={len(got)}'])
      except OSError as e:
        bad.append([p, _err(e)['err']])
      if len(bad) > 20:
        break
    return {'bad': bad}

  def op_unlink_many(self, dirpath, prefix, count):
    fails = []
    for i in range(count):
      p = os.path.join(dirpath, f'{prefix}{i}')
      try:
        os.unlink(p)
      except OSError as e:
        fails.append([p, _err(e)['err']])
        if len(fails) > 20:
          break
    return {'fails': fails}

  def op_meta_storm(self, dirpath, prefix, count, procs=8, mode='create'):
    """Run a metadata storm from `procs` processes in parallel on one
    directory: process p handles names f'{prefix}{p}_{i}' for i < count.
    mode: 'create' (O_CREAT|O_EXCL), 'stat' or 'unlink'.  Returns the op
    count, the wall seconds and the first failures."""
    script = r'''
import os, sys, json, time
d, prefix, p, count, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
fails = []
t0 = time.time()
for i in range(count):
  path = os.path.join(d, f'{prefix}{p}_{i}')
  try:
    if mode == 'create':
      os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
    elif mode == 'stat':
      os.stat(path)
    else:
      os.unlink(path)
  except OSError as e:
    fails.append([path, os.strerror(e.errno)])
    if len(fails) > 10: break
print(json.dumps({'fails': fails, 'secs': time.time() - t0}))
'''
    t0 = time.time()
    ps = [subprocess.Popen([sys.executable, '-c', script, dirpath, prefix,
                            str(p), str(count), mode],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          for p in range(procs)]
    fails = []
    for pr in ps:
      out, err = pr.communicate()
      try:
        fails += json.loads(out.decode())['fails']
      except Exception:  # noqa: BLE001 -- report the child's crash
        fails.append(['<proc>', (err or out).decode()[-300:]])
    return {'ops': procs * count, 'secs': time.time() - t0,
            'fails': fails[:20], 'nfails': len(fails)}

  def op_rename_storm(self, root, dirs, iters, seed):
    """Rename random directories of `dirs` (paths relative to root) into one
    another for `iters` rounds; the set of names each directory has is
    tracked so later rounds keep hitting live directories.  Returns errno
    counts; the caller checks the tree afterwards."""
    rng = random.Random(seed)
    names = list(dirs)
    counts = {}
    for _ in range(iters):
      a, b = rng.choice(names), rng.choice(names)
      src = os.path.join(root, a)
      leaf = f'm{seed}_{rng.randrange(1 << 30)}'
      dst = os.path.join(root, b, leaf)
      try:
        os.rename(src, dst)
        counts['ok'] = counts.get('ok', 0) + 1
        # Anything under a moved: its relative names change.
        rel = os.path.relpath(dst, root)
        names = [rel + x[len(a):] if x == a or x.startswith(a + '/') else x
                 for x in names]
      except OSError as e:
        k = errno_mod.errorcode.get(e.errno, str(e.errno))
        counts[k] = counts.get(k, 0) + 1
        if e.errno == errno_mod.ENOENT:
          # Another node moved it: learn the tree as it is now.
          names = [os.path.relpath(os.path.join(dp, d), root)
                   for dp, dns, _ in os.walk(root) for d in dns] or names
    return counts

  def op_race_dirs(self, paths, mode, seed):
    """One side of an rmdir-vs-create race: visit `paths` in a seeded random
    order and either rmdir each (mode 'rmdir') or create a file inside each
    (mode 'create').  Returns {path: errno name or 'ok'}."""
    order = list(paths)
    random.Random(seed).shuffle(order)
    out = {}
    for d in order:
      try:
        if mode == 'rmdir':
          os.rmdir(d)
        else:
          os.close(os.open(os.path.join(d, 'x'), os.O_CREAT | os.O_WRONLY,
                           0o644))
        out[d] = 'ok'
      except OSError as e:
        out[d] = errno_mod.errorcode.get(e.errno, str(e.errno))
    return out

  def op_readdir_check(self, path, rounds=1):
    """List `path` `rounds` times; report the entry counts and any names
    listed twice within one listing."""
    sizes, dups = [], []
    for _ in range(rounds):
      names = os.listdir(path)
      sizes.append(len(names))
      if len(set(names)) != len(names):
        seen = set()
        dups += [x for x in names if x in seen or seen.add(x)][:5]
    return {'sizes': sizes, 'dups': dups[:20]}

  def op_tree_manifest(self, root):
    """Walk root; return {relpath: [type, size, perm, sha256|target]}."""
    out = {}
    for dp, dns, fns in os.walk(root):
      for d in dns:
        p = os.path.join(dp, d)
        s = os.lstat(p)
        if stat_mod.S_ISLNK(s.st_mode):
          out[os.path.relpath(p, root)] = ['l', 0, 0, os.readlink(p)]
        else:
          out[os.path.relpath(p, root)] = ['d', 0, stat_mod.S_IMODE(
              s.st_mode), '']
      for f in fns:
        p = os.path.join(dp, f)
        s = os.lstat(p)
        rel = os.path.relpath(p, root)
        if stat_mod.S_ISLNK(s.st_mode):
          out[rel] = ['l', 0, 0, os.readlink(p)]
        else:
          h = hashlib.sha256()
          with open(p, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 20), b''):
              h.update(b)
          out[rel] = ['f', s.st_size, stat_mod.S_IMODE(s.st_mode),
                      h.hexdigest()]
    return out

  # -- shell ---------------------------------------------------------------
  def op_sh(self, cmd, cwd=None, env=None, timeout=600):
    """Run a shell command; return rc and the tails of stdout/stderr."""
    e = dict(os.environ)
    if env:
      e.update(env)
    p = subprocess.run(['bash', '-c', cmd], cwd=cwd, env=e,
                       capture_output=True, timeout=timeout)
    return {'rc': p.returncode,
            'out': p.stdout.decode(errors='replace')[-6000:],
            'err': p.stderr.decode(errors='replace')[-6000:]}

  def op_abort_mount(self, mnt):
    """Abort the FUSE connection behind mnt, then lazily unmount it.

    Reads /proc/self/mountinfo (never stat()s the possibly-hung mount) to
    find the device minor, which is the fusectl connection number.
    """
    aborted = False
    with open('/proc/self/mountinfo') as f:
      for line in f:
        parts = line.split()
        if parts[4] == mnt and 'fuse' in line:
          minor = parts[2].split(':')[1]
          ap = f'/sys/fs/fuse/connections/{minor}/abort'
          try:
            with open(ap, 'w') as af:
              af.write('1')
            aborted = True
          except OSError:
            pass
    subprocess.run(['fusermount3', '-u', '-z', mnt], capture_output=True)
    return aborted

  def op_is_mounted(self, mnt):
    with open('/proc/self/mountinfo') as f:
      return any(line.split()[4] == mnt for line in f)

  def op_hostname(self):
    return os.uname().nodename

  def op_ping(self):
    return True


def _compare(got, want, base_off):
  """Return None when equal, else a description of the first difference."""
  if got == want:
    return None
  n = min(len(got), len(want))
  first = next((i for i in range(n) if got[i] != want[i]), n)
  zero_run = got[first:first + 64] == bytes(len(got[first:first + 64]))
  return {'offset': base_off + first, 'got_len': len(got),
          'want_len': len(want), 'got_is_zero': zero_run}


def main():
  agent = Agent()
  out_lock = threading.Lock()

  def reply(obj):
    with out_lock:
      sys.stdout.write(json.dumps(obj) + '\n')
      sys.stdout.flush()

  reply({'id': 0, 'ok': True, 'ret': {'host': os.uname().nodename,
                                      'pid': os.getpid()}})
  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    req = json.loads(line)
    rid = req.get('id')
    fn = getattr(agent, 'op_' + req['op'], None)
    if fn is None:
      reply({'id': rid, 'ok': False, 'err': f'unknown op {req["op"]}'})
      continue
    if req['op'] == 'exit':
      break
    box = {}

    def run(fn=fn, args=req.get('args', {}), box=box):
      t0 = time.monotonic()
      try:
        box['r'] = {'ok': True, 'ret': fn(**args)}
      except OSError as e:
        box['r'] = _err(e)
      except subprocess.TimeoutExpired as e:
        box['r'] = {'ok': False, 'hang': True, 'err': f'sh timeout {e}'}
      except Exception as e:  # pylint: disable=broad-except
        box['r'] = {'ok': False, 'err': f'{type(e).__name__}: {e}'}
      box['ms'] = (time.monotonic() - t0) * 1000.0

    th = threading.Thread(target=run, daemon=True)
    t0 = time.monotonic()
    th.start()
    th.join(req.get('timeout', 120))
    if th.is_alive():
      reply({'id': rid, 'ok': False, 'hang': True,
             'err': f'op {req["op"]} exceeded {req.get("timeout", 120)}s',
             'ms': (time.monotonic() - t0) * 1000.0})
      continue
    r = box['r']
    r['id'] = rid
    r['ms'] = box.get('ms', 0.0)
    reply(r)


if __name__ == '__main__':
  main()
