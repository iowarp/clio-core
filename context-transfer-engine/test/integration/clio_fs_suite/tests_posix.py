"""Single-node POSIX semantics through one FUSE mount.

These are the "general-purpose filesystem" checks: every one compares the
mount's behaviour with what POSIX / Linux (ext4, xfs) returns for the same
sequence.  They run on node 0 only (max_nodes=1).
"""

import errno
import os
import stat
import time

from suite import test, TestUnsupported

MiB = 1 << 20
KiB = 1 << 10


def _rt(ctx, path, size, seed, chunk=MiB):
  ctx.ok(0, 'write_file', path=path, size=size, seed=seed, chunk=chunk)
  v = ctx.ok(0, 'verify_file', path=path, size=size, seed=seed)
  ctx.check(v['ok'], f'{path} size={size}: {v}')


@test('rw_sizes', 'posix', max_nodes=1)
def t_rw_sizes(ctx):
  """Write/read-back around page (4K) and blob (1M) boundaries."""
  sizes = [0, 1, 511, 4095, 4096, 4097, 65535, 65536, MiB - 1, MiB, MiB + 1,
           3 * MiB + 17, 17 * MiB]
  for i, s in enumerate(sizes):
    _rt(ctx, ctx.p(f'f{s}'), s, 100 + i)


@test('rw_odd_chunks', 'posix', max_nodes=1)
def t_rw_odd_chunks(ctx):
  """Sequential writes in unaligned chunk sizes."""
  for i, ch in enumerate([1, 7, 1000, 4097, 131071, 1048577]):
    size = min(ch * 50, 5 * MiB + 3)
    _rt(ctx, ctx.p(f'c{ch}'), size, 200 + i, chunk=ch)


@test('large_file_2g', 'posix', max_nodes=1, timeout=1800)
def t_large(ctx):
  """2 GiB streaming write + verify (crosses 2^31)."""
  _rt(ctx, ctx.p('big'), 2 * 1024 * MiB + 12345, 7, chunk=4 * MiB)


@test('pwrite_overlap', 'posix', max_nodes=1)
def t_pwrite_overlap(ctx):
  """Overlapping pwrites across a blob boundary: last writer wins per byte."""
  p = ctx.p('ov')
  h = ctx.ok(0, 'open', path=p, flags='rwc')
  ops = [(0, 3 * MiB, 1), (MiB - 100, 200, 2), (2 * MiB - 5, MiB + 10, 3),
         (123, 4096, 4), (MiB - 1, 2, 5)]
  for off, ln, sd in ops:
    ctx.ok(0, 'fpwrite', h=h, off=off, length=ln, seed=sd)
  # Rebuild expected image locally from the same pattern generator.
  from agent import pattern
  img = bytearray(3 * MiB + 5)
  end = 0
  for off, ln, sd in ops:
    img[off:off + ln] = pattern(sd, off, ln)
    end = max(end, off + ln)
  img = bytes(img[:end])
  got = bytes.fromhex(ctx.ok(0, 'fpread_hex', h=h, off=0, length=end + 10))
  ctx.ok(0, 'close', h=h)
  ctx.check(len(got) == end, f'size {len(got)} != {end}')
  if got != img:
    first = next(i for i in range(end) if got[i] != img[i])
    ctx.check(False, f'content differs at byte {first}')


@test('sparse_holes', 'posix', max_nodes=1)
def t_sparse(ctx):
  """Writes past EOF leave zero-filled holes; holes read as zeros."""
  p = ctx.p('sparse')
  h = ctx.ok(0, 'open', path=p, flags='rwc')
  ctx.ok(0, 'fpwrite', h=h, off=5 * MiB + 3, length=100, seed=9)
  ctx.ok(0, 'fpwrite', h=h, off=10, length=10, seed=8)
  st = ctx.ok(0, 'fstat', h=h)
  ctx.check(st['size'] == 5 * MiB + 103, f'size {st["size"]}')
  z = bytes.fromhex(ctx.ok(0, 'fpread_hex', h=h, off=20, length=5 * MiB - 17))
  ctx.check(z == bytes(len(z)), 'hole not zero-filled')
  ctx.ok(0, 'close', h=h)


@test('truncate_semantics', 'posix', max_nodes=1)
def t_truncate(ctx):
  """Shrink, re-extend (must read zeros, not old data), truncate to 0."""
  p = ctx.p('tr')
  _rt(ctx, p, 3 * MiB, 11)
  ctx.ok(0, 'truncate', path=p, size=MiB + 10)
  ctx.check(ctx.ok(0, 'stat', path=p)['size'] == MiB + 10, 'shrink size')
  ctx.ok(0, 'truncate', path=p, size=3 * MiB)
  h = ctx.ok(0, 'open', path=p, flags='r')
  tail = bytes.fromhex(ctx.ok(0, 'fpread_hex', h=h, off=MiB + 10,
                              length=2 * MiB - 10))
  ctx.check(tail == bytes(len(tail)),
            'stale data resurfaced after shrink+extend (must be zeros)')
  mm = ctx.ok(0, 'fpread_verify', h=h, off=0, length=MiB + 10, seed=11)
  ctx.check(mm is None, f'prefix corrupted by truncate: {mm}')
  ctx.ok(0, 'close', h=h)
  ctx.ok(0, 'truncate', path=p, size=0)
  ctx.check(ctx.ok(0, 'stat', path=p)['size'] == 0, 'truncate 0')
  ctx.check(ctx.ok(0, 'read_hex', path=p) == '', 'data after truncate 0')
  # ftruncate on an open fd, then write after it.
  h = ctx.ok(0, 'open', path=p, flags='rw')
  ctx.ok(0, 'fpwrite', h=h, off=0, length=2 * MiB, seed=12)
  ctx.ok(0, 'ftruncate', h=h, size=100)
  ctx.ok(0, 'fpwrite', h=h, off=200, length=50, seed=13)
  got = bytes.fromhex(ctx.ok(0, 'fpread_hex', h=h, off=0, length=1000))
  from agent import pattern
  want = pattern(12, 0, 100) + bytes(100) + pattern(13, 200, 50)
  ctx.check(got == want, 'ftruncate+write image wrong')
  ctx.ok(0, 'close', h=h)


@test('o_trunc_o_excl', 'posix', max_nodes=1)
def t_open_flags(ctx):
  """O_CREAT|O_EXCL, O_TRUNC, ENOENT, EISDIR, ENOTDIR."""
  p = ctx.p('x')
  ctx.ok(0, 'write_file', path=p, size=5000, seed=1)
  ctx.err(0, 'open', 'EEXIST', path=p, flags='wcx')
  h = ctx.ok(0, 'open', path=p, flags='wt')
  ctx.ok(0, 'close', h=h)
  ctx.check(ctx.ok(0, 'stat', path=p)['size'] == 0, 'O_TRUNC did not truncate')
  ctx.err(0, 'open', 'ENOENT', path=ctx.p('nope'), flags='r')
  ctx.err(0, 'open', 'ENOENT', path=ctx.p('nodir', 'f'), flags='wc')
  ctx.err(0, 'open', 'ENOTDIR', path=ctx.p('x', 'child'), flags='wc')
  ctx.ok(0, 'mkdir', path=ctx.p('d'))
  ctx.err(0, 'open', 'EISDIR', path=ctx.p('d'), flags='w')
  ctx.err(0, 'open', 'ENOTDIR', path=p, flags='rd')


@test('append_mode', 'posix', max_nodes=1)
def t_append(ctx):
  """O_APPEND from two fds interleaves records without overwriting."""
  p = ctx.p('log')
  h1 = ctx.ok(0, 'open', path=p, flags='wca')
  h2 = ctx.ok(0, 'open', path=p, flags='wa')
  recs = []
  for i in range(200):
    rec = f'{i:05d}{"ab"[i % 2] * 27}\n'.encode()
    recs.append(rec)
    ctx.ok(0, 'fwrite', h=h1 if i % 2 == 0 else h2, data_hex=rec.hex())
  ctx.ok(0, 'close', h=h1)
  ctx.ok(0, 'close', h=h2)
  got = bytes.fromhex(ctx.ok(0, 'read_hex', path=p))
  ctx.check(got == b''.join(recs),
            f'append image wrong: len {len(got)} vs {sum(map(len, recs))}')


@test('dir_ops', 'posix', max_nodes=1)
def t_dirs(ctx):
  """mkdir/rmdir/readdir errors: EEXIST, ENOTEMPTY, ENOTDIR, exact listing."""
  d = ctx.p('d')
  ctx.ok(0, 'mkdir', path=d)
  ctx.err(0, 'mkdir', 'EEXIST', path=d)
  ctx.check(ctx.ok(0, 'listdir', path=d) == [], 'new dir not empty')
  ctx.check(ctx.ok(0, 'stat', path=d)['isdir'], 'not a dir')
  names = [f'n{i}' for i in range(50)] + ['.hidden', 'sp ace', 'ünï',
                                          'a' * 255]
  for n in names:
    ctx.ok(0, 'write_file', path=f'{d}/{n}', size=10, seed=1)
  ctx.ok(0, 'mkdir', path=f'{d}/sub')
  got = ctx.ok(0, 'listdir', path=d)
  ctx.check(got == sorted(names + ['sub']),
            f'listing mismatch: missing={set(names) - set(got)} '
            f'extra={set(got) - set(names) - {"sub"}}')
  ctx.err(0, 'rmdir', 'ENOTEMPTY', path=d)
  ctx.err(0, 'rmdir', 'ENOTDIR', path=f'{d}/n0')
  ctx.err(0, 'unlink', ['EISDIR', 'EPERM'], path=f'{d}/sub')
  ctx.err(0, 'write_file', 'ENAMETOOLONG', path=f'{d}/{"b" * 256}', size=1,
          seed=1)
  ctx.ok(0, 'rmtree', path=d)
  ctx.err(0, 'stat', 'ENOENT', path=d)
  ctx.err(0, 'rmdir', 'ENOENT', path=d)


@test('deep_tree', 'posix', max_nodes=1)
def t_deep(ctx):
  """64-level deep directory chain; create, stat, remove bottom-up."""
  parts = [ctx.dir] + [f'l{i}' for i in range(64)]
  path = '/'.join(parts)
  ctx.ok(0, 'makedirs', path=path)
  ctx.ok(0, 'write_file', path=path + '/leaf', size=4096, seed=3)
  v = ctx.ok(0, 'verify_file', path=path + '/leaf', size=4096, seed=3)
  ctx.check(v['ok'], f'leaf {v}')
  ctx.ok(0, 'rmtree', path=ctx.p('l0'))
  ctx.check(ctx.ok(0, 'listdir', path=ctx.dir) == [], 'tree not removed')


@test('rename_file', 'posix', max_nodes=1)
def t_rename_file(ctx):
  """rename: move, replace existing atomically, cross-dir, onto self."""
  a, b = ctx.p('a'), ctx.p('b')
  ctx.ok(0, 'write_file', path=a, size=MiB + 5, seed=1)
  ctx.ok(0, 'rename', src=a, dst=b)
  ctx.err(0, 'stat', 'ENOENT', path=a)
  ctx.check(ctx.ok(0, 'verify_file', path=b, size=MiB + 5, seed=1)['ok'],
            'data lost in rename')
  ctx.ok(0, 'write_file', path=a, size=300, seed=2)
  ctx.ok(0, 'rename', src=a, dst=b)  # replace
  ctx.check(ctx.ok(0, 'verify_file', path=b, size=300, seed=2)['ok'],
            'rename-replace left old content/size')
  ctx.ok(0, 'mkdir', path=ctx.p('d1'))
  ctx.ok(0, 'rename', src=b, dst=ctx.p('d1', 'b'))
  ctx.check(ctx.ok(0, 'listdir', path=ctx.dir) == ['d1'], 'cross-dir rename')
  ctx.ok(0, 'rename', src=ctx.p('d1', 'b'), dst=ctx.p('d1', 'b'))
  ctx.check(ctx.ok(0, 'verify_file', path=ctx.p('d1', 'b'), size=300,
                   seed=2)['ok'], 'rename onto self damaged file')
  ctx.err(0, 'rename', 'ENOENT', src=ctx.p('none'), dst=ctx.p('z'))


@test('rename_dir', 'posix', max_nodes=1)
def t_rename_dir(ctx):
  """Directory rename carries its subtree; EINVAL into self; ENOTEMPTY."""
  ctx.ok(0, 'makedirs', path=ctx.p('A', 'x', 'y'))
  for i in range(20):
    ctx.ok(0, 'write_file', path=ctx.p('A', 'x', f'f{i}'), size=1000 + i,
           seed=i)
  ctx.ok(0, 'write_file', path=ctx.p('A', 'x', 'y', 'deep'), size=MiB + 1,
         seed=99)
  ctx.ok(0, 'rename', src=ctx.p('A'), dst=ctx.p('B'))
  ctx.err(0, 'stat', 'ENOENT', path=ctx.p('A'))
  for i in range(20):
    v = ctx.ok(0, 'verify_file', path=ctx.p('B', 'x', f'f{i}'),
               size=1000 + i, seed=i)
    ctx.check(v['ok'], f'B/x/f{i} {v}')
  v = ctx.ok(0, 'verify_file', path=ctx.p('B', 'x', 'y', 'deep'),
             size=MiB + 1, seed=99)
  ctx.check(v['ok'], f'deep {v}')
  ctx.err(0, 'rename', 'EINVAL', src=ctx.p('B'), dst=ctx.p('B', 'x', 'in'))
  ctx.ok(0, 'makedirs', path=ctx.p('C', 'full'))
  ctx.err(0, 'rename', ['ENOTEMPTY', 'EEXIST'], src=ctx.p('B', 'x'),
          dst=ctx.p('C'))
  ctx.ok(0, 'mkdir', path=ctx.p('E'))
  ctx.ok(0, 'rename', src=ctx.p('B', 'x'), dst=ctx.p('E'))  # over empty dir
  ctx.check('f0' in ctx.ok(0, 'listdir', path=ctx.p('E')),
            'rename over empty dir lost content')
  ctx.ok(0, 'write_file', path=ctx.p('file'), size=1, seed=1)
  ctx.err(0, 'rename', 'EISDIR', src=ctx.p('file'), dst=ctx.p('E'))
  ctx.err(0, 'rename', 'ENOTDIR', src=ctx.p('E'), dst=ctx.p('file'))


@test('rename_noreplace_exchange', 'posix', max_nodes=1)
def t_renameat2(ctx):
  """renameat2 RENAME_NOREPLACE (supported), RENAME_EXCHANGE (documented)."""
  a, b = ctx.p('a'), ctx.p('b')
  ctx.ok(0, 'write_file', path=a, size=10, seed=1)
  ctx.ok(0, 'write_file', path=b, size=20, seed=2)
  ctx.err(0, 'rename', 'EEXIST', src=a, dst=b, flags=1)
  ctx.ok(0, 'rename', src=a, dst=ctx.p('c'), flags=1)
  r = ctx.call(0, 'rename', src=ctx.p('c'), dst=b, flags=2)
  if not r['ok']:
    ctx.note(f'RENAME_EXCHANGE unsupported: {r["err"]}')


@test('unlink_open_file', 'posix', max_nodes=1)
def t_unlink_open(ctx):
  """An unlinked-but-open file stays readable/writable through its fd."""
  p = ctx.p('orphan')
  ctx.ok(0, 'write_file', path=p, size=2 * MiB, seed=5)
  h = ctx.ok(0, 'open', path=p, flags='rw')
  ctx.ok(0, 'unlink', path=p)
  ctx.err(0, 'stat', 'ENOENT', path=p)
  mm = ctx.call(0, 'fpread_verify', h=h, off=0, length=2 * MiB, seed=5)
  ctx.check(mm.get('ok') and mm['ret'] is None,
            f'read of unlinked open file: {mm.get("err") or mm.get("ret")}')
  w = ctx.call(0, 'fpwrite', h=h, off=0, length=4096, seed=6)
  ctx.check(w.get('ok'), f'write to unlinked open file: {w.get("err")}')
  ctx.ok(0, 'close', h=h)
  # Re-create under same name must be a fresh empty file.
  ctx.ok(0, 'write_file', path=p, size=0, seed=0)
  ctx.check(ctx.ok(0, 'stat', path=p)['size'] == 0,
            'recreated name inherited orphan data')


@test('hardlinks', 'posix', max_nodes=1)
def t_hardlinks(ctx):
  """link(): shared data & inode, nlink counts, survives unlink of original."""
  a, b = ctx.p('a'), ctx.p('b')
  ctx.ok(0, 'write_file', path=a, size=100, seed=1)
  ctx.ok(0, 'link', src=a, dst=b)
  sa, sb = ctx.ok(0, 'stat', path=a), ctx.ok(0, 'stat', path=b)
  ctx.check(sa['ino'] == sb['ino'], f'inode differs {sa["ino"]} {sb["ino"]}')
  ctx.check(sb['nlink'] == 2, f'nlink {sb["nlink"]} != 2')
  ctx.ok(0, 'write_file', path=b, size=MiB + 7, seed=2, flags='w')
  v = ctx.ok(0, 'verify_file', path=a, size=MiB + 7, seed=2)
  ctx.check(v['ok'], f'write via link not seen via original: {v}')
  ctx.ok(0, 'unlink', path=a)
  ctx.check(ctx.ok(0, 'verify_file', path=b, size=MiB + 7, seed=2)['ok'],
            'link lost data after unlinking original')
  ctx.check(ctx.ok(0, 'stat', path=b)['nlink'] == 1, 'nlink after unlink')
  ctx.err(0, 'link', 'EEXIST', src=b, dst=b)
  ctx.ok(0, 'mkdir', path=ctx.p('d'))
  ctx.err(0, 'link', ['EPERM', 'EISDIR'], src=ctx.p('d'), dst=ctx.p('dl'))


@test('symlinks', 'posix', max_nodes=1)
def t_symlinks(ctx):
  """symlink/readlink/lstat, follow through, dangling, dir symlink."""
  ctx.ok(0, 'write_file', path=ctx.p('t'), size=33, seed=1)
  ctx.ok(0, 'symlink', target='t', path=ctx.p('rel'))
  ctx.ok(0, 'symlink', target=ctx.p('t'), path=ctx.p('abs'))
  ctx.ok(0, 'symlink', target='missing', path=ctx.p('dang'))
  ctx.check(ctx.ok(0, 'readlink', path=ctx.p('rel')) == 't', 'readlink rel')
  ctx.check(ctx.ok(0, 'lstat', path=ctx.p('rel'))['islnk'], 'lstat not link')
  for n in ('rel', 'abs'):
    ctx.check(ctx.ok(0, 'verify_file', path=ctx.p(n), size=33, seed=1)['ok'],
              f'follow {n}')
  ctx.err(0, 'stat', 'ENOENT', path=ctx.p('dang'))
  ctx.check(ctx.ok(0, 'lstat', path=ctx.p('dang'))['islnk'], 'dangling lstat')
  ctx.ok(0, 'makedirs', path=ctx.p('dir', 'in'))
  ctx.ok(0, 'symlink', target='dir', path=ctx.p('dl'))
  ctx.check(ctx.ok(0, 'listdir', path=ctx.p('dl')) == ['in'], 'dir symlink')
  ctx.err(0, 'symlink', 'EEXIST', target='x', path=ctx.p('rel'))
  ctx.ok(0, 'unlink', path=ctx.p('rel'))
  ctx.check(ctx.ok(0, 'exists', path=ctx.p('t')), 'unlink symlink removed '
            'target')


@test('attrs_mode_times', 'posix', max_nodes=1)
def t_attrs(ctx):
  """chmod / utimens / mtime-on-write / ctime-on-chmod / chown(self)."""
  p = ctx.p('f')
  ctx.ok(0, 'write_file', path=p, size=10, seed=1, mode=0o640)
  st = ctx.ok(0, 'stat', path=p)
  ctx.check(st['perm'] == 0o640, f'create mode {oct(st["perm"])} (umask?)')
  for m in (0o600, 0o755, 0o4711, 0o000, 0o644):
    ctx.ok(0, 'chmod', path=p, mode=m)
    got = ctx.ok(0, 'stat', path=p)['perm']
    ctx.check(got == m, f'chmod {oct(m)} -> {oct(got)}')
  t = 1_500_000_000_123_456_789
  ctx.ok(0, 'utime', path=p, atime_ns=t, mtime_ns=t + 1000)
  st = ctx.ok(0, 'stat', path=p)
  ctx.check(abs(st['mtime_ns'] - (t + 1000)) < 1_000_000,
            f'mtime {st["mtime_ns"]} != {t + 1000}')
  ctx.check(abs(st['atime_ns'] - t) < 1_000_000, f'atime {st["atime_ns"]}')
  before = ctx.ok(0, 'stat', path=p)
  import time as _t
  _t.sleep(1.1)
  ctx.ok(0, 'write_file', path=p, size=20, seed=2, flags='w')
  after = ctx.ok(0, 'stat', path=p)
  ctx.check(after['mtime_ns'] > before['mtime_ns'], 'write did not bump mtime')
  _t.sleep(1.1)
  ctx.ok(0, 'chmod', path=p, mode=0o600)
  a2 = ctx.ok(0, 'stat', path=p)
  ctx.check(a2['ctime_ns'] > after['ctime_ns'], 'chmod did not bump ctime')
  ctx.ok(0, 'chown_self', path=p)
  st = ctx.ok(0, 'stat', path=p)
  ctx.check(st['uid'] == os.getuid(), f'uid {st["uid"]} != {os.getuid()}')
  d = ctx.p('d')
  ctx.ok(0, 'mkdir', path=d, mode=0o700)
  ctx.check(ctx.ok(0, 'stat', path=d)['perm'] == 0o700, 'mkdir mode')
  m0 = ctx.ok(0, 'stat', path=d)['mtime_ns']
  _t.sleep(1.1)
  ctx.ok(0, 'write_file', path=f'{d}/c', size=1, seed=1)
  ctx.check(ctx.ok(0, 'stat', path=d)['mtime_ns'] > m0,
            'create did not bump parent dir mtime')


@test('permission_enforcement', 'posix', max_nodes=1)
def t_perm_enforce(ctx):
  """A 0000 file must not be openable by its (non-root) owner."""
  p = ctx.p('locked')
  ctx.ok(0, 'write_file', path=p, size=10, seed=1)
  ctx.ok(0, 'chmod', path=p, mode=0o000)
  r = ctx.call(0, 'open', path=p, flags='r')
  if r['ok']:
    ctx.ok(0, 'close', h=r['ret'])
    ctx.ok(0, 'chmod', path=p, mode=0o644)
    raise TestUnsupported('permission bits are not enforced (no '
                          'default_permissions / access check): a mode-0000 '
                          'file opened for read by its owner')
  ctx.check(r['errno'] == errno.EACCES, f'expected EACCES got {r["err"]}')
  ctx.ok(0, 'chmod', path=p, mode=0o644)


@test('xattrs', 'posix', max_nodes=1)
def t_xattr(ctx):
  """user.* xattrs: set/get/list/remove, XATTR_CREATE/REPLACE, ENODATA."""
  p = ctx.p('x')
  ctx.ok(0, 'write_file', path=p, size=1, seed=1)
  big = os.urandom(3000).hex()
  ctx.ok(0, 'setxattr', path=p, name='user.a', value_hex='68656c6c6f')
  ctx.ok(0, 'setxattr', path=p, name='user.big', value_hex=big)
  ctx.ok(0, 'setxattr', path=p, name='user.empty', value_hex='')
  ctx.check(ctx.ok(0, 'getxattr', path=p, name='user.a') == '68656c6c6f',
            'getxattr value')
  ctx.check(ctx.ok(0, 'getxattr', path=p, name='user.big') == big, 'big xattr')
  ctx.check(ctx.ok(0, 'getxattr', path=p, name='user.empty') == '', 'empty')
  names = ctx.ok(0, 'listxattr', path=p)
  ctx.check({'user.a', 'user.big', 'user.empty'} <= set(names),
            f'listxattr {names}')
  ctx.err(0, 'setxattr', 'EEXIST', path=p, name='user.a', value_hex='00',
          flags=1)
  ctx.err(0, 'setxattr', 'ENODATA', path=p, name='user.nx', value_hex='00',
          flags=2)
  ctx.ok(0, 'removexattr', path=p, name='user.a')
  ctx.err(0, 'getxattr', 'ENODATA', path=p, name='user.a')
  ctx.err(0, 'removexattr', 'ENODATA', path=p, name='user.a')
  # xattrs follow renames and die with the file.
  ctx.ok(0, 'rename', src=p, dst=ctx.p('y'))
  ctx.check(ctx.ok(0, 'getxattr', path=ctx.p('y'), name='user.big') == big,
            'xattr lost on rename')
  ctx.ok(0, 'unlink', path=ctx.p('y'))
  ctx.ok(0, 'write_file', path=ctx.p('y'), size=1, seed=1)
  names = ctx.ok(0, 'listxattr', path=ctx.p('y'))
  ctx.check('user.big' not in names,
            f'recreated file inherited xattrs of unlinked one: {names}')


@test('fsync_statfs', 'posix', max_nodes=1)
def t_fsync(ctx):
  """fsync/fdatasync succeed; statvfs reports sane capacity/namemax."""
  h = ctx.ok(0, 'open', path=ctx.p('f'), flags='rwc')
  ctx.ok(0, 'fpwrite', h=h, off=0, length=MiB, seed=1)
  ctx.ok(0, 'fsync', h=h)
  ctx.ok(0, 'fdatasync', h=h)
  ctx.ok(0, 'close', h=h)
  s = ctx.ok(0, 'statvfs', path=ctx.dir)
  ctx.check(s['blocks'] > 0 and s['bsize'] > 0, f'statvfs {s}')
  ctx.check(s['bavail'] <= s['blocks'], f'statvfs avail>total {s}')
  ctx.check(s['namemax'] >= 255, f'namemax {s["namemax"]}')
  ctx.metrics['statvfs_GiB'] = round(s['blocks'] * s['bsize'] / 2**30, 1)


@test('fallocate', 'posix', max_nodes=1)
def t_fallocate(ctx):
  """posix_fallocate extends size; punch-hole zeros a range (or EOPNOTSUPP)."""
  p = ctx.p('fa')
  h = ctx.ok(0, 'open', path=p, flags='rwc')
  ctx.ok(0, 'fallocate', h=h, off=0, length=3 * MiB)
  ctx.check(ctx.ok(0, 'fstat', h=h)['size'] == 3 * MiB, 'fallocate size')
  ctx.ok(0, 'fpwrite', h=h, off=0, length=3 * MiB, seed=4)
  r = ctx.call(0, 'fallocate', h=h, off=MiB, length=MiB, flags=1 | 2)
  if r['ok']:
    z = bytes.fromhex(ctx.ok(0, 'fpread_hex', h=h, off=MiB, length=MiB))
    ctx.check(z == bytes(MiB), 'punched hole not zero')
    ctx.check(ctx.ok(0, 'fstat', h=h)['size'] == 3 * MiB, 'punch changed size')
  else:
    ctx.note(f'punch hole: {r["err"]}')
  ctx.ok(0, 'close', h=h)


@test('seek_data_hole', 'posix', max_nodes=1)
def t_seek(ctx):
  """lseek SEEK_END/SEEK_DATA/SEEK_HOLE are consistent with the file size."""
  p = ctx.p('s')
  h = ctx.ok(0, 'open', path=p, flags='rwc')
  ctx.ok(0, 'fpwrite', h=h, off=4 * MiB, length=10, seed=1)
  end = ctx.ok(0, 'lseek', h=h, off=0, whence='end')
  ctx.check(end == 4 * MiB + 10, f'SEEK_END {end}')
  d = ctx.ok(0, 'lseek', h=h, off=0, whence='data')
  hole = ctx.ok(0, 'lseek', h=h, off=0, whence='hole')
  ctx.check(d <= 4 * MiB and hole <= 4 * MiB + 10,
            f'SEEK_DATA={d} SEEK_HOLE={hole}')
  ctx.err(0, 'lseek', 'ENXIO', h=h, off=4 * MiB + 11, whence='data')
  ctx.ok(0, 'close', h=h)


@test('mmap_shared', 'posix', max_nodes=1)
def t_mmap(ctx):
  """MAP_SHARED write + msync is visible to a later read()."""
  p = ctx.p('m')
  h = ctx.ok(0, 'open', path=p, flags='rwc')
  ctx.ok(0, 'ftruncate', h=h, size=4 * MiB)
  mm = ctx.ok(0, 'mmap_rw', h=h, off=MiB - 100, length=2 * MiB, seed=7)
  ctx.check(mm is None, f'mmap readback {mm}')
  ctx.ok(0, 'close', h=h)
  h = ctx.ok(0, 'open', path=p, flags='r')
  mm = ctx.ok(0, 'fpread_verify', h=h, off=MiB - 100, length=2 * MiB, seed=7)
  ctx.check(mm is None, f'mmap data not in file after msync+close: {mm}')
  ctx.ok(0, 'close', h=h)


@test('locks_flock_posix', 'posix', max_nodes=1)
def t_locks(ctx):
  """flock / fcntl locks conflict between two open file descriptions."""
  p = ctx.p('lk')
  ctx.ok(0, 'write_file', path=p, size=10, seed=1)
  h1 = ctx.ok(0, 'open', path=p, flags='rw')
  h2 = ctx.ok(0, 'open', path=p, flags='rw')
  ctx.ok(0, 'flock', h=h1, kind='ex')
  r = ctx.call(0, 'flock', h=h2, kind='ex_nb')
  ctx.ok(0, 'flock', h=h1, kind='un')
  ctx.ok(0, 'close', h=h1)
  ctx.ok(0, 'close', h=h2)
  if r['ok']:
    raise TestUnsupported('flock does not conflict across descriptions')
  ctx.check(r['errno'] in (errno.EWOULDBLOCK, errno.EAGAIN),
            f'flock conflict errno {r["err"]}')


@test('many_small_files', 'posix', max_nodes=1, timeout=1800)
def t_many(ctx):
  """10k 1 KiB files: create, stat, verify, readdir count, unlink."""
  n = 10000
  ctx.ok(0, 'mkdir', path=ctx.p('d'))
  import time as _t
  t0 = _t.time()
  r = ctx.ok(0, 'create_many', dirpath=ctx.p('d'), prefix='f', count=n,
             size=1024, seed=0, timeout=1200)
  ctx.check(not r['fails'], f'create fails {r["fails"][:3]}')
  t1 = _t.time()
  ctx.check(ctx.ok(0, 'scandir_count', path=ctx.p('d'), timeout=300) == n,
            'readdir count')
  r = ctx.ok(0, 'verify_many', dirpath=ctx.p('d'), prefix='f', count=n,
             size=1024, timeout=1200)
  ctx.check(not r['bad'], f'verify bad {r["bad"][:3]}')
  t2 = _t.time()
  r = ctx.ok(0, 'unlink_many', dirpath=ctx.p('d'), prefix='f', count=n,
             timeout=1200)
  ctx.check(not r['fails'], f'unlink fails {r["fails"][:3]}')
  t3 = _t.time()
  ctx.check(ctx.ok(0, 'listdir', path=ctx.p('d')) == [], 'dir not empty')
  ctx.metrics.update({'create_per_s': round(n / (t1 - t0)),
                      'readverify_per_s': round(n / (t2 - t1)),
                      'unlink_per_s': round(n / (t3 - t2))})


def _nettrace_totals(cl):
  """Cumulative cross-node message counters from every node's runtime log
  when the daemons run with CLIO_NET_TRACE=1: {'sendin': requests sent,
  'sendout': responses sent, 'recv': messages received}, summed over nodes.
  The runtime dumps a line every 32 ops, so each figure is accurate to
  32 per node. The logs are on NFS: wait a moment before sampling."""
  import re
  time.sleep(3)
  tot = {'sendin': 0, 'sendout': 0, 'recv': 0}
  pat = re.compile(r'\[NETTRACE (\w+)\] n=(\d+) .*sendout\(.*? n=(\d+)\) '
                   r'recv\(.*? n=(\d+)\)')
  for h in cl.hosts:
    best = {'sendin': 0, 'sendout': 0, 'recv': 0}
    try:
      with open(cl.log_path(h, 'runtime'), errors='replace') as f:
        for ln in f:
          m = pat.search(ln)
          if not m:
            continue
          tag, n, so, rc = m.group(1), int(m.group(2)), int(m.group(3)), \
              int(m.group(4))
          if tag == 'sendin':
            best['sendin'] = max(best['sendin'], n)
          best['recv'] = max(best['recv'], rc)
          best['sendout'] = max(best['sendout'], so)
    except OSError:
      pass
    for k in tot:
      tot[k] += best[k]
  return tot


@test('many_small_files_traced', 'posix', max_nodes=1, timeout=1800)
def t_many_traced(ctx):
  """many_small_files with per-phase cross-node message counts (#1159):
  the same 10k 1 KiB files from one client, and after each phase the
  cluster-wide CLIO_NET_TRACE counters, so the metrics give requests sent,
  responses sent and messages received PER OPERATION for create, for
  stat+read, and for unlink, at this cluster size."""
  n = 10000
  cl = ctx.cl
  ctx.metrics['cluster_nodes'] = len(cl.hosts)
  ctx.ok(0, 'mkdir', path=ctx.p('d'))
  snaps = [_nettrace_totals(cl)]
  import time as _t
  t0 = _t.time()
  r = ctx.ok(0, 'create_many', dirpath=ctx.p('d'), prefix='f', count=n,
             size=1024, seed=0, timeout=1200)
  ctx.check(not r['fails'], f'create fails {r["fails"][:3]}')
  t1 = _t.time()
  snaps.append(_nettrace_totals(cl))
  ctx.check(ctx.ok(0, 'scandir_count', path=ctx.p('d'), timeout=300) == n,
            'readdir count')
  r = ctx.ok(0, 'verify_many', dirpath=ctx.p('d'), prefix='f', count=n,
             size=1024, timeout=1200)
  ctx.check(not r['bad'], f'verify bad {r["bad"][:3]}')
  t2 = _t.time()
  snaps.append(_nettrace_totals(cl))
  r = ctx.ok(0, 'unlink_many', dirpath=ctx.p('d'), prefix='f', count=n,
             timeout=1200)
  ctx.check(not r['fails'], f'unlink fails {r["fails"][:3]}')
  t3 = _t.time()
  snaps.append(_nettrace_totals(cl))
  ctx.check(ctx.ok(0, 'listdir', path=ctx.p('d')) == [], 'dir not empty')
  ctx.metrics.update({'create_per_s': round(n / (t1 - t0)),
                      'readverify_per_s': round(n / (t2 - t1)),
                      'unlink_per_s': round(n / (t3 - t2))})
  for i, ph in enumerate(('create', 'readverify', 'unlink')):
    for k in ('sendin', 'sendout', 'recv'):
      d = snaps[i + 1][k] - snaps[i][k]
      ctx.metrics[f'{ph}_{k}_per_op'] = round(d / n, 2)
  if snaps[-1]['recv'] == 0 and len(cl.hosts) > 1:
    ctx.note('no [NETTRACE] lines in the runtime logs: run the daemons with '
             'CLIO_SUITE_PASS_CLIO_NET_TRACE=1 for the message counts')


@test('fsx_random_ops', 'posix', max_nodes=1, timeout=1800)
def t_fsx(ctx):
  """Model-checked random pwrite/truncate/read/fallocate on one file."""
  from fsx_model import run_fsx
  run_fsx(ctx, nodes=[0], ops=3000, seed=1234, maxlen=6 * MiB)


@test('readdir_during_mutation', 'posix', max_nodes=1)
def t_readdir_mut(ctx):
  """Names that exist for the whole scan must appear exactly once."""
  d = ctx.p('d')
  ctx.ok(0, 'mkdir', path=d)
  ctx.ok(0, 'create_many', dirpath=d, prefix='keep', count=500)
  for rnd in range(5):
    ctx.ok(0, 'create_many', dirpath=d, prefix=f'churn{rnd}_', count=100)
    names = ctx.ok(0, 'listdir', path=d)
    keep = [x for x in names if x.startswith('keep')]
    ctx.check(len(keep) == 500 and len(set(keep)) == 500,
              f'round {rnd}: {len(keep)} keep entries')
    ctx.ok(0, 'unlink_many', dirpath=d, prefix=f'churn{rnd}_', count=100)


@test('unlinked_open_file', 'posix', max_nodes=1)
def t_unlinked_open_file(ctx):
  """The temp-file pattern (SQLite, Python's TemporaryFile): create, write,
  unlink, then keep using the descriptor -- fstat (nlink 0, the size),
  read back, ftruncate, write, fstat again. Afterwards no hidden leftover
  may remain in the directory."""
  d = ctx.p('tmpf')
  ctx.ok(0, 'mkdir', path=d)
  code = (
      "import os,json; p=%r; fd=os.open(p, os.O_RDWR|os.O_CREAT, 0o600); "
      "os.write(fd, b'x'*100000); os.unlink(p); r={}; st=os.fstat(fd); "
      "r['nlink']=st.st_nlink; r['size']=st.st_size; "
      "r['read']=len(os.pread(fd, 200000, 0)); os.ftruncate(fd, 5000); "
      "r['size_after_trunc']=os.fstat(fd).st_size; "
      "os.pwrite(fd, b'y'*10, 7000); r['size_after_write']=os.fstat(fd).st_size; "
      "r['tail']=os.pread(fd, 10, 7000).decode(); os.close(fd); "
      "r['left']=sorted(os.listdir(%r)); print(json.dumps(r))"
      % (f'{d}/t', d))
  r = ctx.ok(0, 'sh', cmd=f'python3 -c "{code}"', timeout=120)
  ctx.check(r['rc'] == 0, f'the unlinked open file failed: {r["err"][-400:]}')
  import json
  got = json.loads(r['out'].strip().splitlines()[-1])
  want = {'nlink': 0, 'size': 100000, 'read': 100000, 'size_after_trunc': 5000,
          'size_after_write': 7010, 'tail': 'y' * 10, 'left': []}
  bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
  ctx.check(not bad, f'unlinked open file: (got, want) {bad}')
