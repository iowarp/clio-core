"""Capacity: what a full clio-fs looks like to applications.

Not in the default groups: filling the cluster takes a long time on a
normally sized deployment. Run it on a small one, e.g.

  suite.py --groups capacity --disk-gb 2 ...

A full filesystem must behave like ext4's: the write (or the close/fsync that
reports it) fails with ENOSPC -- never EIO, which tells an application its
data or device is broken -- every byte acknowledged before that stays intact,
and deleting files makes room again.
"""

import time

from suite import test

MiB = 1 << 20
FILE_SIZE = 256 * MiB
MAX_FILES = 400  # 100 GiB: past any sensible --disk-gb for this group


@test('capacity_full_is_enospc', 'capacity', min_nodes=1, redeploy_after=True,
      timeout=7200)
def t_full_is_enospc(ctx):
  """Fill the cluster from node0, 256 MiB at a time, until a write fails.
  The failure is ENOSPC; every complete file verifies; after deleting them
  a new file can be written and read back."""
  written = []
  err = None
  for k in range(MAX_FILES):
    path = ctx.p(f'fill{k}')
    r = ctx.call(0, 'write_file', path=path, size=FILE_SIZE, seed=k,
                 fsync=True, timeout=900)
    if not r['ok']:
      err = r.get('err', '')
      break
    written.append((path, k))
  ctx.metrics['files_before_full'] = len(written)
  ctx.metrics['bytes_before_full_mib'] = len(written) * FILE_SIZE // MiB
  ctx.check(err is not None,
            f'wrote {len(written)} x 256 MiB without the cluster filling; '
            f'rerun with a smaller --disk-gb')
  ctx.metrics['full_error'] = (err or '')[:120]
  ctx.check(err is not None and err.startswith('ENOSPC'),
            f'a full cluster failed the write with {err!r}, not ENOSPC')
  bad = []
  for path, k in written:
    v = ctx.call(0, 'verify_file', path=path, size=FILE_SIZE, seed=k,
                 timeout=900)
    if not v['ok'] or not v['ret']['size_ok'] or v['ret']['mismatch']:
      bad.append((path, v.get('err') or v.get('ret')))
  ctx.check(not bad, f'{len(bad)} files acknowledged before the cluster '
                     f'filled are damaged, e.g. {bad[:3]}')
  ctx.call(0, 'unlink', path=ctx.p(f'fill{len(written)}'))  # the partial one
  for path, _ in written:
    ctx.ok(0, 'unlink', path=path, timeout=300)
  # Room again: space freed by unlink is reusable (purges may lag a little).
  again = ctx.p('after_free')
  ok = False
  for _ in range(30):
    r = ctx.call(0, 'write_file', path=again, size=FILE_SIZE, seed=999,
                 fsync=True, timeout=900)
    if r['ok']:
      ok = True
      break
    time.sleep(10)
  ctx.check(ok, 'after deleting every file the cluster still refuses a '
                '256 MiB write')
  if ok:
    v = ctx.ok(0, 'verify_file', path=again, size=FILE_SIZE, seed=999,
               timeout=900)
    ctx.check(v['size_ok'] and not v['mismatch'],
              f'the file written after freeing space reads back wrong: {v}')


@test('capacity_writeback_full_is_enospc', 'capacity', min_nodes=1,
      redeploy_after=True, timeout=7200)
def t_writeback_full_is_enospc(ctx):
  """The same, the way ordinary programs write: no fsync, so the bytes go
  through the batched write-behind path and a full store surfaces at write
  or close. That failure, a mkdir and a create on the full filesystem all
  fail ENOSPC -- never EIO -- and deleting files makes room again."""
  written, err = [], None
  for k in range(MAX_FILES):
    path = ctx.p(f'wb{k}')
    r = ctx.call(0, 'write_file', path=path, size=FILE_SIZE, seed=k,
                 timeout=900)
    if not r['ok']:
      err = r.get('err', '')
      break
    written.append(path)
  ctx.metrics['files_before_full'] = len(written)
  ctx.check(err is not None, 'the cluster never filled; use a smaller '
                             '--disk-gb')
  ctx.metrics['full_error'] = (err or '')[:120]
  ctx.check(err is not None and err.startswith('ENOSPC'),
            f'a full cluster failed a buffered write with {err!r}, not ENOSPC')
  # Metadata on a full filesystem: success (there was room) or ENOSPC.
  for op, args in (('mkdir', {'path': ctx.p('full_dir')}),
                   ('write_file', {'path': ctx.p('full_small'), 'size': 4096,
                                   'seed': 1})):
    r = ctx.call(0, op, **args)
    e = r.get('err') or ''
    ctx.check(r['ok'] or e.startswith('ENOSPC'),
              f'{op} on a full filesystem failed with {e!r}, not ENOSPC')
  for p in [ctx.p(f'wb{len(written)}'), ctx.p('full_small')]:
    ctx.call(0, 'unlink', path=p)
  for p in written:
    ctx.ok(0, 'unlink', path=p, timeout=300)
  ok = False
  for _ in range(30):
    if ctx.call(0, 'write_file', path=ctx.p('wb_again'), size=FILE_SIZE,
                seed=999, timeout=900)['ok']:
      ok = True
      break
    time.sleep(10)
  ctx.check(ok, 'after deleting every file a 256 MiB write still fails')
