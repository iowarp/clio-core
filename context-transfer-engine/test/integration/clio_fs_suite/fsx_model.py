"""fsx-style model checker for clio-fs.

Applies a seeded random sequence of pwrite / truncate / fallocate / read /
reopen / fsync operations to one file and to an in-memory model image, and
compares every read with the model.  With several nodes the operations
rotate across nodes with close-to-open between them (every op opens and
closes its own descriptor on its node), which is the consistency contract a
distributed POSIX filesystem has to honour.
"""

import hashlib
import random

from agent import pattern


def run_fsx(ctx, nodes, ops, seed, maxlen, path=None, full_every=250):
  """Run `ops` random operations; raise TestFailure on first divergence."""
  rng = random.Random(seed)
  path = path or ctx.p(f'fsx_{seed}')
  model = bytearray()
  ctx.ok(nodes[0], 'write_file', path=path, size=0, seed=0)
  hist = []
  for k in range(ops):
    node = nodes[k % len(nodes)] if len(nodes) > 1 else nodes[0]
    r = rng.random()
    if r < 0.40:
      off = rng.randrange(0, maxlen)
      ln = rng.choice([1, rng.randrange(1, 4096), rng.randrange(1, 1 << 17),
                       rng.randrange(1, 1 << 21)])
      ln = min(ln, maxlen - off) or 1
      sd = seed * 100000 + k
      h = ctx.ok(node, 'open', path=path, flags='rw')
      ctx.ok(node, 'fpwrite', h=h, off=off, length=ln, seed=sd)
      ctx.ok(node, 'close', h=h)
      if off > len(model):
        model += bytes(off - len(model))
      data = pattern(sd, off, ln)
      model[off:off + ln] = data
      hist.append(f'{k}@n{node} pwrite off={off} len={ln}')
    elif r < 0.52:
      size = rng.choice([0, rng.randrange(0, maxlen), len(model),
                         max(0, len(model) - rng.randrange(0, 5000)),
                         len(model) + rng.randrange(0, 1 << 20)])
      size = min(size, maxlen)
      ctx.ok(node, 'truncate', path=path, size=size)
      if size < len(model):
        del model[size:]
      else:
        model += bytes(size - len(model))
      hist.append(f'{k}@n{node} truncate {size}')
    elif r < 0.57:
      off = rng.randrange(0, maxlen // 2)
      ln = rng.randrange(1, maxlen // 4)
      h = ctx.ok(node, 'open', path=path, flags='rw')
      fr = ctx.call(node, 'fallocate', h=h, off=off, length=ln)
      ctx.ok(node, 'close', h=h)
      if fr['ok'] and off + ln > len(model):
        model += bytes(off + ln - len(model))
      hist.append(f'{k}@n{node} fallocate off={off} len={ln} ok={fr["ok"]}')
    else:
      st = ctx.ok(node, 'stat', path=path)
      if st['size'] != len(model):
        raise_div(ctx, hist, k, node,
                  f'size {st["size"]} != model {len(model)}')
      if not model:
        continue
      off = rng.randrange(0, len(model))
      ln = min(rng.choice([1, 4096, 65536, 1 << 18]), len(model) - off)
      h = ctx.ok(node, 'open', path=path, flags='r')
      got = bytes.fromhex(ctx.ok(node, 'fpread_hex', h=h, off=off,
                                 length=ln + 8))
      ctx.ok(node, 'close', h=h)
      want = bytes(model[off:off + ln])
      # Asked for ln+8 bytes: the extra 8 must come back only if the model
      # has them (catches reads past EOF as well as short reads).
      want = bytes(model[off:off + ln + 8])
      if got != want:
        n = min(len(got), len(want))
        first = next((i for i in range(n) if got[i] != want[i]), n)
        raise_div(ctx, hist, k, node,
                  f'read off={off} len={ln}: first diff at +{first} '
                  f'(abs {off + first}); got_len={len(got)} '
                  f'got_zero={got[first:first+32] == bytes(32)}')
      hist.append(f'{k}@n{node} read off={off} len={ln}')
    if full_every and k % full_every == full_every - 1:
      full_check(ctx, node, path, model, hist, k)
  full_check(ctx, nodes[-1], path, model, hist, ops)
  ctx.metrics['fsx_ops'] = ops
  ctx.metrics['fsx_final_size'] = len(model)


def full_check(ctx, node, path, model, hist, k):
  r = ctx.ok(node, 'sha256', path=path)
  want = hashlib.sha256(bytes(model)).hexdigest()
  if r['size'] != len(model) or r['sha256'] != want:
    raise_div(ctx, hist, k, node,
              f'full-file check: size {r["size"]} vs {len(model)}, '
              f'sha mismatch={r["sha256"] != want}')


def raise_div(ctx, hist, k, node, msg):
  from suite import TestFailure
  raise TestFailure(f'fsx diverged at op {k} on node{node}: {msg}; '
                    f'last ops: {hist[-6:]}')
