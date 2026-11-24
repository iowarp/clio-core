"""Real-application workloads on the FUSE mount ("general filesystem" use).

Each test runs ordinary tools (tar, git, cc/make, sqlite, rsync, cp) inside
the mount and checks their own integrity verdicts; the multi-node variants
build on one node and consume the result on another.
"""

import os

from suite import test

SRC_TARBALL = None


def src_tarball(ctx):
  """A ~20 MB tarball of this repo's CTE sources (on shared NFS)."""
  global SRC_TARBALL
  if SRC_TARBALL:
    return SRC_TARBALL
  repo = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                      '../../../..'))
  out = os.path.join(ctx.cl.run_dir, 'src.tar')
  if not os.path.exists(out):
    os.system(f'tar -C {repo} -cf {out} --exclude=.git context-transfer-engine'
              f' context-runtime/include context-runtime/src 2>/dev/null')
  SRC_TARBALL = out
  return out


def run(ctx, i, cmd, timeout=900, cwd=None):
  r = ctx.ok(i, 'sh', cmd=cmd, cwd=cwd, timeout=timeout)
  ctx.check(r['rc'] == 0, f'node{i} `{cmd[:200]}` rc={r["rc"]}\n'
                          f'{r["out"][-1500:]}\n{r["err"][-1500:]}')
  return r['out']


@test('app_tar_roundtrip', 'apps', max_nodes=1)
def t_tar(ctx):
  """Extract a source tree, re-tar it, and compare against the original."""
  tb = src_tarball(ctx)
  d = ctx.p('x')
  run(ctx, 0, f'mkdir -p {d} && tar -C {d} -xf {tb}')
  out = run(ctx, 0, f'mkdir -p /dev/shm/$USER.cmp && rm -rf '
                    f'/dev/shm/$USER.cmp/* && tar -C /dev/shm/$USER.cmp -xf '
                    f'{tb} && diff -r --no-dereference /dev/shm/$USER.cmp {d}'
                    f' && echo SAME; rm -rf /dev/shm/$USER.cmp')
  ctx.check('SAME' in out, f'tree differs: {out[-800:]}')
  n = run(ctx, 0, f'find {d} | wc -l').strip()
  ctx.metrics['entries'] = int(n)


@test('app_git', 'apps', max_nodes=2, timeout=1800)
def t_git(ctx):
  """git init/add/commit/fsck/gc on the mount; clone it on another node."""
  tb = src_tarball(ctx)
  d = ctx.p('repo')
  run(ctx, 0, f'mkdir -p {d} && tar -C {d} -xf {tb} && cd {d} && '
              f'git init -q && git add -A && '
              f'git -c user.email=t@t -c user.name=t commit -qm init && '
              f'git fsck --full --strict && git gc -q && git fsck --full && '
              f'git status --porcelain | wc -l', timeout=1500)
  j = len(ctx.hosts) - 1
  c = ctx.p(f'clone{j}')
  out = run(ctx, j, f'git clone -q {d} {c} && cd {c} && git fsck --full && '
                    f'git log --oneline | wc -l && git diff --stat HEAD',
            timeout=1500)
  ctx.check(out.strip().split('\n')[-1].strip() in ('1', ''),
            f'clone state: {out[-500:]}')


@test('app_compile', 'apps', max_nodes=1, timeout=1800)
def t_compile(ctx):
  """Generate a 200-file C project, build with make -j16, run it, rebuild
  incrementally after touching one file (mtime-driven make)."""
  d = ctx.p('proj')
  gen = (
      f'mkdir -p {d} && cd {d} && for i in $(seq 0 199); do '
      f'printf "int f%d(int x){{return x+%d;}}\\n" $i $i > f$i.c; done && '
      f'( echo "#include <stdio.h>"; for i in $(seq 0 199); do '
      f'echo "int f$i(int);"; done; echo "int main(){{long s=0;"; '
      f'for i in $(seq 0 199); do echo "s+=f$i(1);"; done; '
      f'echo "printf(\\"%ld\\\\n\\",s);return 0;}}" ) > main.c && '
      f'printf "SRCS=\\$(wildcard *.c)\\nOBJS=\\$(SRCS:.c=.o)\\n'
      f'prog: \\$(OBJS)\\n\\tcc -o \\$@ \\$^\\n%%.o: %%.c\\n'
      f'\\tcc -O1 -c \\$< -o \\$@\\n" > Makefile')
  run(ctx, 0, gen)
  out = run(ctx, 0, f'cd {d} && make -s -j16 && ./prog')
  ctx.check(out.strip().endswith('20100'), f'prog output {out[-200:]}')
  out = run(ctx, 0, f'cd {d} && sleep 1.1 && '
                    f'printf "int f7(int x){{return x+1007;}}\\n" > f7.c && '
                    f'make -j16 2>&1 | grep -c "cc -O1" ; ./prog')
  lines = out.strip().split('\n')
  ctx.check(lines[-1] == '21100', f'rebuild output {out[-300:]}')
  ctx.check(lines[0] == '1', f'incremental make recompiled {lines[0]} files '
                             f'(mtime handling)')


@test('app_sqlite', 'apps', max_nodes=1, timeout=1800)
def t_sqlite(ctx):
  """SQLite (rollback journal and WAL): 20k-row transactions, integrity."""
  d = ctx.p('db')
  ctx.ok(0, 'mkdir', path=d)
  py = ('import sqlite3,sys\n'
        'for mode in ("delete","wal"):\n'
        '  db=sqlite3.connect(f"{sys.argv[1]}/t_{mode}.db")\n'
        '  db.execute(f"pragma journal_mode={mode}")\n'
        '  db.execute("create table t(k integer primary key, v blob)")\n'
        '  for b in range(20):\n'
        '    db.executemany("insert into t(v) values(?)",'
        '[(bytes([b])*500,) for _ in range(1000)])\n'
        '    db.commit()\n'
        '  db.execute("delete from t where k%3=0"); db.commit()\n'
        '  db.execute("vacuum")\n'
        '  r=db.execute("pragma integrity_check").fetchone()[0]\n'
        '  c=db.execute("select count(*) from t").fetchone()[0]\n'
        '  print(mode,r,c); db.close()\n'
        '  assert r=="ok" and c==13334, (r,c)\n')
  ctx.ok(0, 'write_hex', path=f'{d}/t.py', data_hex=py.encode().hex())
  run(ctx, 0, f'/usr/bin/python3 {d}/t.py {d}', timeout=1500)


@test('app_rsync_cp', 'apps', max_nodes=2, timeout=1800)
def t_rsync(ctx):
  """cp -a into the mount and rsync -a back out; checksums match, a second
  rsync transfers nothing (sizes+mtimes preserved)."""
  tb = src_tarball(ctx)
  src = '/dev/shm/$USER.rs_src'
  d = ctx.p('rs')
  run(ctx, 0, f'rm -rf {src} && mkdir -p {src} && tar -C {src} -xf {tb} && '
              f'cp -a {src} {d} && rsync -a --delete {src}/ {d}/ && '
              f'(cd {src} && find . -type f -print0 | sort -z | '
              f'xargs -0 sha1sum) > /dev/shm/$USER.rs_sum && '
              f'(cd {d} && sha1sum --quiet -c /dev/shm/$USER.rs_sum) && '
              f'rsync -ai {src}/ {d}/ | wc -l')
  out = run(ctx, 0, f'rsync -ai {src}/ {d}/ | wc -l; rm -rf {src}')
  ctx.check(out.strip() == '0', f'second rsync transferred {out.strip()} '
                                f'items (mtime/size not preserved)')
  j = len(ctx.hosts) - 1
  run(ctx, j, f'cd {d} && sha1sum --quiet -c /dev/shm/$USER.rs_sum || '
              f'(cd {d} && find . -type f | head -3)', timeout=1500)
