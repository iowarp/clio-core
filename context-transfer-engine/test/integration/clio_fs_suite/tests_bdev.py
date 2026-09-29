"""Storage-layer fault tests: the safe_bdev array under whole-node loss.

These drive C++ clients (built beside the runtime) against the deployed
cluster rather than the FUSE mount: the array's disks are file bdevs on
different nodes, and a node dying takes one of them with it.
"""

import os
import subprocess
import time

from cluster import parallel, sh
from suite import test


def _drive(ctx, binary, env_extra, on_wait):
  """Run a driver on node 0, answering its SBD_WAIT handshakes.

  The driver prints "SBD_WAIT <step>" and blocks until <sync>/<step>.go
  exists; `on_wait(step)` performs the step (e.g. kills a node) first.
  Returns (exit code, output tail).
  """
  cl = ctx.cl
  sync = os.path.join(os.path.dirname(cl.conf), 'sbd_sync')  # shared
  os.makedirs(sync, exist_ok=True)
  for f in os.listdir(sync):
    os.remove(os.path.join(sync, f))
  env = cl.env_prefix() + ' CLIO_WITH_RUNTIME=0 CLIO_SBD_SYNC=' + sync
  for k, v in env_extra.items():
    env += f' {k}={v}'
  cmd = f'{env} timeout 1500 {cl.bin_dir}/{binary}'
  p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', cl.hosts[0], cmd],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
  tail = []
  for line in p.stdout:
    tail = (tail + [line.rstrip()])[-60:]
    if line.startswith('SBD_WAIT '):
      step = line.split()[1]
      on_wait(step)
      open(os.path.join(sync, step + '.go'), 'w').close()
  p.wait()
  return p.returncode, '\n'.join(tail)


@test('safe_bdev_node_loss', 'bdev', min_nodes=4, redeploy_after=True,
      timeout=1800)
def t_safe_bdev_node_loss(ctx):
  """A safe_bdev array spread over four nodes (grown 1 -> 4 disks, two
  parity disks) loses a whole node: every byte still reads, overwrites land
  while it is gone, and its disk is rebuilt on a spare elsewhere."""
  cl = ctx.cl
  local = f'{cl.local_root}/data/sbd'
  parallel(lambda h: sh(h, f'rm -rf {local}; mkdir -p {local}'), cl.hosts)

  def on_wait(step):
    if step == 'kill3':
      victim = cl.hosts[3]
      cl.kill_fuse(victim)
      cl.kill_runtime(victim)
      time.sleep(25)  # past failure detection: the node is declared dead

  rc, tail = _drive(ctx, 'clio_safe_bdev_dist_stress node_loss',
                    {'CLIO_SBD_NODES': str(len(ctx.hosts)),
                     'CLIO_SAFE_STRESS_DIR': local}, on_wait)
  for line in tail.split('\n'):
    if 'stress [' in line or 'marked faulty' in line:
      ctx.note(line[-200:])
  ctx.check(rc == 0 and '[PASS]' in tail,
            f'driver failed rc={rc}: ' + tail[-1500:])


@test('safe_bdev_live_growth_rejoin', 'bdev', min_nodes=4, redeploy_after=True,
      timeout=1800)
def t_safe_bdev_live_growth_rejoin(ctx):
  """Writers keep going while a safe_bdev array grows from one node's disk
  to four nodes'; a node then dies, overwrites land degraded, and when the
  node is repaired and restarted its disk is rebuilt there."""
  cl = ctx.cl
  local = f'{cl.local_root}/data/sbd'
  parallel(lambda h: sh(h, f'rm -rf {local}; mkdir -p {local}'), cl.hosts)
  victim = cl.hosts[2]

  def on_wait(step):
    if step == 'kill2':
      cl.kill_fuse(victim)
      cl.kill_runtime(victim)
      time.sleep(25)
    elif step == 'restart2':
      cl.start_runtime(victim, 'restart')
      ctx.check(cl.runtime_up(victim), f'{victim} did not restart')
      time.sleep(15)  # rejoin: the node is declared alive again

  rc, tail = _drive(ctx, 'clio_safe_bdev_dist_stress live_growth',
                    {'CLIO_SBD_NODES': str(len(ctx.hosts)),
                     'CLIO_SAFE_STRESS_DIR': local}, on_wait)
  for line in tail.split('\n'):
    if 'stress [' in line:
      ctx.note(line[-200:])
  ctx.check(rc == 0 and '[PASS]' in tail,
            f'driver failed rc={rc}: ' + tail[-1500:])
