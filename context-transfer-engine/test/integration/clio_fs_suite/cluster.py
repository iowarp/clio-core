#!/usr/bin/env python3
"""Deploy / tear down / fault-inject a clio-fs cluster over ssh.

One clio_run daemon per node, the filesystem + CTE chain composed from the
server config on every node (so `clio_run start` and `clio_run restart`
bring the whole stack back with no separate compose step), and one
clio_cte_fuse mount per node at the same node-local path.

All state a node writes (storage tiers, metadata WAL, indexer log, memfd
links, logs of the daemons) lives under a node-local directory, never on
the shared NFS home: a shared WAL / memfd dir lets daemons on different
nodes clobber each other.
"""

import json
import os
import shlex
import subprocess
import threading
import time

SSH = ['env', '-u', 'LD_LIBRARY_PATH', 'ssh', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=no', '-o', 'LogLevel=ERROR',
       '-o', 'ConnectTimeout=10']


def sh(host, cmd, timeout=120, check=False):
  """Run cmd on host via ssh (locally when host is None)."""
  argv = SSH + [host, cmd] if host else ['bash', '-c', cmd]
  try:
    p = subprocess.run(argv, capture_output=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    out = p.stdout.decode(errors='replace') + p.stderr.decode(errors='replace')
    rc = p.returncode
  except subprocess.TimeoutExpired:
    out, rc = f'ssh timeout after {timeout}s: {cmd}', 124
  if check and rc != 0:
    raise RuntimeError(f'[{host}] rc={rc}: {cmd}\n{out[-3000:]}')
  return rc, out


def parallel(fn, items):
  """Run fn(item) for every item concurrently; return results in order."""
  res = [None] * len(items)

  def run(i, it):
    try:
      res[i] = fn(it)
    except Exception as e:  # pylint: disable=broad-except
      res[i] = e
  ths = [threading.Thread(target=run, args=(i, it))
         for i, it in enumerate(items)]
  for t in ths:
    t.start()
  for t in ths:
    t.join()
  return res


class AgentConn:
  """A JSON-lines connection to agent.py running on one node."""

  def __init__(self, host, agent_py, env_prefix=''):
    self.host = host
    cmd = f'{env_prefix} exec /usr/bin/python3 -u {agent_py}'
    self.p = subprocess.Popen(SSH + [host, cmd], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
    self.next_id = 1
    self.lock = threading.Lock()
    hello = self._readline(60)
    if hello is None:
      raise RuntimeError(f'agent on {host} did not start')
    self.pid = hello['ret']['pid']
    self.dead = False

  def _readline(self, timeout):
    box = {}

    def rd():
      box['l'] = self.p.stdout.readline()
    t = threading.Thread(target=rd, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or not box.get('l'):
      return None
    return json.loads(box['l'])

  def call(self, op, timeout=120, **args):
    """Invoke op on the node; returns the reply dict (never raises)."""
    with self.lock:
      if self.dead:
        return {'ok': False, 'err': 'agent dead', 'agent_dead': True}
      rid = self.next_id
      self.next_id += 1
      try:
        self.p.stdin.write((json.dumps({'id': rid, 'op': op, 'args': args,
                                        'timeout': timeout}) + '\n').encode())
        self.p.stdin.flush()
      except (BrokenPipeError, OSError):
        self.dead = True
        return {'ok': False, 'err': 'agent pipe broken', 'agent_dead': True}
      # The agent itself enforces `timeout`; allow slack for the reply.
      r = self._readline(timeout + 30)
      if r is None:
        self.dead = True
        return {'ok': False, 'err': 'agent unresponsive', 'agent_dead': True,
                'hang': True}
      return r

  def close(self):
    try:
      self.p.stdin.write(b'{"id":-1,"op":"exit"}\n')
      self.p.stdin.flush()
    except OSError:
      pass
    try:
      self.p.wait(5)
    except subprocess.TimeoutExpired:
      self.p.kill()


class Cluster:
  """Owns the daemons, mounts and agents of an N-node clio-fs deployment."""

  def __init__(self, hosts, bin_dir, run_dir, profile='persistent',
               port=9519, attr_cache_s=None, num_threads=8,
               ram_gb=8, disk_gb=40, local_root=None, net_suffix='-40g',
               extra_env=None, replicate_period_ms=0):
    self.hosts = list(hosts)
    self.bin_dir = bin_dir
    self.run_dir = run_dir            # shared (NFS): configs, logs, results
    self.profile = profile
    self.port = port
    self.attr_cache_s = attr_cache_s
    self.num_threads = num_threads
    self.ram_gb = ram_gb
    self.disk_gb = disk_gb
    self.net_suffix = net_suffix
    user = os.environ.get('USER', 'user')
    self.local_root = local_root or f'/mnt/nvme/{user}/clio_fs_suite'
    self.mnt = f'{self.local_root}/mnt'
    self.extra_env = extra_env or {}
    # 0 = synchronous write-through to the persistent replica (a put acks
    # only once its durable copy exists); >0 = async sweep every N ms.
    self.replicate_period_ms = replicate_period_ms
    self.agents = {}
    self.agent_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'agent.py')
    os.makedirs(run_dir, exist_ok=True)
    self.conf = os.path.join(run_dir, 'clio_server.yaml')
    self.hostfile = os.path.join(run_dir, 'hostfile')

  # -- configuration -------------------------------------------------------
  def env(self):
    """Environment every daemon/mount/agent on a node runs with."""
    e = {
        'CLIO_SERVER_CONF': self.conf,
        'CLIO_WITH_RUNTIME': '0',
        'CLIO_IPC_MODE': 'SHM',
        'CLIO_MEMFD_DIR': f'{self.local_root}/memfd',
        'CTP_LOG_LEVEL': os.environ.get('CLIO_SUITE_LOG_LEVEL', 'warning'),
        'PATH': f'{self.bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin',
        # The login shell's LD_LIBRARY_PATH (e.g. another build's bin dir)
        # would override the binaries' RUNPATH and load mismatched libs.
        'LD_LIBRARY_PATH': self.bin_dir,
    }
    if self.attr_cache_s is not None:
      e['CLIO_FUSE_ATTR_CACHE_S'] = str(self.attr_cache_s)
    e.update(self.extra_env)
    return e

  def env_prefix(self):
    return ' '.join(f'{k}={shlex.quote(v)}' for k, v in self.env().items())

  def write_config(self):
    """Generate the hostfile and the server config (with compose)."""
    with open(self.hostfile, 'w') as f:
      for h in self.hosts:
        f.write(f'{h}{self.net_suffix}\n')
    lr = self.local_root
    storage = [
        f'      - path: "ram::clio_fs_ram"\n'
        f'        bdev_type: "ram"\n'
        f'        capacity_limit: "{self.ram_gb}GB"\n'
        f'        score: 1.0\n']
    perf = ''
    chain = ''
    fs_next = '512.0'
    if self.profile == 'persistent':
      storage.append(
          f'      - path: "{lr}/data/cte_disk_tier.dat"\n'
          f'        bdev_type: "file"\n'
          f'        capacity_limit: "{self.disk_gb}GB"\n'
          f'        score: 0.2\n'
          f'        persistence_level: "temporary"\n')
      perf = (f'    performance:\n'
              f'      metadata_log_path: "{lr}/data/cte_metadata_log"\n'
              f'      transaction_log_capacity: "256MB"\n')
      chain = ('  - mod_name: clio_cte_replication\n'
               '    pool_name: clio_cte_replication\n'
               '    pool_query: local\n'
               '    pool_id: "561.0"\n'
               '    next_pool_id: "512.0"\n'
               '    num_replicas: 1\n'
               f'    replicate_period_ms: {self.replicate_period_ms}\n'
               '    cache_score: 1.0\n'
               '    replica_score: 0.2\n')
      fs_next = '561.0'
    cfg = f"""# Generated by clio_fs_suite/cluster.py -- profile {self.profile}
memory:
  main_segment_size: auto
  client_data_segment_size: 4GB
networking:
  port: {self.port}
  hostfile: {self.hostfile}
  wait_for_restart: 60
  wait_for_restart_poll_period: 1
runtime:
  num_threads: {self.num_threads}
  queue_depth: 1024
  conf_dir: {lr}/conf
compose:
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    storage:
{''.join(storage)}{perf}    dpe:
      dpe_type: "max_bw"
    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
{chain}  - mod_name: clio_cte_filesystem
    pool_name: clio_cte_filesystem
    pool_query: local
    pool_id: "560.0"
    next_pool_id: "{fs_next}"
"""
    with open(self.conf, 'w') as f:
      f.write(cfg)

  # -- per-node lifecycle --------------------------------------------------
  def log_path(self, host, what):
    d = os.path.join(self.run_dir, 'logs')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{host}.{what}.log')

  def wipe(self, host):
    """Remove all node-local state (storage, WAL, memfd links)."""
    self.force_unmount(host)
    sh(host, f'pkill -9 -u $USER -f "[c]lio_cte_fuse" ; '
             f'pkill -9 -u $USER -f "[c]lio_run" ; sleep 0.5; '
             f'rm -rf {self.local_root}/data {self.local_root}/memfd '
             f'{self.local_root}/conf; '
             f'mkdir -p {self.local_root}/data; {self.seal_mnt_cmd()}',
       timeout=60)

  def seal_mnt_cmd(self):
    """Shell snippet leaving an EMPTY raw mountpoint (only when nothing is
    mounted there; fusermount needs it writable, so it cannot be sealed). A test op that runs after its FUSE mount died
    must fail loudly -- otherwise it silently writes into the node-local
    directory underneath and later runs read that junk back as if it were
    clio-fs state."""
    m = self.mnt
    return (f'if ! grep -q " {m} " /proc/self/mountinfo; then '
            f'chmod -R u+w {m} 2>/dev/null; rm -rf {m}; mkdir -p {m}; fi')

  def start_runtime(self, host, mode='start'):
    """Launch `clio_run <mode>` (start|restart) detached on host."""
    log = self.log_path(host, 'runtime')
    # CLIO_SUITE_GDB=1 runs the daemon under gdb and dumps every thread's
    # stack into the runtime log if it crashes (silent SIGSEGV otherwise).
    cmd = (f'{self.env_prefix()} nohup {self._gdb("runtime")}'
           f'{self.bin_dir}/clio_run {mode} '
           f'--no-viz </dev/null >>{log} 2>&1 &')
    sh(host, f'echo "=== {time.ctime()} clio_run {mode}" >> {log}; {cmd}')

  def gdb_prefix(self):
    """Command prefix running a daemon under gdb when CLIO_SUITE_GDB=1: a
    crash (or a SIGUSR2 sent to a hung daemon) then dumps every thread's
    stack into the daemon's log instead of dying silently."""
    return self._gdb('1')

  def _gdb(self, which):
    want = os.environ.get('CLIO_SUITE_GDB', '')
    if want not in ('1', which):
      return ''
    return ('gdb -q -batch -nx -ex "set startup-with-shell off" '
            '-ex "handle SIGUSR1 nostop noprint pass" '
            '-ex "handle SIGPIPE nostop noprint pass" '
            '-ex run -ex "thread apply all bt 25" --args ')

  def runtime_up(self, host, timeout=180):
    """Wait until the daemon on host listens on its port."""
    t0 = time.time()
    while time.time() - t0 < timeout:
      rc, _ = sh(host, f'ss -ltn | grep -q ":{self.port} "', timeout=20)
      if rc == 0:
        return True
      time.sleep(1)
    return False

  def runtime_pid(self, host):
    rc, out = sh(host, 'pgrep -u $USER -f "^[^ ]*[c]lio_run (start|restart)"')
    return [int(x) for x in out.split()] if rc == 0 else []

  def stop_runtime(self, host, timeout=60):
    """Graceful `clio_run stop`; falls back to SIGKILL after timeout."""
    sh(host, f'{self.env_prefix()} timeout {timeout} '
             f'{self.bin_dir}/clio_run stop', timeout=timeout + 15)
    t0 = time.time()
    while time.time() - t0 < timeout and self.runtime_pid(host):
      time.sleep(1)
    if self.runtime_pid(host):
      self.kill_runtime(host)
      return False
    return True

  def kill_runtime(self, host, sig='KILL'):
    sh(host, f'pkill -{sig} -u $USER -f "[c]lio_run (start|restart)"')

  def mount(self, host, timeout=90):
    """Start clio_cte_fuse on host and wait for a usable mount."""
    log = self.log_path(host, 'fuse')
    # Any mount still at the mountpoint is a stale one (dead daemon): the
    # setuid fusermount3 then fails with EACCES on it. Clear it first.
    for _ in range(10):
      rc, _ = sh(host, f'grep -q " {self.mnt} " /proc/self/mountinfo')
      if rc != 0:
        break
      self.force_unmount(host)
      time.sleep(0.5)
    sh(host, f'{self.seal_mnt_cmd()}; echo "=== {time.ctime()} mount" >> {log};'
             f' {self.env_prefix()} nohup {self._gdb("fuse")}'
             f'{self.bin_dir}/clio_cte_fuse '
             f'{self.mnt} -f </dev/null >>{log} 2>&1 &')
    t0 = time.time()
    while time.time() - t0 < timeout:
      rc, _ = sh(host, f'grep -q " {self.mnt} " /proc/self/mountinfo && '
                       f'timeout 20 stat -f {self.mnt} >/dev/null',
                 timeout=40)
      if rc == 0:
        return True
      time.sleep(1)
    return False

  def fuse_pid(self, host):
    rc, out = sh(host, 'pgrep -u $USER -f "[c]lio_cte_fuse"')
    return [int(x) for x in out.split()] if rc == 0 else []

  def force_unmount(self, host):
    """Abort the FUSE connection (unsticks D-state callers), lazy-unmount."""
    script = (
        f'for m in $(awk \'$5=="{self.mnt}" {{split($3,a,":"); print a[2]}}\' '
        f'/proc/self/mountinfo); do echo 1 > /sys/fs/fuse/connections/$m/abort'
        f' 2>/dev/null; done; fusermount3 -u -z {self.mnt} 2>/dev/null; true')
    sh(host, script, timeout=30)

  def unmount(self, host):
    """Clean unmount, then make sure the fuse process is gone."""
    sh(host, f'timeout 30 fusermount3 -u {self.mnt}', timeout=45)
    t0 = time.time()
    while time.time() - t0 < 20 and self.fuse_pid(host):
      time.sleep(0.5)
    if self.fuse_pid(host):
      self.force_unmount(host)
      sh(host, 'pkill -9 -u $USER -f "[c]lio_cte_fuse"')

  def kill_fuse(self, host):
    sh(host, 'pkill -9 -u $USER -f "[c]lio_cte_fuse"')
    self.force_unmount(host)

  # -- agents --------------------------------------------------------------
  def agent(self, host):
    a = self.agents.get(host)
    if a is None or a.dead:
      if a is not None:
        a.close()
      a = AgentConn(host, self.agent_py, self.env_prefix())
      self.agents[host] = a
    return a

  def close_agents(self):
    for a in self.agents.values():
      a.close()
    self.agents = {}

  # -- whole-cluster -------------------------------------------------------
  def up(self, wipe=True, mode='start'):
    """Bring every node up; return (ok, message)."""
    self.write_config()
    if wipe:
      parallel(self.wipe, self.hosts)
    parallel(lambda h: self.start_runtime(h, mode), self.hosts)
    ups = parallel(self.runtime_up, self.hosts)
    bad = [h for h, ok in zip(self.hosts, ups) if ok is not True]
    if bad:
      return False, f'runtime did not come up on {bad}'
    # Compose happens inside start; give the pools a moment on every node.
    time.sleep(3)
    mounts = parallel(self.mount, self.hosts)
    bad = [h for h, ok in zip(self.hosts, mounts) if ok is not True]
    if bad:
      return False, f'mount failed on {bad}'
    return True, 'ok'

  def down(self, graceful=True):
    self.close_agents()
    parallel(self.unmount, self.hosts)
    if graceful:
      parallel(self.stop_runtime, self.hosts)
    else:
      parallel(self.kill_runtime, self.hosts)
    parallel(lambda h: sh(h, 'pkill -9 -u $USER -f "[c]lio_run"'), self.hosts)

  def health(self):
    """Return {host: problem} for nodes whose mount or daemon is not OK."""
    def chk(h):
      if not self.runtime_pid(h):
        return 'runtime not running'
      if not self.fuse_pid(h):
        return 'fuse not running'
      rc, out = sh(h, f'timeout 15 stat -f {self.mnt} >/dev/null && '
                      f'timeout 15 ls {self.mnt} >/dev/null', timeout=40)
      return None if rc == 0 else f'mount unusable rc={rc} {out[-200:]}'
    res = parallel(chk, self.hosts)
    return {h: r for h, r in zip(self.hosts, res) if r}
