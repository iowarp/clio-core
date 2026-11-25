"""Start and stop a single-node CLIO runtime for one composition (inside the container).

Recomposition is restart-based: CLIO has no in-place verb to rewire next_pool_id, so the
actuator writes a new compose YAML and restarts the runtime. time_recompose() reports
what that costs.
"""

import os
import subprocess
import time

import compose

BIN = os.environ.get("CLIO_BIN", "/build/bin")
ROOT = os.environ.get("SSC_ROOT", "/tmp/ssc_live")


def env_for(c):
    e = dict(os.environ)
    e["CLIO_SERVER_CONF"] = os.path.join(ROOT, "clio.yaml")
    e["CLIO_STORAGE_ROOT"] = ROOT
    e["CLIO_CTE_POOL"] = compose.chain_top(c)
    e["CLIO_REPO_PATH"] = BIN
    e["LD_LIBRARY_PATH"] = BIN + ":" + e.get("LD_LIBRARY_PATH", "")
    e["PYTHONPATH"] = BIN + ":" + e.get("PYTHONPATH", "")
    return e


def start(c, fresh=True, wait_s=60):
    os.makedirs(ROOT, exist_ok=True)
    tier_dir = os.environ.get("SSC_DISK_DIR")
    if fresh and tier_dir:
        # the second tier lives outside ROOT (e.g. on the parallel FS): a fresh start must not
        # inherit the previous runtime's tier file or allocation log
        import glob
        for f in glob.glob(os.path.join(tier_dir, "cte_disk_tier.dat*")):
            os.remove(f)
    with open(os.path.join(ROOT, "clio.yaml"), "w") as f:
        f.write(compose.render(c, ROOT))
    log = open(os.path.join(ROOT, "runtime.log"), "a")
    args = [os.path.join(BIN, "clio_run"), "start"] + (["--fresh"] if fresh else []) + ["--disk", ROOT]
    proc = subprocess.Popen(args, env=env_for(c), stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        r = subprocess.run([os.path.join(BIN, "clio_run"), "status"], env=env_for(c),
                           capture_output=True, text=True)
        if r.returncode == 0:
            return proc, time.time() - t0
        if proc.poll() is not None:
            raise RuntimeError(f"runtime exited with {proc.returncode}; see {ROOT}/runtime.log")
        time.sleep(0.5)
    raise TimeoutError("runtime did not become ready")


def stop(proc, c, wait_s=30):
    subprocess.run([os.path.join(BIN, "clio_run"), "stop"], env=env_for(c), capture_output=True)
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
