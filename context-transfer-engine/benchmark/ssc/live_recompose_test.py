"""Does CLIO's live recomposition work? (clio_run compose start/stop on a running runtime)

Starts the runtime with core only, then without restarting it:
  1. compose start  compressor(lz4) -> core      put/get a text blob through the compressor
  2. compose rm     the compressor
  3. compose start  indexer -> core               put/get through the indexer
  4. compose rm     the indexer, then put/get through core again
Each step checks the data round-trips and times the compose command.
"""

import os
import subprocess
import sys
import time

import compose
import lifecycle

BIN = lifecycle.BIN
ROOT = lifecycle.ROOT


def interceptor_yaml(entries):
    """Compose file holding only interceptor pools (core keeps running)."""
    return "compose:\n" + "".join(entries)


COMPRESSOR = f"""  - mod_name: clio_cte_compressor
    pool_name: clio_cte_compressor
    pool_query: local
    pool_id: "{compose.COMPRESSOR}"
    next_pool_id: "{compose.CORE}"
    tracking_enabled: false
    default_compress: "lz4"
    default_preset: "fast"
"""
INDEXER = f"""  - mod_name: clio_cte_indexer
    pool_name: clio_cte_indexer
    pool_query: local
    pool_id: "{compose.INDEXER}"
    next_pool_id: "{compose.CORE}"
    index_log_path: "{ROOT}/cte_indexer_index"
"""

CLIENT = r"""
import os, sys, clio_cte_core_ext as cte
assert cte.clio_init(cte.RuntimeMode.kClient, False)
assert cte.initialize_cte(os.environ["CLIO_SERVER_CONF"], cte.PoolQuery.Dynamic())
client = cte.get_cte_client()
tag = cte.Tag(sys.argv[1])
data = (b"2026-11-05 INFO request served in 12 ms by worker 7\n" * 20000)[:1 << 20]
tag.PutBlob("b0", data)
back = bytes(tag.GetBlob("b0", len(data)))
client.AsyncStatTargets().wait()
used = {}
for n in client.AsyncListTargets().result():
    used[n] = client.AsyncGetTargetInfo(n).result()
stored = sum(i.bytes_written for i in used.values())
print("ROUNDTRIP", back == data, "len", len(back), "tier bytes_written so far", stored)
"""


def run(cmd, env):
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return r, time.time() - t0


def client(pool, tag):
    env = lifecycle.env_for(compose.DEFAULT)
    env["CLIO_CTE_POOL"] = pool
    r = subprocess.run([sys.executable, "-c", CLIENT, tag], env=env, capture_output=True, text=True, timeout=120)
    out = [l for l in r.stdout.splitlines() if l.startswith("ROUNDTRIP")]
    return (out[-1] if out else f"FAILED rc={r.returncode}: {r.stderr[-400:]}")


REREAD = r"""
import os, sys, clio_cte_core_ext as cte
assert cte.clio_init(cte.RuntimeMode.kClient, False)
assert cte.initialize_cte(os.environ["CLIO_SERVER_CONF"], cte.PoolQuery.Dynamic())
data = (b"2026-11-05 INFO request served in 12 ms by worker 7\n" * 20000)[:1 << 20]
try:
    back = bytes(cte.Tag(sys.argv[1]).GetBlob("b0", len(data)))
    print("REREAD", back == data, "first bytes", back[:24])
except Exception as e:
    print("REREAD error", str(e)[:200])
"""


def reread(tag):
    env = lifecycle.env_for(compose.DEFAULT)
    r = subprocess.run([sys.executable, "-c", REREAD, tag], env=env, capture_output=True, text=True, timeout=120)
    out = [l for l in r.stdout.splitlines() if l.startswith("REREAD")]
    return out[-1] if out else f"FAILED rc={r.returncode}: {r.stderr[-300:]}"


def main():
    subprocess.run(["rm", "-rf", ROOT])
    proc, t_start = lifecycle.start(compose.DEFAULT)
    env = lifecycle.env_for(compose.DEFAULT)
    clio = os.path.join(BIN, "clio_run")
    print(f"runtime started (core only) in {t_start:.2f} s")
    try:
        print("core only:", client(compose.CORE, "t_core"))
        for name, entry, pool in (("compressor", COMPRESSOR, compose.COMPRESSOR),
                                  ("indexer", INDEXER, compose.INDEXER)):
            path = os.path.join(ROOT, f"{name}.yaml")
            open(path, "w").write(interceptor_yaml([entry]))
            r, dt = run([clio, "compose", "start", path], env)
            print(f"compose start {name}: rc={r.returncode} {dt:.2f} s  {r.stdout.strip()[-200:]} {r.stderr.strip()[-300:]}")
            r2, _ = run([clio, "compose", "list"], env)
            print("  list:", " | ".join(l.strip() for l in r2.stdout.splitlines() if l.strip())[-400:])
            print(f"  via {name}:", client(pool, f"t_{name}"))
            r, dt = run([clio, "compose", "rm", path], env)
            print(f"compose rm {name}: rc={r.returncode} {dt:.2f} s  {r.stderr.strip()[-300:]}")
            print(f"  read {name}'s blob via core after rm:", reread(f"t_{name}"))
        print("core again:", client(compose.CORE, "t_core2"))
        print("runtime still the same process:", proc.poll() is None)
    finally:
        lifecycle.stop(proc, compose.DEFAULT)


if __name__ == "__main__":
    main()
