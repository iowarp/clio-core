# Safe-BDev interactive integration test

A single-node, fully containerized demo of safe-bdev erasure-coding recovery,
driven from the context-visualizer dashboard.

## What it launches

Inside one container (`node_entrypoint.sh`):

1. **clio runtime** composing (`clio_conf.yaml`):
   - 3 file-backed block devices, 256 MB each (`/mnt/bdev0..2.dat`) — the
     safe-bdev **data** members
   - **safe0** — the safe-bdev, `max_failures: 1`
   - **cte_main** — a CTE bound to `safe0`
   - **clio_fs** — a POSIX filesystem over the CTE
2. a 4th 256 MB device added as **parity** (`safe_bdev_add_bdev`) → 3 data + 1
   parity = tolerates **1 drive failure** using exactly 4 devices
3. the **clio-fs FUSE mount** at `/mnt/clio_fs`
4. the **context-visualizer** dashboard on **:5000**

## Run it (interactive)

```bash
cd context-runtime/test/integration/safe-bdev
./run_tests.sh
```

Then open **http://localhost:5000/safe_bdev** and:

1. Write data through the filesystem (in the container or via the mount):
   `dd if=/dev/urandom of=/mnt/clio_fs/f bs=1M count=64`
2. On the dashboard, click **Remove** on a data member (e.g. `302.0`).
3. Click **Replace + recover** and give a path (e.g. `/mnt/bdev_new.dat`).
4. Watch the **Recovery** panel: **ops in flight** vs **remaining**, and the
   progress bar climb to 100% as shards are rebuilt onto the replacement.

Tear down: `./run_tests.sh -c`

## Run it (automated smoke / CI)

```bash
MODE=smoke ./run_tests.sh
```

Brings the stack up, writes 32 MB through clio-fs, scripts a member
remove+replace via the dashboard API, asserts the recovery counters advanced
(`recovery_ops_completed == recovery_ops_total > 0`), verifies the data is
still readable, then tears down. This is the form registered with ctest as
`cr_safe_bdev_integration_docker` (under `CLIO_CORE_ENABLE_DOCKER_CI`).

## Requirements

- Docker with `--cap-add SYS_ADMIN` and `/dev/fuse`
- A build with `-DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON` (provides `clio_cte_fuse`)
  and the `clio_runtime_ext` python module (dashboard controls)
- A container image carrying `fuse3` + `flask` (default
  `iowarp/cte-xfstests:latest`; override with `IOWARP_DOCKER_IMAGE`)
