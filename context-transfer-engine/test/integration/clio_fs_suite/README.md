# clio-fs distributed POSIX / scale / fault suite

Black-box tests of clio-fs through the FUSE adapter (`clio_cte_fuse`), on 1..N
nodes, with real daemons and real crashes. The goal is evidence that clio-fs
behaves as a general-purpose filesystem, as a shared namespace across nodes,
and that it survives daemon crashes, node loss and restarts.

## Layout

| file | role |
|---|---|
| `cluster.py` | deploys `clio_run` + CTE chain + FUSE mount per node over ssh; kill / restart / force-unmount helpers |
| `agent.py` | per-node POSIX agent (JSON over ssh stdin/stdout); every op has a deadline and a HANG is reported instead of wedging |
| `suite.py` | orchestrator, test registry, report (`results.jsonl`, `summary.md`) |
| `tests_posix.py` | single-node POSIX semantics (27 tests) |
| `tests_dist.py` | cross-node coherence, races, shared files, storms (14 tests) |
| `tests_apps.py` | tar / git / make / sqlite / rsync on the mount |
| `tests_perf.py` | mdtest/IOR-shaped scaling measurements |
| `tests_fault.py` | crash + restart, node loss (partial availability of the hash-sharded namespace), FUSE crash, chaos |
| `fsx_model.py` | fsx-style model checker (single node or rotating across nodes) |

## Running

Inside a SLURM allocation (hosts come from `SLURM_JOB_NODELIST`), or with
`--hosts a,b,c`:

```bash
python3 suite.py --bin <build>/bin --out ~/clio_fs_suite/run1 \
    --groups posix,dist,apps,perf,fault [--nodes 8] [--only t1,t2]
```

* The binaries are snapshotted into `<out>/bin`, so rebuilding during a run
  cannot mix builds.
* Node-local state lives in `/mnt/nvme/$USER/clio_fs_suite` (storage tiers,
  metadata WAL, memfd links, mountpoint). Never on shared NFS.
* `--profile persistent` (default): RAM tier + file tier + metadata WAL +
  replication chain. `--profile ram`: RAM only (no restart guarantees).
* `--attr-cache S` sets `CLIO_FUSE_ATTR_CACHE_S` (0 = strict coherence).
* `CLIO_SUITE_GDB=1|runtime|fuse` runs the daemons under gdb: a crash (or a
  `SIGUSR2` sent to a hung daemon) dumps every thread's stack into
  `<out>/logs/<host>.<runtime|fuse>.log`.

## Result classes

`PASS`, `FAIL` (wrong answer), `HANG` (an op exceeded its deadline; the FUSE
connection is aborted via `/sys/fs/fuse/connections/*/abort` so nothing is
left in D state, then the cluster is redeployed), `ERROR` (harness), `UNSUP`
(documented limitation).

## xfstests

`scripts/xfstests/run_generic_sweep.sh` runs the generic group against the
single-node FUSE mount; see `scripts/xfstests/` for baselines.
