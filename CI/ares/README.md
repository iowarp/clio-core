# CI/ares: reliability and tiering-performance suite for the Ares cluster

One command runs clio-core on real Ares nodes, breaks it on purpose (killed
daemons, lost nodes, dead / replaced / newly plugged-in drives), measures the
storage tiers, and writes a report that says what failed and what got slower
than the last run you compare against. Run it before and after major updates.

It is manual for now (no cron, no self-hosted runner).

## Run it

```bash
# 1. Build (any Release build with tests; the suite snapshots bin/).
cmake --preset release && cmake --build build -j

# 2. Run. Outside an allocation it submits itself (sbatch --wait) and waits.
CI/ares/run.sh --build build --suite all --nodes 6

# 3. Next time, compare against that run:
CI/ares/run.sh --build build --suite all --nodes 6 \
    --baseline ~/clio_ares_runs/<previous run dir>
```

| option | default | meaning |
|---|---|---|
| `--suite` | `all` | `reliability`, `tiermix`, `perf`, `all`, or `smoke` (about 30 min; checks the harness) |
| `--nodes` | 6 | nodes to allocate (the evaluation target is 6 or more; several tests need 3-4) |
| `--tier-budget-mb` | 16384 | per-node capacity split across DRAM/NVMe/SSD/HDD by the `tiermix` legs |
| `--baseline` | none | earlier run dir; enables regression detection |
| `--threshold` | 0.20 | a metric worse by more than this fraction is a regression |
| `--out` | `~/clio_ares_runs/<date>-<sha>-<suite>` | results directory |
| `--partition`, `--time` | `compute`, `24:00:00` | SLURM settings |

Inside an existing `salloc`, the same command runs on the allocation directly.

Exit status 0 means every test passed and nothing regressed. Open
`<out>/report.md` first; per-leg detail is in `<out>/<leg>/summary.md`, daemon
logs in `<out>/<leg>/logs/`.

## What it covers

Each leg is a fresh deployment of `clio_run` + CTE + replication + cache +
clio-fs (FUSE) on every node, driven by
`context-transfer-engine/test/integration/clio_fs_suite`.

### Reliability (`--suite reliability`)

| leg | profile | what breaks / what is checked |
|---|---|---|
| `ctest_components` | n/a | CTE core, cache, replication, safe_bdev, clio-fs unit/integration ctests (first node) |
| `posix_dist` | persistent | POSIX semantics, cross-node coherence and races, directories, real apps (tar/git/make/sqlite/rsync) |
| `node_faults` | persistent | the `repl` group (below), graceful and crash restarts, double crash, node loss with bounded survivors, node killed mid-I/O, FUSE client crash, chaos cycles, page owner down, hand-off across successor restart, truncate during failover, open handles across home loss |
| `stress_tiers` | tiered | tier overflow with the organizer migrating, node loss during writes, cache coherence under pressure |
| `safe_bdev` | safe | disks dying during writes, disk dies then crash, disk comes back, disk replaced and rebuilt (RecoverBdev), rebuild interrupted by a crash, disk swapped while the node is down, parity replaced with data down, two nodes down never lies, beyond tolerance never returns wrong bytes, partition during writes, rolling restarts degraded, chaos of disks and nodes, cache coherent after (partial) crash, **new data + parity drives plugged into a running array** (`safe_new_drive_plugged_in`) |

Every fault test writes self-verifying records and checks that every fsynced
version reads back intact. Past the tolerance, reads may fail (EIO) but must
never return wrong bytes. Result classes: PASS, FAIL (wrong answer), HANG
(an op exceeded its deadline), ERROR (harness), UNSUP (documented limit).

Drive failure is injected per member with the file bdev's `<file>.fail`
marker (every I/O fails, as on a dead drive). Replacing a drive uses
`clio_safe_bdev_recover --failed`; plugging in a new one uses
`clio_safe_bdev_recover --add data|parity`.

### Aimed node faults for replication and cache (group `repl`)

`placement.py` is the Python twin of the CTE's `BlobHash` (checked against
the C++ hash), so these tests know which node owns each 1 MiB page of a file
and which node holds its remote copy, and aim each fault at that role.

| test | fault |
|---|---|
| `repl_holder_outage_then_owner_loss` | the page's copy holder goes down, the page is overwritten, the holder returns; later the owner goes down. Reads must return the overwrite, never the holder's pre-outage copy |
| `repl_rolling_single_failures` | every node in turn goes down and returns while files are rewritten with fsync (a rolling upgrade); one failure at a time must lose nothing |
| `cache_holder_crash_misses_overwrite` | a node with cached pages crashes, the file is overwritten meanwhile; after restart it must read the new bytes |
| `cache_holder_partitioned_misses_overwrite` | a node with cached pages is partitioned (alive, cache intact) during an overwrite; after the heal it must converge to the new bytes within 30 s |

### Tier fractions (`--suite tiermix`)

Three `ares_tiers` legs split `--tier-budget-mb` per node across
DRAM:NVMe:SSD:HDD and run the `tier` studies plus the `stress` group (which
writes 1.5x the DRAM + NVMe capacity, forcing data down to SSD/HDD):

| leg | DRAM:NVMe:SSD:HDD | also |
|---|---|---|
| `tiermix_dram` | 40:30:20:10 | |
| `tiermix_balanced` | 25:25:25:25 | `repl` node faults |
| `tiermix_hdd` | 5:10:15:70 | `repl` node faults |

### Tiering performance (`--suite perf`)

Ares compute nodes have four device tiers: RAM, NVMe (`/mnt/nvme`), SATA SSD
(`/mnt/ssd`) and HDD (`/mnt/hdd`). Profile `ares_tiers` makes each one a
separate CTE target on its own device. The `tier_default` leg runs the same
studies on the suite's default layout (RAM + one NVMe tier) as the comparison
point.

| test | measures |
|---|---|
| `tier_small_io` | 4 KiB sequential/random writes (fsynced) and reads: IOPS, MB/s, p99 latency |
| `tier_large_io` | 4 MiB sequential write (fsynced) and cold/hot read of a RAM-tier-sized file per node |
| `tier_spill_to_slow_tiers` | writing 1.5x (RAM + NVMe) per node, so data lands on SSD/HDD; cold read of oldest vs newest data; where the bytes went (`tier_usage`) |
| `tier_hot_set_under_streaming` | hot-file random-read latency while a cold stream pushes past the fast tiers (does the organizer keep the hot set up?) |
| `tier_producer_consumer` | node 0 publishes fsynced items, the last node waits for, reads and verifies them; small (64 KiB) and large (8 MiB) items: throughput and publish-to-read lag |
| `perf` group | mdtest/IOR-shaped metadata rates and file-per-process / shared-file bandwidth |

All bytes are pattern-verified, so a fast wrong answer fails.
`compare.py` judges metrics by name: `*_mb_s`, `*_iops`, `*_ops_s` higher is
better; `*_us`, `*_ms`, `*_s` lower is better.

Caveats:
- "Cold" reads drop the page cache only with passwordless `sudo`; otherwise
  the test notes that the read may have come from the FUSE page cache.
- Producer/consumer lag uses wall clocks on two nodes, so it assumes NTP sync.
- Performance on a shared cluster is noisy. Compare runs on the same node
  count, and treat a single regression near the threshold as a prompt to
  re-run, not a verdict.

## Files

| file | role |
|---|---|
| `run.sh` | allocation, legs, ctest leg, report |
| `compare.py` | failures and regressions vs a baseline, `report.md` |
