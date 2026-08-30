# Distributed paged-workload harness

Runs one paged workload benchmark across a 2-node Clio cluster in docker and
checks that it still computes the right answer.

```
BUILD_DIR=build-gv ./run_workloads_distributed.sh {kmeans|weights|all}
```

## What this covers that the md harnesses do not

The other five gpu_vector harnesses are all md: four run
`clio_lammps_md_paged_bench`, and `workloads/` runs the external
LAMMPS/GROMACS/LBANN forks. kmeans, weights, gmx, grayscott and lbann had no
distributed harness at all, so their `--nodes` support could be written but
never exercised. This one runs whichever bench `GVW_BENCH` names.

It has its own port (9425) and subnet (172.26.0.0/24), so it can be up at the
same time as the md harnesses.

## Why a distributed run of these was impossible before

Every paged bench used to write its own config and `Setenv` `CLIO_SERVER_CONF`
with `overwrite=1`, so a harness-supplied cluster config was clobbered a line
later and each node stood up its own single-host runtime on the same port. Two
of those on one network collide with `Address already in use`, and the
survivor waits forever for a peer that never arrives. All six now leave an
already-set `CLIO_SERVER_CONF` alone.

## The gate is against a single-node reference, not against exit 0

Each workload is run single-node first and the distributed result compared
against it. These benches all report a checksum whose entire purpose is to be
configuration-independent, so a 2-node run that merely exits 0 proves nothing:
the failure mode they actually have is a plausible, wrong number.

Measured:

| workload | 1 node | 2 nodes | gate |
|----------|--------|---------|------|
| kmeans   | `30719.999040` | `30719.999676` | rel `2.07e-08` vs 1e-4 tolerance |
| weights  | `checksum=OK`  | `checksum=OK`  | exact |

kmeans is the stronger of the two: `--data-mb` is the GLOBAL problem, split in
half, so the distributed run has to reproduce the single-node centroids over
the same point set. The tolerance is relative because atomicAdd makes the
float summation order depend on the page and block layout; this bench
documents ~1e-8 spread across configurations, so an equality test would fail
correct runs.

## checksum=OK alone is a vacuous gate, and weights needed a witness

Each node compares its own partial `got` against its own partial `want`, so a
reduction that silently did nothing still reports OK. The bench therefore also
prints `checksum_total` -- the reduced, whole-model number -- and the harness
fails if it is unchanged from the single-node run.

The measured 2-node/1-node ratio is **2.00002**. That number is the proof:

- `1.0` would mean the reduction never happened
- exactly `2.0` would mean both nodes summed the *same* shard
- `2.00002` means they summed *different, adjacent* shards and combined them

Both nodes report the same total, which is the ascending-order reduction in
`bench_dist.h` holding up -- accumulating from the local partial and then
peers `0,1,2...` makes each node sum in a different order, and float addition
is not associative.

## Traps this harness already stepped in

- **libnvcomp is not in the deps image.** The containers died at exec with
  exit 127 before any clio code ran. It is mounted, as the md harness does.
- **The cluster compose schema is not the one the benches write inline.** Pool
  ids are `"major.minor"` STRINGS -- a bare `pool_id: 512` is rejected at load
  with `Invalid UniqueId format` -- and a core pool declares `storage:`, not
  `tiers:`.
- **Neither node may leave while its peer is still paging.** The runtime dies
  with the process and a peer whose pages live on the departing node waits
  forever, so each node touches `.done_<id>` on the shared mount and waits for
  its peer rather than sleeping a fixed time.

## CI

There is deliberately no `add_test` here, matching the other cluster
harnesses: this needs a GPU and a cluster, and GitHub-hosted runners have
neither. CI parses the script and the compose file
(`gpu-tests.yml`, `validate-gpu-vector-harnesses`); the run itself is a script
you invoke on a GPU host. A job that could never run would read as coverage
that does not exist.
