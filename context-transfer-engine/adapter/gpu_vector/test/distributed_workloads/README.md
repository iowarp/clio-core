# Distributed paged-workload harness

Runs a paged workload benchmark across a multi-container Clio cluster and
checks that it still computes the right answer.

```
BUILD_DIR=build-gv ./run_workloads_distributed.sh {kmeans|weights|gmx|grayscott|lbann|lammps_md|all}

GVW_NODES=4                     # 4 containers instead of 2
GVW_VARIANT=sycl                # the SYCL editions (needs their own build tree)
```

From ctest:

```
ctest -L gv_distributed    # paged CUDA, 2 nodes
ctest -L gv_dist4          # 4 nodes, and the baselines at 4 ranks
ctest -L gv_dist_sycl      # paged SYCL, 2 nodes
ctest -L distributed       # all of it -- 38 gates; ctest reports 40,
                           # the extra two being the cleanup fixtures it
                           # pulls in via FIXTURES_REQUIRED
```

The docker gates are opt-in:
`-DCLIO_CORE_ENABLE_DOCKER_TESTS=ON`, and the SYCL ones additionally need
`-DCLIO_GV_SYCL_BUILD_DIR=build-syclreg` because DPC++ and nvcc cannot share a
build tree.

## What this covers that the md harnesses do not

The other harnesses here are md-specific: four of them run
`clio_lammps_md_paged_bench`, and `workloads/` runs the external
LAMMPS/GROMACS/LBANN forks. Every other workload had no distributed harness at
all, so their `--nodes` support could be written and never exercised.

Own port (9425) and subnet (172.26.0.0/24), so it can be up alongside the md
harnesses.

## Why a distributed run of these was impossible before

Every paged bench used to write its own config and `Setenv`
`CLIO_SERVER_CONF` with `overwrite=1`, so a harness-supplied cluster config was
clobbered a line later and each node stood up its own single-host runtime on
the same port. Two of those collide with `Address already in use`, and the
survivor waits forever for a peer that never arrives. All six now leave an
already-set `CLIO_SERVER_CONF` alone.

## The gate is against a single-node reference, not against exit 0

Each workload runs single-node first and the distributed result is compared
against it. These benches report checksums whose entire purpose is to be
configuration-independent, so a run that merely exits 0 proves nothing: the
failure mode they actually have is a plausible, wrong number.

| workload | 2 nodes vs 1 | notes |
|-----------|--------------|-------|
| kmeans    | rel ~2e-09 | `--data-mb` is the GLOBAL problem, split; tolerance is for atomicAdd ordering |
| weights   | exact | integer accumulation commutes, so bit-equal at any node count |
| gmx       | bit-exact | fixed point: CONSERVATION, MESH and GATHER all have no tolerance |
| grayscott | rel 0 | 3D stencil, the only one exchanging a halo every step |
| lbann     | max abs 0 | elementwise vs the dense reference, not a digest |
| lammps_md | rel 0 | `E0 = -592121.595111`, the value `distributed_md_bench/` documents |

All six also pass at 4 nodes, and all six pass in their SYCL edition.

## checksum=OK alone is a vacuous gate -- weights needed a witness

Each node compares its own partial `got` against its own partial `want`, so a
reduction that silently did nothing still reports OK. The bench therefore also
prints `checksum_total`, the reduced whole-model number, and the harness fails
if it is unchanged from the single-node run.

Measured 2-node/1-node ratio: **2.00002**. That number is the proof --
`1.0` would mean no reduction happened, exactly `2.0` would mean both nodes
summed the same shard, and `2.00002` means they summed different adjacent
shards and combined them.

## grayscott has a negative control, because its gate has a tolerance

`GS_NO_HALO=1` forces the halo fetches back to generation 0. The harness runs
it after a pass and FAILS if the answer is unchanged: an exchange whose absence
changes nothing is not load-bearing, and a tolerant checksum would hide that.
With the demand: `36410.579344`. Without: `36104.119147`.

Contrast gmx, whose fixed-point gates are exact and fail loudly on their own --
that is why gmx caught a stale-page bug that grayscott's gate would have
absorbed.

## Traps this harness has already stepped in

- **libnvcomp is not in the deps image.** Containers died at exec with 127
  before any clio code ran. It is mounted, as the md harness does. Same for the
  DPC++ prefix (`DPCPP_HOME`) for the SYCL editions.
- **The cluster compose schema is not the one the benches write inline.** Pool
  ids are `"major.minor"` STRINGS -- a bare `pool_id: 512` is rejected at load
  with `Invalid UniqueId format` -- and a core pool declares `storage:`, not
  `tiers:`.
- **Neither node may leave while its peer is still paging.** The runtime dies
  with the process and a peer whose pages live on the departing node waits
  forever, so each node touches `.done_<id>` on the shared mount and waits.
- **An unset variable is not an absent one.** docker-compose passes
  `GS_NO_HALO=` through as an empty string, and `getenv` returns non-null for
  that, so a presence test enabled the negative control in EVERY run -- a
  correct build reported the control's wrong answer and looked like a
  regression. Flags test for non-empty.
- **Generation 0 means "any version will do".** It was the cause of three
  separate defects: grayscott's stale halo, lbann's stale peer W2, and lbann's
  own verification read, which reported a whole weight update as error while
  the vector held the right bytes. A fetch that does not name a generation
  accepts whatever is lying around.
- **A node's band must own whole pages.** Splitting off a page boundary makes
  two nodes write the same page and clobber each other -- silently for lbann's
  biases, and as a HANG at 4 nodes. lbann rejects such a geometry now.

## CI

No `add_test` for the underlying scripts themselves, matching the other
cluster harnesses -- the ctest entries above wrap them and carry the
environment. These need docker, the nvidia container toolkit and a GPU, which
is why they are opt-in rather than on by default.
