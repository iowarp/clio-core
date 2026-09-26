# Workload cluster tests

Runs LAMMPS, GROMACS and LBANN against a 4-node Clio cluster in docker and
checks that each still computes the right answer.

```
ETERNIA_BIN_DIR=<build root> ./run_workload_cluster.sh {lammps|gromacs|lbann|all}
```

`ETERNIA_BIN_DIR` holds `lmp-build/`, `gmx-build/`, `lbann-build/` and
`clio-inst/`. The workload binaries are not in this repo -- they are large
builds from the forks under `external/` -- so the harness mounts them read-only
at `/sp` rather than expecting them in `build/`.

## What this tests that the single-node harnesses do not

A vector's pages are CTE blobs, and blobs hash across the cluster at the blob
level. With three peers holding storage, most of a workload's page faults are
served from **another node's memory**. Anything in the paging path that works
only when the page is local fails here and nowhere else.

Each workload is checked against the same reference its single-node harness
uses -- so a cluster result that differs is a real difference, not a
differently-configured run.

## Topology, and why the workload is node 4

Nodes 1-3 run `eternia_cluster_node`; node 4 runs the workload.

Putting the workload on node 1 does not work, and the reason is worth keeping:
a workload's runtime does not start when its container starts, it starts when
the workload first reaches the eternia code path. That is seconds in for LAMMPS
and much later for GROMACS and LBANN, which do setup and data loading first. So
peers had nothing to join and hung; and when the workload was short, it finished
before they joined, created every blob locally, and the "cluster" run never left
the node -- passing while testing nothing.

Node 1 is therefore a peer, which listens immediately. Nodes 2 and 3 wait for
its port, and each writes a readiness marker once it is serving. Node 4 waits
for all three markers and **refuses to run** without them:

```
[node4] refusing to run: only 0 peer(s) ready, this would not be a cluster run
```

The runner checks the same thing independently from the log, because a harness
that silently degrades to single-node is the exact failure this exists to catch.

## Results

| workload | reference | cluster result |
|----------|-----------|----------------|
| LAMMPS  | E_pair = -4.785579 (stock `lj/cut`) | -4.7855792 |
| GROMACS | LJ(SR) = -8491.7693 (exact lattice sum) | -8491.76907 |
| LBANN   | objective 1.93652 / 1.87546 / 1.87988 (stock `El::Gemm`) | identical |

All three pass in one `./run_workload_cluster.sh all` invocation with the
defaults -- `3 passed, 0 failed` -- rather than only individually, which is the
claim that matters for reproducing it.

## The settle between runs is load-bearing

`ETERNIA_SETTLE` (default 45s) is not cosmetic. At 20s, cluster formation
failed five times in a row -- including with a trivial `echo` as the workload,
on an idle host with a clean GPU and no stale containers, networks or
processes. At 45s it formed first try. Back-to-back `docker compose` cycles are
where this breaks, so the settle is the difference between a working harness
and one that looks broken.

Formation is still retried (`ETERNIA_FORM_RETRIES`, default 3) because it
remains intermittent. Only FORMATION is retried: a workload that ran and
produced a wrong answer is reported as it stands, since retrying a bad result
until it passes is how a test suite becomes decorative.

A hypothesis that did NOT hold, recorded so it is not retried: that four nodes
composing an `hbm` tier -- four GPU contexts on one physical device -- was
causing the stalls. Removing the tier changed nothing, and the tier is back.

## Are all operations on eternia primitives?

Audited by looking for device allocations that bypass `gv::Vector`. The bulk
arrays are on eternia primitives; what is not, and why:

| workload | on eternia | resident, and why |
|----------|-----------|-------------------|
| LAMMPS  | `x`, `type`, `neigh`, `f` | `offset`/`ilist` index tables (4 B/atom each); per-block scratch; energy/virial/pair accumulators |
| GROMACS | `cj` (pair list), `xq` (coordinates) | `sci` (one entry per supercluster, ~1/64 of an atom array); **forces**; per-block scratch |
| LBANN   | `W`, `dW` | activations `X`, `C`, `dX`, `dC` -- mini-batch sized, and the weights are what dominate a wide layer |

Two of these are deliberate rather than unfinished:

**GROMACS forces cannot be paged as the kernel is decomposed today.** Sci
entries are handed to blocks round-robin (`for s = block; s < numSci; s +=
nblocks`), so the i-atoms a block writes are scattered across the whole array
and two blocks routinely touch the same page. Page caches are per block and
writeback granularity is a page, so paging the force array in that arrangement
would have each block flush its own copy of a shared page and silently lose the
other's contribution. Paging it requires giving each block a page-aligned range
of atoms and assigning it the sci entries that fall inside -- which is what
LAMMPS does, and why LAMMPS's `f` *is* paged.

**LBANN's activations are mini-batch sized**, not weight sized. Paging them
would add fault traffic without lifting the ceiling the weights set.

The remaining items -- per-block scratch, `O(1)` accumulators, and the small
index tables -- are either not proportional to the problem size or a few bytes
per atom. Paging them costs fault traffic for no capacity benefit.
