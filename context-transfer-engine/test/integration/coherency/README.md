# CTE multi-node coherency test (Docker, 4 nodes)

Verifies **memory coherency** of the distributed CTE blob store across a 4-node
cluster for three workloads, with **every node acting as a client** — coherency
only means something when *different nodes touch the same blobs*.

| workload | what it asserts |
|----------|-----------------|
| **write-only** | 4 nodes write 8 disjoint blobs each, concurrently; every node then reads back **all 32** → no lost or cross-clobbered writes |
| **read-only** | node 0 writes 16 blobs; all 4 nodes read the **same** blobs → identical content, no stale or torn reads on non-writer nodes |
| **append-only** | all 4 nodes write disjoint slices into **one shared blob** at increasing offsets; every node verifies all slices → concurrent partial writes assemble instead of clobbering |

Blobs hash-route by `(tag_id, blob_name)` across the pool's containers, so a blob
written on one node generally *lives* on another — that cross-node round trip is
what is being tested.

## Run

```bash
cmake --build build --target clio_run test_cte_coherency   # binaries first
context-transfer-engine/test/integration/coherency/run_tests.sh
```

or through ctest:

```bash
ctest -R cte_coherency_integration_docker
```

Requires docker + docker compose. Useful knobs:

| variable | meaning |
|----------|---------|
| `HOST_WORKSPACE` | repo path mounted at `/workspace` (auto-detected) |
| `IOWARP_DOCKER_IMAGE` | image to run (default `iowarp/deps-cpu:latest`) |
| `KEEP_UP=1` | leave containers up afterwards for inspection |

## How it works

Each container starts a `clio_run` daemon, waits at a **ready barrier** until all
4 daemons are up, runs `test_cte_coherency` as a client, then waits at a **done
barrier** before shutting its daemon down — a node must not disappear while a
peer is still reading blobs that hash-routed to it. Barriers are files on the
shared `coh-rundir` volume (`CTE_RUNDIR`), which `run_tests.sh` deletes between
runs so stale barriers cannot make a run appear to pass.

The runner aggregates **all four** nodes: it fails if any node exits non-zero
**or** reports fewer than 3 passing coherency cases, so a run where nothing
executed is a failure rather than a silent green.

## Relationship to `distributed_slurm/`

`test/integration/distributed_slurm/` runs the same `test_cte_coherency` binary
on a real cluster via Slurm + Apptainer (and additionally covers the GPU
compressed-vector cases, which need real GPUs). That harness is portable only to
Slurm sites. **This suite is the portable one** — it exercises the same coherency
logic anywhere Docker runs, including a laptop.

## Related

`../distributed/` runs the stock CTE distributed test with a client on **one**
node only; this suite differs by making every node a client.
