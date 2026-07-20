# CTE Memory-Coherency Integration Test (4-node Docker)

Runs `test_cte_coherency` (unit source: `test/unit/test_cte_coherency.cc`) as a
**client on all four nodes** of a Docker Compose cluster — different nodes
touching the same blobs, which is what coherency actually means. Blobs
hash-route by `(tag_id, blob_name)`, so a blob written on one node generally
lives on another and must read back identically cross-node.

Three coherency cases (12 assertions across 4 nodes): **write-only** (concurrent
disjoint writes globally visible), **read-only** (all nodes read one dataset
identically), **append-only** (concurrent disjoint slices assemble into one
blob).

## Differs from `test/integration/distributed`

| | distributed | coherency (this suite) |
|---|---|---|
| test client | node1 only | **all 4 nodes** |
| binary | `test_core_functionality` | `test_cte_coherency` |
| cross-node barriers | n/a | shared Docker volume `coh_rundir` at `CTE_RUNDIR` |
| subnet | 172.28.0.0/24 | **172.31.0.0/24** |

## Per-node sequence (`node_entrypoint.sh`)

1. start `clio_run` daemon (background)
2. wait for the daemon's **ChiMod scan** to finish — gate on
   `Loaded ChiMod: clio_cte_core`, not merely `Main server started` (the scan
   that registers `clio_bdev`/`clio_cte_core` runs *after* the server-started
   marker; gating too early lets cross-node pool creation begin before peers
   have loaded `clio_bdev`)
3. **READY barrier** — no further step until all 4 daemons are up and scanned
4. rank 0 creates pool 512.0 via `clio_run compose start` (idempotent), then a
   **COMPOSED barrier** — see "Pool creation" below
5. run `test_cte_coherency` as a client (`CLIO_WITH_RUNTIME=0`, `CTE_RANK`=node-1)
6. **TESTDONE barrier** — a node must not tear down its daemon while peers may
   still be reading blobs that hash-route to it
7. stop the daemon; exit with the test's return code

## Pool creation (why the explicit `compose start`)

`test_cte_coherency` *assumes* the CTE core pool (512.0) already exists (it only
does `GetOrCreateTag`/`PutBlob`/`GetBlob`), unlike `test_core_functionality`
which creates the pool in its own body. Relying on the daemon-startup compose is
racy in a cold 4-node cluster: a node can receive cross-node
`GetOrCreatePool(clio_bdev)` broadcasts *before* its own ChiMod scan has loaded
`clio_bdev`, so the pool silently fails to form. Rank 0 therefore composes the
pool explicitly *after* the READY barrier guarantees every peer has loaded
`clio_bdev`; the COMPOSED barrier makes all ranks wait for the pool before they
connect. Idempotent — returns the existing pool for a single-node run.

## Guards (why this can't fake a pass)

- **Shared volume wiped every run** (`docker compose down -v`): stale
  `READY`/`TESTDONE`/`bar_*` files would let a node skip a barrier and fake a pass.
- **Vacuous run ≠ pass**: the binary exits 0 if a filter matches no cases (prints
  `Passed: 0`). `run_tests.sh` asserts `Total tests: 3 / Passed: 3 / Failed: 0`
  **per node**, not merely exit 0.
- **All four aggregated**: any node failing fails the suite (a single
  `--abort-on-container-exit` code is not enough).
- **Done barrier, not just ready**: peers keep serving until every client finishes.

## Run

```bash
# Prerequisite (Linux binaries, e.g. inside the deps container):
cmake --build build --target clio_run test_cte_coherency

# Then:
ctest -R cte_coherency_integration_docker      # via CTest (needs CLIO_CORE_ENABLE_DOCKER_CI=ON)
./run_tests.sh                                 # or directly
./run_tests.sh --keep                          # keep containers for debugging
./run_tests.sh --cleanup-only                  # wipe a previous run
```

`HOST_WORKSPACE=/abs/path` overrides the workspace bind-mount (devcontainers /
out-of-tree builds). `IOWARP_DOCKER_IMAGE` overrides the image (auto-detected:
`deps-nvidia` if `clio_run` links libcudart, else `deps-cpu`).

## Note on measuring where data landed

If you ever extend this to check placement: the file bdev pre-allocates **sparse**
backing files, so apparent size (`du --apparent-size`, `find -printf '%s'`) reads
identically on every node whether or not data landed there. Count **allocated
blocks** instead (`find -printf '%b'`, 512-byte units).
