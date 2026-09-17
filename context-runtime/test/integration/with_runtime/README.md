# CLIO_WITH_RUNTIME=1 as a cache (issue #1015)

`CLIO_WITH_RUNTIME=1` means *"make sure a runtime is available"*, not *"be the
runtime"*. This suite is the multi-node gate on that behaviour.

## What it exercises

Two Docker nodes with **no clio daemon on either of them**, and `mpirun -np 8`
across both (4 ranks per node). Every rank calls
`CLIO_INIT(kClient, /*with_runtime=*/true)` at the same moment, with no
coordination beyond MPI. The assertions:

1. **init** — every rank's `CLIO_INIT` succeeds. Before the fix the losers of
   the race either failed to bind the runtime's ports or reaped the winner's
   memfd segments out from under it.
2. **one owner per node** — `IsRuntime()` is true on exactly one of each node's
   four ranks. Two runtimes on a port is the failure this issue exists to
   prevent; zero is the opposite failure. Both are caught.
3. **PutBlob / GetBlob** — every rank writes a rank-stamped blob under a shared
   tag, then reads back its own blob *and* its node-local predecessor's, so a
   rank that STARTED the runtime reads bytes a rank that ATTACHED wrote (and
   vice versa). Rank 0 additionally reads a blob written on the other node, so
   the two independently-started runtimes are proven to have formed one cluster.

The port-classification logic underneath (free / live runtime / foreign
listener, plus the start lock) has fast single-process coverage in
`context-runtime/test/unit/test_runtime_probe.cc`.

## Running it

Requires `CLIO_CORE_ENABLE_MPI=ON` (off by default) and
`CLIO_CORE_ENABLE_DOCKER_CI=ON`:

```bash
cmake --preset release -DCLIO_CORE_ENABLE_MPI=ON -DCLIO_CORE_ENABLE_DOCKER_CI=ON
cmake --build build -j"$(nproc)"
./context-runtime/test/integration/with_runtime/run_tests.sh
# or: ctest -R cr_with_runtime_cache_integration_docker
```

The result is node 1's exit code (it launches `mpirun`, which max-reduces every
rank's result onto rank 0). Non-zero codes: `2` a rank's `CLIO_INIT` failed,
`3` a node had != 1 runtime owner, `6`–`9` a PutBlob/GetBlob round trip failed.

## Layout

| file | role |
| --- | --- |
| `test_with_runtime_cache.cc` | the MPI binary carrying all the assertions |
| `docker-compose.yaml` | 2 daemonless nodes on a private network |
| `node1_entrypoint.sh` | sets up SSH, launches `mpirun -np 8` |
| `node2_entrypoint.sh` | serves sshd so mpirun can start ranks 4–7 |
| `mpi_hostfile` | `slots=4` per node — must match `kRanksPerNode` |
| `hostfile` | the clio cluster hostfile (2 nodes) |
| `clio_config.yaml` | port 8090 + the CTE core pool each runtime composes |
