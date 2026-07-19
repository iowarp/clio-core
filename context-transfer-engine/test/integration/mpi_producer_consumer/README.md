# Multi-node MPI producer/consumer test (issue #714)

Dockerized 2-node **producer/consumer** reproducer for the cross-node blob-data
gap tracked in [#714](https://github.com/iowarp/clio-core/issues/714): after
#685 the CTE **tag namespace** is cluster-wide, but a blob's **payload** written
on node A is not retrievable from node B (a cross-node `GetBlob`/`read` returns
0 bytes). Two binaries cover both layers:

| binary | client | producer (rank 0) | consumer (rank 1) |
|---|---|---|---|
| `test_cte_producer_consumer` | CTE core | `PutBlob("0")` | `GetTagSize` / `GetBlobSize("0")` / **`GetBlob("0")`** |
| `test_cfs_producer_consumer` | CFS (filesystem) | `Open`+`Write`+`Close` | `Getattr(size)` / `Open`+**`Read`** |

**MPI** is used only for cross-rank coordination — `MPI_Barrier` orders
write-before-read and (CTE only) `MPI_Bcast` hands the tag id to the consumer.
The data path under test is CLIO's own cross-node routing, never MPI. Each rank
attaches to its **own local daemon** (`CLIO_INIT(kClient)`, `CLIO_WITH_RUNTIME=0`);
distributed memory = one rank per node in its own address space.

The consumer's cross-node `GetBlob`/`Read` is the authoritative assertion:
- **#714 unfixed:** returns 0 bytes → the MPI job exits non-zero → test FAILS
  (this is a reproducer).
- **#714 fixed:** returns the producer's bytes intact → test PASSES.

## Layout

- `test_cte_producer_consumer.cc` / `test_cfs_producer_consumer.cc` — the MPI binaries.
- `docker-compose.yaml` — two containers (`iowarp-node1`/`iowarp-node2`), one clio daemon each.
- `node1_entrypoint.sh` — sets up passwordless SSH, starts the daemon, runs `mpirun -np 2` across both nodes.
- `node2_entrypoint.sh` — starts the daemon and serves `sshd` so `mpirun` can start rank 1 here.
- `clio_config.yaml` — composes the CTE core (512.0) + CFS (560.0) pools on every node.
- `hostfile` — clio runtime hostfile; `mpi_hostfile` — OpenMPI hostfile (1 slot/node).
- `run_tests.sh` — brings the cluster up and keys the result on node 1 (the mpirun launcher).

## Running

Requires Docker + a build with `-DCLIO_CORE_ENABLE_MPI=ON`,
`-DCLIO_CTE_ENABLE_FILESYSTEM=ON`, and `-DCLIO_CORE_ENABLE_DOCKER_CI=ON` (so the
ctest is registered) whose binaries are in `build/bin/`:

```
ctest -R cte_mpi_producer_consumer_integration_docker
# or directly:
./run_tests.sh
```
