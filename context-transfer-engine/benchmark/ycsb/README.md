# YCSB benchmark integration (issue #862)

Benchmarks the CTE blob store with [YCSB](https://github.com/brianfrankcooper/YCSB)
and compares it against Redis, RocksDB, and DynamoDB (Local) on the core
workloads A (50/50 read/update), B (95/5), C (read-only), and F
(read-modify-write). Workloads D/E are excluded: the Clio binding has no
ordered scan (`scan()`/`delete()` return `NOT_IMPLEMENTED`).

## Pieces

| File | Role |
|------|------|
| `ClioClient.java` | YCSB `DB` adapter (`site.ycsb.db.ClioClient`); serializes the YCSB field map into one blob per key under tag `ycsb`, blob name `<table>/<key>` |
| `clio_ycsb_jni.cc` | JNI shim over the CTE SHM client (`libclio_ycsb_jni`, built by CMake when a JDK is present) |
| `run_comparison.sh` | Driver: load+run each workload against each DB with identical parameters, per-DB service lifecycle included |
| `ddb_create_table.py` | Creates YCSB's `usertable` on DynamoDB Local (boto3) |

## Provisioning (done by the devcontainer)

- `external/YCSB` — source clone (reference / future maven builds)
- `external/ycsb-dist/` — extracted 0.17.0 release binding tarballs
  (`redis`, `rocksdb`, `dynamodb`), `dynamodb-local/DynamoDBLocal.jar`,
  and `clio-binding/classes` (the compiled `ClioClient`)
- `redis-server`, a JDK (for `javac` + `jni.h`), python3 `boto3`

Manual setup, if needed:

```bash
DIST=external/ycsb-dist && mkdir -p $DIST && cd $DIST
for b in redis rocksdb dynamodb; do
  curl -sfLO https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-$b-binding-0.17.0.tar.gz
  tar xzf ycsb-$b-binding-0.17.0.tar.gz
done
curl -sfL -o ddb.tar.gz https://d1ni2b6xgvw0s0.cloudfront.net/dynamodb_local_latest.tar.gz
mkdir -p dynamodb-local && tar xzf ddb.tar.gz -C dynamodb-local
mkdir -p clio-binding/classes
javac -cp ycsb-redis-binding-0.17.0/lib/core-0.17.0.jar \
      -d clio-binding/classes \
      ../../context-transfer-engine/benchmark/ycsb/ClioClient.java
```

## Building the shim

`libclio_ycsb_jni.so` builds with the normal tree whenever CMake's
`find_package(JNI)` succeeds (install `openjdk-21-jdk-headless`):

```bash
cmake --build build-cpu --target clio_ycsb_jni
```

## Running

```bash
cd context-transfer-engine/benchmark/ycsb
BUILD_DIR=../../../build-cpu ./run_comparison.sh            # all four DBs
./run_comparison.sh clio redis                              # subset
RECORDS=1000000 OPS=1000000 THREADS=8 ./run_comparison.sh   # bigger run
```

Each DB × workload gets a fresh load phase; the run-phase `[OVERALL]
Throughput` lines are summarized at the end and full YCSB outputs land in
`results/`. The Clio leg starts/stops its own `clio_run` daemon (default
config: SHM client, RAM bdev); Redis and DynamoDB Local are started/stopped
per workload; RocksDB is embedded (fresh temp dir per workload).

## Reference numbers

Ares, one node (2x Xeon Silver 4114, 46 GB), 2026-11-22, `dev` 7c6adbae7.
1M records of ~1.1 KB, zipfian, median of 3 x 20 s runs, run-phase
`[OVERALL] Throughput` in ops/sec at 16 client threads. "Best" = fastest
configuration of that store at that point (`run_comparison.sh` variants):

| Workload | clio (shm cache) | best redis | best rocksdb | dynamodb-local |
|----------|-----------------:|-----------:|-------------:|---------------:|
| A (50/50 read/update) | 557,346 | 164,481 | 438,955 | 4,860 |
| B (95/5)              | 1,665,534 | 160,635 | 738,135 | 6,425 |
| C (read-only)         | 1,630,755 | 158,796 | 812,469 | 6,552 |
| F (read-modify-write) | 1,107,663 | 110,806 | 349,779 | 3,937 |

The full 1/4/16-thread table and the method are on the site
(`site/pages/technologies/context-transfer-engine.html`). CLIO beats the best
Redis 3.4-13x and the best RocksDB 1.3-5.5x at every point.

Harness traps that produced misleading CLIO numbers before (all fixed here):

- The binding updated its scan-key index (one global mutex + `std::set`) on
  every put; 16 writers serialized on it. Only inserts index keys now.
- The index was seeded with a 1M-key `GetContainedBlobs` broadcast at init,
  inside the timed window. It is now seeded on the first scan.
- A client that connects while the runtime is still composing the shm cache
  gets no cache keys and silently stays on the task path. The driver waits
  for the compose to finish; any CLIO run whose client did not bind to the
  cache is recorded as `NOCACHE`.
- The cache writes updates through to the core in the background; a run that
  starts behind that backlog waits for it. The driver runs
  `ClioClient drain` (SyncTag) between runs, outside the timed window.

## Caveats

- DynamoDB Local is not the DynamoDB service: numbers characterize the
  binding + local engine, not AWS-hosted performance.
- The Clio binding's `update` is a client-side read-modify-write (like the
  rocksdb bindings); redis uses hash-field updates natively.
- `RocksDBTunedClient.java` is this repo's tuned RocksDB binding (compile it
  against rocksdbjni 11.x into `$DIST/rocksdb-tuned/classes`, jar in
  `$DIST/rocksdb-tuned/lib`).
- One tag holds all records; blob metadata scale-out across tags is not
  exercised.
