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

First run on the WSL2 dev container (100k records / 100k ops / 4 threads /
~1.1KB records; run-phase `[OVERALL] Throughput` in ops/sec — single-node
noisy-box numbers, magnitudes not gospel):

| Workload | clio (SHM) | redis | rocksdb (embedded) | dynamodb-local |
|----------|-----------:|------:|-------------------:|---------------:|
| A (50/50 read/update) | 6,952 | 21,650 | 49,140 | 1,664 |
| B (95/5)              | 42,553 | 16,787 | 64,809 | 1,867 |
| C (read-only)         | 34,566–55,866 | 21,044 | 62,383 | 1,915 |
| F (read-modify-write) | 7,637 | 14,762 | 27,824 | 1,335 |

Reading: Clio's SHM read path beats Redis ~2x on the read-heavy workloads;
update-heavy A/F pay double because the binding's `update` is a client-side
full-record read + full-record rewrite (see Caveats) where redis writes a
single hash field. Embedded RocksDB (no IPC at all) leads everywhere;
DynamoDB Local's HTTP path trails by ~10-30x. Obvious next lever: a partial
PutBlob-at-offset or server-side field merge for updates.

## Caveats

- DynamoDB Local is not the DynamoDB service: numbers characterize the
  binding + local engine, not AWS-hosted performance.
- The Clio binding's `update` is a client-side read-modify-write (like the
  stock rocksdb binding); redis uses hash-field updates natively.
- One tag holds all records; blob metadata scale-out across tags is not
  exercised.
