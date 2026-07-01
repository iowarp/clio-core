# thallium-bench

Raw **libthallium** async-RPC microbenchmark (issue #647).

This measures how well thallium itself drives async RPCs, independent of the
lightbeam (`ctp::lbm`) transport wrapper. It talks to thallium **directly** over
the **shared-memory transport** (`na+sm`), so it captures the RPC-engine
overhead of an intra-node round trip without involving a real NIC.

The server (RPC handler) and the client (RPC submitter) live in the **same
process and the same `tl::engine`**: the client looks up the engine's own self
address and fires RPCs at it over the `na+sm` shared-memory loopback. The
handler runs on one of the engine's `rpc_thread_count` Argobots execution
streams; a dedicated progress thread drives Mercury.

## Building

The benchmark only builds when thallium is enabled:

```bash
cmake -B build -DCLIO_CORE_ENABLE_THALLIUM=ON -DCLIO_CORE_ENABLE_BENCHMARKS=ON
cmake --build build --target thallium_bench
```

The binary installs to `bin/thallium_bench`.

## Running

```bash
# Defaults: na+sm, 4 RPC threads, batch 16, 100k iters, no payload
./thallium_bench

# Sweep RPC handler threads
./thallium_bench --rpc-threads 1
./thallium_bench --rpc-threads 8

# Sweep async batch size (requests kept in flight at once)
./thallium_bench --batch 1     # synchronous-equivalent, latency-bound
./thallium_bench --batch 64    # deep async pipeline, throughput-bound

# Add a request payload to measure with data
./thallium_bench --payload 4096
```

### Options

| Flag | Meaning | Default |
|------|---------|---------|
| `--protocol P`    | Mercury protocol | `na+sm` |
| `--rpc-threads N` | Engine `rpc_thread_count` (handler ES pool) | `4` |
| `--batch B`       | Async RPCs kept in flight before draining | `16` |
| `--iters M`       | Total timed RPCs | `100000` |
| `--payload S`     | Request payload bytes (`0` = bare int request) | `0` |
| `--warmup W`      | Untimed warmup RPCs | `1000` |

## Metrics

- **Overall IOPS** — timed RPCs / elapsed seconds.
- **Average latency (us)** — `elapsed / iters`. Note that with `--batch > 1`
  this is the throughput-derived amortized latency (many requests are in flight
  at once), not the latency of a single isolated call. Use `--batch 1` to read
  off per-RPC round-trip latency.

## Notes

- `na+sm` is Mercury's shared-memory transport: same-node only, no NIC. Pass
  `--protocol "ofi+tcp;ofi_rxm"` (or a verbs string) to compare against a real
  network provider on the same harness.
