/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * End-of-benchmark CTE data flush.
 *
 * Every paged benchmark ends by asking the CTE to push volatile blocks to
 * persistent tiers (persistence level >= kTemporaryNonVolatile), so a run
 * with a file/NVMe tier configured leaves its dataset durable rather than
 * stranded in RAM. With no qualifying tier the core returns success with
 * nothing flushed -- the call is a clean no-op for RAM-only configs, so
 * every benchmark can make it unconditionally.
 */
#ifndef CLIO_GPU_VECTOR_BENCH_FLUSH_DATA_H_
#define CLIO_GPU_VECTOR_BENCH_FLUSH_DATA_H_

#include <clio_cte/core/core_client.h>

#include <cstdio>

#if !CTP_IS_DEVICE_PASS

/** Flush CTE data to persistent tiers; prints what moved. Returns the task
 *  return code (0 also when no persistent tier exists). */
inline int BenchFlushData() {
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  auto f = cte.AsyncFlushData(clio::run::PoolQuery::Local(),
                              /*target_persistence_level=*/1,
                              /*period_us=*/0);
  f.Wait();
  const int rc = f->GetReturnCode();
  std::printf("  [flush-data] rc=%d blobs=%llu bytes=%llu\n", rc,
              (unsigned long long)f->blobs_flushed_,
              (unsigned long long)f->bytes_flushed_);
  return rc;
}

#endif  // !CTP_IS_DEVICE_PASS

#endif  // CLIO_GPU_VECTOR_BENCH_FLUSH_DATA_H_
