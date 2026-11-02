/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

/**
 * GPU unit tests for the asynchronous copy primitives added to GpuApi for the
 * async memory-bdev transport: StreamQuery (non-blocking stream poll),
 * MemcpyAsync, and MemsetAsync. These mirror exactly the launch-then-poll
 * pattern the mem bdev uses (enqueue copies on a per-task stream, then poll
 * StreamQuery instead of blocking on Synchronize).
 */

#include <catch2/catch_all.hpp>

#include <cstring>
#include <vector>

#include "clio_ctp/util/gpu_api.h"

using ctp::GpuApi;

namespace {
// Poll a stream to completion the way the bdev coroutine does (there it yields
// the worker between polls; here we just spin). Returns the number of polls
// observed before completion.
int PollToCompletion(void *stream) {
  int polls = 0;
  while (!GpuApi::StreamQuery(stream)) {
    ++polls;
  }
  return polls;
}
}  // namespace

TEST_CASE("StreamQuery reports completion on an idle stream", "[gpu][async]") {
  void *stream = GpuApi::CreateStream();
  REQUIRE(stream != nullptr);
  // No work enqueued: a query must immediately report complete.
  REQUIRE(GpuApi::StreamQuery(stream));
  GpuApi::DestroyStream(stream);
}

TEST_CASE("Async H2D/D2H round trip through pinned host memory",
          "[gpu][async]") {
  const size_t n = 8 * 1024 * 1024;  // 8 MiB, big enough to actually be in-flight
  char *h_src = GpuApi::MallocHost<char>(n);
  char *h_dst = GpuApi::MallocHost<char>(n);
  char *d = GpuApi::Malloc<char>(n);
  REQUIRE(h_src != nullptr);
  REQUIRE(h_dst != nullptr);
  REQUIRE(d != nullptr);

  for (size_t i = 0; i < n; ++i) {
    h_src[i] = static_cast<char>((i * 131u + 7u) & 0xFFu);
  }
  std::memset(h_dst, 0, n);

  void *stream = GpuApi::CreateStream();
  GpuApi::MemcpyAsync(d, h_src, n, stream);   // host -> device
  GpuApi::MemcpyAsync(h_dst, d, n, stream);   // device -> host
  PollToCompletion(stream);

  REQUIRE(std::memcmp(h_src, h_dst, n) == 0);

  GpuApi::DestroyStream(stream);
  GpuApi::FreeHost(h_src);
  GpuApi::FreeHost(h_dst);
  GpuApi::Free(d);
}

TEST_CASE("MemsetAsync zeroes device memory", "[gpu][async]") {
  const size_t n = 1u << 20;  // 1 MiB
  char *d = GpuApi::Malloc<char>(n);
  char *h = GpuApi::MallocHost<char>(n);
  REQUIRE(d != nullptr);
  REQUIRE(h != nullptr);

  // Prime the device buffer with a non-zero pattern.
  std::memset(h, 0xFF, n);
  void *stream = GpuApi::CreateStream();
  GpuApi::MemcpyAsync(d, h, n, stream);
  // Zero it on the stream, then copy back into a poisoned host buffer.
  GpuApi::MemsetAsync(d, 0, n, stream);
  std::memset(h, 0xAA, n);
  GpuApi::MemcpyAsync(h, d, n, stream);
  PollToCompletion(stream);

  bool all_zero = true;
  for (size_t i = 0; i < n; ++i) {
    if (h[i] != 0) {
      all_zero = false;
      break;
    }
  }
  REQUIRE(all_zero);

  GpuApi::DestroyStream(stream);
  GpuApi::Free(d);
  GpuApi::FreeHost(h);
}

TEST_CASE("Zero-size async memcpy is a harmless no-op", "[gpu][async]") {
  char *d = GpuApi::Malloc<char>(16);
  char *h = GpuApi::MallocHost<char>(16);
  REQUIRE(d != nullptr);
  REQUIRE(h != nullptr);

  void *stream = GpuApi::CreateStream();
  GpuApi::MemcpyAsync(d, h, 0, stream);  // must not error or hang
  GpuApi::MemsetAsync(d, 0, 0, stream);
  REQUIRE(GpuApi::StreamQuery(stream));

  GpuApi::DestroyStream(stream);
  GpuApi::Free(d);
  GpuApi::FreeHost(h);
}

TEST_CASE("Many concurrent streams complete correctly", "[gpu][async]") {
  // Sanity that multiple in-flight streams (as the bdev creates one per task)
  // each transfer their own data correctly.
  const size_t kStreams = 16;
  const size_t n = 1u << 20;  // 1 MiB each
  std::vector<void *> streams(kStreams);
  std::vector<char *> hsrc(kStreams), hdst(kStreams), dev(kStreams);

  for (size_t s = 0; s < kStreams; ++s) {
    hsrc[s] = GpuApi::MallocHost<char>(n);
    hdst[s] = GpuApi::MallocHost<char>(n);
    dev[s] = GpuApi::Malloc<char>(n);
    REQUIRE(hsrc[s] != nullptr);
    REQUIRE(hdst[s] != nullptr);
    REQUIRE(dev[s] != nullptr);
    std::memset(hsrc[s], static_cast<int>(s + 1), n);
    std::memset(hdst[s], 0, n);
    streams[s] = GpuApi::CreateStream();
  }

  // Launch everything first so the copies genuinely overlap.
  for (size_t s = 0; s < kStreams; ++s) {
    GpuApi::MemcpyAsync(dev[s], hsrc[s], n, streams[s]);
    GpuApi::MemcpyAsync(hdst[s], dev[s], n, streams[s]);
  }
  for (size_t s = 0; s < kStreams; ++s) {
    PollToCompletion(streams[s]);
  }

  for (size_t s = 0; s < kStreams; ++s) {
    REQUIRE(std::memcmp(hsrc[s], hdst[s], n) == 0);
    GpuApi::DestroyStream(streams[s]);
    GpuApi::FreeHost(hsrc[s]);
    GpuApi::FreeHost(hdst[s]);
    GpuApi::Free(dev[s]);
  }
}
