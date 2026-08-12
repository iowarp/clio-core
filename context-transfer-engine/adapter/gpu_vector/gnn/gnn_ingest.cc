/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Stream a raw array from stdin straight into a gpu_vector's CTE backing store.
 *
 * WHY THIS EXISTS. gnn_prep.py turns ogbn-papers100M into flat binaries, and
 * the training path then wants an aggregated copy on top of that. For
 * papers100M that is 53 GiB of features.f32 + 13 GiB of graph.csr + another
 * 53 GiB of agg_features.f32, on top of the 56 GiB download and its unpacked
 * members -- roughly 190 GiB of scratch to load a matrix that the vector is
 * supposed to hold for you. That made sense on the machine it was written for
 * (a 1.5 TB node-local NVMe); it is the wrong shape everywhere else, and it
 * defeats the point of a tiered vector.
 *
 * So: no intermediate files. The producer streams decoded rows on stdout, this
 * reads them a page at a time and puts each page into the CTE through the
 * compressor, exactly as the vector's own flush path would. The CTE places
 * pages across whatever tiers are composed (RAM, then NVMe by score), so the
 * matrix can be far larger than memory. Peak disk is the source archive plus
 * the compressed store -- nothing else.
 *
 * The page naming is the vector's own: page N is blob "p<N>" under the vector's
 * tag (see PageBlobName in gpu_vector/page.h). Get that wrong and the vector
 * faults forever against blobs that exist under names nobody asks for, with no
 * error anywhere -- so it is asserted against the header, not just documented.
 *
 * Usage:
 *   python3 gnn_stream_npz.py ... | gnn_ingest --tag papers100M_feat \
 *       --page-bytes 1048576 [--elems N] [--codec 10] [--preset 2]
 *
 * The runtime must already be composed with the tiers you want (see
 * CLIO_SERVER_CONF); this tool only writes.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/page.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

namespace {

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/** Read exactly n bytes unless EOF. Returns bytes read. */
size_t ReadFull(std::FILE *f, char *buf, size_t n) {
  size_t got = 0;
  while (got < n) {
    size_t r = std::fread(buf + got, 1, n - got, f);
    if (r == 0) break;
    got += r;
  }
  return got;
}

void Usage() {
  std::fprintf(stderr,
      "usage: gnn_ingest --tag NAME [--page-bytes N] [--codec ID] "
      "[--preset P] [--storage-pool MAJOR.MINOR]\n"
      "  reads raw bytes on stdin, writes them as vector pages p0..pK-1\n");
}

}  // namespace

int main(int argc, char **argv) {
  std::string tag_name;
  bool read_mode = false;
  std::uint64_t read_pages = 0;
  std::uint64_t page_bytes = 1u << 20;  // 1 MiB
  int codec = 10;                       // ZSTD
  int preset = 2;                       // BALANCED
  clio::run::PoolId storage(600, 0);    // the compressor pool

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) { Usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--tag") tag_name = next();
    else if (a == "--page-bytes") page_bytes = std::strtoull(next(), nullptr, 10);
    else if (a == "--codec") codec = std::atoi(next());
    else if (a == "--preset") preset = std::atoi(next());
    else if (a == "--read") { read_mode = true; read_pages = std::strtoull(next(), nullptr, 10); }
    else if (a == "--storage-pool") {
      std::string v = next();
      auto dot = v.find('.');
      storage = clio::run::PoolId((clio::run::u32)std::atoi(v.substr(0, dot).c_str()),
                                  (clio::run::u32)std::atoi(v.substr(dot + 1).c_str()));
    } else { Usage(); return 2; }
  }
  if (tag_name.empty() || page_bytes == 0) { Usage(); return 2; }

  // The vector composes page blob names itself; borrow the same function so
  // this can never drift from what the fault path will ask for.
  {
    char probe[32];
    clio::cte::gpu_vector::PageBlobName(1234, probe);
    if (std::string(probe) != "p1234") {
      std::fprintf(stderr,
                   "[ingest] FATAL: PageBlobName produced '%s', expected "
                   "'p1234' -- the vector's naming changed and every page "
                   "written here would be unreachable\n", probe);
      return 1;
    }
  }

  // Pure client: attach to the daemon that owns the composed tiers. Passing
  // default_with_runtime=true here would stand up a SECOND in-process runtime
  // and try to bind the same port, which fails with a "port in use" wall of
  // text that looks nothing like the actual mistake.
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient)) {
    std::fprintf(stderr, "[ingest] runtime init failed\n");
    return 1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[ingest] cte client init failed\n");
    return 1;
  }

  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  auto tf = core.AsyncGetOrCreateTag(tag_name);
  tf.Wait();
  if (tf->GetReturnCode() != 0) {
    std::fprintf(stderr, "[ingest] could not create tag %s\n", tag_name.c_str());
    return 1;
  }
  const auto tag_id = tf->tag_id_;

  clio::cte::core::Client comp(storage);
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_lib_ = codec;
  ctx.compress_preset_ = preset;

  std::fprintf(stderr,
               "[ingest] tag=%s page=%lluB codec=%d preset=%d pool=%u.%u\n",
               tag_name.c_str(), (unsigned long long)page_bytes, codec, preset,
               storage.major_, storage.minor_);

  // Readback: pull pages back out and write them to stdout, so a load can be
  // verified byte-for-byte against its source without involving a GPU.
  if (read_mode) {
    std::vector<char> rbuf((size_t)page_bytes);
    for (std::uint64_t p = 0; p < read_pages; ++p) {
      char name[32];
      clio::cte::gpu_vector::PageBlobName(p, name);
      ctp::ipc::ShmPtr<> dp;
      dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
      dp.off_ = reinterpret_cast<clio::run::u64>(rbuf.data());
      auto gf = comp.AsyncGetBlob(tag_id, name, (clio::run::u64)0,
                                  (clio::run::u64)page_bytes, 0, dp,
                                  clio::run::PoolQuery::Local());
      gf.Wait();
      if (gf->GetReturnCode() != 0) {
        std::fprintf(stderr, "[ingest] read FAILED on page %llu rc=%d\n",
                     (unsigned long long)p, gf->GetReturnCode());
        return 1;
      }
      std::fwrite(rbuf.data(), 1, (size_t)page_bytes, stdout);
    }
    std::fflush(stdout);
    std::fprintf(stderr, "[ingest] read back %llu pages\n",
                 (unsigned long long)read_pages);
    return 0;
  }

  std::vector<char> page((size_t)page_bytes);
  std::uint64_t pg = 0, bytes_in = 0;
  const double t0 = NowSec();
  double last_report = t0;

  for (;;) {
    size_t got = ReadFull(stdin, page.data(), (size_t)page_bytes);
    if (got == 0) break;
    if (got < page_bytes) {
      // Final short page: zero-fill so every page is the same size. The vector
      // addresses pages by fixed stride, so a ragged last page would make the
      // tail unreadable; the caller knows the true element count.
      std::memset(page.data() + got, 0, (size_t)page_bytes - got);
      std::fprintf(stderr, "[ingest] final page %llu short (%zu of %llu B), "
                   "zero-filled\n", (unsigned long long)pg, got,
                   (unsigned long long)page_bytes);
    }

    char name[32];
    clio::cte::gpu_vector::PageBlobName(pg, name);
    ctp::ipc::ShmPtr<> dp;
    dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(page.data());
    auto pf = comp.AsyncPutBlob(tag_id, name, (clio::run::u64)0,
                                (clio::run::u64)page_bytes, dp, 0.5f, ctx, 0,
                                clio::run::PoolQuery::Local());
    pf.Wait();
    if (pf->GetReturnCode() != 0) {
      std::fprintf(stderr, "[ingest] FAILED on page %llu rc=%d (tier full?)\n",
                   (unsigned long long)pg, pf->GetReturnCode());
      return 1;
    }
    ++pg;
    bytes_in += got;

    const double now = NowSec();
    if (now - last_report > 10.0) {
      const double mibs = (bytes_in / 1048576.0) / (now - t0);
      std::fprintf(stderr, "[ingest] %llu pages, %.2f GiB in, %.1f MiB/s\n",
                   (unsigned long long)pg, bytes_in / 1073741824.0, mibs);
      last_report = now;
    }
    if (got < page_bytes) break;  // short read means EOF
  }

  const double dt = NowSec() - t0;
  std::fprintf(stderr,
               "[ingest] DONE: %llu pages, %.2f GiB in %.1fs (%.1f MiB/s)\n",
               (unsigned long long)pg, bytes_in / 1073741824.0, dt,
               dt > 0 ? (bytes_in / 1048576.0) / dt : 0.0);
  std::fprintf(stdout, "%llu %llu\n", (unsigned long long)pg,
               (unsigned long long)bytes_in);
  return 0;
}
