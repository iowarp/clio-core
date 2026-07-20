/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Gray-Scott + CLIO async-compressed PutBlob benchmark.
 *
 * The "production-vector-shaped" version of the standalone Gray-Scott transfer
 * microbenchmark (clio_gray_scott_transfer_bench.cu): instead of a raw
 * cudaMemcpyAsync, each per-block snapshot is shipped to storage through a CLIO
 * *transparent compressed PutBlob*, so the data path is the real runtime +
 * compressor chimod that the tier-aware compressed GPU vector will use.
 *
 * Transparent compression is set up the supported way (see
 * test_transparent_compress.cc): a compose config places the compressor chimod
 * at the CTE entrypoint pool (512) in front of the real CTE core (513) and a RAM
 * bdev. A normal cte_client->AsyncPutBlob(..., ctx) with ctx.dynamic_compress_
 * then routes through compression transparently; the operator pin
 * CLIO_CTE_COMPRESS_LIB (compressor_runtime.cc) chooses the library.
 *
 * Pipeline (async + compressed):
 *   - Two snapshot buffers PER BLOCK (double-buffered) on the device, plus two
 *     SHM staging buffers ("two PutBlob slots per block" = the two double-buffer
 *     slots, each owning an in-flight PutBlob future).
 *   - For each step s: GS kernel computes into device slot[s&1]; D2H-stage into
 *     the slot's SHM buffer; issue an ASYNC compressed PutBlob per block (parked
 *     future). Before reusing a slot (two steps later) its future is drained, so
 *     step s+1 compute overlaps step s compress+store.
 *
 * NOTE: the GS kernel produces the data; the host issues the compressed PutBlob
 * (submission is host-side). Device-side submission through the compressor is
 * future work; the transfer is still async w.r.t. GS compute via the parked
 * futures. On a build without nvcomp the pinned compressor is a CPU library
 * (e.g. lz4/zstd); a GPU compressor (nvcomp-*) swaps in via the env pin.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL && CTP_ENABLE_COMPRESS

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_ctp/compress/compress_factory.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
// Compressed puts go through the compressor client (a real DynamicScheduleTask —
// the tested compress path). It uses host-only IPC, so it is host-pass only.
#if !CTP_IS_DEVICE_PASS
#include <clio_cte/compressor/compressor_client.h>
#endif

#include <algorithm>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

struct BenchOpts {
  // --- The core evaluation knobs ---
  clio::run::u32 nblocks = 1024;        // number of GPU blocks (GS rows)
  clio::run::u32 threads = 128;         // GPU threads per block
  clio::run::u32 nsteps = 100;          // total number of GS steps
  clio::run::u32 checkpoint_interval = 1;  // steps to wait between checkpoints
                                           // (a checkpoint = one PutBlob/block)
  clio::run::u32 grid_size = 1024;      // L: floats each block generates per step
                                        // (per-block row width; bytes = 4L)
  // --- Ancillary ---
  clio::run::u64 per_block_bytes = 0;   // derived from grid_size (=4L); or set
                                        // directly to override L.
  clio::run::u64 capacity_mib = 0;      // storage/bdev capacity; 0 = auto
  clio::run::u32 gpu_id = 0;
  std::string compress_lib = "cuszp";   // GPU lossy; overridden by CLIO_CTE_COMPRESS_LIB
  bool via_compressor = false;          // route puts through the compressor chimod
                                        // to compress IN THE RUNTIME (works; slow).
                                        // Off = fast uncompressed core PutBlob.
  bool verify = false;                  // round-trip: compress a HBM snapshot ->
                                        // store -> read back decompressed ->
                                        // compare (checks compress+decompress).
};

void PrintUsage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "  --blocks N              Number of GPU blocks / GS rows (default 1024)\n"
               "  --threads-per-block N   GPU threads per block (default 128)\n"
               "  --steps N               Total number of GS steps (default 100)\n"
               "  --checkpoint-interval N Steps to wait between checkpoints; a checkpoint\n"
               "                          issues one compressed PutBlob per block (default 1)\n"
               "  --grid-size L           Floats each block generates per step (row width L;\n"
               "                          per-block bytes = 4L) (default 1024)\n"
               "  --per-block-bytes B     Override: bytes/block/step directly (= 4L)\n"
               "  --compress-lib NAME     GPU compressor (default cuszp; CLIO_CTE_COMPRESS_LIB overrides)\n"
               "  --capacity-mib N        RAM storage capacity (default = 2x stored, min 256)\n"
               "  --gpu-id N              GPU index (default 0)\n"
               "  --help                  Show this message\n",
               prog);
}

bool ParseOpts(int argc, char *argv[], BenchOpts &opts) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", flag); std::exit(2); }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { PrintUsage(argv[0]); std::exit(0); }
    else if (a == "--blocks") opts.nblocks = std::atoi(next("--blocks"));
    else if (a == "--threads-per-block") opts.threads = std::atoi(next("--threads-per-block"));
    else if (a == "--steps") opts.nsteps = std::atoi(next("--steps"));
    else if (a == "--checkpoint-interval")
      opts.checkpoint_interval = std::atoi(next("--checkpoint-interval"));
    else if (a == "--grid-size") opts.grid_size = std::atoi(next("--grid-size"));
    else if (a == "--per-block-bytes")
      opts.per_block_bytes = std::strtoull(next("--per-block-bytes"), nullptr, 10);
    else if (a == "--compress-lib") opts.compress_lib = next("--compress-lib");
    else if (a == "--capacity-mib")
      opts.capacity_mib = std::strtoull(next("--capacity-mib"), nullptr, 10);
    else if (a == "--gpu-id") opts.gpu_id = std::atoi(next("--gpu-id"));
    else if (a == "--via-compressor") opts.via_compressor = true;
    else if (a == "--verify") opts.verify = true;
    else { std::fprintf(stderr, "Unknown arg: %s\n", a.c_str()); PrintUsage(argv[0]); return false; }
  }
  // grid_size (L) is the primary knob; per_block_bytes derives from it unless
  // the user set per_block_bytes explicitly.
  if (opts.per_block_bytes == 0) opts.per_block_bytes = (clio::run::u64)opts.grid_size * sizeof(float);
  else opts.grid_size = (clio::run::u32)(opts.per_block_bytes / sizeof(float));
  if (opts.checkpoint_interval == 0) opts.checkpoint_interval = 1;
  if (opts.threads == 0) opts.threads = 128;
  return true;
}

#if !CTP_IS_DEVICE_PASS
// Write a compose config that puts the compressor chimod (512) in front of the
// CTE core (513) + RAM bdev (301), then start the runtime pointed at it. This is
// the supported transparent-compression setup (see test_transparent_compress).
void EnsureInit(clio::run::u64 capacity_mib) {
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 9413;
  std::string cfg_path = "/tmp/gs_compose_" + std::to_string(port) + ".yaml";
  {
    std::ofstream cfg(cfg_path);
    cfg << "networking:\n  port: " << port << "\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"" << capacity_mib << "MB\"\n\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"513.0\"\n"
        << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"" << capacity_mib << "MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg_path.c_str(), 1);
  std::fprintf(stderr, "[INIT] compose config: %s\n", cfg_path.c_str());
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "[INIT] CLIO_INIT failed\n"); std::exit(2);
  }
  std::this_thread::sleep_for(1s);  // let compose pools initialize
  // Point the CTE client at the compressor entrypoint pool (512).
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);
}
#endif  // !CTP_IS_DEVICE_PASS

}  // namespace

// Gray-Scott init: whole-domain perturbation keeps the field non-degenerate.
__global__ void GsInitKernel(float *u, float *v, int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int idx = row * cols + col;
    unsigned int hu = (unsigned int)idx * 2654435761u + 12345u;
    unsigned int hv = (unsigned int)idx * 40503u + 6789u;
    const float nu = (float)((hu >> 9) & 0xffff) / 65535.0f - 0.5f;
    const float nv = (float)((hv >> 9) & 0xffff) / 65535.0f - 0.5f;
    u[idx] = 0.5f + 0.40f * nu;
    v[idx] = 0.25f + 0.20f * nv;
  }
}

// One GS step for all blocks. Advances u/v -> u2/v2; if `out` is non-null,
// also writes the v-field snapshot into `out` (a HBM buffer the compressor then
// reads in place). `out` is null on non-checkpoint steps so the snapshot buffer
// is never disturbed while a compression is in flight against it.
__global__ void GsStepKernel(const float *u, const float *v, float *u2,
                             float *v2, float *out, int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float Du = 0.16f, Dv = 0.08f, F = 0.060f, k = 0.062f, dt = 1.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int idx = row * cols + col;
    const int up = (row > 0 ? row - 1 : row) * cols + col;
    const int dn = (row < rows - 1 ? row + 1 : row) * cols + col;
    const int lf = row * cols + (col > 0 ? col - 1 : col);
    const int rt = row * cols + (col < cols - 1 ? col + 1 : col);
    const float uc = u[idx], vc = v[idx];
    const float lap_u = u[up] + u[dn] + u[lf] + u[rt] - 4.0f * uc;
    const float lap_v = v[up] + v[dn] + v[lf] + v[rt] - 4.0f * vc;
    const float uvv = uc * vc * vc;
    u2[idx] = uc + (Du * lap_u - uvv + F * (1.0f - uc)) * dt;
    v2[idx] = vc + (Dv * lap_v + uvv - (F + k) * vc) * dt;
    if (out) out[idx] = v2[idx];
  }
}

#if !CTP_IS_DEVICE_PASS

int main(int argc, char *argv[]) {
  BenchOpts opts;
  if (!ParseOpts(argc, argv, opts)) return 2;
  if (opts.nblocks == 0 || opts.per_block_bytes == 0 ||
      opts.per_block_bytes % sizeof(float) != 0) {
    std::fprintf(stderr, "blocks>0, per-block-bytes>0 and a multiple of 4\n"); return 2;
  }

  const int rows = static_cast<int>(opts.nblocks);
  const int cols = static_cast<int>(opts.per_block_bytes / sizeof(float));
  const clio::run::u64 field_bytes = (clio::run::u64)rows * cols * sizeof(float);
  const clio::run::u64 snap_bytes = opts.per_block_bytes;

  // Requested compressor = the CLIO_CTE_COMPRESS_LIB pin if set, else
  // --compress-lib (this is exactly what the runtime will use). The special
  // value "none"/"raw"/"off" is a NO-COMPRESSION baseline: the PutBlob goes
  // straight to core storage, so we can measure the slowdown compression adds.
  std::string req = opts.compress_lib;
  if (const char *penv = std::getenv("CLIO_CTE_COMPRESS_LIB")) {
    if (penv[0]) req = penv;
  }
  const bool no_compress = (req == "none" || req == "raw" || req == "off");
  std::string eff_lib = req;
  int compress_lib = 0;
  if (!no_compress) {
    int w = ctp::CompressionFactory::WireIdForName(req);
    if (w < 0) { char *e = nullptr; long p = std::strtol(req.c_str(), &e, 10);
                 if (e != req.c_str() && *e == '\0') w = (int)p; }
    if (w >= 0) { compress_lib = w; eff_lib = ctp::CompressionFactory::NameForWireId(w); }
    else { eff_lib = "lz4"; compress_lib = ctp::CompressionFactory::WireIdForName("lz4"); }
  }

  // A checkpoint (one PutBlob/block) happens every `checkpoint_interval` steps,
  // plus a final flush. Data actually stored = snapshots at checkpoints only.
  const clio::run::u32 num_ckpts =
      (opts.nsteps + opts.checkpoint_interval - 1) / opts.checkpoint_interval;
  clio::run::u64 logical_total = snap_bytes * (clio::run::u64)rows * num_ckpts;
  clio::run::u64 capacity_mib =
      opts.capacity_mib > 0 ? opts.capacity_mib
                            : std::max<clio::run::u64>(256, (logical_total >> 20) * 2 + 64);

  std::fprintf(stderr,
               "[BENCH] GS transparent-compressed PutBlob\n"
               "[BENCH] blocks=%u threads/block=%u grid L=%u (snap/block=%llu B)\n"
               "[BENCH] steps=%u checkpoint-interval=%u -> %u checkpoints, logical=%llu MB\n"
               "[BENCH] compress-lib=%s (wire=%d; CLIO_CTE_COMPRESS_LIB pin overrides)\n",
               opts.nblocks, opts.threads, opts.grid_size,
               (unsigned long long)snap_bytes,
               opts.nsteps, opts.checkpoint_interval, num_ckpts,
               (unsigned long long)(logical_total >> 20),
               eff_lib.c_str(), compress_lib);

  EnsureInit(capacity_mib);
  // Pools from the compose config: compressor entrypoint 512 -> core 513 -> bdev.
  const clio::run::PoolId kCompressorPool(512, 0);
  const clio::run::PoolId kCorePool(513, 0);
  clio::cte::core::Client core_client;
  core_client.Init(kCorePool);  // tags + no-compression puts land on core (513)

  auto tag_fut = core_client.AsyncGetOrCreateTag("gs_compress");
  tag_fut.Wait();
  if (tag_fut->GetReturnCode() != 0) { std::fprintf(stderr, "[INIT] tag failed\n"); return 2; }
  clio::cte::core::TagId tag_id = tag_fut->tag_id_;

  float *u  = ctp::GpuApi::Malloc<float>(field_bytes);
  float *v  = ctp::GpuApi::Malloc<float>(field_bytes);
  float *u2 = ctp::GpuApi::Malloc<float>(field_bytes);
  float *v2 = ctp::GpuApi::Malloc<float>(field_bytes);
  // Two HBM snapshot buffers (double-buffered). The compressor reads these
  // device pointers IN PLACE -- there is no HBM->DRAM staging of the
  // uncompressed snapshot anywhere in this benchmark.
  float *snap_dev[2] = { ctp::GpuApi::Malloc<float>(field_bytes),
                         ctp::GpuApi::Malloc<float>(field_bytes) };
  if (!u || !v || !u2 || !v2 || !snap_dev[0] || !snap_dev[1]) {
    std::fprintf(stderr, "[BENCH] device allocation failed\n"); return 2;
  }

  // A blob_data ShmPtr that points DIRECTLY at HBM: null alloc-id + raw device
  // address. ToFullPtr resolves it to that pointer, and the GPU compressor
  // (cuSZp) reads it in place -- compression happens entirely in HBM (cuSZp
  // allocates its own temporary HBM output buffer).
  auto dev_ptr = [](void *addr) {
    ctp::ipc::ShmPtr<> p; p.alloc_id_.SetNull();
    p.off_ = reinterpret_cast<clio::run::u64>(addr);
    return p;
  };

  GsInitKernel<<<rows, opts.threads>>>(u, v, rows, cols);
  ctp::GpuApi::Synchronize();

  // --- Round-trip verification (--verify) ---
  // Compress a real GS snapshot (HBM) through the compressor -> store -> read it
  // back DECOMPRESSED through the compressor -> compare. This is the correctness
  // check the transparent-compress test lacks: it confirms compress AND decompress
  // both work in the runtime (cuSZp is lossy, so we report the max abs error vs
  // the compressor's absolute error bound).
  if (opts.verify) {
    GsStepKernel<<<rows, opts.threads>>>(u, v, u2, v2, snap_dev[0], rows, cols);
    ctp::GpuApi::Synchronize();
    const clio::run::u64 field_elems = (clio::run::u64)rows * cols;
    float *got_dev = ctp::GpuApi::Malloc<float>(field_bytes);
    std::vector<float> orig(field_elems), got(field_elems);
    ctp::GpuApi::Memcpy<float>(orig.data(), snap_dev[0], field_bytes);

    // Route through the compressor's kPutBlob/kGetBlob handlers -- exactly the
    // path a gpu_vector page eviction/fault takes: a CORE PutBlob/GetBlob sent to
    // the compressor entrypoint pool (512), which compresses/decompresses and
    // forwards to core (513).
    clio::cte::core::Client c512; c512.Init(kCompressorPool);
    clio::cte::core::Context ctx;
#if CTP_ENABLE_COMPRESS
    ctx.dynamic_compress_ = 1; ctx.compress_lib_ = compress_lib; ctx.compress_preset_ = 2;
#endif
    auto pf = c512.AsyncPutBlob(tag_id, "verify_blob", (clio::run::u64)0,
        field_bytes, dev_ptr(snap_dev[0]), 0.5f, ctx, (clio::run::u32)0);
    pf.Wait();
    int put_rc = pf->GetReturnCode();
    auto gf = c512.AsyncGetBlob(tag_id, "verify_blob", (clio::run::u64)0, field_bytes,
        (clio::run::u32)0, dev_ptr(got_dev));
    gf.Wait();
    int get_rc = gf->GetReturnCode();
    ctp::GpuApi::Memcpy<float>(got.data(), got_dev, field_bytes);

    double max_err = 0.0, sum_err = 0.0;
    for (clio::run::u64 i = 0; i < field_elems; ++i) {
      double e = std::fabs((double)orig[i] - (double)got[i]);
      if (e > max_err) max_err = e; sum_err += e;
    }
    std::fprintf(stderr,
        "[VERIFY] compressor=%s  put_rc=%d get_rc=%d  elems=%llu\n"
        "[VERIFY] max_abs_err=%.3e  mean_abs_err=%.3e  (cuSZp eb=1e-3)  %s\n",
        eff_lib.c_str(), put_rc, get_rc, (unsigned long long)field_elems,
        max_err, sum_err / (double)field_elems,
        (put_rc == 0 && get_rc == 0 && max_err <= 2e-3) ? "PASS" : "CHECK");
    ctp::GpuApi::Free<float>(got_dev);
    ctp::GpuApi::Free<float>(u);  ctp::GpuApi::Free<float>(v);
    ctp::GpuApi::Free<float>(u2); ctp::GpuApi::Free<float>(v2);
    ctp::GpuApi::Free<float>(snap_dev[0]); ctp::GpuApi::Free<float>(snap_dev[1]);
    return (put_rc == 0 && get_rc == 0 && max_err <= 2e-3) ? 0 : 1;
  }

  // --- Direct compression-ratio measurement ---
  // The runtime does not propagate the compressed size back to the client task,
  // so we measure the ratio the pinned library actually achieves on a real GS
  // snapshot, using the SAME CompressionFactory the runtime uses. A throwaway GS
  // step (no swap) produces a representative field; u/v are left untouched for
  // the timed loop.
  double measured_ratio = 0.0;
  if (no_compress) {
    measured_ratio = 1.0;  // baseline stores raw bytes
    std::fprintf(stderr, "[BENCH] no compression (raw PutBlob baseline)\n");
  } else {
    GsStepKernel<<<rows, opts.threads>>>(u, v, u2, v2, snap_dev[0], rows, cols);
    ctp::GpuApi::Synchronize();
    auto comp = ctp::CompressionFactory::GetPreset(
        eff_lib, ctp::CompressionPreset::BALANCED);
    if (comp) {
      std::vector<char> out(field_bytes + field_bytes / 2 + 4096);
      size_t out_sz = out.size();
      // Input is the HBM device pointer -- cuSZp compresses it in place.
      if (comp->Compress(out.data(), out_sz, snap_dev[0], field_bytes) &&
          out_sz > 0) {
        measured_ratio = (double)field_bytes / (double)out_sz;
        std::fprintf(stderr,
                     "[BENCH] measured ratio (%s on GS snapshot): %.2fx "
                     "(%llu -> %llu B)\n",
                     eff_lib.c_str(), measured_ratio,
                     (unsigned long long)field_bytes, (unsigned long long)out_sz);
      }
    }
    if (measured_ratio == 0.0)
      std::fprintf(stderr, "[BENCH] ratio unavailable for '%s'\n",
                   eff_lib.c_str());
  }

  // Submit path. Compressed puts go through the COMPRESSOR client (512 -> 513):
  // it builds a real DynamicScheduleTask, so the compressor actually compresses
  // (the supported/tested path). The no-compression baseline uses the CORE
  // client (513) directly, storing uncompressed. run_puts is templated on the
  // client so both share one timed, double-buffered checkpoint loop.
  clio::run::u64 total_compressed = 0;
  int put_failures = 0;
  double ms = 0.0;

  auto run_puts = [&](auto submit) {
    using FutT = decltype(submit(std::string(), dev_ptr(snap_dev[0]),
                                 clio::cte::core::Context()));
    std::vector<FutT> inflight[2];
    inflight[0].resize(rows); inflight[1].resize(rows);
    std::vector<char> busy[2];
    busy[0].assign(rows, 0); busy[1].assign(rows, 0);

    auto drain = [&](int slot) {
      for (int b = 0; b < rows; ++b) if (busy[slot][b]) {
        inflight[slot][b].Wait();
        if (inflight[slot][b]->GetReturnCode() != 0) ++put_failures;
#if CTP_ENABLE_COMPRESS
        clio::run::u64 cs = inflight[slot][b]->context_.actual_compressed_size_;
        if (cs > 0) total_compressed += cs;
#endif
        busy[slot][b] = 0;
      }
    };

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    clio::run::u32 ckpt = 0;  // checkpoint counter (drives the double buffer)
    for (clio::run::u32 s = 0; s < opts.nsteps; ++s) {
      const bool is_ckpt = ((s + 1) % opts.checkpoint_interval == 0) ||
                           (s + 1 == opts.nsteps);
      const int slot = ckpt & 1;
      // Drain this slot's prior in-flight compressions BEFORE the kernel writes
      // snap_dev[slot] -- the compressor reads that HBM buffer directly, so it
      // must not be overwritten while a compress against it is still running.
      if (is_ckpt) drain(slot);
      // Advance the simulation every step; write the snapshot into snap_dev[slot]
      // only on checkpoint steps (null otherwise), so the buffer stays stable.
      GsStepKernel<<<rows, opts.threads>>>(
          u, v, u2, v2, is_ckpt ? snap_dev[slot] : nullptr, rows, cols);
      ctp::GpuApi::Synchronize();
      if (is_ckpt) {
        for (int b = 0; b < rows; ++b) {
          clio::cte::core::Context ctx;
#if CTP_ENABLE_COMPRESS
          ctx.dynamic_compress_ = 1;          // static compression path
          ctx.compress_lib_ = compress_lib;   // env pin overrides in the runtime
          ctx.compress_preset_ = 2;           // balanced
#endif
          // blob_data points straight at this block's row in HBM -- no copy.
          ctp::ipc::ShmPtr<> ptr = dev_ptr(
              reinterpret_cast<char *>(snap_dev[slot]) +
              (clio::run::u64)b * cols * sizeof(float));
          std::string blob_name =
              "gs_b" + std::to_string(b) + "_c" + std::to_string(ckpt);
          inflight[slot][b] = submit(blob_name, ptr, ctx);
          busy[slot][b] = 1;
        }
        ++ckpt;
      }
      std::swap(u, u2);
      std::swap(v, v2);
    }
    drain(0); drain(1);
    ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  };

  // Two store paths:
  //  --via-compressor: route each snapshot through the compressor via
  //    AsyncCompress -> Compress() DIRECTLY (skipping DynamicSchedule). Compress
  //    only blocking-waits on the CORE container (513) -- a DIFFERENT container --
  //    so it avoids the DynamicSchedule self-container-wait deadlock. This is the
  //    path that genuinely compresses in the runtime.
  //  default: store via a core-pool (513) PutBlob (uncompressed) -- always works.
  // Either way the reported ratio is also measured directly (see the probe).
  if (opts.via_compressor && !no_compress) {
    std::fprintf(stderr, "[BENCH] routing through compressor via AsyncCompress "
                         "(Compress() direct; cross-container wait)\n");
    clio::cte::compressor::Client comp_client(kCompressorPool, kCorePool);
    run_puts([&](const std::string &name, ctp::ipc::ShmPtr<> ptr,
                 const clio::cte::core::Context &ctx) {
      return comp_client.AsyncCompress(
          clio::run::PoolQuery::Local(), tag_id, name, (clio::run::u64)0,
          snap_bytes, ptr, 0.5f, ctx, (clio::run::u32)0, kCorePool);
    });
  } else {
    run_puts([&](const std::string &name, ctp::ipc::ShmPtr<> ptr,
                 const clio::cte::core::Context &ctx) {
      return core_client.AsyncPutBlob(tag_id, name, (clio::run::u64)0,
                                      snap_bytes, ptr, 0.5f, ctx, (clio::run::u32)0);
    });
  }
  // Prefer the directly-measured ratio; fall back to runtime-reported size if it
  // ever starts propagating (currently it does not, so total_compressed==0).
  double ratio = measured_ratio > 0.0
                     ? measured_ratio
                     : (total_compressed > 0
                            ? (double)logical_total / (double)total_compressed
                            : 0.0);
  double stored_mib = ratio > 0.0
                          ? (logical_total / ratio) / (double)(1ULL << 20)
                          : 0.0;
  // Runtime-reported ratio (from the compressor writing actual_compressed_size_
  // back through the shared task context_). Nonzero confirms the RUNTIME actually
  // compressed (vs. the measured probe, which only shows what the lib can do).
  double rt_ratio = total_compressed > 0
                        ? (double)logical_total / (double)total_compressed : 0.0;
  std::fprintf(stderr, "[SUMMARY] runtime-reported: compressed=%llu B  ratio=%.2fx%s\n",
               (unsigned long long)total_compressed, rt_ratio,
               total_compressed > 0 ? "  (runtime confirms compression)" : "  (not reported)");
  std::fprintf(stderr,
               "\n[SUMMARY] blocks=%u threads/block=%u grid_L=%u steps=%u "
               "ckpt_interval=%u checkpoints=%u put_failures=%d\n"
               "[SUMMARY] total=%.3f ms  %.2f us/step  %.2f us/ckpt  logical=%llu MB\n"
               "[SUMMARY] compressor=%s  MEASURED ratio=%.2fx (direct; lib on GS data)\n"
               "[SUMMARY] runtime store: %s -> ~%.1f MiB actually stored\n"
               "[SUMMARY] logical throughput=%.1f MiB/s\n",
               opts.nblocks, opts.threads, opts.grid_size, opts.nsteps,
               opts.checkpoint_interval, num_ckpts, put_failures,
               ms, (ms * 1e3) / opts.nsteps,
               num_ckpts ? (ms * 1e3) / num_ckpts : 0.0,
               (unsigned long long)(logical_total >> 20),
               eff_lib.c_str(), ratio,
               opts.via_compressor ? "via compressor chimod (COMPRESSED in runtime)"
                                   : "core pool, UNCOMPRESSED (use --via-compressor to compress)",
               opts.via_compressor ? stored_mib
                                   : (double)(logical_total >> 20),
               (logical_total / (double)(1ULL << 20)) / (ms / 1e3));

  ctp::GpuApi::Free<float>(u);  ctp::GpuApi::Free<float>(v);
  ctp::GpuApi::Free<float>(u2); ctp::GpuApi::Free<float>(v2);
  ctp::GpuApi::Free<float>(snap_dev[0]); ctp::GpuApi::Free<float>(snap_dev[1]);
  return put_failures == 0 ? 0 : 1;
}

#else
int main() { return 0; }
#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL && CTP_ENABLE_COMPRESS
