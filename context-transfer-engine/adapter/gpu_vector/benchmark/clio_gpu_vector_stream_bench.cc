/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Streaming benchmark for the device-paged vector WITH THE COMPRESSOR MODULE
 * ON THE PATH: how much does compression cost, and what does it buy, as the
 * working set outgrows the top storage tier?
 *
 * Both arms of the comparison run against the SAME compose (compressor pool
 * -> core pool). Only the vector's codec differs -- 0 for the raw arm, lz4 for
 * the compressed arm -- so the measurement isolates the codec rather than the
 * presence of an extra pool in the chain.
 *
 * WORKLOAD. Two sequential passes over the whole vector:
 *
 *   write  Each block fills its slice page by page and flushes as it goes,
 *          collecting the previous page's flush while filling the current one
 *          so the store overlaps the fill instead of serializing behind it.
 *   read   A fresh kernel walks the whole vector and accumulates a checksum.
 *          Nothing is resident when it starts (the write pass evicted it), so
 *          every page is fetched -- from the top tier if it still fits there,
 *          over the bus from the spill tier otherwise.
 *
 * They are reported separately because compression moves them in opposite
 * directions: it costs CPU on the write and buys tier residency on the read.
 * One blended number would hide exactly the trade this benchmark exists to
 * show.
 *
 * SYNTHETIC DATA. `--zero-pct X` makes X% of pages all-zero (lz4 crushes them
 * to almost nothing) and the rest full-entropy PRNG output (lz4 cannot shrink
 * them at all -- it will refuse and store them raw). Mixing whole PAGES rather
 * than diluting every page is deliberate: the codec runs per page, so the
 * page is the unit at which compression either pays or does not, and a
 * per-page mix gives a stored size that tracks --zero-pct almost exactly.
 * That makes the compression ratio an INPUT to the experiment instead of an
 * emergent property of some data generator.
 *
 * TIERS. `--vram-mb` sizes the top tier, `--total-mb` the data. Their ratio is
 * the variable that matters: at 1x everything is top-tier resident either way
 * and compression is pure overhead; past that the raw image spills first, and
 * whether the compressed image still fits is what compression is being bought
 * for.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
using clio::run::u32;
using clio::run::u64;

namespace {

const clio::run::PoolId kCompressorPool(512, 0);
const clio::run::PoolId kCorePool(513, 0);
constexpr int kLz4WireId = 4;         // registry: {"lz4", 4, ...}      CPU
constexpr int kNvcompLz4WireId = 11;  // registry: {"nvcomp-lz4", 11, ...} GPU

}  // namespace

/**
 * Is page `p` one of the all-zero (highly compressible) pages?
 *
 * Hashed rather than striped (`p % 100 < pct`) so the compressible pages are
 * scattered through the address space. A contiguous or regularly strided
 * layout would let the tier hold a clean prefix of the vector and turn the
 * residency question into a different, easier one than the mixed workload
 * this is meant to represent.
 */
CTP_INLINE_CROSS_FUN bool PageIsZero(u64 p, u32 zero_pct) {
  u64 h = p * 0x9E3779B97F4A7C15ull;
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  return (h % 100u) < zero_pct;
}

/**
 * Element value at global index `i` on page `p`.
 *
 * The non-zero case is a full-width mixed hash: every bit of every byte
 * varies, which is what makes it incompressible. Anything narrower (a small
 * value set, a slowly varying field) leaves byte-level structure that lz4
 * finds, and then --zero-pct stops controlling the ratio.
 */
/**
 * Position weight for the read checksum.
 *
 * An unweighted sum cannot see a PERMUTATION: if the batched fault path
 * mapped a record to the wrong slot and two pages' contents were exchanged,
 * the total would be bit-identical and the run would still report
 * checksum=OK. That is precisely the failure the batched path can produce,
 * so every term is weighted by its own index. Overflow wraps, which is fine
 * -- host and device do the same u64 arithmetic.
 */
CTP_INLINE_CROSS_FUN unsigned long long PosWeight(u64 idx) {
  return static_cast<unsigned long long>(idx) * 2654435761ull + 1ull;
}

CTP_INLINE_CROSS_FUN u32 Value(u64 p, u64 i, u32 zero_pct) {
  if (PageIsZero(p, zero_pct)) return 0u;
  u64 h = (i + 0x1234567ull) * 0xD6E8FEB86659FD93ull;
  h ^= h >> 32;
  h *= 0xD6E8FEB86659FD93ull;
  h ^= h >> 32;
  return static_cast<u32>(h);
}

/**
 * Write pass: fill every page of this block's slice and flush as we stream.
 *
 * The flush of page k is collected while page k+1 is being filled. Flushing
 * and waiting in the same iteration was measured on the flush benchmark to
 * serialize the store behind the fill; a streaming producer has no reason to
 * wait, and the resident-slot budget (--pages) is what bounds how far ahead it
 * may run.
 */
__global__ void StreamWriteKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceVector<u32> v, u64 pages_per_block,
                                  u32 zero_pct) {
  CLIO_GPU_INIT(info, nullptr);
  const u64 pe = v.h_->elems_per_page_;
  const u64 base_page = static_cast<u64>(blockIdx.x) * pages_per_block;
  long long prev = -1;
  for (u64 k = 0; k < pages_per_block; ++k) {
    const u64 p = base_page + k;
    const u64 off = p * pe;
    v.HoldPage(off, pe);
    for (u64 i = threadIdx.x; i < pe; i += blockDim.x) {
      v[off + i] = Value(p, off + i, zero_pct);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      // Collect the PREVIOUS page's store, then start this one: the store of
      // page k-1 has been in flight for the whole fill of page k.
      if (prev >= 0) {
        v.WaitFlush(static_cast<u64>(prev) * pe, pe);
      }
      v.BeginFlush(off, pe);
      prev = static_cast<long long>(p);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && prev >= 0) {
    v.WaitFlush(static_cast<u64>(prev) * pe, pe);
  }
}

/**
 * Read pass: stream the whole slice back and checksum it.
 *
 * `depth` pages are kept in flight ahead of the one being consumed. A fault is
 * a synchronous round trip to the CPU, so consuming page k while nothing else
 * is outstanding exposes that latency in full, once per page. It matters much
 * more for a compressed vector: a GPU codec runs in its own CUDA context and
 * each fault waits on a driver context switch (~7.6 ms measured), which is
 * latency to be overlapped, not bandwidth to be saved. depth=0 keeps the old
 * strictly-synchronous behaviour for comparison.
 */
__global__ void StreamReadKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<u32> v, u64 pages_per_block,
                                 u32 depth, unsigned long long *sum) {
  CLIO_GPU_INIT(info, nullptr);
  const u64 pe = v.h_->elems_per_page_;
  const u64 base_page = static_cast<u64>(blockIdx.x) * pages_per_block;
  unsigned long long acc = 0;
  // Prime the pipeline.
  if (threadIdx.x == 0) {
    for (u64 d = 0; d < depth && d < pages_per_block; ++d) {
      v.BeginFetch(base_page + d);
    }
  }
  __syncthreads();
  for (u64 k = 0; k < pages_per_block; ++k) {
    const u64 off = (base_page + k) * pe;
    // Start the page `depth` ahead before touching this one, so the fault for
    // it is already in flight by the time we get there.
    if (threadIdx.x == 0 && depth > 0 && k + depth < pages_per_block) {
      v.BeginFetch(base_page + k + depth);
    }
    __syncthreads();
    v.HoldPage(off, pe);
    for (u64 i = threadIdx.x; i < pe; i += blockDim.x) {
      acc += static_cast<unsigned long long>(v.at(off + i)) *
             PosWeight(off + i);
    }
    __syncthreads();
  }
  atomicAdd(sum, acc);
}

/**
 * Streaming read that faults in CHUNKS: one batched get per `chunk` pages,
 * then a fault-free walk over them.
 *
 * Why this exists: a fault costs ~110us of round trip against ~6us of actual
 * 256 KB device-to-device copy, so ~95% of a read is the submission itself.
 * While every page costs one, compressing cannot pay for itself no matter how
 * well it compresses -- there are simply no bytes on the critical path to
 * save. Amortizing the submission over `chunk` pages is what lets a smaller
 * stored size turn into a shorter read.
 */
__global__ void StreamReadBatchedKernel(clio::run::IpcManagerGpuInfo info,
                                        gv::DeviceVector<u32> v,
                                        u64 pages_per_block, u32 chunk,
                                        unsigned long long *sum) {
  CLIO_GPU_INIT(info, nullptr);
  const u64 pe = v.h_->elems_per_page_;
  const u64 base_page = static_cast<u64>(blockIdx.x) * pages_per_block;
  unsigned long long acc = 0;
  for (u64 k = 0; k < pages_per_block; k += chunk) {
    u64 n = pages_per_block - k;
    if (n > chunk) n = chunk;
    if (threadIdx.x == 0) {
      gv::DeviceVectorTestAccess::FetchPagesBatched(v, base_page + k, static_cast<u32>(n));
    }
    __syncthreads();
    // The chunk is resident now, so this walk faults on nothing.
    for (u64 j = 0; j < n; ++j) {
      const u64 off = (base_page + k + j) * pe;
      v.HoldPage(off, pe);
      for (u64 i = threadIdx.x; i < pe; i += blockDim.x) {
        acc += static_cast<unsigned long long>(v.at(off + i)) *
               PosWeight(off + i);
      }
      __syncthreads();
    }
  }
  atomicAdd(sum, acc);
}

#if !CTP_IS_DEVICE_PASS

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

double GbPerSec(u64 bytes, double ms) {
  if (ms <= 0.0) return 0.0;
  return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) /
         (ms / 1000.0);
}

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 16;
  u64 pages_per_block_resident = 8;  // page-cache slots each block may hold
  u64 page_kb = 256;
  u64 total_mb = 512;
  u32 zero_pct = 30;
  u64 vram_mb = 256;      // top ("VRAM") tier capacity
  u64 spill_mb = 16384;   // tier below it
  u32 threads = 256;
  u64 stored_sample = 1024;  // pages probed for the stored-size estimate
  u32 prefetch_depth = 0;    // pages kept in flight ahead of the reader
  u32 read_batch = 0;        // pages per batched get (0 = per-page faults)
  u32 rt_threads = 8;        // runtime worker threads
  bool compressed = false;
  bool gpu_codec = false;  // nvcomp: decompress ON the GPU
  const char *tier_type = "pinned";  // pageable staging halves device I/O
  // The tier BELOW the top one. "ram" keeps the spill in host memory, which is
  // still fast -- so compression only buys capacity there, never read
  // bandwidth. Point this at a file to model the reason compression exists:
  // when the overflow medium is slow, compressing means fewer bytes actually
  // read back, and the compressed read can beat the raw one outright.
  const char *spill_type = "ram";
  const char *spill_path = "";

  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> const char * {
      return (i + 1 < argc) ? argv[++i] : "0";
    };
    if (f == "--blocks") blocks = static_cast<u32>(std::atoll(next()));
    else if (f == "--pages") pages_per_block_resident = std::atoll(next());
    else if (f == "--page-kb") page_kb = std::atoll(next());
    else if (f == "--total-mb") total_mb = std::atoll(next());
    else if (f == "--zero-pct") zero_pct = static_cast<u32>(std::atoll(next()));
    else if (f == "--vram-mb") vram_mb = std::atoll(next());
    else if (f == "--spill-mb") spill_mb = std::atoll(next());
    else if (f == "--threads") threads = static_cast<u32>(std::atoll(next()));
    else if (f == "--stored-sample") stored_sample = std::atoll(next());
    else if (f == "--prefetch") prefetch_depth = static_cast<u32>(std::atoll(next()));
    else if (f == "--read-batch") read_batch = static_cast<u32>(std::atoll(next()));
    else if (f == "--rt-threads") rt_threads = static_cast<u32>(std::atoll(next()));
    else if (f == "--compressed") compressed = true;
    else if (f == "--gpu-codec") { compressed = true; gpu_codec = true; }
    else if (f == "--tier-type") tier_type = next();
    else if (f == "--spill-type") spill_type = next();
    else if (f == "--spill-path") spill_path = next();
    else if (f == "--help") {
      std::printf(
          "usage: %s [--blocks N] [--pages P] [--page-kb K] [--total-mb M]\n"
          "          [--zero-pct X] [--vram-mb V] [--spill-mb S]\n"
          "          [--threads T] [--compressed] [--tier-type ram|pinned]\n"
          "\n"
          "  --blocks    CUDA blocks streaming in parallel\n"
          "  --pages     resident page-cache slots PER BLOCK\n"
          "  --page-kb   page granularity (the codec's unit)\n"
          "  --total-mb  total data streamed\n"
          "  --zero-pct  %% of pages that are all-zero (compressible)\n"
          "  --vram-mb   top-tier capacity: the 'VRAM' the data is sized against\n"
          "  --spill-type  tier below the top one: ram|file (default ram)\n"
          "  --spill-path  backing file when --spill-type file\n"
          "  --prefetch    pages kept in flight ahead of the reader (0 = sync)\n"
          "  --read-batch  pages per BATCHED get (0 = one fault per page)\n"
          "  --rt-threads  runtime worker threads (default 8)\n",
          argv[0]);
      return 0;
    }
  }

  const u64 page_bytes = page_kb * 1024;
  const u64 page_elems = page_bytes / sizeof(u32);
  const u64 total_bytes = total_mb * 1024ull * 1024ull;
  u64 total_pages = total_bytes / page_bytes;
  // Every block gets the same number of pages so the slices tile the vector
  // exactly; a ragged tail would leave the last block doing less I/O than the
  // others and skew the aggregate bandwidth.
  const u64 pages_per_block = total_pages / blocks;
  if (pages_per_block == 0) {
    std::fprintf(stderr,
                 "stream: %llu MB of %llu KB pages does not cover %u blocks\n",
                 (unsigned long long) total_mb, (unsigned long long) page_kb,
                 blocks);
    return 1;
  }
  total_pages = pages_per_block * blocks;
  const u64 n = total_pages * page_elems;
  const u64 logical = total_pages * page_bytes;

  {
    std::ofstream cfg("gv_stream_bench.yaml");
    cfg << "networking:\n  port: 9439\n\n"
        // Workers that fall asleep add their wake-up latency to every page
        // fault, which is a synchronous round trip; that noise is the same
        // order as the effect being measured.
        << "runtime:\n  num_threads: " << rt_threads
        << "\n  queue_depth: 8192\n"
        << "  first_busy_wait: 2000000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"2GB\"\n\n"
        // The compressor sits in FRONT of the core in both arms, so the raw
        // arm pays whatever the chain itself costs and the delta is the codec.
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"513.0\"\n"
        << "    storage:\n"
        << "      - path: \"gv_stream_top\"\n"
        << "        bdev_type: \"" << tier_type << "\"\n"
        << "        capacity_limit: \"" << vram_mb << "MB\"\n"
        << "        score: 1.0\n"
        << "      - path: \""
        << (std::string(spill_type) == "file"
                ? (std::string(spill_path).empty()
                       ? std::string("/tmp/gv_stream_spill.bin")
                       : std::string(spill_path))
                : std::string("ram::gv_stream_spill"))
        << "\"\n"
        << "        bdev_type: \"" << spill_type << "\"\n"
        << "        capacity_limit: \"" << spill_mb << "MB\"\n"
        << "        score: 0.2\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_stream_bench.yaml", 1);

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "stream: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "stream: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  const std::string tag =
      std::string("gvs_") + (!compressed ? "raw" : (gpu_codec ? "nvlz4" : "lz4")) + "_" +
      std::to_string(total_mb) + "_" + std::to_string(vram_mb) + "_" +
      std::to_string(zero_pct) + "_" + std::to_string(page_kb);

  // A prefetch depth at or above the per-block slot count deadlocks: every
  // slot ends up pinned by an in-flight fetch, so the page actually being read
  // can never claim one and the block spins forever. Leave at least one slot
  // for the reader.
  if (pages_per_block_resident > 1 &&
      prefetch_depth > pages_per_block_resident - 1) {
    prefetch_depth = static_cast<u32>(pages_per_block_resident - 1);
    std::fprintf(stderr, "stream: prefetch clamped to %u (slots/blk=%llu)\n",
                 prefetch_depth,
                 (unsigned long long) pages_per_block_resident);
  } else if (pages_per_block_resident <= 1) {
    prefetch_depth = 0;
  }

  gv::Vector<u32> vec(tag, {0}, page_bytes, blocks,
                      static_cast<u32>(pages_per_block_resident), n,
                      kCompressorPool,
                      !compressed ? 0
                                  : (gpu_codec ? kNvcompLz4WireId : kLz4WireId));

  vec.EnableStats();
  auto dev = vec.GetDevice(0);

  // ---- write stream --------------------------------------------------------
  const double w0 = NowMs();
  StreamWriteKernel<<<blocks, threads>>>(gpu, dev, pages_per_block, zero_pct);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "stream: write kernel failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  const double write_ms = NowMs() - w0;
  {
    // Counters at the PHASE BOUNDARY. Totals alone cannot say whether a
    // get_error happened while writing (a first touch of a page that does not
    // exist yet -- benign) or while reading (a failed fault, which is not).
    const auto st = vec.ReadStats(0);
    std::fprintf(stderr,
                 "GVSTAT-WRITE faults=%llu puts=%llu evicts=%llu "
                 "get_errors=%llu\n",
                 (unsigned long long) st.faults, (unsigned long long) st.puts,
                 (unsigned long long) st.evicts,
                 (unsigned long long) st.get_errors);
  }

  // ---- what it actually cost to store --------------------------------------
  // Summing the stored blob sizes is the only honest compression ratio: it is
  // what the tier is charged for, and it accounts for pages the codec refused
  // to shrink (which lz4 stores raw, plus our header).
  //
  // Sampled, not exhaustive: each GetBlobSize is a synchronous round trip, and
  // at 4 GB / 256 KB that is 16k of them -- minutes of wall clock bolted onto
  // a benchmark whose phases take seconds. The sample is strided rather than
  // random so it covers the address space uniformly, and the zero/non-zero
  // split is a deterministic hash of the page id, so a uniform sample of pages
  // reproduces the population ratio closely.
  u64 stored = 0;
  u64 sampled_pages = 0;
  {
    clio::cte::core::Client core(kCorePool);
    const u64 stride =
        (total_pages > stored_sample) ? (total_pages / stored_sample) : 1;
    for (u64 p = 0; p < total_pages; p += stride) {
      char name[32];
      gv::PageBlobName(p, name);
      auto sz = core.AsyncGetBlobSize(vec.TagId(), name);
      sz.Wait();
      if (sz->GetReturnCode() == 0) {
        stored += sz->size_;
        ++sampled_pages;
      }
    }
    // Scale the sample back up to the whole vector.
    if (sampled_pages > 0 && sampled_pages < total_pages) {
      stored = static_cast<u64>(
          (static_cast<double>(stored) / static_cast<double>(sampled_pages)) *
          static_cast<double>(total_pages));
    }
  }

  // ---- read stream ---------------------------------------------------------
  unsigned long long *d_sum = nullptr;
  cudaMalloc(&d_sum, sizeof(unsigned long long));
  cudaMemset(d_sum, 0, sizeof(unsigned long long));
  const double r0 = NowMs();
  if (read_batch > 0) {
    // The chunk has to fit the cache -- a batch larger than the block's slots
    // would evict pages the same batch just claimed -- and one task holds at
    // most kPodMultiMax records.
    if (read_batch > pages_per_block_resident) {
      read_batch = static_cast<u32>(pages_per_block_resident);
    }
    if (read_batch > clio::cte::core::kPodMultiMax) {
      read_batch = clio::cte::core::kPodMultiMax;
    }
    StreamReadBatchedKernel<<<blocks, threads>>>(gpu, dev, pages_per_block,
                                                 read_batch, d_sum);
  } else {
    StreamReadKernel<<<blocks, threads>>>(gpu, dev, pages_per_block,
                                          prefetch_depth, d_sum);
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "stream: read kernel failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  const double read_ms = NowMs() - r0;
  unsigned long long got = 0;
  cudaMemcpy(&got, d_sum, sizeof(got), cudaMemcpyDeviceToHost);
  cudaFree(d_sum);

  unsigned long long want = 0;
  for (u64 p = 0; p < total_pages; ++p) {
    if (PageIsZero(p, zero_pct)) continue;  // contributes nothing
    const u64 off = p * page_elems;
    for (u64 i = 0; i < page_elems; ++i) {
      want += static_cast<unsigned long long>(Value(p, off + i, zero_pct)) *
              PosWeight(off + i);
    }
  }

  // Pager counters. Read these carefully before concluding anything:
  //
  //   faults      ~2x the page count, because the WRITE phase faults each
  //               page before writing it (write-allocate) and the read phase
  //               faults it again.
  //   get_errors  ~1x the page count, and BENIGN: the write phase's first
  //               touch of a page asks the CTE for a blob that does not exist
  //               yet. It looks like a total failure of the read path and is
  //               not one -- verify against num_ok on the batch before
  //               believing otherwise.
  //   prefetch    counts batched claims, so ~1x the page count on a read pass.
  {
    const auto st = vec.ReadStats(0);
    std::fprintf(stderr,
                 "GVSTAT faults=%llu puts=%llu evicts=%llu prefetch=%llu "
                 "get_errors=%llu\n",
                 (unsigned long long) st.faults, (unsigned long long) st.puts,
                 (unsigned long long) st.evicts,
                 (unsigned long long) st.prefetches,
                 (unsigned long long) st.get_errors);
  }

  const bool ok = (got == want);
  const double ratio =
      (stored == 0) ? 0.0
                    : static_cast<double>(logical) / static_cast<double>(stored);
  std::printf(
      "GVS mode=%s blocks=%u threads=%u pages/blk=%llu slots/blk=%llu "
      "page=%lluKB zero=%u%% vram=%lluMB total=%.0fMB stored=%.0fMB "
      "ratio=%.2fx oversub=%.2fx "
      "write_ms=%.1f write_GBps=%.2f read_ms=%.1f read_GBps=%.2f checksum=%s\n",
      !compressed ? "raw" : (gpu_codec ? "nvcomp-lz4" : "lz4"), blocks, threads,
      (unsigned long long) pages_per_block,
      (unsigned long long) pages_per_block_resident,
      (unsigned long long) page_kb, zero_pct, (unsigned long long) vram_mb,
      logical / (1024.0 * 1024.0), stored / (1024.0 * 1024.0), ratio,
      static_cast<double>(logical) /
          (static_cast<double>(vram_mb) * 1024.0 * 1024.0),
      write_ms, GbPerSec(logical, write_ms), read_ms,
      GbPerSec(logical, read_ms), ok ? "OK" : "MISMATCH");
  return ok ? 0 : 1;
}

#endif  // !CTP_IS_DEVICE_PASS
