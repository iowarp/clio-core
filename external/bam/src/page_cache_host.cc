/**
 * page_cache_host.cc -- Host-side page cache manager implementation
 */
#include <bam/page_cache_host.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

namespace bam {

PageCache::PageCache(const PageCacheConfig &config)
    : config_(config),
      d_cache_mem_(nullptr),
      d_page_states_(nullptr),
      d_page_tags_(nullptr),
      d_page_locks_(nullptr),
      d_page_refs_(nullptr),
      d_bus_addrs_(nullptr),
      host_buf_(nullptr),
      host_buf_size_(0) {
  memset(&dev_state_, 0, sizeof(dev_state_));

  if (init_gpu_memory() != 0) {
    fprintf(stderr, "bam::PageCache: failed to init GPU memory\n");
    return;
  }

  if (config_.backend == BackendType::kNvme) {
    if (init_nvme_backend() != 0) {
      fprintf(stderr, "bam::PageCache: NVMe init failed, falling back to host memory\n");
      config_.backend = BackendType::kHostMemory;
    }
  }
}

PageCache::~PageCache() {
  if (d_cache_mem_)   cudaFree(d_cache_mem_);
  if (d_page_states_) cudaFree(d_page_states_);
  if (d_page_tags_)   cudaFree(d_page_tags_);
  if (d_page_locks_)  cudaFree(d_page_locks_);
  if (d_page_refs_)   cudaFree(d_page_refs_);
  if (d_bus_addrs_)   cudaFree(d_bus_addrs_);

  if (host_buf_) {
    cudaFreeHost(host_buf_);
    host_buf_ = nullptr;
  }

  ctrl_.reset();
}

int PageCache::init_gpu_memory() {
  size_t page_size = config_.page_size;
  uint32_t num_pages = config_.num_pages;

  // Compute page_shift
  uint32_t page_shift = 0;
  size_t s = page_size;
  while (s > 1) { s >>= 1; page_shift++; }

  // Allocate cache pages in GPU VRAM
  cudaError_t err;
  err = cudaMalloc(&d_cache_mem_, page_size * num_pages);
  if (err != cudaSuccess) {
    fprintf(stderr, "bam::PageCache: cudaMalloc cache_mem: %s\n",
            cudaGetErrorString(err));
    return -1;
  }

  // Allocate metadata arrays
  err = cudaMalloc(&d_page_states_, num_pages * sizeof(uint32_t));
  if (err != cudaSuccess) goto fail;
  err = cudaMalloc(&d_page_tags_, num_pages * sizeof(uint64_t));
  if (err != cudaSuccess) goto fail;
  err = cudaMalloc(&d_page_locks_, num_pages * sizeof(uint32_t));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_page_refs_, num_pages * sizeof(uint32_t));
  if (err != cudaSuccess) goto fail;

  // Initialize metadata to zero
  cudaMemset(d_page_states_, 0, num_pages * sizeof(uint32_t));
  cudaMemset(d_page_refs_, 0, num_pages * sizeof(uint32_t));
  cudaMemset(d_page_tags_, 0xFF, num_pages * sizeof(uint64_t));  // Invalid tag
  cudaMemset(d_page_locks_, 0, num_pages * sizeof(uint32_t));

  // Fill device state
  dev_state_.cache_mem = d_cache_mem_;
  dev_state_.page_states = d_page_states_;
  dev_state_.page_tags = d_page_tags_;
  dev_state_.page_locks = d_page_locks_;
  dev_state_.page_refs = d_page_refs_;
  dev_state_.page_size = page_size;
  dev_state_.num_pages = num_pages;
  dev_state_.page_shift = page_shift;

  return 0;

fail:
  fprintf(stderr, "bam::PageCache: cudaMalloc metadata: %s\n",
          cudaGetErrorString(err));
  if (d_cache_mem_)   { cudaFree(d_cache_mem_);   d_cache_mem_ = nullptr; }
  if (d_page_states_) { cudaFree(d_page_states_); d_page_states_ = nullptr; }
  if (d_page_tags_)   { cudaFree(d_page_tags_);   d_page_tags_ = nullptr; }
  if (d_page_locks_)  { cudaFree(d_page_locks_);  d_page_locks_ = nullptr; }
  return -1;
}

int PageCache::init_nvme_backend() {
  if (!config_.nvme_dev) {
    fprintf(stderr, "bam::PageCache: nvme_dev not specified\n");
    return -1;
  }

  ctrl_ = std::make_unique<NvmeController>();
  if (ctrl_->attach(config_.nvme_dev) != 0) {
    ctrl_.reset();
    return -1;
  }

  if (ctrl_->mmap_doorbells() != 0) {
    ctrl_.reset();
    return -1;
  }

  // Create queue pairs
  uint32_t nq = config_.num_queues > 0 ? config_.num_queues : 1;
  uint32_t qdepth = config_.queue_depth > 0 ? config_.queue_depth : 64;
  for (uint32_t i = 0; i < nq; i++) {
    if (ctrl_->create_queue(qdepth, qdepth) < 0) {
      fprintf(stderr, "bam::PageCache: failed to create queue %u\n", i);
      ctrl_.reset();
      return -1;
    }
  }

  // Allocate one DMA buffer per cache page for NVMe I/O
  std::vector<uint64_t> bus_addrs(config_.num_pages);
  for (uint32_t i = 0; i < config_.num_pages; i++) {
    int buf_idx = ctrl_->alloc_buffer(config_.page_size);
    if (buf_idx < 0) {
      fprintf(stderr, "bam::PageCache: failed to alloc buffer %u\n", i);
      ctrl_.reset();
      return -1;
    }
    bus_addrs[i] = ctrl_->buffers()[buf_idx].bus_addr;
  }

  // Copy bus addresses to GPU
  cudaError_t err = cudaMalloc(&d_bus_addrs_,
                                config_.num_pages * sizeof(uint64_t));
  if (err != cudaSuccess) {
    ctrl_.reset();
    return -1;
  }
  cudaMemcpy(d_bus_addrs_, bus_addrs.data(),
             config_.num_pages * sizeof(uint64_t),
             cudaMemcpyHostToDevice);

  return 0;
}

int PageCache::init_host_backend() {
  // Host backend is initialized lazily via alloc_host_backing()
  return 0;
}

int PageCache::alloc_host_backing(size_t total_bytes) {
  if (host_buf_) {
    if (host_buf_size_ >= total_bytes) return 0;
    cudaFreeHost(host_buf_);
    host_buf_ = nullptr;
  }

  cudaError_t err = cudaMallocHost(&host_buf_, total_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "bam::PageCache: cudaMallocHost: %s\n",
            cudaGetErrorString(err));
    return -1;
  }
  host_buf_size_ = total_bytes;
  memset(host_buf_, 0, total_bytes);
  return 0;
}

QueuePairDevice PageCache::queue_pair_device(uint32_t idx) const {
  QueuePairDevice qpd;
  memset(&qpd, 0, sizeof(qpd));

  if (!ctrl_ || idx >= ctrl_->queues().size()) return qpd;

  const auto &qp = ctrl_->queues()[idx];
  qpd.sq = qp.sq_ptr;
  qpd.cq = qp.cq_ptr;
  qpd.sq_doorbell = reinterpret_cast<volatile uint32_t *>(
      static_cast<char *>(ctrl_->doorbell_base()) + qp.sq_db_offset);
  qpd.cq_doorbell = reinterpret_cast<volatile uint32_t *>(
      static_cast<char *>(ctrl_->doorbell_base()) + qp.cq_db_offset);
  qpd.sq_depth = qp.sq_depth;
  qpd.cq_depth = qp.cq_depth;
  qpd.nsid = ctrl_->nsid();

  // Compute lba_shift
  uint32_t lba = ctrl_->lba_size();
  qpd.lba_shift = 0;
  while (lba > 1) { lba >>= 1; qpd.lba_shift++; }

  return qpd;
}

}  // namespace bam

namespace bam {


}  // namespace bam
