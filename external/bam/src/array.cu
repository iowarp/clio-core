/**
 * array.cu -- bam::Array<T> host-side implementation
 *
 * Template instantiations for common types.
 */
#include <bam/array.cuh>
#include <bam/page_cache_host.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

namespace bam {

template <typename T>
Array<T>::Array(uint64_t num_elements, PageCache &cache)
    : num_elements_(num_elements), cache_(&cache) {
  memset(&dev_, 0, sizeof(dev_));

  dev_.cache_state = cache.device_state();
  dev_.num_elements = num_elements;
  dev_.total_bytes = num_elements * sizeof(T);
  dev_.backend = cache.backend();

  if (cache.backend() == BackendType::kNvme && cache.controller()) {
    dev_.qp = cache.queue_pair_device(0);
    dev_.buf_bus_addrs = cache.device_bus_addrs();
    dev_.host_base = nullptr;
  } else {
    // Host memory fallback — allocate backing store if needed
    size_t total = num_elements * sizeof(T);
    if (cache.alloc_host_backing(total) != 0) {
      fprintf(stderr, "bam::Array: failed to alloc host backing store\n");
    }
    dev_.host_base = cache.host_buffer();
    dev_.buf_bus_addrs = nullptr;
    memset(&dev_.qp, 0, sizeof(dev_.qp));
  }
}

template <typename T>
Array<T>::~Array() {
  // Nothing to free — cache owns all GPU memory
}

template <typename T>
int Array<T>::load_from_host(const T *host_data, uint64_t count) {
  if (count > num_elements_) count = num_elements_;
  size_t bytes = count * sizeof(T);

  if (cache_->backend() == BackendType::kHostMemory) {
    // Copy directly into the pinned host buffer
    uint8_t *buf = cache_->host_buffer();
    if (!buf) {
      fprintf(stderr, "bam::Array::load_from_host: no host buffer\n");
      return -1;
    }
    memcpy(buf, host_data, bytes);
    return 0;
  }

  // NVMe mode: write data to NVMe storage via the controller buffers
  // For now, use the DMA buffers page-by-page
  NvmeController *ctrl = cache_->controller();
  if (!ctrl || !ctrl->is_attached()) return -1;

  size_t page_size = cache_->config().page_size;
  const auto &buffers = ctrl->buffers();
  size_t offset = 0;

  while (offset < bytes) {
    size_t chunk = (bytes - offset < page_size) ? (bytes - offset) : page_size;
    uint32_t buf_idx = (offset / page_size) % buffers.size();

    // Copy to DMA buffer, then NVMe would write it
    // For basic version, just copy to the mmap'd DMA buffer
    memcpy(buffers[buf_idx].host_ptr, reinterpret_cast<const char *>(host_data) + offset, chunk);

    // TODO: Submit NVMe write via host-side admin path
    // For now this is a placeholder — real NVMe writes would go through
    // the kernel module or a host-side I/O submission path
    offset += chunk;
  }

  return 0;
}

// Explicit template instantiations for common types
template class Array<float>;
template class Array<double>;
template class Array<int32_t>;
template class Array<uint32_t>;
template class Array<int64_t>;
template class Array<uint64_t>;
template class Array<uint8_t>;
template class Array<char>;
template class Array<int8_t>;
template class Array<int16_t>;
template class Array<uint16_t>;

}  // namespace bam
