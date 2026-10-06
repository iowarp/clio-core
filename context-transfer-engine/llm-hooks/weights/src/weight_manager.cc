#include "clio_llm/weights/weight_manager.h"

#include <algorithm>
#include <cstdio>

namespace clio_llm {
namespace weights {

WeightManager::WeightManager() : cfg_(Config{}) {}

WeightManager::WeightManager(const Config& cfg) : cfg_(cfg) {}

WeightManager::~WeightManager() { Shutdown(); }

bool WeightManager::Init() {
  if (ready_) return true;

  clio::cte::uvm::GpuVmmConfig vmm_cfg;
  vmm_cfg.va_size_bytes  = cfg_.va_size_bytes;
  vmm_cfg.page_size      = cfg_.page_size;
  vmm_cfg.device         = cfg_.device;
  vmm_cfg.prefetch_window = static_cast<size_t>(cfg_.prefetch_window);
  vmm_cfg.use_cte        = cfg_.use_cte;
  vmm_cfg.cte_tag_name   = cfg_.cte_tag_name;

  // A failed init (e.g. use_cte with no reachable CTE) must fail Init, not
  // leave a manager that pages into nothing.
  if (vmm_.init(vmm_cfg) != CUDA_SUCCESS) {
    fprintf(stderr, "WeightManager: GpuVmm init failed
");
    return false;
  }
  ready_ = true;
  return true;
}

void WeightManager::Shutdown() {
  if (!ready_) return;
  vmm_.destroy();
  ready_ = false;
}

void WeightManager::RegisterLayer(int layer_idx,
                                   size_t page_start,
                                   size_t page_count) {
  layer_ranges_.push_back({layer_idx, page_start, page_count});
}

void WeightManager::PrepareLayer(int layer_idx) {
  if (!ready_) return;

  // Touch all pages for this layer synchronously.
  for (auto& lr : layer_ranges_) {
    if (lr.layer_idx != layer_idx) continue;
    for (size_t p = lr.page_start; p < lr.page_start + lr.page_count; ++p) {
      vmm_.touchPage(p);
    }
    break;
  }

  // Kick off async prefetch for the next prefetch_window layers.
  for (size_t w = 1; w <= cfg_.prefetch_window; ++w) {
    int next = layer_idx + static_cast<int>(w);
    for (auto& lr : layer_ranges_) {
      if (lr.layer_idx != next) continue;
      for (size_t p = lr.page_start; p < lr.page_start + lr.page_count; ++p) {
        vmm_.touchPageAsync(p);
      }
      break;
    }
  }
}

void WeightManager::ReleaseLayer(int layer_idx) {
  if (!ready_) return;
  for (auto& lr : layer_ranges_) {
    if (lr.layer_idx != layer_idx) continue;
    for (size_t p = lr.page_start; p < lr.page_start + lr.page_count; ++p) {
      vmm_.evictPageAsync(p);
    }
    break;
  }
}

void* WeightManager::BaseAddress() const {
  // GpuVmm exposes the reserved VA base via getBasePtr() (CUdeviceptr).
  return reinterpret_cast<void*>(vmm_.getBasePtr());
}

size_t WeightManager::PageSize() const { return cfg_.page_size; }

bool WeightManager::IsReady() const { return ready_; }

clio::cte::uvm::GpuVirtualMemoryManager& WeightManager::Vmm() { return vmm_; }

}  // namespace weights
}  // namespace clio_llm
