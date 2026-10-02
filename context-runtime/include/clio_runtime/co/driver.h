/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * Host side of the portable GPU coroutine mechanism: stack layout and the
 * relaunch loop.
 *
 * DELIBERATELY BACKEND-AGNOSTIC. This header allocates nothing. The caller
 * supplies the memory -- sycl::malloc_shared, cudaMallocManaged, hipMallocManaged
 * or plain operator new for the host emulation -- and this file only lays out
 * the group regions and reads the headers back. That is why there is no `#if`
 * in it: allocation is the one part of the mechanism that genuinely differs per
 * backend, so it is the one part that stays outside.
 *
 * The headers must be host-readable after a launch completes. Shared/managed
 * memory is the simple way; a device allocation plus a copy-back of just the
 * GroupHeader array works identically and is what a latency-sensitive driver
 * should do.
 */
#ifndef CLIO_RUNTIME_CO_DRIVER_H_
#define CLIO_RUNTIME_CO_DRIVER_H_

#include <cstring>

#include <clio_runtime/co/coro.h>

namespace clio::co {

/**
 * Lays out a coroutine stack over caller-supplied memory.
 *
 * Layout, per group: [GroupHeader][frames], where the frame area is the sum
 * along the deepest suspending call chain -- kStackBytes_<entry> -- times the
 * group size, because each work-item owns a private slice of every frame.
 */
class Stack {
 public:
  /** @param mem            caller-owned, device-visible, host-readable
   *  @param n_groups       work-groups in the grid
   *  @param group_size     work-items per group
   *  @param frame_bytes    kStackBytes_<entry>: per-item bytes for the whole
   *                        deepest chain, NOT multiplied by group_size */
  Stack(void *mem, u32 n_groups, u32 group_size, u32 frame_bytes)
      : mem_(static_cast<char *>(mem)) {
    view_.base = mem_;
    view_.n_groups = n_groups;
    view_.group_size = group_size;
    view_.bytes_per_group =
        static_cast<u32>(sizeof(GroupHeader)) + frame_bytes * group_size;
    Reset();
  }

  /** Bytes the caller must allocate for this configuration. */
  static std::size_t BytesNeeded(u32 n_groups, u32 group_size,
                                 u32 frame_bytes) {
    return static_cast<std::size_t>(n_groups) *
           (sizeof(GroupHeader) + static_cast<std::size_t>(frame_bytes) *
                                      group_size);
  }

  /** Put every group back to kFresh. Frame bytes are left alone: they are
   *  meaningless until a Push writes them, and zeroing them would only hide a
   *  save-list bug that the differential test is there to find. */
  void Reset() {
    for (u32 g = 0; g < view_.n_groups; ++g) {
      GroupHeader *h = view_.Header(g);
      std::memset(h, 0, sizeof(*h));
      h->status = kFresh;
    }
  }

  StackView View() const { return view_; }
  GroupHeader *Header(u32 g) const { return view_.Header(g); }
  u32 NumGroups() const { return view_.n_groups; }

 private:
  char *mem_;
  StackView view_;
};

/** What a launch left behind. */
struct Progress {
  u32 parked = 0;
  u32 done = 0;
  u32 fresh = 0;
  bool Pending() const { return parked != 0 || fresh != 0; }
};

/**
 * The relaunch loop's bookkeeping.
 *
 * Usage:
 *     Driver drv(stack);
 *     do {
 *       Launch(...);  q.wait();
 *     } while (drv.Step(servicer).Pending());
 *
 * `servicer(group, wait_tag)` is called once per parked group, and is where
 * the host does whatever the group is waiting for. It runs between launches,
 * which is the entire reason suspension is a kernel exit: no concurrency
 * between the servicer and the kernel means no system-scope atomics and no
 * fences, on any backend (invariant I8).
 */
class Driver {
 public:
  explicit Driver(const Stack &st) : st_(st) {}

  template <class ServiceFn>
  Progress Step(ServiceFn &&servicer) {
    Progress p;
    for (u32 g = 0; g < st_.NumGroups(); ++g) {
      GroupHeader *h = st_.Header(g);
      switch (h->status) {
        case kParked:
          ++p.parked;
          servicer(g, h->wait_tag);
          break;
        case kDone:
          ++p.done;
          break;
        default:
          ++p.fresh;
          break;
      }
    }
    ++launches_;
    return p;
  }

  u64 Launches() const { return launches_; }

 private:
  const Stack &st_;
  u64 launches_ = 0;
};

}  // namespace clio::co

#endif  // CLIO_RUNTIME_CO_DRIVER_H_
