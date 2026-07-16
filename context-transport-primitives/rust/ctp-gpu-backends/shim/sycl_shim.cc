// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// SYCL shim for the ctp-gpu-backends `sycl` feature. SYCL 2020 is a
// C++-only API (templates, ADL-driven `sycl::` free functions, RAII
// `sycl::queue`), so Rust cannot bind it directly the way ctp-gpu binds
// the CUDA driver's C ABI. This file exposes the GpuApi SYCL branch of
// clio_ctp/util/gpu_api.h as a flat C ABI over opaque handles, modelled
// on ctp-coroutine/shim/boost_fiber_shim.cc.
//
// Design notes:
//  - EXCEPTIONS MUST NOT ESCAPE. SYCL reports errors by throwing
//    sycl::exception (e.g. queue construction on a host with no GPU, or
//    an async error surfaced by wait_and_throw). Unwinding across the C
//    ABI into Rust frames is UB, so every entry point is wrapped in
//    guard() and returns a CtpSyclStatus code instead.
//  - NO thread_local and NO mutable globals (project rule; header
//    thread_locals also duplicate per-DLL on Windows). Rather than a
//    thread_local "last error", every fallible entry point takes a
//    caller-owned (err, errlen) buffer that guard() fills from what().
//    This keeps the shim reentrant with zero hidden state.
//  - No process-wide default queue. gpu_api.h's GpuApi::SyclQueue() is a
//    function-static `sycl::queue` that is never destroyed; the Rust side
//    models queue lifetime explicitly (RAII) instead, so the shim only
//    ever operates on a handle the caller owns.
//
// Build: must be compiled by a SYCL compiler (Intel oneAPI DPC++ `icpx
// -fsycl`, or AdaptiveCpp `acpp`) against SYCL 2020 headers, and linked
// against the SYCL runtime (-lsycl). See the module docs in src/sycl.rs.

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// Status codes. Mirrored by `Status` in src/sycl.rs — keep in sync.
#define CTP_SYCL_OK 0
#define CTP_SYCL_ERR_EXCEPTION 1
#define CTP_SYCL_ERR_INVALID 2
#define CTP_SYCL_ERR_ALLOC 3
#define CTP_SYCL_ERR_UNKNOWN 4

namespace {

/** Copy `msg` into the caller's buffer, truncating; always NUL-terminates. */
void SetErr(char *err, size_t errlen, const char *msg) {
  if (err == nullptr || errlen == 0) {
    return;
  }
  size_t n = std::strlen(msg);
  if (n >= errlen) {
    n = errlen - 1;  // reserve the NUL
  }
  std::memcpy(err, msg, n);
  err[n] = '\0';
}

/**
 * Run `fn` and translate any C++ exception into a status code + message.
 * Templates cannot have C linkage, so this lives outside `extern "C"`.
 */
template <typename Fn>
int Guard(char *err, size_t errlen, Fn &&fn) {
  try {
    return fn();
  } catch (const sycl::exception &e) {
    SetErr(err, errlen, e.what());
    return CTP_SYCL_ERR_EXCEPTION;
  } catch (const std::exception &e) {
    SetErr(err, errlen, e.what());
    return CTP_SYCL_ERR_EXCEPTION;
  } catch (...) {
    SetErr(err, errlen, "unknown C++ exception");
    return CTP_SYCL_ERR_UNKNOWN;
  }
}

sycl::queue *AsQueue(void *q) { return static_cast<sycl::queue *>(q); }

}  // namespace

extern "C" {

/**
 * Number of SYCL GPU devices across all platforms. `GpuApi::GetDeviceCount`
 * analog. A host with no SYCL GPU yields 0 (not an error).
 */
int ctp_sycl_device_count(int *out_count, char *err, size_t errlen) {
  if (out_count == nullptr) {
    SetErr(err, errlen, "ctp_sycl_device_count: null out_count");
    return CTP_SYCL_ERR_INVALID;
  }
  *out_count = 0;
  return Guard(err, errlen, [&] {
    int ngpu = 0;
    for (const auto &p : sycl::platform::get_platforms()) {
      ngpu += static_cast<int>(p.get_devices(sycl::info::device_type::gpu).size());
    }
    *out_count = ngpu;
    return CTP_SYCL_OK;
  });
}

/**
 * Create a heap `sycl::queue` on the default GPU device and return an
 * opaque handle. `in_order != 0` adds sycl::property::queue::in_order,
 * matching GpuApi::CreateStream(); `in_order == 0` matches the
 * out-of-order GpuApi::SyclQueue(). Fails (does not abort) when no GPU
 * device is available — queue construction throws in that case.
 */
int ctp_sycl_queue_create(int in_order, void **out_queue, char *err,
                          size_t errlen) {
  if (out_queue == nullptr) {
    SetErr(err, errlen, "ctp_sycl_queue_create: null out_queue");
    return CTP_SYCL_ERR_INVALID;
  }
  *out_queue = nullptr;
  return Guard(err, errlen, [&] {
    sycl::queue *q;
    if (in_order != 0) {
      q = new sycl::queue(sycl::gpu_selector_v,
                          sycl::property::queue::in_order{});
    } else {
      q = new sycl::queue(sycl::gpu_selector_v);
    }
    *out_queue = q;
    return CTP_SYCL_OK;
  });
}

/** Destroy a queue handle. `GpuApi::DestroyStream` analog. Null-safe. */
void ctp_sycl_queue_destroy(void *queue) {
  if (queue == nullptr) {
    return;
  }
  // ~queue is not specified to throw, but a destructor must never let one
  // escape into Rust; swallow defensively.
  try {
    delete AsQueue(queue);
  } catch (...) {
  }
}

/**
 * Block until all work submitted to `queue` completes.
 * `GpuApi::Synchronize(stream)` analog. Uses wait_and_throw() so async
 * errors surface here as a status code rather than being swallowed.
 */
int ctp_sycl_queue_wait(void *queue, char *err, size_t errlen) {
  if (queue == nullptr) {
    SetErr(err, errlen, "ctp_sycl_queue_wait: null queue");
    return CTP_SYCL_ERR_INVALID;
  }
  return Guard(err, errlen, [&] {
    AsQueue(queue)->wait_and_throw();
    return CTP_SYCL_OK;
  });
}

/** Device name for `queue`, truncated into `buf`. Always NUL-terminates. */
int ctp_sycl_queue_device_name(void *queue, char *buf, size_t buflen,
                               char *err, size_t errlen) {
  if (queue == nullptr || buf == nullptr || buflen == 0) {
    SetErr(err, errlen, "ctp_sycl_queue_device_name: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  return Guard(err, errlen, [&] {
    std::string name =
        AsQueue(queue)->get_device().get_info<sycl::info::device::name>();
    SetErr(buf, buflen, name.c_str());
    return CTP_SYCL_OK;
  });
}

/** `sycl::malloc_device` — device USM; NOT host-dereferenceable. */
int ctp_sycl_malloc_device(void *queue, size_t bytes, void **out_ptr,
                           char *err, size_t errlen) {
  if (queue == nullptr || out_ptr == nullptr) {
    SetErr(err, errlen, "ctp_sycl_malloc_device: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  *out_ptr = nullptr;
  return Guard(err, errlen, [&] {
    void *p = sycl::malloc_device(bytes, *AsQueue(queue));
    if (p == nullptr) {
      SetErr(err, errlen, "sycl::malloc_device returned null");
      return CTP_SYCL_ERR_ALLOC;
    }
    *out_ptr = p;
    return CTP_SYCL_OK;
  });
}

/** `sycl::malloc_shared` — USM shared; host- and device-accessible. */
int ctp_sycl_malloc_shared(void *queue, size_t bytes, void **out_ptr,
                           char *err, size_t errlen) {
  if (queue == nullptr || out_ptr == nullptr) {
    SetErr(err, errlen, "ctp_sycl_malloc_shared: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  *out_ptr = nullptr;
  return Guard(err, errlen, [&] {
    void *p = sycl::malloc_shared(bytes, *AsQueue(queue));
    if (p == nullptr) {
      SetErr(err, errlen, "sycl::malloc_shared returned null");
      return CTP_SYCL_ERR_ALLOC;
    }
    *out_ptr = p;
    return CTP_SYCL_OK;
  });
}

/** `sycl::malloc_host` — pinned host USM; host- and device-accessible. */
int ctp_sycl_malloc_host(void *queue, size_t bytes, void **out_ptr, char *err,
                         size_t errlen) {
  if (queue == nullptr || out_ptr == nullptr) {
    SetErr(err, errlen, "ctp_sycl_malloc_host: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  *out_ptr = nullptr;
  return Guard(err, errlen, [&] {
    void *p = sycl::malloc_host(bytes, *AsQueue(queue));
    if (p == nullptr) {
      SetErr(err, errlen, "sycl::malloc_host returned null");
      return CTP_SYCL_ERR_ALLOC;
    }
    *out_ptr = p;
    return CTP_SYCL_OK;
  });
}

/**
 * `sycl::free(ptr, queue)`. Must be the queue whose context owns `ptr`.
 * Null `ptr` is a no-op.
 */
int ctp_sycl_free(void *queue, void *ptr, char *err, size_t errlen) {
  if (queue == nullptr) {
    SetErr(err, errlen, "ctp_sycl_free: null queue");
    return CTP_SYCL_ERR_INVALID;
  }
  if (ptr == nullptr) {
    return CTP_SYCL_OK;
  }
  return Guard(err, errlen, [&] {
    sycl::free(ptr, *AsQueue(queue));
    return CTP_SYCL_OK;
  });
}

/**
 * Blocking USM memcpy: submit then wait_and_throw on the event, exactly
 * as GpuApi::Memcpy's SYCL branch does. Handles any USM direction.
 */
int ctp_sycl_memcpy(void *queue, void *dst, const void *src, size_t bytes,
                    char *err, size_t errlen) {
  if (queue == nullptr || dst == nullptr || src == nullptr) {
    SetErr(err, errlen, "ctp_sycl_memcpy: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  if (bytes == 0) {
    return CTP_SYCL_OK;
  }
  return Guard(err, errlen, [&] {
    AsQueue(queue)->memcpy(dst, src, bytes).wait_and_throw();
    return CTP_SYCL_OK;
  });
}

/**
 * Async USM memcpy: submit and return without waiting. The returned event
 * is dropped — completion is observed via ctp_sycl_queue_wait(), which is
 * sufficient for both queue kinds (an in_order queue also orders it
 * against later submissions, matching CUDA stream semantics).
 *
 * NOTE: GpuApi::MemcpyAsync has NO SYCL branch, so under a SYCL build the
 * C++ MemcpyAsync silently copies nothing. This shim implements the copy;
 * the divergence is documented in src/sycl.rs.
 */
int ctp_sycl_memcpy_async(void *queue, void *dst, const void *src,
                          size_t bytes, char *err, size_t errlen) {
  if (queue == nullptr || dst == nullptr || src == nullptr) {
    SetErr(err, errlen, "ctp_sycl_memcpy_async: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  if (bytes == 0) {
    return CTP_SYCL_OK;
  }
  return Guard(err, errlen, [&] {
    AsQueue(queue)->memcpy(dst, src, bytes);
    return CTP_SYCL_OK;
  });
}

/** Async USM memset. `GpuApi::MemsetAsync` SYCL branch analog. */
int ctp_sycl_memset_async(void *queue, void *dst, int value, size_t bytes,
                          char *err, size_t errlen) {
  if (queue == nullptr || dst == nullptr) {
    SetErr(err, errlen, "ctp_sycl_memset_async: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  if (bytes == 0) {
    return CTP_SYCL_OK;
  }
  return Guard(err, errlen, [&] {
    AsQueue(queue)->memset(dst, value, bytes);
    return CTP_SYCL_OK;
  });
}

/**
 * `GpuApi::IsDevicePointer` analog: 1 when `ptr` is device USM in this
 * queue's context, else 0. Null is not a device pointer.
 */
int ctp_sycl_is_device_ptr(void *queue, const void *ptr, int *out_is_device,
                           char *err, size_t errlen) {
  if (queue == nullptr || out_is_device == nullptr) {
    SetErr(err, errlen, "ctp_sycl_is_device_ptr: invalid argument");
    return CTP_SYCL_ERR_INVALID;
  }
  *out_is_device = 0;
  if (ptr == nullptr) {
    return CTP_SYCL_OK;
  }
  return Guard(err, errlen, [&] {
    sycl::usm::alloc kind =
        sycl::get_pointer_type(ptr, AsQueue(queue)->get_context());
    *out_is_device = (kind == sycl::usm::alloc::device) ? 1 : 0;
    return CTP_SYCL_OK;
  });
}

}  // extern "C"
