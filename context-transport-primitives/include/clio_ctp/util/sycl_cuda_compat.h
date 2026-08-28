/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The CUDA device-code spellings, provided for SYCL.
 *
 * WHY THIS EXISTS RATHER THAN A SECOND device_vector.h
 *
 * The yieldable page-cache headers (yield_stack.h, yield_coro.h,
 * device_vector.h) are ~4000 lines whose CUDA surface is, when you actually
 * enumerate it, TWELVE NAMES:
 *
 *     threadIdx blockIdx blockDim gridDim  __syncthreads __syncthreads_or
 *     atomicAdd atomicCAS atomicSub atomicExch
 *     __threadfence __threadfence_system __trap  (+ printf, __nanosleep)
 *
 * Everything else in those files is ordinary C++. Porting by rewriting each
 * call site into a neutral macro would touch several hundred lines of the
 * most bug-prone code in the repository -- code whose failure mode is
 * silent wrong data, not a compile error -- for the benefit of a backend
 * that did not exist yesterday. Supplying the twelve names instead touches
 * the CUDA path in exactly ZERO places: this header is inert unless
 * CTP_IS_SYCL_COMPILER, and under CUDA these names are the real builtins.
 *
 * So the port is additive, and there is ONE device_vector.h.
 *
 * WHAT IT ASSUMES
 *
 *   - 1-D nd_range kernels. `threadIdx.y/z` and `blockDim.y/z` read as 1
 *     and `blockIdx.y/z` as 0, which is true of every yieldable kernel
 *     here; a 2-D kernel would silently get the wrong answer, so this
 *     header deliberately does NOT try to serve one.
 *   - The kernel is launched over an nd_range, so the free-function
 *     work-item queries resolve. They are what make this possible at all:
 *     without `this_work_item::get_nd_item()` every one of these would need
 *     to be a parameter, and the shared header would be impossible.
 *
 * WHAT IT DOES NOT PROVIDE
 *
 *   - `__shared__` / `extern __shared__`. There is no free-function
 *     equivalent, and the one consumer (YieldTls) is given a SYCL branch of
 *     its own that puts the block's state in global memory instead. See
 *     yield_stack.h.
 *   - `__global__`. SYCL kernels are lambdas submitted to a queue, so
 *     SYCL kernel bodies and their host drivers are written separately;
 *     only the device machinery below them is shared.
 */
#ifndef CTP_UTIL_SYCL_CUDA_COMPAT_H_
#define CTP_UTIL_SYCL_CUDA_COMPAT_H_

#include <clio_ctp/constants/macros.h>

/* CTP_ENABLE_SYCL, not CTP_IS_SYCL_COMPILER.
 *
 * A SYCL program's HOST translation units are compiled without -fsycl, so
 * CTP_IS_SYCL_COMPILER is 0 there -- but they still include device_vector.h
 * (through gpu_vector.h, for the host Vector's device view) and so still
 * have to PARSE `threadIdx.x` and `atomicCAS`. Under CUDA they get away
 * with it because nvcc/clang-CUDA supply the builtins in both passes; a
 * plain host compiler supplies neither.
 *
 * They never CALL any of this -- the work-item queries would be meaningless
 * on a CPU -- they only need the names to exist. Defining them everywhere
 * SYCL is enabled is what makes one device_vector.h serve both passes.
 */
#if CTP_ENABLE_SYCL

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

#include <cstdint>
#include <type_traits>

namespace clio::run::gpu::sycl_compat {

namespace twi = ::sycl::ext::oneapi::this_work_item;

/** What `threadIdx` / `blockIdx` / `blockDim` evaluate to. Built per read
 *  from the work-item query -- there is no cheaper form, and the query is a
 *  register read on every backend that matters. */
struct Dim3Ref {
  unsigned x, y, z;
};

inline Dim3Ref ThreadIdx() {
  return Dim3Ref{static_cast<unsigned>(twi::get_nd_item<1>().get_local_id(0)),
                 0u, 0u};
}
inline Dim3Ref BlockIdx() {
  return Dim3Ref{static_cast<unsigned>(twi::get_nd_item<1>().get_group(0)), 0u,
                 0u};
}
inline Dim3Ref BlockDim() {
  return Dim3Ref{
      static_cast<unsigned>(twi::get_nd_item<1>().get_local_range(0)), 1u, 1u};
}
/** gridDim: work-GROUPS in the launch, i.e. CUDA blocks. Needed by every
 *  grid-stride loop (`i += gridDim.x * blockDim.x`), which is how the dense
 *  reference kernels in the benchmarks walk their arrays. */
inline Dim3Ref GridDim() {
  return Dim3Ref{
      static_cast<unsigned>(twi::get_nd_item<1>().get_group_range(0)), 1u, 1u};
}

inline void SyncThreads() {
  ::sycl::group_barrier(twi::get_work_group<1>());
}

/** CUDA's __syncthreads_or: barrier + OR-reduction of the predicate over the
 *  block, returning nonzero if ANY thread passed a nonzero predicate. */
inline int SyncThreadsOr(int pred) {
  return ::sycl::any_of_group(twi::get_work_group<1>(), pred != 0) ? 1 : 0;
}

/* -------------------------------------------------------------------- */
/* Atomics.                                                             */
/*                                                                      */
/* DEVICE SCOPE, matching what the CUDA code these serve actually uses   */
/* (plain atomicAdd/atomicCAS, not the _system variants). The one place  */
/* the target is host memory -- the fatal channel -- pairs a device      */
/* atomic with an explicit system fence on both paths, so the scope here */
/* is the same statement CUDA is making.                                */
/* -------------------------------------------------------------------- */

/** `NonDeduced<T>` on every value parameter is load-bearing. CUDA's atomics
 *  are OVERLOADS, not templates, so `atomicCAS(ull_ptr, kFatalNone, kCode)`
 *  converts the enum arguments to unsigned long long implicitly. A plain
 *  template deduces T from all three and fails on exactly that call. Fixing
 *  T from the pointer restores CUDA's behaviour. */
// Hand-rolled rather than std::type_identity: some TUs in this build are
// still compiled as C++17, where that trait does not exist.
template <typename T>
struct NonDeducedT {
  using type = T;
};
template <typename T>
using NonDeduced = typename NonDeducedT<T>::type;

template <typename T>
inline T AtomicAdd(T *p, NonDeduced<T> v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::device>
      r(*p);
  return r.fetch_add(v);
}

template <typename T>
inline T AtomicSub(T *p, NonDeduced<T> v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::device>
      r(*p);
  return r.fetch_sub(v);
}

template <typename T>
inline T AtomicExch(T *p, NonDeduced<T> v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::device>
      r(*p);
  return r.exchange(v);
}

/**
 * CUDA's atomicCAS: returns the OLD value, whether or not the swap happened.
 *
 * compare_exchange_strong reports success as a bool and writes the observed
 * value back into `expected` on failure -- so on failure `expected` IS the
 * old value, and on success it is still the value that compared equal, which
 * is also the old value. Returning it either way is exactly CUDA's contract,
 * and callers here rely on it (`atomicCAS(p, 0, 1) == 0` means "I won").
 */
template <typename T>
inline T AtomicCas(T *p, NonDeduced<T> cmp, NonDeduced<T> val) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  // TOOLCHAIN BUG, NOT A PREFERENCE. A 64-bit compare_exchange through
  // sycl::atomic_ref emits a call to __clc_atomic_compare_exchange that
  // libspirv does not define for nvptx64, and the shared-library device
  // link dies at:
  //
  //   ptxas fatal : Unresolved extern function
  //                 '_Z29__clc_atomic_compare_exchangePmmmiii'
  //
  // (DPC++ nightly-2026-08-27; reproduced on a four-line kernel, and only
  // for 64-bit types -- 32-bit compare_exchange links fine.) The PTX
  // instruction is right there, so use it. Delete this branch once the
  // builtin resolves; the atomic_ref path below is the one we want.
  if constexpr (sizeof(T) == 8) {
    unsigned long long out;
    asm volatile("atom.cas.b64 %0, [%1], %2, %3;"
                 : "=l"(out)
                 : "l"(p), "l"(static_cast<unsigned long long>(cmp)),
                   "l"(static_cast<unsigned long long>(val))
                 : "memory");
    return static_cast<T>(out);
  }
#endif
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::device>
      r(*p);
  T expected = cmp;
  r.compare_exchange_strong(expected, val);
  return expected;
}

template <typename T>
inline T AtomicOr(T *p, NonDeduced<T> v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::device>
      r(*p);
  return r.fetch_or(v);
}

template <typename T>
inline T AtomicAnd(T *p, NonDeduced<T> v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::device>
      r(*p);
  return r.fetch_and(v);
}

template <typename T>
inline T AtomicXor(T *p, NonDeduced<T> v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::device>
      r(*p);
  return r.fetch_xor(v);
}

/* The _system variants. SYSTEM scope really is the point of these -- the
 * reader is the CPU across PCIe -- so they are NOT aliases of the device
 * forms. ctp::ipc::rocm_atomic uses them for exactly one thing: a GPU->CPU
 * ring counter the host polls, where device scope would let the store sit
 * in L2 forever. */
template <typename T>
inline T AtomicAddSystem(T *p, NonDeduced<T> v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::system>
      r(*p);
  return r.fetch_add(v);
}

template <typename T>
inline T AtomicCasSystem(T *p, NonDeduced<T> cmp, NonDeduced<T> val) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::system>
      r(*p);
  T expected = cmp;
  r.compare_exchange_strong(expected, val);
  return expected;
}

inline void FenceDevice() {
  ::sycl::atomic_fence(::sycl::memory_order::seq_cst,
                       ::sycl::memory_scope::device);
}
inline void FenceSystem() {
  ::sycl::atomic_fence(::sycl::memory_order::seq_cst,
                       ::sycl::memory_scope::system);
}

/**
 * CUDA's __trap: kill the kernel now.
 *
 * There is no portable SYCL equivalent. On NVPTX the asm is the real thing;
 * anywhere else this degrades to a null write, which faults the kernel
 * rather than trapping it. Either way the run stops, which is the contract
 * the callers depend on -- they have already latched the reason into the
 * host-memory fatal channel by the time they get here (see
 * gpu-device-trap-diagnostics: nothing else survives).
 */
inline void Trap() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  asm volatile("trap;");
#elif defined(__SYCL_DEVICE_ONLY__)
  *reinterpret_cast<volatile int *>(0) = 0;
#endif
}

inline void NanoSleep(unsigned) { /* no SYCL equivalent; spin */ }

/**
 * CUDA's __shfl_sync: broadcast one lane's value to the whole warp.
 *
 * SYCL's sub_group is CUDA's warp (32 lanes on NVPTX), and
 * select_from_group is the same operation. The `mask` argument is dropped
 * because SYCL sub-group collectives are convergent by definition -- there
 * is no partial-warp form to name.
 */
template <typename T>
inline T ShflSync(unsigned /*mask*/, T val, int src_lane) {
  return ::sycl::select_from_group(twi::get_sub_group(), val,
                                   static_cast<size_t>(src_lane));
}

/**
 * clock64(). SYCL has no portable cycle counter, but this one is not
 * optional: gpu_vector stamps `last_access` with it and the eviction policy
 * uses that to break score ties (LRU). A constant would not fail to
 * compile, it would silently turn the tiebreak off and change which frame
 * gets evicted -- a policy change disguised as a port.
 *
 * On NVPTX the register is right there. Anywhere else this returns 0 and
 * the tiebreak really is inert, which is why that case says so out loud.
 */
inline long long Clock64() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  long long r;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(r));
  return r;
#else
  return 0;  // no cycle counter: LRU tie-breaking degrades to "no order"
#endif
}

}  // namespace clio::run::gpu::sycl_compat

/* -------------------------------------------------------------------- */
/* The names themselves.                                                */
/*                                                                      */
/* Macros rather than functions for threadIdx/blockIdx/blockDim because  */
/* the CUDA spelling is a VARIABLE (`threadIdx.x`), not a call. The rest */
/* could be plain functions, but macros keep the whole surface in one    */
/* place and out of the overload set for host code that never wanted it. */
/* -------------------------------------------------------------------- */

#define threadIdx (::clio::run::gpu::sycl_compat::ThreadIdx())
#define blockIdx (::clio::run::gpu::sycl_compat::BlockIdx())
#define blockDim (::clio::run::gpu::sycl_compat::BlockDim())
#define gridDim (::clio::run::gpu::sycl_compat::GridDim())

#define __syncthreads() ::clio::run::gpu::sycl_compat::SyncThreads()
#define __syncthreads_or(p) ::clio::run::gpu::sycl_compat::SyncThreadsOr(p)

#define atomicAdd(p, v) ::clio::run::gpu::sycl_compat::AtomicAdd(p, v)
#define atomicSub(p, v) ::clio::run::gpu::sycl_compat::AtomicSub(p, v)
#define atomicExch(p, v) ::clio::run::gpu::sycl_compat::AtomicExch(p, v)
#define atomicCAS(p, c, v) ::clio::run::gpu::sycl_compat::AtomicCas(p, c, v)
#define atomicOr(p, v) ::clio::run::gpu::sycl_compat::AtomicOr(p, v)
#define atomicAnd(p, v) ::clio::run::gpu::sycl_compat::AtomicAnd(p, v)
#define atomicXor(p, v) ::clio::run::gpu::sycl_compat::AtomicXor(p, v)
#define atomicAdd_system(p, v) \
  ::clio::run::gpu::sycl_compat::AtomicAddSystem(p, v)
#define atomicCAS_system(p, c, v) \
  ::clio::run::gpu::sycl_compat::AtomicCasSystem(p, c, v)

#define __threadfence() ::clio::run::gpu::sycl_compat::FenceDevice()
#define __threadfence_system() ::clio::run::gpu::sycl_compat::FenceSystem()
#define __trap() ::clio::run::gpu::sycl_compat::Trap()
#define __nanosleep(n) ::clio::run::gpu::sycl_compat::NanoSleep(n)
#define clock64() ::clio::run::gpu::sycl_compat::Clock64()
#define __shfl_sync(m, v, l) ::clio::run::gpu::sycl_compat::ShflSync(m, v, l)

/**
 * Device printf, injected by USING-DECLARATION rather than by a macro.
 *
 * SYCL device code cannot call a C-variadic function -- a bare printf in a
 * kernel is a hard error, "SYCL device code does not support variadic
 * functions". The oneAPI builtin is a variadic TEMPLATE that lowers to
 * __builtin_printf on NVPTX, so the FATAL diagnostics in device_vector.h
 * survive the port intact.
 *
 * It must NOT be `#define printf(...)`. A SYCL TU compiles host and device
 * from the same text, so that macro would also rewrite every host-side
 * `printf` in the translation unit -- including `std::printf`, which the
 * preprocessor happily mangles into `std::sycl::ext::...`. Injecting the
 * name into the namespaces whose device code calls it means unqualified
 * lookup finds it there and stops, and everything else in the TU keeps the
 * C library's printf.
 *
 * Reopening these namespaces here is deliberate: it keeps the whole SYCL
 * accommodation in one file instead of sprinkling #ifdefs through the
 * shared headers.
 */
namespace clio::run::gpu {
using ::sycl::ext::oneapi::experimental::printf;
}
namespace clio::cte::gpu_vector {
using ::sycl::ext::oneapi::experimental::printf;
}

/**
 * The execution-space keywords, as nothing.
 *
 * SYCL has no host/device function attributes -- a function is device code
 * if a kernel reaches it, and that is decided by the compiler, not by the
 * author. So `__device__` is not merely emulated here, it is genuinely
 * meaningless, and erasing it is the correct translation rather than a
 * shortcut. This is what lets yield_coro.h keep its 41 `__device__`
 * spellings and still compile under SYCL untouched.
 *
 * `__global__` is erased for the same reason, but a function it marks is
 * NOT a SYCL kernel -- it becomes an ordinary function nobody calls. SYCL
 * kernel bodies are lambdas submitted to a queue, which is why the SYCL
 * kernels and their host drivers are written separately from the CUDA ones
 * even though everything below them is shared.
 */
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif

/** Alignment and inlining spellings the shared headers use. */
#ifndef __align__
#define __align__(n) alignas(n)
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

#endif  // CTP_ENABLE_SYCL

/* ==================================================================== */
/* HOST-side names, needed WITHOUT -fsycl.                              */
/*                                                                      */
/* A SYCL program's host translation units are compiled by an ordinary  */
/* C++ compiler with CTP_ENABLE_SYCL=1 but NO -fsycl -- so              */
/* CTP_IS_SYCL_COMPILER is 0 there and everything above is inert. They  */
/* still use the shared host relaunch driver, which spells its launch   */
/* geometry in dim3. Gated on the CUDA/ROCm backends being off so this  */
/* can never shadow the real CUDA type.                                 */
/* ==================================================================== */
#if CTP_ENABLE_SYCL && !CTP_ENABLE_CUDA && !CTP_ENABLE_ROCM

/**
 * dim3, for the HOST side.
 *
 * The relaunch driver (Yieldable::Round in yieldable.h) is already
 * backend-neutral -- it allocates through ctp::GpuApi and hands the caller
 * `launch(dim3(blocks), dim3(threads), view)`. dim3 is the one CUDA name it
 * still spells, so supplying it lets the ENTIRE host driver -- round
 * compaction, wait-tag logic, timers -- be shared rather than rewritten. A
 * SYCL launch callback turns it into an nd_range.
 */
struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
      : x(x_), y(y_), z(z_) {}
};

#endif  // CTP_ENABLE_SYCL && !CUDA && !ROCM

#endif  // CTP_UTIL_SYCL_CUDA_COMPAT_H_
