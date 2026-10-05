/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * CycleNow(): a cheap monotonic cycle stamp for hot-path probes.
 *
 * The evlat/phase probes deliberately use a counter INSTRUCTION, not a
 * clock call: std::chrono::steady_clock on the gpu2cpu completion path
 * measurably shifted timing enough to hang the runtime (lost-wake). The
 * x86-only spelling (<x86intrin.h> + __rdtsc) broke every non-x86 build --
 * arm64 macOS and the aarch64 wheels died in ia32intrin.h -- so the per-arch
 * choice lives here and nowhere else. Units are arch-defined cycles; only
 * DELTAS of stamps from the same core are meaningful, which is how every
 * caller uses them.
 */
#ifndef CLIO_RUNTIME_CYCLE_COUNTER_H_
#define CLIO_RUNTIME_CYCLE_COUNTER_H_

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
namespace clio::run {
inline unsigned long long CycleNow() { return __rdtsc(); }
}  // namespace clio::run
#elif defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
namespace clio::run {
inline unsigned long long CycleNow() {
  return _ReadStatusReg(ARM64_SYSREG(3, 3, 14, 0, 2));  // CNTVCT_EL0
}
}  // namespace clio::run
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
namespace clio::run {
inline unsigned long long CycleNow() { return __rdtsc(); }
}  // namespace clio::run
#elif defined(__aarch64__)
namespace clio::run {
inline unsigned long long CycleNow() {
  unsigned long long v;
  asm volatile("mrs %0, cntvct_el0" : "=r"(v));
  return v;
}
}  // namespace clio::run
#else
#include <chrono>
namespace clio::run {
inline unsigned long long CycleNow() {
  // Last resort for exotic targets; none of the probe paths are hot there.
  return static_cast<unsigned long long>(
      std::chrono::steady_clock::now().time_since_epoch().count());
}
}  // namespace clio::run
#endif

#endif  // CLIO_RUNTIME_CYCLE_COUNTER_H_
