/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_RUN_INLINE_H_
#define CLIO_RUNTIME_INCLUDE_RUN_INLINE_H_

/**
 * CLIO_RUN_INLINE (issue #862): execute a task's handler INLINE on the
 * current worker fiber instead of routing it through the task queues, when
 * that is provably equivalent:
 *
 *   - the process IS the runtime (co-located handlers, same address space),
 *   - the task resolves to a LOCAL container (single-node deployment for
 *     this first cut — multi-node routing keeps the queue path),
 *   - we are on a worker fiber (nested handler calls are plain calls in the
 *     Boost fiber backend; their awaits suspend the enclosing fiber), and
 *   - we are NOT in a unit test (suites assert queue/scheduling semantics;
 *     simple_test.h marks test processes via CLIO_UNIT_TEST).
 *
 * Motivation, measured: a CTE put spends most of its ~11-14us of server-side
 * cost on task round trips whose payload is a local RAM-bdev memcpy — the
 * dispatch/queue/schedule/complete cycle dwarfs the work. Inlining runs the
 * handler as a nested call and returns an already-COMPLETE future (the same
 * synthesized-task contract as the SHM read fast path), so callers keep the
 * exact Future/Wait() surface.
 *
 * First adopters: bdev AsyncWrite / AsyncRead. The mechanism is general —
 * any NewTask+Send pair whose handler tolerates nested execution can switch
 * to CLIO_RUN_INLINE(task).
 *
 * Kill switch: CLIO_DISABLE_INLINE_RUN=1 restores the queue path everywhere.
 */

#include <cstdlib>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/container.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_runtime/task.h"

namespace clio::run::detail {

/** Static (process-wide) part of the inline-eligibility check.
 *
 *  OPT-IN (CLIO_ENABLE_INLINE_RUN=1): inline execution measured
 *  throughput-neutral on the YCSB workloads at every pipeline depth, and the
 *  always-on first cut broke the file-bdev/adapters paths in CI
 *  (ModifyExistingData "wrote 0 bytes" — Catch2 test processes do not carry
 *  the simple_test CLIO_UNIT_TEST marker, and the file-backed bdev write
 *  handler is not yet validated under nested execution). The mechanism stays
 *  for latency-sensitive callers to enable deliberately. */
inline bool InlineRunEnvEnabled() {
  static const bool v = [] {
    if (std::getenv("CLIO_ENABLE_INLINE_RUN") == nullptr) return false;
    if (std::getenv("CLIO_UNIT_TEST") != nullptr) return false;
    return true;
  }();
  return v;
}

/**
 * Run `task`'s handler inline when eligible (see header comment); otherwise
 * fall back to the normal Send. Returns a Future either way — already
 * complete on the inline path.
 */
template <typename TaskT>
inline clio::run::Future<TaskT> RunInlineOrSend(
    clio::run::shared_ptr<TaskT> &task) {
  auto *ipc_manager = CLIO_CPU_IPC;
#if CTP_IS_HOST
  // Host-only: Container::Run / TaskResume / the RunContext accessors are
  // host-side types (the CUDA device pass sees only forward declarations), so
  // the device pass compiles just the Send fallback below.
  if (InlineRunEnvEnabled() && ipc_manager != nullptr &&
      CLIO_RUNTIME_MANAGER != nullptr && CLIO_RUNTIME_MANAGER->IsRuntime() &&
      ipc_manager->GetNumHosts() <= 1 && CLIO_CUR_WORKER != nullptr) {
    // The REAL container, not the static one: this path runs the module's
    // handler, which needs the container that Create initialized. The static
    // container holds only the pool's task-stat model (issue #956).
    clio::run::DynamicContainer dc =
        CLIO_POOL_MANAGER->GetRealOrStaticContainer(task->pool_id_);
    clio::run::ContainerHold c = dc.get();
    if (c != nullptr) {
      // Same synthesized-completed-future shape as the batch-merged sink and
      // the TryShmGet fast path: give the task a RunContext + future, run the
      // handler as a nested call on THIS fiber (its awaits suspend the
      // enclosing fiber, per the Boost backend contract), then mark complete.
      task->BeginRunContext();
      clio::run::Future<TaskT> fut(task->pool_id_, task->method_, task);
      fut.GetFutureShm()->origin_ = clio::run::ClientOrigin::kClientShm;
      task->RunFuture() = fut.template Cast<clio::run::Task>();
      clio::run::shared_ptr<clio::run::Task> base =
          task.template Cast<clio::run::Task>();
      c->Run(task->method_, base);
      task->SetComplete();
      return fut;
    }
  }
#endif  // CTP_IS_HOST
  return ipc_manager->Send(task);
}

}  // namespace clio::run::detail

/** Execute inline when safely equivalent, else Send. Yields a Future. */
#define CLIO_RUN_INLINE(task) clio::run::detail::RunInlineOrSend(task)

#endif  // CLIO_RUNTIME_INCLUDE_RUN_INLINE_H_
