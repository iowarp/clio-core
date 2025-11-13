/**
 * Task implementation
 */

#include "chimaera/task.h"

#include <algorithm>

#include "chimaera/container.h"
#include "chimaera/singletons.h"
#include "chimaera/worker.h"

// Namespace alias for boost::context::detail
namespace bctx = boost::context::detail;

namespace chi {

void Task::Wait(double block_time_us, bool from_yield) {
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager && chimaera_manager->IsRuntime()) {
    // Runtime implementation: Estimate load and yield execution

    // Get current run context from worker
    Worker *worker = CHI_CUR_WORKER;
    RunContext *run_ctx = worker ? worker->GetCurrentRunContext() : nullptr;

    if (!worker || !run_ctx) {
      // No worker or run context available, fall back to client implementation
      while (!IsComplete()) {
        YieldBase();
      }
      return;
    }

    // Check if task is already blocked - this should never happen
    if (run_ctx->is_blocked) {
      HELOG(kFatal,
            "Worker {}: Task is already blocked when calling Wait()! "
            "Task ptr: {:#x}, Pool: {}, Method: {}, TaskId: {}.{}.{}.{}.{}",
            worker->GetId(), reinterpret_cast<uintptr_t>(this), pool_id_,
            method_, task_id_.pid_, task_id_.tid_, task_id_.major_,
            task_id_.replica_id_, task_id_.unique_);
      std::abort();
    }

    // Add this task to the current task's waiting_for_tasks list
    // This ensures AreSubtasksCompleted() properly tracks this subtask
    // Skip if called from yield to avoid double tracking
    if (!from_yield) {
      auto alloc = HSHM_MEMORY_MANAGER->GetDefaultAllocator<CHI_MAIN_ALLOC_T>();
      hipc::FullPtr<Task> this_task_ptr(alloc, this);
      run_ctx->waiting_for_tasks.push_back(this_task_ptr);
    }

    // Store blocking duration in RunContext (use provided value directly)
    // block_time_us is passed by the caller - no estimation
    run_ctx->block_time_us = block_time_us;

    // Yield execution back to worker in loop until task completes
    // Add to blocked queue before each yield
    // NOTE(llogan): This will only be unblocked when all subtasks are complete
    // No need for a while loop here.
    worker->AddToBlockedQueue(run_ctx);
    YieldBase();

    // After yielding, check if task is complete
    // If not complete, set task_did_work_ to false to indicate blocked work
    if (!IsComplete()) {
      worker->SetTaskDidWork(false);
    }
  } else {
    // Client implementation: Wait loop using Yield()
    while (!IsComplete()) {
      YieldBase();
    }
  }
}

void Task::YieldBase() {
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager && chimaera_manager->IsRuntime()) {
    // Get current run context from worker
    Worker *worker = CHI_CUR_WORKER;
    RunContext *run_ctx = worker ? worker->GetCurrentRunContext() : nullptr;

    if (!run_ctx) {
      // No run context available, fall back to client implementation
      HSHM_THREAD_MODEL->Yield();
      return;
    }

    // Mark this task as blocked
    run_ctx->is_blocked = true;

    // Jump back to worker using boost::fiber

    // Jump back to worker - the task has been added to blocked queue
    // Store the result (task's yield point) in resume_context for later
    // resumption Use temporary variables to store the yield context before
    // jumping
    bctx::fcontext_t yield_fctx = run_ctx->yield_context.fctx;
    void *yield_data = run_ctx->yield_context.data;

    // Jump back to worker and capture the result
    bctx::transfer_t yield_result = bctx::jump_fcontext(yield_fctx, yield_data);

    // CRITICAL: Update yield_context with the new worker context from the
    // resume operation This ensures that subsequent yields or completion
    // returns to the correct worker location
    run_ctx->yield_context = yield_result;

    // Store where we can resume from for the next yield cycle
    run_ctx->resume_context = yield_result;
  } else {
    // Outside runtime mode, just yield
    HSHM_THREAD_MODEL->Yield();
  }
}

void Task::Yield(double block_time_us) {
  // New public Yield function that calls Wait with from_yield=true
  // to avoid adding subtasks to RunContext
  Wait(block_time_us, true);
}

bool Task::IsComplete() const {
  // Completion check (works for both client and runtime modes)
  return is_complete_.load() != 0;
}

void Task::Aggregate(const hipc::FullPtr<Task> &replica_task) {
  // If replica task has non-zero return code, propagate it to this task
  if (!replica_task.IsNull() && replica_task->GetReturnCode() != 0) {
    SetReturnCode(replica_task->GetReturnCode());
  }
  // Copy the completer from the replica task
  if (!replica_task.IsNull()) {
    SetCompleter(replica_task->GetCompleter());
  }
  HILOG(kDebug, "[COMPLETER] Aggregated task {} with completer {}", task_id_, GetCompleter());
}

size_t Task::EstCpuTime() const {
  // Calculate: io_size / 4GBps + compute + 5
  // 4 GBps = 4 * 1024 * 1024 * 1024 bytes/second = 4294967296 bytes/second
  // Convert to microseconds: (io_size / 4294967296) * 1000000
  size_t io_time_us = (stat_.io_size_ * 1000000) / 4294967296ULL;
  return io_time_us + stat_.compute_ + 5;
}

} // namespace chi