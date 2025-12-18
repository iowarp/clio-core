#ifndef CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_
#define CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_

#include <atomic>
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/data_structures/ipc/shm_container.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "chimaera/types.h"

namespace chi {

// Forward declarations
class Task;
class IpcManager;

/**
 * FutureShm - Shared memory container for task future state
 *
 * This container holds the serialized task data and completion status
 * for asynchronous task operations.
 */
template<typename AllocT = CHI_MAIN_ALLOC_T>
class FutureShm : public hipc::ShmContainer<AllocT> {
 public:
  /** Pool ID for the task */
  PoolId pool_id_;

  /** Method ID for the task */
  u32 method_id_;

  /** Serialized task data */
  hipc::vector<char, AllocT> serialized_task_;

  /** Atomic completion flag (0=not complete, 1=complete) */
  std::atomic<u32> is_complete_;

  /**
   * SHM default constructor
   */
  explicit FutureShm(AllocT* alloc)
      : hipc::ShmContainer<AllocT>(alloc),
        serialized_task_(alloc) {
    pool_id_ = PoolId::GetNull();
    method_id_ = 0;
    is_complete_.store(0);
  }
};

/**
 * Future - Template class for asynchronous task operations
 *
 * Future provides a handle to an asynchronous task operation, allowing
 * the caller to check completion status and retrieve results.
 *
 * @tparam TaskT The task type (e.g., CreateTask, CustomTask)
 * @tparam AllocT The allocator type (defaults to CHI_MAIN_ALLOC_T)
 */
template<typename TaskT, typename AllocT = CHI_MAIN_ALLOC_T>
class Future {
 public:
  using FutureT = FutureShm<AllocT>;

 private:
  /** FullPtr to the task (wraps private memory with null allocator) */
  hipc::FullPtr<TaskT> task_ptr_;

  /** FullPtr to the shared FutureShm object */
  hipc::FullPtr<FutureT> future_shm_;

 public:
  /**
   * Constructor with allocator - allocates new FutureShm
   * @param alloc Allocator to use for FutureShm allocation
   * @param task_ptr FullPtr to the task (wraps private memory with null allocator)
   */
  Future(AllocT* alloc, hipc::FullPtr<TaskT> task_ptr)
      : task_ptr_(task_ptr) {
    // Allocate FutureShm object
    future_shm_ = alloc->template NewObj<FutureT>(alloc).template Cast<FutureT>();
    // Copy pool_id to FutureShm
    if (!task_ptr_.IsNull() && !future_shm_.IsNull()) {
      future_shm_->pool_id_ = task_ptr_->pool_id_;
      future_shm_->method_id_ = task_ptr_->method_;
    }
  }

  /**
   * Constructor with allocator and existing FutureShm
   * @param alloc Allocator (for FullPtr construction)
   * @param task_ptr FullPtr to the task (wraps private memory with null allocator)
   * @param future_shm ShmPtr to existing FutureShm object
   */
  Future(AllocT* alloc, hipc::FullPtr<TaskT> task_ptr, hipc::ShmPtr<FutureT> future_shm)
      : task_ptr_(task_ptr),
        future_shm_(alloc, future_shm) {}

  /**
   * Constructor from FullPtr<FutureShm> and FullPtr<Task>
   * @param future_shm FullPtr to existing FutureShm object
   * @param task_ptr FullPtr to the task (wraps private memory with null allocator)
   */
  Future(hipc::FullPtr<FutureT> future_shm, hipc::FullPtr<TaskT> task_ptr)
      : task_ptr_(task_ptr),
        future_shm_(future_shm) {
    // No need to copy pool_id - FutureShm already has it
  }

  /**
   * Default constructor - creates null future
   */
  Future() {}

  /**
   * Get raw pointer to the task
   * @return Pointer to the task object
   */
  TaskT* get() const {
    return task_ptr_.ptr_;
  }

  /**
   * Get the FullPtr to the task (non-const version)
   * @return FullPtr to the task object
   */
  hipc::FullPtr<TaskT>& GetTaskPtr() {
    return task_ptr_;
  }

  /**
   * Get the FullPtr to the task (const version)
   * @return FullPtr to the task object
   */
  const hipc::FullPtr<TaskT>& GetTaskPtr() const {
    return task_ptr_;
  }

  /**
   * Dereference operator - access task members
   * @return Reference to the task object
   */
  TaskT& operator*() const {
    return *task_ptr_.ptr_;
  }

  /**
   * Arrow operator - access task members
   * @return Pointer to the task object
   */
  TaskT* operator->() const {
    return task_ptr_.ptr_;
  }

  /**
   * Check if the task is complete
   * @return True if task has completed, false otherwise
   */
  bool IsComplete() const {
    if (future_shm_.IsNull()) {
      return false;
    }
    return future_shm_->is_complete_.load() != 0;
  }

  /**
   * Wait for task completion (blocking)
   * Calls IpcManager::Recv() to handle task completion and deserialization
   */
  void Wait();

  /**
   * Mark the task as complete
   */
  void Complete() {
    if (!future_shm_.IsNull()) {
      future_shm_->is_complete_.store(1);
    }
  }

  /**
   * Mark the task as complete (alias for Complete)
   */
  void SetComplete() {
    Complete();
  }

  /**
   * Check if this future is null
   * @return True if future is null, false otherwise
   */
  bool IsNull() const {
    return task_ptr_.IsNull();
  }

  /**
   * Get the FutureShm FullPtr
   * @return FullPtr to the FutureShm object
   */
  hipc::FullPtr<FutureT>& GetFutureShm() {
    return future_shm_;
  }

  /**
   * Get the FutureShm FullPtr (const version)
   * @return FullPtr to the FutureShm object
   */
  const hipc::FullPtr<FutureT>& GetFutureShm() const {
    return future_shm_;
  }

  /**
   * Get the pool ID from the FutureShm
   * @return Pool ID for the task
   */
  PoolId GetPoolId() const {
    if (future_shm_.IsNull()) {
      return PoolId::GetNull();
    }
    return future_shm_->pool_id_;
  }

  /**
   * Set the pool ID in the FutureShm
   * @param pool_id Pool ID to set
   */
  void SetPoolId(const PoolId& pool_id) {
    if (!future_shm_.IsNull()) {
      future_shm_->pool_id_ = pool_id;
    }
  }
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_
