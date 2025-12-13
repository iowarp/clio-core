/**
 * TaskQueue implementation - helper functions for task queue operations
 */

#include "chimaera/task_queue.h"

namespace chi {

/**
 * Emplace a FutureShm into a task lane
 * @param lane_ptr Pointer to the task lane
 * @param future_shm_ptr Pointer to the FutureShm to enqueue
 * @return true if successful, false otherwise
 */
bool TaskQueue_EmplaceTask(hipc::FullPtr<TaskLane>& lane_ptr, hipc::ShmPtr<FutureShm<CHI_MAIN_ALLOC_T>> future_shm_ptr) {
  if (lane_ptr.IsNull() || future_shm_ptr.IsNull()) {
    return false;
  }

  // Push to the lane
  lane_ptr->Push(future_shm_ptr);

  return true;
}

/**
 * Pop a FutureShm from a task lane
 * @param lane_ptr Pointer to the task lane
 * @param future_shm_ptr Reference to store the popped FutureShm
 * @return true if a FutureShm was popped, false otherwise
 */
bool TaskQueue_PopTask(hipc::FullPtr<TaskLane>& lane_ptr, hipc::ShmPtr<FutureShm<CHI_MAIN_ALLOC_T>>& future_shm_ptr) {
  if (lane_ptr.IsNull()) {
    return false;
  }

  return lane_ptr->Pop(future_shm_ptr);
}

/**
 * Pop a FutureShm from a task lane (overload for raw pointer)
 * @param lane_ptr Raw pointer to the task lane
 * @param future_shm_ptr Reference to store the popped FutureShm
 * @return true if a FutureShm was popped, false otherwise
 */
bool TaskQueue_PopTask(TaskLane *lane_ptr, hipc::ShmPtr<FutureShm<CHI_MAIN_ALLOC_T>>& future_shm_ptr) {
  if (!lane_ptr) {
    return false;
  }

  return lane_ptr->Pop(future_shm_ptr);
}

}  // namespace chi