#ifndef CHIMAERA_INCLUDE_CHIMAERA_TASK_QUEUE_H_
#define CHIMAERA_INCLUDE_CHIMAERA_TASK_QUEUE_H_

#include "chimaera/types.h"
#include "chimaera/future.h"

namespace chi {

// Forward declaration for WorkOrchestrator static method
class WorkOrchestrator;

// Forward declarations
class Worker;
// class TaskQueue;
class WorkOrchestrator;

/**
 * Custom header for tracking lane state (stored per-lane)
 */
struct TaskQueueHeader {
  PoolId pool_id;
  WorkerId assigned_worker_id;
  u32 task_count;        // Number of tasks currently in the queue
  bool is_enqueued;      // Whether this queue is currently enqueued in worker

  TaskQueueHeader() : pool_id(), assigned_worker_id(0), task_count(0), is_enqueued(false) {}
  TaskQueueHeader(PoolId pid, WorkerId wid = 0)
    : pool_id(pid), assigned_worker_id(wid), task_count(0), is_enqueued(false) {}
};

// Type alias for individual lanes with per-lane headers (moved outside TaskQueue class)
// Worker queues now store FutureShm pointers instead of Task pointers
using TaskLane =
    hipc::multi_mpsc_ring_buffer<hipc::ShmPtr<FutureShm<CHI_MAIN_ALLOC_T>>,
                                 CHI_MAIN_ALLOC_T>::ring_buffer_type;

/**
 * Simple wrapper around hipc::multi_mpsc_ring_buffer
 *
 * This wrapper adds custom enqueue and dequeue functions while maintaining
 * compatibility with existing code that expects the multi_mpsc_ring_buffer
 * interface.
 */
typedef hipc::multi_mpsc_ring_buffer<hipc::ShmPtr<FutureShm<CHI_MAIN_ALLOC_T>>, CHI_MAIN_ALLOC_T>
    TaskQueue;

// template <typename 
// class TaskQueue {
// public:
//   /**
//    * Constructor using CtxAllocator (preferred pattern)
//    * @param alloc Context allocator containing memory context
//    * @param num_lanes Number of lanes
//    * @param num_prios Number of priorities
//    * @param depth_per_lane Depth per lane
//    */
//   TaskQueue(CHI_MAIN_ALLOC_T* alloc,
//             u32 num_lanes, u32 num_prios, u32 depth_per_lane);

//   /**
//    * Destructor
//    */
//   ~TaskQueue() = default;


//   /**
//    * Forward all other methods to underlying queue for compatibility
//    */
//   TaskLane& GetLane(u32 lane_id, u32 prio_id) { 
//     return queue_.GetLane(lane_id, prio_id); 
//   }
//   const TaskLane& GetLane(u32 lane_id, u32 prio_id) const { 
//     return queue_.GetLane(lane_id, prio_id); 
//   }

//   /**
//    * Check if queue is null (for compatibility)
//    */
//   bool IsNull() const { return queue_.IsNull(); }

//   /**
//    * Get number of lanes in the queue
//    */
//   u32 GetNumLanes() const { return queue_.GetNumLanes(); }

//   /**
//    * Get number of priorities in the queue
//    */
//   u32 GetNumPriorities() const { return queue_.GetNumPriorities(); }



//   /**
//    * Static method to emplace a task into a specific lane
//    * @param lane_ptr FullPtr to the lane (as returned by GetLane)
//    * @param task_ptr TypedPointer to the task to emplace
//    * @return true if emplace successful
//    */
//   static bool EmplaceTask(hipc::FullPtr<TaskLane>& lane_ptr, hipc::ShmPtr<Task> task_ptr);


//   /**
//    * Static method to pop a task from a specific lane
//    * @param lane_ptr FullPtr to the lane
//    * @param task_ptr Output TypedPointer to the popped task
//    * @return true if pop successful
//    */
//   static bool PopTask(hipc::FullPtr<TaskLane>& lane_ptr, hipc::ShmPtr<Task>& task_ptr);

// private:
//   chi::ipc::multi_mpsc_ring_buffer<hipc::ShmPtr<Task>, TaskQueueHeader> queue_; // Underlying queue

// };

/**
 * Pop a FutureShm from a task lane (FullPtr overload)
 * @param lane_ptr FullPtr to the task lane
 * @param future_shm_ptr Reference to store the popped FutureShm
 * @return true if a FutureShm was popped, false otherwise
 */
bool TaskQueue_PopTask(hipc::FullPtr<TaskLane>& lane_ptr, hipc::ShmPtr<FutureShm<CHI_MAIN_ALLOC_T>>& future_shm_ptr);

/**
 * Pop a FutureShm from a task lane (raw pointer overload)
 * @param lane_ptr Raw pointer to the task lane
 * @param future_shm_ptr Reference to store the popped FutureShm
 * @return true if a FutureShm was popped, false otherwise
 */
bool TaskQueue_PopTask(TaskLane *lane_ptr, hipc::ShmPtr<FutureShm<CHI_MAIN_ALLOC_T>>& future_shm_ptr);

} // namespace chi

#endif // CHIMAERA_INCLUDE_CHIMAERA_TASK_QUEUE_H_