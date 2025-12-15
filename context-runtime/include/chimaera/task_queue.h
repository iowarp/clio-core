#ifndef CHIMAERA_INCLUDE_CHIMAERA_TASK_QUEUE_H_
#define CHIMAERA_INCLUDE_CHIMAERA_TASK_QUEUE_H_

#include "chimaera/types.h"
#include "chimaera/future.h"

namespace chi {

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

} // namespace chi

#endif // CHIMAERA_INCLUDE_CHIMAERA_TASK_QUEUE_H_