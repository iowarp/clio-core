/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/singletons.h"

#include <deque>
#include <mutex>
#include <vector>

namespace clio::run {

//==============================================================================
// RuntimeRecv: poll ZMQ transports for incoming client tasks
//==============================================================================

bool IpcCpu2CpuZmq::RuntimeRecv(IpcManager *ipc, u32 &tasks_received) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  bool did_work = false;
  tasks_received = 0;

  // Instrumentation: cumulative count of client requests this daemon has
  // accepted from RuntimeRecv. Printed every 256 to keep log volume sane
  // but still bracket the 24×128 = 3072-req IOR read phase.
  static std::atomic<size_t> recv_counter{0};

  // Process both TCP and IPC servers
  for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
    IpcMode mode = (mode_idx == 0) ? IpcMode::kTcp : IpcMode::kIpc;
    ctp::lbm::Transport *transport = ipc->GetClientTransport(mode);
    if (!transport) continue;

    // Drain all pending messages from this transport. Unbounded `while`
    // is intentional: RuntimeRecv is invoked by the ClientRecv periodic
    // which runs on its own dedicated net_recv worker (see
    // project_net_worker_split.md / DefaultScheduler::DivideWorkers),
    // so a hot client stream here doesn't starve any other periodic.
    // EAGAIN ends the loop when the transport buffer drains.
    while (true) {
      LoadTaskArchive archive;
      auto recv_info = transport->Recv(archive);
      int rc = recv_info.rc;
      if (rc == EAGAIN) break;
      if (rc != 0) {
        // -1 here is overwhelmingly "peer closed the socket" (RecvExact
        // returns -1 on EOF). The lightbeam SocketTransport already cleaned
        // up the dead fd inside Recv(); breaking out and retrying is the
        // correct behavior. Demote to kDebug so a routine client exit
        // doesn't spam the runtime log with kError lines.
        HLOG(kDebug, "IpcCpu2CpuZmq::RuntimeRecv: Recv failed: {}", rc);
        break;
      }

      const auto &task_infos = archive.GetTaskInfos();
      if (task_infos.empty()) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeRecv: No task_infos in message");
        continue;
      }

      const auto &info = task_infos[0];
      PoolId pool_id = info.pool_id_;
      u32 method_id = info.method_id_;

      // Get container for deserialization
      Container *container = pool_manager->GetStaticContainer(pool_id);
      if (!container) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeRecv: Container not found "
             "for pool_id {}", pool_id);
        continue;
      }

      // Allocate and deserialize the task
      ctp::ipc::FullPtr<Task> task_ptr =
          container->AllocLoadTask(method_id, archive);

      // SerializeIn copied any zmq-owned BULK_XFER payloads into
      // CHI-owned buffers (LoadTaskArchive::bulk), so the zmq_msg_t
      // handles in archive.recv[*].desc are now unreferenced. Free them
      // here — this is the only place that closes them on the server
      // inbound path; without it every inbound TCP bulk leaks one
      // zmq_msg_t + its payload. Safe to call unconditionally: the zmq
      // ClearRecvHandles only closes/deletes desc handles and leaves the
      // (now CHI-owned) data buffers alone; SHM recv has desc==null so
      // this is a no-op there.
      transport->ClearRecvHandles(archive);

      if (task_ptr.IsNull()) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeRecv: Failed to deserialize task");
        continue;
      }

      // If SerializeIn copied any ZMQ-owned BULK_XFER input into a fresh
      // CHI buffer, the task now owns that buffer. Promote the count to
      // TASK_DATA_OWNER so the task destructor frees it (mirrors admin
      // RecvIn). Without this the copied buffer leaks one io_size
      // allocation per inbound TCP/IPC bulk.
      if (archive.daemon_allocated_bulk_count_ > 0) {
        task_ptr->SetFlags(TASK_DATA_OWNER);
      }

      // Create FutureShm for the task (server-side)
      ctp::ipc::FullPtr<FutureShm> future_shm = ipc->NewObj<FutureShm>();
      future_shm->pool_id_ = pool_id;
      future_shm->method_id_ = method_id;
      future_shm->origin_ = (mode == IpcMode::kTcp)
                                ? FutureShm::FUTURE_CLIENT_TCP
                                : FutureShm::FUTURE_CLIENT_IPC;
      future_shm->client_task_vaddr_ = info.task_id_.net_key_;
      future_shm->client_pid_ = info.task_id_.pid_;
      // Store transport and routing info for response
      future_shm->response_transport_ = transport;
      future_shm->response_fd_ = recv_info.fd_;
      // Store ZMQ identity from recv frame for response routing
      if (!recv_info.identity_.empty() &&
          recv_info.identity_.size() <=
              sizeof(future_shm->response_identity_)) {
        std::memcpy(future_shm->response_identity_,
                    recv_info.identity_.data(),
                    recv_info.identity_.size());
        future_shm->response_identity_len_ =
            static_cast<u32>(recv_info.identity_.size());
      }
      // Mark as copied so EndTask routes back via lightbeam
      future_shm->flags_.SetBits(FutureShm::FUTURE_WAS_COPIED);

      // Create Future and enqueue to worker lane
      Future<Task> future(future_shm.shm_, task_ptr);
      LaneId lane_id =
          ipc->GetScheduler()->ClientMapTask(ipc, future);
      auto *worker_queues = ipc->GetTaskQueue();
      auto &lane_ref = worker_queues->GetLane(lane_id, 0);
      lane_ref.Push(future);
      // Always signal — see ipc_cpu2cpu_impl.h for the race.
      ipc->AwakenWorker(&lane_ref);

      did_work = true;
      tasks_received++;
      size_t total = recv_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if ((total & 0xff) == 0) {
        HLOG(kDebug,
             "[CountRecv] cumulative client requests received = {} "
             "(mode={}, latest method_id={}, pool_id={})",
             total, mode_idx, method_id, pool_id);
      }
    }
  }

  return did_work;
}

//==============================================================================
// EnqueueRuntimeSend: worker-inline enqueue to net_queue_
//==============================================================================

void IpcCpu2CpuZmq::EnqueueRuntimeSend(IpcManager *ipc, RunContext *run_ctx,
                                         u32 origin) {
  if (origin == FutureShm::FUTURE_CLIENT_TCP) {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendTcp);
  } else {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendIpc);
  }
}

//==============================================================================
// RuntimeSend: net-worker serialize and send response via ZMQ
//==============================================================================

bool IpcCpu2CpuZmq::RuntimeSend(
    IpcManager *ipc, u32 &tasks_sent,
    std::vector<ctp::ipc::FullPtr<Task>> & /*deferred_deletes — unused*/) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  bool did_work = false;
  tasks_sent = 0;
  static std::atomic<size_t> send_counter{0};
  static std::atomic<size_t> send_fail_counter{0};

  // Task lifetime across the zero-copy ZMQ send is handled entirely
  // inside lightbeam now: each Send() takes an LbmContext::on_send_complete
  // callback, and the transport keeps the task buffer alive (via an
  // atomic refcount on its internal SendCompletion record) until ZMQ
  // confirms every bulk frame has flushed.  When that happens the
  // transport parks the callback on its ready-completions list, and
  // the NEXT Send() drains the list — running each callback on the net
  // worker's thread (i.e. THIS thread), so DelTask can safely touch
  // coroutine-aware container state.  No per-invocation deferral
  // queue, no time-window guessing, no extra mutex on the hot path.
  //
  // The deferred_deletes parameter is kept in the API signature for
  // ABI back-compat with any out-of-tree caller that still passes one;
  // we never read or write it.

  // Process both TCP and IPC queues
  for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
    NetQueuePriority priority =
        (mode_idx == 0) ? NetQueuePriority::kClientSendTcp
                        : NetQueuePriority::kClientSendIpc;
    IpcMode mode =
        (mode_idx == 0) ? IpcMode::kTcp : IpcMode::kIpc;

    Future<Task> queued_future;
    // Snapshot queue depth at function entry and drain exactly that
    // many — see admin_runtime.cc Send for the same pattern. Bounds
    // the per-priority drain so neither kClientSendTcp nor
    // kClientSendIpc can starve the other (or starve the deferred-
    // delete reclaim above) when one side has a hot producer.
    const size_t client_send_bound = ipc->GetNetQueueSize(priority);
    for (size_t send_i = 0; send_i < client_send_bound; ++send_i) {
      if (!ipc->TryPopNetTask(priority, queued_future)) break;
      auto origin_task = queued_future.GetTaskPtr();
      if (origin_task.IsNull()) continue;

      auto future_shm = queued_future.GetFutureShm();
      if (future_shm.IsNull()) continue;

      // Get container to serialize outputs
      Container *container =
          pool_manager->GetStaticContainer(origin_task->pool_id_);
      if (!container) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeSend: Container not found "
             "for pool_id {}", origin_task->pool_id_);
        continue;
      }

      // Get response transport and routing info from FutureShm
      ctp::lbm::Transport *response_transport =
          future_shm->response_transport_;
      if (!response_transport) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeSend: No response transport "
             "for mode {} pid {}", mode_idx, future_shm->client_pid_);
        continue;
      }

      // Preserve client's net_key for response routing
      origin_task->task_id_.net_key_ = future_shm->client_task_vaddr_;

      // Serialize task outputs
      SaveTaskArchive archive(MsgType::kSerializeOut, response_transport);
      container->SaveTask(origin_task->method_, archive, origin_task);

      // Set routing info for the response
      if (mode == IpcMode::kTcp) {
        if (future_shm->response_identity_len_ > 0) {
          archive.client_info_.identity_ =
              std::string(future_shm->response_identity_,
                          future_shm->response_identity_len_);
        } else {
          u32 client_pid = future_shm->client_pid_;
          archive.client_info_.identity_ = std::string(
              reinterpret_cast<const char *>(&client_pid), sizeof(client_pid));
        }
      } else if (mode == IpcMode::kIpc) {
        archive.client_info_.fd_ = future_shm->response_fd_;
      }

      // SYNC send: lightbeam copies bulks into ZMQ inside this call and
      // holds no reference to origin_task's buffers after it returns, so
      // we DelTask synchronously below.  No async callback, no I/O-thread
      // race with the task's destructor.
      //
      // On read responses each task ships a 1 MiB bulk frame; at high
      // concurrency the ROUTER socket can transiently return EAGAIN.
      // Without retry the response is lost and the client spins on
      // FUTURE_COMPLETE forever, so re-queue on failure (without
      // DelTask — task lifetime stays with the queued_future).
      ctp::lbm::LbmContext send_ctx(ctp::lbm::LBM_SYNC);
      int rc = response_transport->Send(archive, send_ctx);
      if (rc != 0) {
        size_t fail_total =
            send_fail_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        HLOG(kError,
             "[CountSend] Send rc={} fail#{} — re-queueing client response "
             "(priority={})",
             rc, fail_total, static_cast<int>(priority));
        ipc->EnqueueNetTask(queued_future, priority);
        continue;
      }

      // Send succeeded — caller-side buffers are no longer referenced
      // by ZMQ, so it's safe to delete the task here.
      {
        auto *pm = CLIO_POOL_MANAGER;
        auto *del_container =
            pm ? pm->GetStaticContainer(origin_task->pool_id_) : nullptr;
        if (del_container) {
          del_container->DelTask(origin_task->method_, origin_task);
        }
      }

      did_work = true;
      tasks_sent++;
      size_t total = send_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if ((total & 0xff) == 0) {
        HLOG(kDebug,
             "[CountSend] cumulative client responses sent = {} "
             "(mode={}, fails so far = {})",
             total, mode_idx,
             send_fail_counter.load(std::memory_order_relaxed));
      }
    }
  }

  return did_work;
}

}  // namespace clio::run
