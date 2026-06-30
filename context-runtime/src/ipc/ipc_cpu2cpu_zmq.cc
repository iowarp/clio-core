/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_ctp/introspect/system_info.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace clio::run {

//==============================================================================
// RecvIn: poll ZMQ transports for incoming client tasks
//==============================================================================

bool IpcCpu2CpuZmq::RecvIn(IpcManager *ipc, u32 &tasks_received) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  bool did_work = false;
  tasks_received = 0;

  // Instrumentation: cumulative count of client requests this daemon has
  // accepted from RecvIn. Printed every 256 to keep log volume sane
  // but still bracket the 24×128 = 3072-req IOR read phase.
  static std::atomic<size_t> recv_counter{0};

  // Process both TCP and IPC servers
  for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
    IpcMode mode = (mode_idx == 0) ? IpcMode::kTcp : IpcMode::kIpc;
    ctp::lbm::Transport *transport = ipc->GetClientTransport(mode);
    if (!transport) continue;

    // Drain all pending messages from this transport. Unbounded `while`
    // is intentional: RecvIn is invoked by the ClientRecv periodic
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
        HLOG(kDebug, "IpcCpu2CpuZmq::RecvIn: Recv failed: {}", rc);
        break;
      }

      const auto &task_infos = archive.GetTaskInfos();
      if (task_infos.empty()) {
        HLOG(kError, "IpcCpu2CpuZmq::RecvIn: No task_infos in message");
        continue;
      }

      const auto &info = task_infos[0];
      PoolId pool_id = info.pool_id_;
      u32 method_id = info.method_id_;

      // Get container for deserialization
      auto container = pool_manager->GetStaticContainer(pool_id).get();
      if (!container) {
        HLOG(kError, "IpcCpu2CpuZmq::RecvIn: Container not found "
             "for pool_id {}", pool_id);
        continue;
      }

      // Allocate and deserialize the task
      clio::run::shared_ptr<clio::run::Task> task_ptr =
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
        HLOG(kError, "IpcCpu2CpuZmq::RecvIn: Failed to deserialize task");
        continue;
      }

      // This transport serves external user clients (TCP/IPC); runtime peers
      // use the run2run path. Tag the task for per-RPC access control. The flag
      // is in SerializeIn, so it rides along if the task is forwarded to a
      // remote container owner.
      task_ptr->SetFlags(TASK_EXTERNAL_CLIENT);

      // If SerializeIn copied any ZMQ-owned BULK_XFER input into a fresh
      // CHI buffer, the task now owns that buffer. Promote the count to
      // TASK_DATA_OWNER so the task destructor frees it (mirrors admin
      // RecvIn). Without this the copied buffer leaks one io_size
      // allocation per inbound TCP/IPC bulk.
      if (archive.daemon_allocated_bulk_count_ > 0) {
        task_ptr->SetFlags(TASK_DATA_OWNER);
      }

      // Create the Future (owns the FutureShm via shared_ptr; pushing onto the
      // lane copies it so the FutureShm outlives this scope).
      Future<Task> future(pool_id, method_id, task_ptr);
      auto future_shm = future.GetFutureShm();
      future_shm->origin_ = (mode == IpcMode::kTcp)
                                ? ClientOrigin::kClientTcp
                                : ClientOrigin::kClientIpc;
      // Capture the client's net_key so SendOut can stamp it back onto the
      // response (AllocLoadTask reassigns the server task's identity).
      future_shm->client_net_key_ = info.task_id_.net_key_;
      future_shm->client_pid_ = info.task_id_.pid_;
      future_shm->response_fd_ = recv_info.fd_;
      // Resolve the response transport. TCP clients advertise an ephemeral
      // response-listener port (archive.client_port_); open (or reuse from the
      // connection cache) a dedicated dial-back DEALER to <identity-host>:<port>
      // and route the response there instead of echoing back over the inbound
      // ROUTER. The DEALER's identity is "hostname:pid", so the host part plus
      // the advertised port is the listener address. IPC clients keep replying
      // over the same connection-oriented unix socket.
      if (mode == IpcMode::kTcp) {
        const std::string &identity = recv_info.identity_;
        int client_port = archive.client_port_;
        // Fast path: open (or reuse) a dedicated dial-back DEALER to the
        // client's ephemeral response listener at <identity-host>:<client_port>
        // and route the response there, off the inbound ROUTER's sock_mtx_. A
        // DEALER has a single peer so it auto-routes with no identity frame.
        // This requires the client to advertise a response port (client_port_)
        // AND present a parseable "hostname:pid" routing identity.
        ctp::lbm::Transport *dial_back = nullptr;
        size_t colon = identity.find(':');
        const bool parseable_identity =
            colon != std::string::npos &&
            identity.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789.-:") == std::string::npos;
        if (client_port > 0 && parseable_identity) {
          std::string host = identity.substr(0, colon);
          // Same-host client: dial loopback. The host part of the identity is
          // the client's gethostname(); when it matches ours the client is on
          // this machine, so 127.0.0.1 is both always resolvable and free of
          // the LAN-interface firewall rules an external hostname would need.
          // The cache key stays the full identity, so distinct clients never
          // alias.
          if (host == ctp::SystemInfo::GetHostname()) {
            host = "127.0.0.1";
          }
          dial_back =
              ipc->GetOrCreateClientByIdentity(identity, host, client_port);
        }
        // The inbound ROUTER is recv-only: responses NEVER go back over it (a
        // worker Send racing the recv thread on the same non-thread-safe ZMQ
        // socket is exactly what forced sock_mtx_ and deadlocked force_net).
        // Every live client opens a response listener (client_port_ > 0) and
        // connects with a "hostname:pid" identity, so dial-back always
        // resolves. If it ever doesn't, the response is undeliverable — log and
        // drop rather than echo over the ROUTER.
        if (dial_back) {
          future_shm->response_transport_ = dial_back;
          future_shm->response_identity_len_ = 0;  // DEALER: no identity frame
        } else {
          HLOG(kError,
               "IpcCpu2CpuZmq::RecvIn: TCP client {} has no dial-back route "
               "(client_port={}, identity='{}') — response undeliverable",
               future_shm->client_pid_, client_port, identity);
          future_shm->response_transport_ = nullptr;
          future_shm->response_identity_len_ = 0;
        }
      } else {
        future_shm->response_transport_ = transport;
        future_shm->response_identity_len_ = 0;
      }

      // Allocate the task's RunContext (and resolve its container) now that it
      // is deserialized, so RouteTask / the worker have an active RunContext.
      future.GetTaskPtr()->BeginRunContext();

      // Enqueue to worker lane
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
// EnqueueSendOut: worker-inline enqueue to net_queue_
//==============================================================================

void IpcCpu2CpuZmq::EnqueueSendOut(IpcManager *ipc,
                                         const clio::run::shared_ptr<Task> &task,
                                         ClientOrigin origin) {
  // Enqueue a future that OWNS the task. task->RunFuture()'s task_ptr_ may be a
  // NON-OWNING self-handle: RouteGlobal / RouteManyToOne rebind it non-owning to
  // break the leak cycle, so a collective/broadcast origin (e.g. GetOrCreatePool
  // creating one container per node) reaches here with a non-owning RunFuture.
  // Enqueuing that handle puts a non-owning reference on the net queue, so the
  // origin task can be freed before net_send_worker drains it — GetFutureShm()
  // then resolves through the freed task, returns null, the SendOut loop skips
  // the response, and the client's Wait() hangs forever (the multi-node AllToOne
  // / Distributed cluster tests). An owning copy keeps the task alive until the
  // response is on the wire; run_ctx_->future_ stays non-owning, so this adds no
  // leak — the net-queue copy drops once the response is sent.
  clio::run::Future<Task> owning = task->RunFuture();
  owning.GetTaskPtr() = task;
  if (origin == ClientOrigin::kClientTcp) {
    ipc->EnqueueNetTask(owning, NetQueuePriority::kClientSendTcp);
  } else {
    ipc->EnqueueNetTask(owning, NetQueuePriority::kClientSendIpc);
  }
}

//==============================================================================
// SendOut: net-worker serialize and send response via ZMQ
//==============================================================================

bool IpcCpu2CpuZmq::SendOut(
    IpcManager *ipc, u32 &tasks_sent,
    std::vector<clio::run::shared_ptr<Task>> & /*deferred_deletes — unused*/) {
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
      auto container =
          pool_manager->GetStaticContainer(origin_task->pool_id_).get();
      if (!container) {
        HLOG(kError, "IpcCpu2CpuZmq::SendOut: Container not found "
             "for pool_id {}", origin_task->pool_id_);
        continue;
      }

      // Get response transport and routing info from FutureShm
      ctp::lbm::Transport *response_transport =
          future_shm->response_transport_;
      if (!response_transport) {
        HLOG(kError, "IpcCpu2CpuZmq::SendOut: No response transport "
             "for mode {} pid {}", mode_idx, future_shm->client_pid_);
        continue;
      }

      // Restore the client's net_key so the serialized response matches the
      // pending future the ZMQ recv thread keyed by it.
      origin_task->task_id_.net_key_ = future_shm->client_net_key_;

      // Serialize task outputs
      SaveTaskArchive archive(MsgType::kSerializeOut, response_transport);
      container->SaveTask(origin_task->method_, archive, origin_task);

      // Routing. TCP responses always go over the dedicated dial-back DEALER
      // resolved at RecvIn (response_transport_): a DEALER has exactly one
      // connected peer (the client's response ROUTER), so it auto-routes — no
      // identity frame, and crucially never touches the recv-only inbound
      // ROUTER. IPC replies still carry the client's socket fd.
      if (mode == IpcMode::kIpc) {
        archive.client_info_.fd_ = future_shm->response_fd_;
      }

      // SYNC send: lightbeam copies bulks into ZMQ inside this call and
      // holds no reference to origin_task's buffers after it returns, so
      // the task can be released as soon as this iteration ends (RAII).
      // No async callback, no I/O-thread race with the task's destructor.
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

      // Send succeeded — caller-side buffers are no longer referenced by ZMQ.
      // The task itself frees via RAII: queued_future (and the origin_task
      // shared_ptr copy) drop at the end of this loop iteration.

      // The server-side FutureShm is owned by the queued Future's shared_ptr
      // (created in RecvIn); when queued_future goes out of scope at the end of
      // this loop iteration the FutureShm is freed automatically. No manual
      // FreeBuffer is needed (and CleanupResponseArchive is client-side only).

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
