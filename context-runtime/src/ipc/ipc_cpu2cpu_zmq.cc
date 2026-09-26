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
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <vector>

namespace clio::run {

// #722: bound the re-queue of an undeliverable client response. A client that
// submitted a task over TCP/IPC and then disconnected leaves the daemon holding
// a response whose network Send fails persistently. Without a cap the owning
// future is re-enqueued forever (the reporter saw 22.9M retries / an 8.9 GB
// log). Drop the response — freeing the task via RAII — after whichever of these
// bounds trips first. Mirrors the run2run retry-timeout in ipc_run2run.cc.
static constexpr u32 kMaxClientResponseRetries = 32;
static constexpr double kClientResponseRetryDropSec = 5.0;

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
        // Nothing took ownership of the received bulks, so release them here.
        // The ClearRecvHandles below is reached only on the success path; a
        // message abandoned here or at the missing-container check kept its
        // transport-allocated payload forever (LeakSanitizer: 1 MB per
        // inbound bulk in cr_shutdown_bt_transports, where a client's last
        // request arrives after its pool's container is already destroyed).
        transport->ClearRecvHandles(archive);
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
        // No AllocLoadTask ran, so the bulks are still transport-owned; see
        // the note above.
        transport->ClearRecvHandles(archive);
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
      // #968: also keep the client's non-recyclable identity so SendOut can
      // echo it and the client can corroborate the net_key match. AllocLoadTask
      // has already reassigned the server task's own task_id_, so this is the
      // only surviving record of who actually asked.
      future_shm->client_task_unique_ = info.task_id_.unique_;
      future_shm->client_task_major_ = info.task_id_.major_;
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

      // issue #781: ClientMapTask removed. Recv threads deposit on the shared
      // ingress lane (0); the runtime maps to a worker via RuntimeMapTask.
      LaneId lane_id = 0;
      auto *worker_queues = ipc->GetTaskQueue();
      auto &lane_ref = worker_queues->GetLane(lane_id, 0);
      lane_ref.Push(future);
      // Always signal — see ipc_cpu2cpu_impl.h for the race.
      ipc->AwakenWorker(&lane_ref);
      HLOG(kDebug, "[TRACE768] t={} RecvIn ingested task mode={} lane_tid={}",
           std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(), (int)mode, lane_ref.GetTid());

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

// Client-response send instrumentation (#968). At namespace scope rather than
// function-local statics so LogSendTally() can report the finals at teardown;
// SendOut is driven from a periodic admin task, so it has no exit of its own to
// hook. Relaxed throughout -- read once at shutdown, nothing branches on them.
static std::atomic<size_t> send_counter{0};
static std::atomic<size_t> send_fail_counter{0};
// #968 anomaly counters. Each has a loud per-occurrence kError above; these
// exist so the teardown tally states the totals even if the per-occurrence
// lines are lost to log rotation or a truncated tail.
static std::atomic<size_t> premature_send_counter{0};
static std::atomic<size_t> duplicate_send_counter{0};

// Timestamp of the last MaybeLogSendTally() report, as steady-clock
// nanoseconds; 0 means "not yet seeded".
//
// Deliberately a bare atomic and not the mutex-guarded trio of function-local
// statics this started as, because a function-local std::mutex here is only
// accidentally safe. SendOut still runs during teardown: the runtime finalizes
// from atexit handlers, and ServerFinalize -> DrainPendingTasks drains with the
// workers live, so Runtime::ClientSend drives at least one more pass through
// here. Whether a function-local static is still alive for that pass depends on
// whether its __cxa_atexit registration (first pass, on a worker thread) beat
// the main thread's std::atexit(CLIO_RUNTIME_FINALIZE) at the end of init --
// handlers run LIFO over one shared list, so registering later means being
// destroyed earlier. Measured on Linux the worker wins that race comfortably
// (the ClientSend periodic ticks hundreds of times during ClientInit), but
// nothing enforces it.
//
// Losing it costs nothing on libstdc++, where ~std::mutex() is trivial when
// __GTHREAD_MUTEX_INIT is defined, so the object is never really destroyed.
// libc++ does call pthread_mutex_destroy(), and Darwin then fails the late lock
// with EINVAL -- std::mutex::lock() throws inside a coroutine, whose promise
// unhandled_exception() std::terminate()s the process.
//
// A trivially destructible atomic has no destructor to order, so the teardown
// pass is safe by construction. It is also cheaper: this runs on every SendOut.
static std::atomic<i64> last_tally_ns{0};

/**
 * #968: emit LogSendTally() on a wall-clock cadence from the SendOut pass.
 *
 * LogSendTally() was originally hooked to ~IpcManager(). That destructor never
 * runs in the deployments this instrumentation was built for: the benchmark
 * pipeline tears the daemon down with `clio_run runtime stop --force` followed
 * by `pkill -9`, and there is no SIGTERM handler, so the process dies without
 * unwinding and the tally was never printed once. SendOut is driven by a
 * periodic admin task that keeps running whether or not there is traffic, so
 * gating on elapsed time here reports the totals for the life of the daemon
 * regardless of how it eventually dies -- at worst kReportPeriodSec stale.
 * @param force emit now, ignoring the cadence (used at teardown).
 */
static void MaybeLogSendTally(bool force) {
  static constexpr i64 kReportPeriodNs = 30LL * 1000 * 1000 * 1000;
  i64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
  // 0 is the "unseeded" sentinel, so never store it as a real stamp.
  if (now == 0) now = 1;
  while (true) {
    i64 last = last_tally_ns.load(std::memory_order_relaxed);
    if (last != 0 && !force && now - last < kReportPeriodNs) {
      return;
    }
    if (!last_tally_ns.compare_exchange_weak(last, now,
                                             std::memory_order_relaxed)) {
      continue;  // another worker moved the stamp; re-test against its value
    }
    // Exactly one caller wins each period, which is what the mutex bought.
    if (last == 0 && !force) {
      return;  // nothing has happened yet on the first pass
    }
    break;
  }
  IpcCpu2CpuZmq::LogSendTally();
}

void IpcCpu2CpuZmq::LogSendTally() {
  const size_t sent = send_counter.load(std::memory_order_relaxed);
  const size_t failed = send_fail_counter.load(std::memory_order_relaxed);
  if (sent == 0 && failed == 0) {
    return;  // this process never sent a client response
  }
  // Retries are the leading suspect for a DUPLICATED response in #968, so a
  // non-zero count is promoted to kWarning: it must survive a log level that
  // hides kInfo, and it is the number to correlate against the client's
  // [CountClientRecv] miss tally.
  if (failed > 0) {
    HLOG(kWarning,
         "[CountSend] TOTAL client responses sent={} send_failures={} -- each "
         "failure re-queued the response, so a duplicate delivery is possible; "
         "see #968",
         sent, failed);
  } else {
    HLOG(kInfo, "[CountSend] TOTAL client responses sent={} send_failures=0",
         sent);
  }
  // #968 anomaly totals. Both are zero in a healthy run, so a non-zero line is
  // the headline result of the run and must survive a kInfo log level.
  const size_t premature =
      premature_send_counter.load(std::memory_order_relaxed);
  const size_t duplicate =
      duplicate_send_counter.load(std::memory_order_relaxed);
  if (premature > 0 || duplicate > 0) {
    HLOG(kError,
         "[CountSend] TOTAL ANOMALIES premature_responses={} "
         "duplicate_responses={} (of {} sent) -- see #968",
         premature, duplicate, sent);
  } else {
    HLOG(kInfo,
         "[CountSend] TOTAL ANOMALIES premature_responses=0 "
         "duplicate_responses=0 (of {} sent)",
         sent);
  }
}

bool IpcCpu2CpuZmq::SendOut(
    IpcManager *ipc, u32 &tasks_sent,
    std::vector<clio::run::shared_ptr<Task>> & /*deferred_deletes — unused*/) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  bool did_work = false;
  tasks_sent = 0;

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
      HLOG(kDebug, "[TRACE768] t={} SendOut popped queued response prio={}",
           std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(), (int)priority);
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
      // pending future the ZMQ recv thread keyed by it, and echo back the
      // client's own task identity (#968) so the recv thread can tell a reply
      // to the task it is waiting for from a late reply to a freed task that
      // used to live at this address. TaskInfo is built from task_id_ inside
      // SaveTask, so both must be stamped before serializing. This is the last
      // use of origin_task's identity -- it is freed by RAII at the end of this
      // iteration -- so overwriting it here is safe.
      origin_task->task_id_.net_key_ = future_shm->client_net_key_;
      origin_task->task_id_.unique_ = future_shm->client_task_unique_;
      origin_task->task_id_.major_ = future_shm->client_task_major_;

      // #968 PREMATURE-COMPLETION PROBE. The observed failure is a client that
      // deserialized a well-formed response whose OUT fields were all at their
      // runtime-side pre-completion defaults -- i.e. the archive was serialized
      // before the handler's coroutine reached the line that fills them in.
      // coro_completed_ is set by the top-level coroutine's final_suspend, so
      // this is a direct test of that: a response leaving here with the flag
      // clear IS the bug, and names the task it happened to.
      //
      // Two BENIGN paths also complete a task without running its coroutine to
      // the end, and are expected to appear here: the early access-control
      // rejection in Worker::ExecTask (rc = EACCES) and the batch-merge
      // broadcast in Worker::CompleteBatchParents. Both are correct -- there
      // genuinely is no handler output to wait for. So pool, method and rc are
      // logged: the #968 signature is rc=0 on a handler that DOES produce OUT
      // fields, which neither benign path can produce.
      if (!origin_task->IsCoroCompleted()) {
        premature_send_counter.fetch_add(1, std::memory_order_relaxed);
        HLOG(kError,
             "[#968] PREMATURE RESPONSE: serializing a reply for a task whose "
             "coroutine has NOT completed -- OUT fields are still at their "
             "defaults (net_key={} client_unique={} pool={} method={} rc={} "
             "worker={}). rc=0 here means the client reads a silent failure; "
             "rc=13 (EACCES) or a batch-merge pool is the benign case.",
             future_shm->client_net_key_, future_shm->client_task_unique_,
             origin_task->pool_id_, origin_task->method_,
             origin_task->GetReturnCode(), origin_task->RunWorkerId());
      }

      // #968 DUPLICATE-RESPONSE PROBE. One request must produce exactly one
      // reply. A second one is delivered against a net_key the client may have
      // already recycled onto a different task, which is the other way the
      // observed symptom can arise. Counted on the future itself so a repeat is
      // reported at the moment it is created, not inferred afterwards from the
      // client's miss tally.
      if (future_shm->responses_sent_ > 0) {
        duplicate_send_counter.fetch_add(1, std::memory_order_relaxed);
        HLOG(kError,
             "[#968] DUPLICATE RESPONSE: this future has already shipped {} "
             "reply(ies) (net_key={} client_unique={} pool={} method={} "
             "send_fail_count={}). A second delivery lands on whatever owns "
             "this net_key now.",
             future_shm->responses_sent_, future_shm->client_net_key_,
             future_shm->client_task_unique_, origin_task->pool_id_,
             origin_task->method_, future_shm->send_fail_count_);
      }

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
        send_fail_counter.fetch_add(1, std::memory_order_relaxed);
        // #722: bound the re-queue. Track failures on the response future's
        // RunContext (it rides along across re-queues). Stamp the first-failure
        // time once, then drop after N attempts OR T seconds — whichever first.
        if (future_shm->send_fail_count_ == 0) {
          future_shm->first_send_fail_.Now();
        }
        future_shm->send_fail_count_++;
        ctp::Timepoint now;
        now.Now();
        double elapsed_sec =
            future_shm->first_send_fail_.GetUsecFromStart(now) / 1e6;
        if (future_shm->send_fail_count_ > kMaxClientResponseRetries ||
            elapsed_sec >= kClientResponseRetryDropSec) {
          // Undeliverable: log ONCE at kWarning and DROP. Not re-enqueuing
          // means the next TryPopNetTask overwrites queued_future, dropping the
          // last owner of the task → freed by RAII (no leak, no infinite loop).
          HLOG(kWarning,
               "IpcCpu2CpuZmq::SendOut: dropping undeliverable client response "
               "for pid {} after {} retries / {}s (last rc={}, priority={})",
               future_shm->client_pid_, future_shm->send_fail_count_,
               elapsed_sec, rc, static_cast<int>(priority));
          continue;
        }
        // Bounded retry: re-enqueue. #722 put this at kDebug because logging
        // EVERY attempt at kError floods; but kDebug is COMPILED OUT of a
        // default build (CTP_LOG_LEVEL defaults to kInfo and HLOG gates on
        // `if constexpr`), so in practice the retry path was invisible in
        // every deployed build -- including the #968 sweeps, where a retried
        // (hence duplicated) response is a leading suspect.
        //
        // Log the FIRST failure per response at kWarning and the rest at
        // kDebug. That keeps #722's anti-flood property -- one line per
        // struggling response, not one per attempt -- while making the
        // phenomenon visible at all without a debug build.
        HLOG(kDebug,
             "IpcCpu2CpuZmq::SendOut: Send rc={} attempt {} — re-queueing "
             "client response (priority={})",
             rc, future_shm->send_fail_count_, static_cast<int>(priority));
        if (future_shm->send_fail_count_ == 1) {
          HLOG(kWarning,
               "IpcCpu2CpuZmq::SendOut: Send rc={} — re-queueing client "
               "response for pid {} (first retry for this response; "
               "subsequent attempts log at kDebug, priority={})",
               rc, future_shm->client_pid_, static_cast<int>(priority));
        }
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
      // #968: this future has now put a reply on the wire. A later iteration
      // that reaches the duplicate probe above with this non-zero is shipping a
      // second reply for one request.
      future_shm->responses_sent_++;
      size_t total = send_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if ((total & 0xff) == 0) {
        // kInfo, not kDebug: already rate-limited to 1-in-256, so this is ~10
        // lines per benchmark row -- cheap enough to keep in a default build,
        // and it is the only running record of send_fail_counter, which is
        // otherwise unobservable outside a debug build that nobody deploys.
        HLOG(kInfo,
             "[CountSend] cumulative client responses sent = {} "
             "(mode={}, fails so far = {})",
             total, mode_idx,
             send_fail_counter.load(std::memory_order_relaxed));
      }
    }
  }

  // #968: periodic totals, so the tally exists in the log even though this
  // daemon is killed rather than shut down. See MaybeLogSendTally.
  MaybeLogSendTally(/*force=*/false);

  return did_work;
}

}  // namespace clio::run
