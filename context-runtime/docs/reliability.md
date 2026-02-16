# Reliability Subsystem

This document describes the algorithms and mechanisms that keep a Chimaera
cluster correct and available when nodes join, leave, migrate containers, or
fail.  The subsystem spans several runtime components:

| Component | Key files |
|-----------|-----------|
| Address mapping | `pool_manager.h`, `pool_manager.cc` |
| Address consistency / WAL | `pool_manager.cc`, `admin_runtime.cc` |
| Heartbeat (SWIM) | `admin_runtime.cc` (HeartbeatProbe) |
| Recovery | `admin_runtime.cc` (TriggerRecovery, RecoverContainers) |
| Leadership | `ipc_manager.h` (GetLeaderNodeId, IsLeader) |
| Migration | `admin_runtime.cc` (MigrateContainers) |
| Restart | `admin_runtime.cc` (RestartContainers) |
| Container callbacks | `container.h` |
| Retry queues | `admin_runtime.cc` (ProcessRetryQueues) |

---

## 1  Address Mapping

Every pool has a global **address table** that maps each logical container ID
to the physical node that currently hosts it.  The table lives in `PoolInfo`:

```
struct PoolInfo {
  // ALL containers across the cluster
  unordered_map<ContainerId, u32>        address_map_;

  // Containers physically present on THIS node
  unordered_map<ContainerId, Container*> containers_;

  Container* static_container_;   // stateless ops (serialize/alloc)
  Container* local_container_;    // default container on this node
};
```

`address_map_` is replicated on every node in the cluster.  Updates are applied
through broadcast `ChangeAddressTable` tasks so that every node converges to the
same view.

### 1.1  Routing Resolution

When a worker picks up a task from its lane it resolves the task's `PoolQuery`
into one or more concrete targets before dispatching.  The resolution chain
(`Worker::ResolvePoolQuery` in `worker.cc`) handles every routing mode:

| Mode | Resolution |
|------|-----------|
| `Local` | Execute on current node. |
| `Physical(node_id)` | Already resolved; send to `node_id`. |
| `DirectId(container_id)` | Look up `address_map_[container_id]` to get node. If the container is local, short-circuit to `Local`. |
| `DirectHash(hash)` | Compute `container_id = hash % num_containers`, then resolve as `DirectId(container_id)`.  This preserves the container ID through the routing chain so retry-after-recovery can re-resolve it. |
| `Range(offset, count)` | Fan out into sub-queries, one per container or per neighbourhood chunk.  Single-container ranges resolve as `DirectId`. |
| `Broadcast` | Expand to `Range(0, num_containers)`, then resolve that range. |
| `Dynamic` | Execute locally first (the container's `Run` sets a new `pool_query_` in the RunContext) then re-route. |

A resolved query that targets a remote node becomes a `Physical` or `DirectId`
query stored in `RunContext::pool_queries_`.  The network worker's `SendIn`
then inspects each query to determine the target node ID.

### 1.2  address_map_ vs containers_

`address_map_` and `containers_` serve different purposes and can diverge after
migration or recovery:

| Map | Scope | Updated by |
|-----|-------|-----------|
| `address_map_` | Global container→node mapping, replicated on all nodes | `ChangeAddressTable` (broadcast), `RecoverContainers` (broadcast) |
| `containers_` | Local container objects physically present on this node | `RegisterContainer` / `UnregisterContainer` (local only) |

After migration, a container may be in `address_map_` pointing to a node that
does **not** have it in `containers_`.  This happens because:

1. Migration broadcasts `ChangeAddressTable` to update `address_map_` on all
   nodes (container X → destination node).
2. The destination node does not physically create the container—it simply
   becomes the routing target.  Tasks arriving at the destination use
   `GetContainer(pool_id, container_id)`, which falls back to
   `local_container_` when `container_id` is not in `containers_`.

Similarly, during recovery, `RecoverContainers` updates `address_map_` on all
nodes first, then creates the container only on the destination node.

**Key invariant**: `HasContainer(pool_id, cid)` checks `containers_` (local
only), while `GetContainerNodeId(pool_id, cid)` checks `address_map_` (global).
Routing logic must consult both to avoid forwarding loops (see §1.4).

### 1.3  Why DirectHash Resolves to DirectId

If `DirectHash` resolved directly to `Physical(node_id)`, the container ID
would be lost.  When the target node dies and the container is recovered to a
different node, the retry queue would have no way to re-resolve the target
because it only has a stale node ID.  By resolving to `DirectId(container_id)`,
the retry logic can call `GetContainerNodeId` against the updated
`address_map_` and discover the new location.

### 1.4  Local Node Check (Forwarding Loop Prevention)

Both `ResolveDirectHashQuery` and `ResolveRangeQuery` perform a two-step local
check before returning `DirectId`:

```
ResolveDirectHashQuery(hash):
  container_id = hash % num_containers

  // Step 1: container physically present on this node?
  if HasContainer(pool_id, container_id):     // checks containers_
    return Local()

  // Step 2: address_map_ says this node owns the container?
  if GetContainerNodeId(pool_id, container_id) == self_node_id:
    return Local()                            // checks address_map_

  // Step 3: remote — preserve container_id for retry-after-recovery
  return DirectId(container_id)
```

Step 2 prevents an **infinite forwarding loop** that occurs when:

1. A container is in `address_map_` pointing to this node (e.g., after
   migration or recovery).
2. The container is **not** in `containers_` (no physical object registered).
3. Without step 2, the resolver returns `DirectId(container_id)`.
4. `IsTaskLocal` returns false for DirectId → `RouteGlobal` → `SendIn`.
5. `SendIn` calls `GetContainerNodeId` → finds this node → sends to self.
6. The task re-enters the worker → step 3 again → infinite loop.

With step 2, the resolver detects that `address_map_` maps the container to the
local node and returns `Local()`.  The task then executes via `RouteLocal`,
which calls `GetContainer` — this falls back to `local_container_` when the
specific container ID is not in `containers_`.

---

## 2  Address Consistency

All `address_map_` mutations go through a two-step protocol:

1. **WAL write** -- persist the change to disk before applying it.
2. **Broadcast** -- every node applies the same update.

### 2.1  ChangeAddressTable Task

```
ChangeAddressTable(pool_id, container_id, new_node_id):
  1. old_node = GetContainerNodeId(pool_id, container_id)
  2. WriteAddressTableWAL(pool_id, container_id, old_node, new_node)
  3. UpdateContainerNodeMapping(pool_id, container_id, new_node)
```

This task is always sent with `PoolQuery::Broadcast()`, so every alive node
executes the same steps.  The result is a consistent `address_map_` on all
nodes.

### 2.2  Write-Ahead Log (WAL)

WAL entries are appended to per-pool binary files under `<conf_dir>/wal/`:

```
Path: <conf_dir>/wal/domain_table.<pool_major>.<pool_minor>.<node_id>.bin
```

Each entry is a fixed-size record:

| Field | Type | Bytes |
|-------|------|-------|
| timestamp | u64 (nanoseconds since epoch) | 8 |
| pool_id | PoolId (major + minor) | 8 |
| container_id | u32 | 4 |
| old_node | u32 | 4 |
| new_node | u32 | 4 |

The WAL is append-only and written synchronously before the in-memory mapping
is updated.  On recovery, the WAL can be replayed to reconstruct the address
table without needing cluster-wide communication.

### 2.3  New Node Integration (AddNode)

When a new node joins:

```
AddNode(ip, port):
  1. ipc_manager->AddNode(ip, port)          // assign node_id
  2. For each pool:
       container->Expand(new_host)           // callback
  3. Return new_node_id
```

The new node does **not** receive a snapshot of existing address tables in this
path.  It bootstraps by loading its own WAL or by having containers created on
it via `GetOrCreatePool` broadcasts.

---

## 3  Heartbeat (SWIM Failure Detection)

Node liveness is monitored with a SWIM-inspired protocol implemented in
`Runtime::HeartbeatProbe`, a periodic admin task.

### 3.1  Node State Machine

```
                 direct probe OK
        +-----------------------------+
        |                             |
        v                             |
   +---------+   direct timeout   +-------------+
   |  Alive  | -----------------> | ProbeFailed |
   +---------+   (5 seconds)     +-------------+
        ^                             |
        |   indirect probe OK         |  all indirect probes timeout
        +-----------------------------+  (3 seconds each)
                                      |
                                      v
                                +-----------+
                                | Suspected |
                                +-----------+
                                      |
                                      |  suspicion timeout (10 seconds)
                                      v
                                  +------+
                                  | Dead |
                                  +------+
```

State transitions are tracked with `state_changed_at` timestamps in the `Host`
struct.

### 3.2  Algorithm (Five Steps per Invocation)

The `HeartbeatProbe` task runs periodically (default 2-second interval, or
configured via `heartbeat_interval` in the YAML config).  Each invocation
executes five steps in order:

**Step 1 -- Check pending direct probes**

For each entry in `pending_direct_probes_`:
- If the heartbeat future completed: set target state to `kAlive`, remove entry.
- If elapsed > `kDirectProbeTimeoutSec` (5s): escalate to indirect probing.
  - Set target state to `kProbeFailed`.
  - Pick `kIndirectProbeHelpers` (3) random alive helpers (excluding self and target).
  - Send `AsyncProbeRequest` to each helper.
  - Add entries to `pending_indirect_probes_`.
  - Remove from `pending_direct_probes_`.

**Step 2 -- Check pending indirect probes**

For each entry in `pending_indirect_probes_`:
- If the future completed with `probe_result_ == 0` (alive):
  - Set target state to `kAlive`.
  - Remove **all** pending indirect probes for this target.
- If the future completed with `probe_result_ == -1` or timed out (3s):
  - Remove this entry.
  - If no more pending indirect probes remain for this target: set state to
    `kSuspected`.

**Step 3 -- Check suspicion timeouts**

For each host in `kSuspected` state:
- If `time_since_state_change >= kSuspicionTimeoutSec` (10s):
  - Call `TriggerRecovery(node_id)`.
  - Call `SetDead(node_id)`.

**Step 4 -- Self-fencing (partition detection)**

Count the number of other nodes that are suspected or dead.  If a majority are
unreachable (`bad_count * 2 > other_count`), the node **fences itself** to
prevent split-brain:

```
SetSelfFenced(true)
```

A self-fenced node will not initiate recovery.  The fence is cleared when
connectivity is restored.

**Step 5 -- Send a new direct probe (round-robin)**

Select the next alive node (round-robin, skipping self, dead, suspected, and
probe-failed nodes) and send `AsyncHeartbeat(Physical(target))`.  Only one new
probe is sent per invocation to spread load.

Nodes in `kSuspected` or `kProbeFailed` states are **skipped** in step 5 to
prevent a re-probing cycle that would reset `state_changed_at` and prevent the
suspicion timeout from ever firing.

### 3.3  ProbeRequest (Indirect Probe)

When a node receives a `ProbeRequest` task it probes the target on behalf of
the requester:

```
ProbeRequest(target_node_id):
  future = AsyncHeartbeat(Physical(target_node_id))
  while !future.IsComplete() and elapsed < 3s:
    co_await yield()
  if future.IsComplete():
    probe_result_ = 0    // alive
  else:
    probe_result_ = -1   // unreachable
```

### 3.4  Timing Summary

| Constant | Value | Purpose |
|----------|-------|---------|
| `kDirectProbeTimeoutSec` | 5.0s | Time before escalating to indirect probes |
| `kIndirectProbeTimeoutSec` | 3.0s | Per-helper indirect probe timeout |
| `kIndirectProbeHelpers` | 3 | Number of helpers for indirect probing |
| `kSuspicionTimeoutSec` | 10.0s | Time in suspected state before confirming dead |
| `kRetryTimeoutSec` | 30.0s | Max time a task can sit in retry queue |

Worst-case detection latency: 5s (direct) + 3s (indirect) + 10s (suspicion) =
**18 seconds**.

---

## 4  Leadership

Leadership is deterministic and requires no election protocol.  Every node
computes the same leader from its local `Host` table:

```
GetLeaderNodeId():
  return min(node_id) where host.IsAlive()
```

```
IsLeader():
  return GetNodeId() == GetLeaderNodeId()
```

Because all nodes agree on which nodes are alive (the SWIM protocol converges
within seconds), they agree on the leader.  If the current leader dies, the
node with the next-lowest ID takes over automatically.

The leader is the only node that initiates recovery (`TriggerRecovery` checks
`IsLeader()` before proceeding).

---

## 5  Recovery

Recovery is triggered when the SWIM protocol confirms a node is dead (step 3
of HeartbeatProbe).  Only the leader initiates recovery.

### 5.1  TriggerRecovery

```
TriggerRecovery(dead_node_id):
  if !IsLeader(): return
  if already_initiated(dead_node_id): return
  if IsSelfFenced(): return      // partition safety

  assignments = ComputeRecoveryPlan(dead_node_id)
  if assignments.empty(): return

  AsyncRecoverContainers(Broadcast(), assignments, dead_node_id)
```

### 5.2  ComputeRecoveryPlan

The leader scans every pool's `address_map_` for containers that were on the
dead node and assigns each to a surviving node using round-robin:

```
ComputeRecoveryPlan(dead_node_id):
  alive_nodes = [h.node_id for h in GetAllHosts() if h.IsAlive()]
  rr_idx = 0

  for each pool in GetAllPoolIds():
    for each (container_id, node_id) in pool.address_map_:
      if node_id == dead_node_id:
        assignment = RecoveryAssignment {
          pool_id, chimod_name, pool_name, chimod_params,
          container_id, dead_node_id,
          dest_node_id = alive_nodes[rr_idx++ % len(alive_nodes)]
        }
        assignments.append(assignment)

  return assignments
```

### 5.3  RecoverContainers (Broadcast)

This task is broadcast to **all alive nodes**.  Every node executes the same
handler:

```
RecoverContainers(assignments):
  for each assignment in assignments:
    // ALL NODES: update address table
    UpdateContainerNodeMapping(pool_id, container_id, dest_node_id)
    WriteAddressTableWAL(pool_id, container_id, dead_node, dest_node)

    // ONLY DEST NODE: create the container
    if self_node_id == dest_node_id:
      container = CreateContainer(chimod_name, pool_id, pool_name)
      container->Restart(pool_id, pool_name, container_id)  // callback
      RegisterContainer(pool_id, container_id, container)
```

**Key property**: The address table update is applied on all nodes for
consistency, but the container is only physically created on the destination
node.

### 5.4  Container Callback: Restart

```cpp
virtual void Restart(const PoolId& pool_id, const std::string& pool_name,
                     u32 container_id = 0) {
  Init(pool_id, pool_name, container_id);
}
```

The default `Restart` implementation calls `Init`, which reinitializes the
container with a clean state.  Modules that need to restore persistent state
(e.g., from a checkpoint or WAL) should override `Restart` to load that state
after calling `Init`.

**When invoked**: During `RecoverContainers` on the destination node, after
`CreateContainer` allocates the container but before it begins accepting tasks.

---

## 6  Migration

Migration moves a live container from one node to another without killing
either node.

### 6.1  MigrateContainers Flow

```
MigrateContainers(migrations):
  for each MigrateInfo(pool_id, container_id, dest_node) in migrations:

    1. Plug the container (stop accepting new tasks)
       PlugContainer(pool_id, container_id)

    2. Call container->Migrate(dest_node)      // callback
       (serialize/transfer state if needed)

    3. Broadcast ChangeAddressTable to all nodes
       co_await AsyncChangeAddressTable(Broadcast(),
                  pool_id, container_id, dest_node)

    4. Unregister container on source node
       UnregisterContainer(pool_id, container_id)
```

### 6.2  Container Callbacks During Migration

**Plug** (`Container::SetPlugged`):
- Sets `CONTAINER_PLUG` flag.  Workers check `IsPlugged()` before dispatching
  tasks to a container.  Plugged containers reject new work.
- Callers wait for `GetWorkRemaining() == 0` before proceeding.

**Migrate** (`Container::Migrate`):
```cpp
virtual void Migrate(u32 dest_node_id) {
  (void)dest_node_id;  // default: no-op
}
```
Override to serialize and transfer container state to the destination node.
Called after the container is plugged and all in-flight work has drained.

**When invoked**: `Migrate` is called on the **source node** after plug and
drain.  The destination node receives the container via the pool creation
pathway (the pool already exists; the address table update directs future tasks
to the new node).

### 6.3  Unregister on Source

After the address table is updated on all nodes, the source node calls:

```
UnregisterContainer(pool_id, container_id)
```

This removes the container from `containers_` (so `HasContainer` returns
false) and recalculates `local_container_`.  The `static_container_` pointer
is preserved so that in-flight task serialization still works.  Subsequent
`DirectHash` queries that previously resolved locally now fall through to
`address_map_` lookup, which returns the new destination node.

---

## 7  Restart (Warm Start)

The `RestartContainers` task restores pools from saved YAML configurations
on disk.

### 7.1  Algorithm

```
RestartContainers():
  restart_dir = <conf_dir>/restart/

  if !exists(restart_dir): return

  for each .yaml file in restart_dir:
    pool_configs = LoadYaml(file).compose_config.pools_
    for each pool_config in pool_configs:
      co_await AsyncCompose(pool_config)
      containers_restarted_++
```

Each pool configuration YAML captures the full pool spec (module name, pool
name, pool ID, number of containers, module-specific parameters).  The
`Compose` pathway creates the pool on all nodes, which internally calls
`Container::Init` for each new container.

### 7.2  When Invoked

`RestartContainers` is typically called during runtime startup after the basic
infrastructure (IPC, pool manager, admin container) is initialized.  It
re-creates pools that were active before the previous shutdown.

---

## 8  Retry Queues

When `SendIn` or `SendOut` cannot reach a target node (dead or transport
failure), the task is placed in a retry queue rather than being dropped.

### 8.1  Data Structures

```cpp
struct RetryEntry {
  FullPtr<Task> task;
  u64 target_node_id;
  steady_clock::time_point enqueued_at;
};

deque<RetryEntry> send_in_retry_;    // failed input sends
deque<RetryEntry> send_out_retry_;   // failed output sends
```

### 8.2  ProcessRetryQueues

Called on every `Send` task invocation (the periodic network worker loop).

**For each entry in `send_in_retry_`:**

```
if elapsed >= 30s:
  task->SetReturnCode(kNetworkTimeoutRC)    // fail permanently
  erase entry

else if IsAlive(original_target):
  if RetrySendToNode(entry, original_target):
    erase entry                             // success

else:  // original target still dead
  new_node = RerouteRetryEntry(entry)
  if new_node != 0 and IsAlive(new_node):
    entry.target_node_id = new_node
    if RetrySendToNode(entry, new_node):
      erase entry                           // re-routed successfully
```

**For each entry in `send_out_retry_`:**

Same logic but without re-routing (outputs must go back to the original
requesting node).  If the requesting node is dead for 30s, the entry is
dropped (the client will eventually time out).

### 8.3  RerouteRetryEntry

Re-resolves the target for a retried task by consulting the updated
`address_map_`:

```
RerouteRetryEntry(entry):
  query = entry.task->pool_query_

  if query.IsDirectIdMode():
    container_id = query.GetContainerId()
    new_node = GetContainerNodeId(pool_id, container_id)
    return new_node if new_node != original_target, else 0

  if query.IsRangeMode():
    container_id = query.GetRangeOffset()
    new_node = GetContainerNodeId(pool_id, container_id)
    return new_node if new_node != original_target, else 0

  return 0  // cannot re-route broadcast/physical/etc.
```

This is why `DirectHash` resolves to `DirectId` rather than `Physical` -- it
preserves the `container_id` needed for re-resolution after recovery updates
the address map.

### 8.4  ScanSendMapTimeouts

Scans `send_map_` for origin tasks whose replicas target nodes that have been
dead for more than `kRetryTimeoutSec` (30s).  These tasks are failed with
`kNetworkTimeoutRC` and completed via `EndTask`.

---

## 9  Network Transport (Send/Recv)

All distributed task execution flows through four helper methods on the admin
runtime.  A single dedicated network worker processes all of them, eliminating
the need for locks on `send_map_` and `recv_map_`.

### 9.1  SendIn (Send Task Inputs)

```
SendIn(origin_task, rctx):
  send_map_key = ptr(origin_task)
  send_map_[send_map_key] = origin_task

  for each (i, query) in rctx.pool_queries_:
    target_node = resolve(query)        // DirectId, Range, Physical, etc.

    task_copy = NewCopyTask(origin_task, deep=true)
    subtasks_[i] = task_copy
    task_copy.net_key_ = send_map_key   // for response matching
    task_copy.replica_id_ = i
    task_copy.SetReturnNode(self)

    if !IsAlive(target_node):
      send_in_retry_.push(task_copy, target_node)
      continue

    archive = SaveTaskArchive(kSerializeIn, transport)
    container->SaveTask(method, archive, task_copy)
    transport->Send(archive)            // non-blocking
```

### 9.2  RecvIn (Receive Task Inputs)

```
RecvIn(archive):
  for each task_info in archive:
    task = container->AllocLoadTask(method, archive)
    task.SetFlags(TASK_REMOTE | TASK_DATA_OWNER)
    task.ClearFlags(TASK_PERIODIC | TASK_FORCE_NET | TASK_ROUTED |
                    TASK_RUN_CTX_EXISTS | TASK_STARTED)

    recv_key = net_key ^ (replica_id * hash_constant)
    recv_map_[recv_key] = task

    ipc_manager->Send(task)             // enqueue for local execution
```

### 9.3  SendOut (Send Task Outputs)

```
SendOut(completed_task):
  recv_key = net_key ^ (replica_id * hash_constant)
  recv_map_.erase(recv_key)

  return_node = completed_task.GetReturnNode()

  if !IsAlive(return_node):
    send_out_retry_.push(completed_task, return_node)
    return

  archive = SaveTaskArchive(kSerializeOut, transport)
  container->SaveTask(method, archive, completed_task)
  transport->Send(archive)              // non-blocking

  deferred_deletes_.push(completed_task)  // zero-copy safety
```

### 9.4  RecvOut (Receive Task Outputs)

Two-pass algorithm:

**Pass 1 -- Deserialize outputs:**
```
for each task_info in archive:
  origin_task = send_map_[net_key]
  replica = origin_task.subtasks_[replica_id]
  container->LoadTask(method, archive, replica)   // exposes bulk buffers
```

**Pass 2 -- Aggregate and complete:**
```
for each task_info in archive:
  origin_task = send_map_[net_key]
  replica = origin_task.subtasks_[replica_id]
  container->Aggregate(method, origin_task, replica)
  completed_replicas_++

  if completed_replicas_ == total_replicas:
    delete all replicas
    send_map_.erase(net_key)
    EndTask(origin_task)                // unblocks waiting coroutine
```

### 9.5  Network Key Matching

| Field | Set in | Used in | Purpose |
|-------|--------|---------|---------|
| `net_key_` | SendIn (= ptr of origin task) | RecvOut (lookup in send_map_) | Match response to origin |
| `replica_id_` | SendIn (= index in pool_queries_) | RecvOut (index into subtasks_) | Identify which replica returned |
| `return_node_` | SendIn (= self node ID) | SendOut (= destination for outputs) | Route outputs back |

---

## 10  Container Callback Summary

| Callback | When Invoked | Default Behavior |
|----------|-------------|-----------------|
| `Init(pool_id, pool_name, container_id)` | Pool creation, start of `Restart` | Initialize base fields, clear flags |
| `Restart(pool_id, pool_name, container_id)` | Recovery (`RecoverContainers` on dest node), warm start (`RestartContainers`) | Calls `Init`. Override to restore persistent state. |
| `Expand(new_host)` | `AddNode` -- a new node joined the cluster | No-op. Override to re-partition data. |
| `Migrate(dest_node_id)` | `MigrateContainers` -- after plug and drain on source node | No-op. Override to serialize and transfer state. |
| `GetWorkRemaining()` | Migration drain check, shutdown drain | Pure virtual. Return count of pending work units. |
| `SetPlugged()` / `IsPlugged()` | Migration start (plug), followed by drain | Sets/checks atomic `CONTAINER_PLUG` flag. |

### 10.1  Correctness Guarantees for Recovery Callbacks

1. **Restart is called after CreateContainer**: The container object exists and
   has been allocated by `ModuleManager::CreateContainer` before `Restart` is
   invoked.  The container is not yet registered with `PoolManager`, so no
   tasks can reach it during initialization.

2. **RegisterContainer happens after Restart**: Only after `Restart` completes
   does the container become visible to the routing system via
   `RegisterContainer`.  This prevents tasks from reaching a half-initialized
   container.

3. **Address table is updated before container creation**: All nodes (including
   the destination) update `address_map_` before the destination node creates
   the container.  Tasks that arrive at the destination before the container is
   registered will find `HasContainer() == false` and be queued for retry.

4. **WAL is written before in-memory update**: Both `ChangeAddressTable` and
   `RecoverContainers` write WAL entries before calling
   `UpdateContainerNodeMapping`.  A crash between WAL write and in-memory
   update can be recovered by replaying the WAL.

### 10.2  Correctness Guarantees for Migration Callbacks

1. **Plug before Migrate**: The container is plugged (no new tasks accepted)
   and all in-flight work is drained (`GetWorkRemaining() == 0`) before
   `Migrate` is called.

2. **ChangeAddressTable before Unregister**: The address table is updated on
   all nodes (via broadcast) before the source node unregisters the container.
   This ensures no window where tasks are routed to a node that no longer has
   the container.

3. **Unregister preserves static_container_**: After unregistering, the source
   node keeps `static_container_` alive so that in-flight serialization of
   tasks (e.g., tasks already in the network pipeline) can still use the
   container's `SaveTask`/`LoadTask` methods.

---

## 11  End-to-End Recovery Timeline

```
t=0     Node 4 crashes
t=0-5s  Direct probe to node 4 times out
        → State: kProbeFailed
t=5-8s  3 indirect probes sent via helper nodes
        All indirect probes fail (3s timeout each)
        → State: kSuspected
t=8-18s Suspicion timeout (10s) expires
        → State: kDead
t=18s   Leader calls TriggerRecovery(node_4)
          ComputeRecoveryPlan: scan all pools, assign containers
          Broadcast RecoverContainers:
            All nodes: update address_map_
            Dest nodes: CreateContainer + Restart + RegisterContainer
t=18s+  Retry queues re-resolve: RerouteRetryEntry finds new node
        Tasks that were waiting for dead node route to recovered container
        Normal operation resumes
```

With `heartbeat_interval: 500` (500ms probe interval), detection is faster
because probes are sent more frequently, reducing the time between node death
and the first failed probe.
