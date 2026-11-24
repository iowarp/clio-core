# Capacity and storage-target bindings

These Python methods wrap existing CTE native tasks. They do not change task
serialization, method IDs, placement policy, storage limits, or daemon
configuration. They return the same `Future` handle as other asynchronous client
methods. Submission and waiting release the Python GIL.

## Diagnostic methods

| Method | Default routing | Result |
| --- | --- | --- |
| `AsyncGetCapacity(pool_query=PoolQuery.Local())` | Local node | `.result(max_sec)` returns a read-only `CapacityInfo` snapshot |
| `AsyncListTargets(pool_query=PoolQuery.Local())` | Local node | `.result(max_sec)` returns registered target names as `list[str]` |
| `AsyncGetTargetInfo(target_name, pool_query=PoolQuery.Local())` | Local node | `.result(max_sec)` returns a read-only `TargetInfo` snapshot |
| `AsyncStatTargets(pool_query=PoolQuery.Local())` | Local node | One-shot refresh; `.wait(max_sec)` returns the task return code |

`CapacityInfo` has `total_capacity` and `remaining_capacity`, both integers in
bytes. Total capacity is the sum of configured capacities of the selected
node's registered targets. Remaining capacity is the native remaining value,
including reclaimable cache copies. It is not the host filesystem's free space.
`PoolQuery.Broadcast()` sums capacity across reachable selected containers using
the existing native aggregation behavior.

After daemon restart, registration can initially report the configured capacity
as free before restored allocations reach the canonical target statistics. Before
using capacity for a storage decision, request and await a one-shot refresh:

```python
refresh = client.AsyncStatTargets().wait(max_sec=5.0)
if refresh is None:
    raise TimeoutError("Target-statistics refresh did not complete")
if refresh != 0:
    raise RuntimeError(f"Target-statistics refresh failed: {refresh}")
capacity = client.AsyncGetCapacity().result(max_sec=5.0)
```

Even a successful refresh can retain last-known values for a target that failed
to answer. It does not establish target health or placement eligibility.

`TargetInfo` contains only the fields supplied by the native task:

- `target_score`: floating-point target performance score.
- `remaining_space`: remaining target space in bytes.
- `bytes_read`, `bytes_written`: byte counters.
- `ops_read`, `ops_written`: operation counters.

There is no per-target total-capacity, last-update timestamp, TTL, health, or
placement-eligibility field in this task. Separate query results are separate
snapshots, not one atomic view. Listing a name does not establish target health.
Broadcast listing appends native replica results without deduplication. A
successful `AsyncStatTargets` task can retain last-known statistics for a target
that did not answer, so its success does not prove that every target refreshed.
The binding intentionally exposes only a one-shot refresh, not an indefinitely
scheduled periodic task.

```python
import clio_cte_core_ext as cte

# Use an already initialized CTE client; these calls do not start a daemon.
client = cte.get_cte_client()
capacity = client.AsyncGetCapacity().result(max_sec=5.0)
print(capacity.total_capacity, capacity.remaining_capacity)

for name in client.AsyncListTargets().result(max_sec=5.0):
    info = client.AsyncGetTargetInfo(name).result(max_sec=5.0)
    print(name, info.remaining_space)
```

For the three data-returning diagnostic methods, `.result()` raises
`RuntimeError` on a nonzero native task return code or a missing task. It never
turns a failed query into a zero-capacity or empty-list success. `.wait()` still
returns the native code, and a timeout returns `None`; `.result()` raises
`TimeoutError` if its deadline expires. Callers should bound waits and preserve
diagnostic failures.

Write return code 12 means that placement found no eligible target. Insufficient
remaining capacity can cause it, but these snapshots cannot rule out stale or
unhealthy targets and placement constraints. Free capacity, a successful refresh,
or a registered name does not establish eligibility. Do not treat code 12 alone
as permission to grow storage or retry writes indefinitely.

## Explicit target registration

```python
future = client.AsyncRegisterTarget(
    target_name,
    cte.BdevType.kFile,
    capacity_bytes,
    target_query=cte.PoolQuery.Local(),
    bdev_id=stable_pool_id,
    pool_query=cte.PoolQuery.Local(),
    attach_existing=0,
    growth_unit=1 << 30,
)
return_code = future.wait(max_sec=10.0)
if return_code is None:
    raise TimeoutError("Target registration did not complete")
if return_code != 0:
    raise RuntimeError(f"Target registration failed: {return_code}")
```

The complete signature is:

```python
AsyncRegisterTarget(
    target_name,
    bdev_type,
    total_size,
    target_query=PoolQuery.Local(),
    bdev_id=PoolId.GetNull(),
    pool_query=PoolQuery.Dynamic(),
    attach_existing=0,
    growth_unit=1 << 30,
)
```

The original three-argument call remains accepted with its original native
defaults. A null `bdev_id` does not allocate a new pool ID and can fail native
registration. Applications adding managed targets must supply a stable,
collision-free ID and explicit routing. `attach_existing=0` creates a bdev;
`attach_existing=1` attaches a pool that already implements the bdev interface.
`growth_unit` is passed through to the existing native method, in bytes.

Registration does not resize an existing target and does not persist its
configuration. An application that owns the storage configuration must persist
the target's exact path, capacity, routing and ID before registering it, then
reuse those values on restart. For CTE's ordinary configured storage entries,
the native startup convention is pool ID `(512 + storage_index, 1 + node_index)`
and target name `configured_path + "_node" + node_index`. Applications must not
reuse an occupied storage index or apply that convention to an unrelated pool.

Both synchronous `RegisterTarget` overloads now return the integer from
`GetReturnCode()`. They no longer attempt to convert the native atomic return
field into a Python object after registration has already completed.
