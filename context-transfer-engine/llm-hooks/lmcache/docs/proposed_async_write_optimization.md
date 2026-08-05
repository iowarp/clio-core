# Proposed LMCache Asynchronous Write Optimization

Status: discussion draft; not implemented.

This note records the proposed optimization for the CLIO CTE LMCache write
path. It should not be read as a description of implemented behavior.

## Current Behavior

The existing `LMCacheStore::PutManyRecords` implementation already follows the
basic recommendation to submit multiple `AsyncPutBlob` operations before
waiting:

1. Allocate a CLIO buffer for a record.
2. Copy the header, metadata, and payload into that buffer.
3. Call `AsyncPutBlob`.
4. Retain the record index, buffer, and returned future in a window.
5. Repeat until `max_inflight` operations have been submitted.
6. Wait for every future in the window and free each buffer after completion.
7. Submit the next window.

The default plugin value of `max_inflight` is 16. Waiting for the futures
sequentially after all 16 have been submitted does not serialize the CTE
operations. Operations 2 through 16 remain in flight while the caller waits
for operation 1.

The current behavior is therefore:

```text
submit up to 16 -> drain all 16 -> submit the next window
```

The integration sequence diagram should show separate submission and drain
phases. Placing completion and buffer release inside one undifferentiated
per-record loop incorrectly suggests that every `AsyncPutBlob` is immediately
followed by a wait.

The single-record `PutBytes` method does submit and immediately wait, but the
LMCache plugin's batch write path uses `PutManyRecords`.

## Remaining Performance Limitation

The current implementation uses a barrier between windows. It drains the
entire current window before submitting any operation from the next window.
This can produce a sawtooth submission pattern and leave capacity unused when
operation latency varies.

## Recommended Rolling Window

Replace the barriered windows with a rolling bounded queue:

```text
fill max_inflight slots
wait for and remove the oldest operation
free its completed payload buffer
immediately submit one replacement
repeat
drain the remaining operations at the end
```

The first implementation should use FIFO reclamation. Completion-order
reclamation could poll `Future::Wait(0)` or `Future::IsComplete`, but it adds
polling, fairness, and CPU-overhead concerns. It should only be considered if
benchmarking shows significant head-of-line blocking in the FIFO design.

### Proposed C++ Structure

1. Replace the batch-oriented `std::vector<PutInFlight>` with a bounded
   `std::deque<PutInFlight>`.
2. Add a helper that waits for and finalizes one queue entry:
   - wait for its future;
   - record its return status;
   - update profiling counters;
   - free its CLIO buffer only after completion; and
   - remove the entry from the queue.
3. After each submission, if the queue has reached `max_inflight`, finalize
   only its oldest entry.
4. Continue submitting replacements until all records have been submitted.
5. Drain the remaining queue at the end.
6. Move each future into the queue instead of copying it.
7. Apply the same structure to `PutMany` and `PutManyRecords` through shared
   helpers where practical.

Functions introduced by this work should remain small, have Doxygen comments,
and preserve the existing profiling fields and result ordering.

## Required Lifetime Rules

Every pending queue entry must retain both:

- the `Future<PutBlobTask>`; and
- the `FullPtr<char>` for the CLIO payload buffer.

The CTE runtime continues using the `ShmPtr` supplied to `AsyncPutBlob` after
submission. The payload buffer must not be freed when `AsyncPutBlob` returns.
It may only be freed after that operation has completed.

The Python payload views are needed while C++ assembles each record. Once the
record has been copied into its CLIO buffer, the CTE operation depends on the
CLIO buffer rather than the original Python payload.

## Timeout Correctness

`WaitForFuture` currently has a 30-second timeout. Timing out does not cancel
the underlying CTE operation, so freeing its payload buffer would be unsafe.
Stack unwinding also drops the local future state, which deserves a separate
lifetime review.

The rolling-window implementation should not weaken the current no-free rule.
A robust solution should do one of the following:

1. Retain timed-out futures and buffers in a quarantine/reaper owned by
   `LMCacheStore` until each operation completes.
2. Add a real CTE cancellation mechanism with a completion guarantee that
   makes buffer reclamation safe.

The retained-operation approach is the smaller initial design. `Close` would
need to stop new submissions and drain or safely transfer responsibility for
all retained operations before closing the CTE client.

## Python-Level Asynchrony

The rolling C++ window improves concurrency inside one batch, but
`batched_submit_put_task` still returns only after the complete batch commits.
The plugin's `async_batched_submit_put_task` uses `asyncio.to_thread`, which
keeps the event loop responsive but still awaits the blocking batch operation.

Making the synchronous LMCache API return before persistence completes is a
different and larger change. It would require:

- a bounded background executor or work queue;
- incrementing the reference count of every accepted `MemoryObj` before the
  caller can release it;
- releasing those references after completion;
- clear callback and partial-failure behavior;
- admission backpressure based on operations or bytes;
- safe serialization around `LMCacheStore` and CLIO singleton use; and
- a `Close` implementation that stops admission and drains outstanding work.

This background-queue design is not required to satisfy the recommendation of
keeping 16 CTE puts in flight. It should only be pursued if LMCache needs
fire-and-forget behavior across separate batch calls.

## Proposed Validation

Before and after the rolling-window change:

1. Install the changed CLIO build before running any tests or benchmarks.
2. Run the existing LMCache C++ and Python integration tests.
3. Verify successful, missing, and partial-failure result ordering.
4. Exercise final windows smaller than `max_inflight`.
5. Exercise `max_inflight` values of 1, 8, 16, and 32.
6. Verify that buffers remain valid until their corresponding futures finish.
7. Exercise timeout handling without freeing memory still used by CTE.
8. Compare barriered and rolling windows using `CLIO_LMCACHE_PROFILE`, including
   total, allocation, copy, submission, wait, and free times in milliseconds.
9. Measure throughput and tail latency with uniform and deliberately uneven
   storage-operation latency.

## Proposed Decision

The recommended initial scope for later agreement is:

1. Correct the integration sequence diagram.
2. Implement a FIFO rolling bounded window in `PutMany` and
   `PutManyRecords`.
3. Preserve future and payload-buffer ownership until completion.
4. Add a safe retained-operation policy for timeouts.
5. Benchmark before considering completion-order polling or a Python
   background queue.

