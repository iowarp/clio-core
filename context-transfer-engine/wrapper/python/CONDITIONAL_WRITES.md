# Conditional primary blob writes from Python

`clio_cte_core_ext.Client` exposes the CTE owner's existing scalar conditional
write operations. These bindings add no Admin methods, ownership service, ledger,
task fields or wire protocol. `AsyncPutBlob` and `AsyncGetBlob` retain their
existing return types and behavior.

| Method | `Future.result(max_sec)` |
| --- | --- |
| `AsyncPutBlobIfAbsent(tag_id, blob_name, data, off=0)` | `{"code": int, "version": int}` |
| `AsyncPutBlobIfVersion(tag_id, blob_name, data, version, off=0)` | `{"code": int, "version": int}` |
| `AsyncGetBlobVersioned(tag_id, blob_name, data_size, off=0)` | `{"code": int, "version": int, "data": bytes or None}` |

All three return the existing `Future`; `wait(max_sec)` returns the native status
code, or `None` if its deadline expires. `result(max_sec)` raises `TimeoutError`
on expiration. Submission and waiting release the GIL. The binding retains
immutable put bytes and private read buffers until the typed task completes.

`code == 0` means success. The module exports the existing native conflict codes:

- `PUT_EXISTS_RC` (60): create-if-absent found an existing blob.
- `PUT_VERSION_MISMATCH_RC` (61): compare-and-swap found another current version.

A successful put returns its new version. A conflict's returned version is
diagnostic: native response aggregation takes the maximum of input/output
versions, so a supplied expected version larger than the current version can
survive in a routed conflict response. Refetch with `AsyncGetBlobVersioned` after
any conflict before retrying; do not treat the failed put's version as a new
ownership proof.

Other native status codes remain failures and must not be interpreted as absence
or lock contention. A failed get returns `data=None`; its version is not a usable
ownership proof. A normal missing blob returns code 1 and version 0. A refused
submission raises `RuntimeError`, rather than turning an empty native future into
a successful operation. Empty put payloads and zero-length gets raise
`ValueError`; unsigned native parameters reject negative values.

## What the version means

The runtime checks both conditional modes under its existing per-blob write token
at the primary owner. Competing create-if-absent writes have one winner. Competing
writes with the same expected version have one winner; successful scalar puts
advance that blob's version strictly (`max(clock_tick, previous_version + 1)`).
An absent blob has version 0. These operations use ordinary owner routing, not a
replica selector or a batched put (which does not honor these conditional flags).
Coordination requires a fixed authoritative pool with successor failover disabled;
these bindings do not turn a changing replica/failover topology into consensus.

Versioned reads submit the existing `GetBlobTask` through the owner, using the
existing `kNoFault` flag to inspect stored primary records without invoking a
tag's lazy materialization handler. They bypass the ordinary SHM read
optimization, whose synthesized response does not populate
the content version. Native reads pin a coherent byte snapshot, but capture the
version before the read pin: a concurrent write can return newer bytes carrying
an older version. Using that returned version in a conditional put then conflicts
safely. Retry the complete read/validate/CAS operation on a conflict; do not claim
that a read alone establishes ownership.

Puts overwrite a byte range and do not truncate an existing longer blob. Use a
fixed-size, padded record for coordination; reading an arbitrary oversized range
can include zero padding because native reads can be short. Keep the tag, blob
name, record size and padding protocol identical for every participant.

## A coordination record built with these operations

For example, a fixed-size record can contain an opaque owner nonce and a released
marker. Acquire a previously absent record with `AsyncPutBlobIfAbsent`. For an
existing released record, read and validate its format, then acquire with
`AsyncPutBlobIfVersion` using the returned version. Only code 0 grants ownership.
Release with a version-checked write of a released record, after verifying the
owner nonce. Retain the record; deleting and recreating it discards its version
history. Another process cannot release or replace a live owner merely because a
local timeout expired. A lost reply is an uncertain result: inspect the same
record and nonce before deciding whether to retry or recover.

These primitives alone do **not** supply a restart-stable fencing token,
consensus, lease expiry, automatic lock stealing or a fenced filesystem rename.
Versions use the runtime's clock domain; metadata recovery translates persisted
wall timestamps back into that domain. RAM-only storage can lose payloads after
a restart, and deleting/recreating a blob also discards its version history.
The coordination layer must have an explicit authority/session epoch and recovery
policy, fail closed when that authority or data is unavailable, and enforce
ownership at its publication boundary. Do not treat local elapsed time as proof
that an earlier owner is dead or assume that a CAS authorizes an unrelated file
write after ownership has changed.

## Example

This example assumes the existing runtime and CTE client are initialized. Every
record is exactly 256 bytes, and the caller retains the returned version and
nonce for its later release.

```python
import json
import secrets

import clio_cte_core_ext as cte

client = cte.get_cte_client()
tag_future = client.AsyncGetOrCreateTag("coordination-example")
assert tag_future.wait(10.0) == 0
tag_id = tag_future.result()
nonce = secrets.token_hex(16)
payload = json.dumps({"owner": nonce, "released": False}).encode().ljust(256, b" ")
created = client.AsyncPutBlobIfAbsent(tag_id, "resource-a", payload).result(10.0)
if created["code"] == cte.PUT_EXISTS_RC:
    raise RuntimeError("Resource already has a coordination record")
if created["code"] != 0:
    raise RuntimeError(f"Acquire failed: {created['code']}")

released_payload = json.dumps({"owner": nonce, "released": True}).encode().ljust(256, b" ")
released = client.AsyncPutBlobIfVersion(
    tag_id, "resource-a", released_payload, created["version"]
).result(10.0)
if released["code"] != 0:
    raise RuntimeError("Release did not establish the released state")
```

The real multi-process regression driver is `test_cte_conditional.py`, registered
as `python_conditional_put_test` in CTest. It runs against a separate native runtime,
with real independent Python clients and no emulated production implementation.
The Linux unit workflow requires exact test discovery and one passing execution
without retries, and treats a skipped or absent test as failure.
