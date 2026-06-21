# Globus CAE Integration Test Results

**Date**: 2026-03-23
**Branch**: fix/cte-bench-node-distribution
**Test**: `context-assimilation-engine/test/integration/globus_matsci/run_test.sh`

## Status: PASSED ✓

End-to-end Globus HTTPS file download through the Clio runtime works correctly.

---

## What Was Tested

- **Endpoint**: Globus Tutorial Collection 1 (`6c54cade-bde5-45c1-bdea-f4bd71dba2cc`)
- **HTTPS server**: `https://m-d3a2c3.collection1.tutorials.globus.org`
- **Source file**: `/home/share/godata/file1.txt`
- **Destination**: `/tmp/globus_afrl/file1.txt`
- **OMNI file**: `afrl_globus_omni.yaml`

## Download Steps

| Step | Description | Result |
|------|-------------|--------|
| 1/4 | Query endpoint metadata via Transfer API | ✓ |
| 2/4 | Parse HTTPS server from endpoint JSON | ✓ |
| 3/4 | Initiate HTTPS download via curl subprocess | ✓ |
| 4/4 | Write file to local filesystem | ✓ `"one"` (4 bytes) |

---

## Key Fix: NSS SIGSEGV in Clio Worker Thread

### Root Cause

Calling `Poco::Net::HTTPSClientSession` from inside a clio_run worker thread
triggered `getaddrinfo()` → glibc NSS hostname resolution → **SIGSEGV** at
`nss_action.h:64` (glibc 2.39, offset `0x160a8d` in libc.so.6).

- Crash address: `segfault at 2` — null `nss_action_list` accessed at field offset 2
- NSS lazy-init is broken in clio_run's dlopen'd module context (hshm allocator interference)
- Not a catchable C++ exception; kills the runtime process immediately
- Crash happens in both worker threads and background threads; standalone `std::thread` DNS works fine

### Fix Applied

**File**: `context-assimilation-engine/core/src/factory/globus_file_assimilator.cc`

Replaced all POCO HTTPS calls with `fork()`/`execvp()`/`curl` subprocess calls.
Each HTTP request forks a child process running `curl` in a clean address space,
completely bypassing the NSS/POCO issue.

**Removed includes**:
```
Poco/Net/Context.h, Poco/Net/HTTPRequest.h, Poco/Net/HTTPResponse.h,
Poco/Net/HTTPSClientSession.h, Poco/StreamCopier.h, Poco/URI.h
```

**Added includes**: `<unistd.h>`, `<sys/wait.h>`, `<fcntl.h>`

**Two static helpers added**:
- `RunCurlCapture(args)` — fork+exec curl, capture stdout as string
- `RunCurlExec(args)` — fork+exec curl, return exit code only

**Why fork/exec and not in-process libcurl**: libcurl also calls `getaddrinfo()`
internally and would hit the same NSS crash. The subprocess isolation is essential.

---

## Token Setup

Globus HTTPS downloads require two separate tokens — the HTTPS token is
**collection-specific** and cannot be reused across collections.

```bash
cd context-assimilation-engine/test/integration/globus_matsci
python3 get_oauth_token.py \
    --client-id 9f60b3dc-f895-4c68-8961-fd431399f523 \
    --with-data-access \
    <collection_id>
source /tmp/globus_tokens.sh
```

| Variable | Purpose |
|----------|---------|
| `GLOBUS_ACCESS_TOKEN` | Transfer API token — works for any endpoint metadata query |
| `GLOBUS_HTTPS_ACCESS_TOKEN` | Collection HTTPS token — specific to one collection |

---

## Running the Test

```bash
cd context-assimilation-engine/test/integration/globus_matsci

# Default (SEM_103 collection — requires access grant from collection owner)
bash run_test.sh

# Tutorial Collection 1 (publicly accessible)
export OMNI_FILE=/path/to/afrl_globus_omni.yaml
export OUTPUT_DIR=/tmp/globus_afrl
bash run_test.sh
```

Both `OMNI_FILE` and `OUTPUT_DIR` can be overridden via environment variables.

---

## Endpoint Notes

| Endpoint | ID | Status |
|----------|----|--------|
| SEM_103 Materials Science | `e8cf0e9a-f96a-11ed-9a83-83ef71fbf0ae` | 403 — identity not authorized |
| AFRL Challenge Data (old) | `4b116d3c-1aed-11e3-bb9e-22000b97608d` | 404 — endpoint no longer exists |
| ALCF Eagle (new AFRL home) | `05d2c76a-e867-4f67-aa57-76edeb0beda0` | Requires `data_access` token + ALCF account |
| Globus Tutorial Collection 1 | `6c54cade-bde5-45c1-bdea-f4bd71dba2cc` | ✓ Public, tested successfully |
