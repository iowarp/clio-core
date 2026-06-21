/**
 * External CTE Core Integration Test
 *
 * Validates the C++ example in the top-level README.md against the
 * installed `clio-core` package surface.  The test deliberately mirrors
 * the README snippet line-for-line (singleton CLIO_CTE_CLIENT + async
 * AsyncGetOrCreateTag / AsyncPutBlob / AsyncGetBlob / AsyncDelTag) so
 * that any README drift surfaces as a build/test failure here.
 *
 * If you edit this file, please mirror the change to README.md's
 * "Context Transfer Engine C++ Example" section, and vice versa.
 */

#include <clio_cte/core/core_client.h>
#include <clio_runtime/ipc_manager.h>

#include <cstring>
#include <iostream>
#include <vector>

#include <clio_ctp/util/logging.h>

namespace {

// Run the exact code path documented in README.md.  Returns 0 on
// success (matches the README's `return 0;` semantics so the test
// binary's exit code is the README example's exit code).
int RunReadmeExample() {
  // ---- begin README example ----------------------------------------
  // (Keep the body of this block identical to the README's
  //  "Context Transfer Engine C++ Example" code block.)

  // 1. Initialize the CTE client.  This auto-connects to the runtime
  //    started by `clio_run start` and creates the CTE pool on the
  //    first call (no separate CLIO_INIT / runtime-mode setup needed
  //    in the consumer process).  Storage targets are configured
  //    declaratively via the runtime's compose YAML — no
  //    RegisterTarget call needed here either.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return 1;
  auto *cte = CLIO_CTE_CLIENT;

  // 2. Get-or-create a named container for blobs.  The async APIs
  //    return a clio::run::Future immediately; Wait() blocks for completion.
  auto tag_future = cte->AsyncGetOrCreateTag("my_tag");
  tag_future.Wait();
  auto tag_id = tag_future->tag_id_;

  // 3. Stage blob data into a CTE-managed shm buffer and submit the
  //    PutBlob asynchronously.  Submit-then-Wait is the canonical
  //    pattern; multiple AsyncPutBlob calls can be in flight before
  //    the first Wait() to pipeline I/O.  The async signatures take a
  //    type-erased `ShmPtr<>` so we wrap `put_buf.shm_` (a typed
  //    `ShmPtr<char>`) in the void-typed view.
  constexpr size_t kSize = 4096;
  auto put_buf = CLIO_IPC->AllocateBuffer(kSize);
  std::memset(put_buf.ptr_, 'A', kSize);
  ctp::ipc::ShmPtr<> put_data(put_buf.shm_);
  auto put_future = cte->AsyncPutBlob(tag_id, "my_blob",
                                       /*offset=*/0, kSize,
                                       put_data);
  put_future.Wait();
  CLIO_IPC->FreeBuffer(put_buf);

  // 4. Pre-allocate the receive buffer in shm, fire an async GetBlob,
  //    then Wait — the buffer holds the blob data on return.
  auto get_buf = CLIO_IPC->AllocateBuffer(kSize);
  ctp::ipc::ShmPtr<> get_data(get_buf.shm_);
  auto get_future = cte->AsyncGetBlob(tag_id, "my_blob",
                                       /*offset=*/0, kSize,
                                       /*flags=*/0,
                                       get_data);
  get_future.Wait();
  // get_buf.ptr_ now holds the retrieved bytes.

  // ---- end of README example body ---------------------------------
  // The README returns here; below the line is test-only verification
  // that the round-trip actually transferred the expected bytes.

  // Verify the GetBlob returned the data we wrote.
  for (size_t i = 0; i < kSize; ++i) {
    if (get_buf.ptr_[i] != 'A') {
      HLOG(kError,
           "GetBlob roundtrip mismatch at byte {}: expected 'A', got {}",
           i, static_cast<int>(get_buf.ptr_[i]));
      CLIO_IPC->FreeBuffer(get_buf);
      cte->AsyncDelTag(tag_id).Wait();
      return 2;
    }
  }
  CLIO_IPC->FreeBuffer(get_buf);

  // 5. Clean up.
  cte->AsyncDelTag(tag_id).Wait();
  return 0;
}

}  // namespace

int main() {
  HLOG(kInfo, "=== External CTE Integration Test (README example) ===");
  HLOG(kInfo,
       "Validating that the top-level README.md C++ example builds and "
       "runs against the installed clio-core surface.");

  const int rc = RunReadmeExample();
  if (rc == 0) {
    HLOG(kInfo, "PASS: README CTE example completed successfully.");
  } else {
    HLOG(kError,
         "FAIL: README CTE example returned non-zero exit code: {}", rc);
  }
  return rc;
}
