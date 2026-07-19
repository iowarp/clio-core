/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * test_cloud_factory_guard.cc - Always-on test of AssimilatorFactory protocol
 * dispatch for the cloud (s3://, gs://, gcs://) backends.
 *
 * This test does NOT touch any cloud service. It only checks the factory's
 * compile-time guard behaviour:
 *   - An unsupported protocol returns nullptr.
 *   - The always-available `file::` protocol returns a backend.
 *   - `s3://`   returns a backend iff CAE_ENABLE_S3 was compiled in, else nullptr.
 *   - `gs://` / `gcs://` return a backend iff CAE_ENABLE_GCS was compiled in.
 *
 * The factory is built with a null CTE client: the guard/unsupported paths
 * never dereference it, and the enabled paths only store it in the backend.
 * The test CMake passes CAE_ENABLE_S3 / CAE_ENABLE_GCS to this target so the
 * expectations track the build configuration.
 */

#include <memory>
#include <string>

#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/factory/assimilator_factory.h>

namespace {

/** Report one expectation; returns 0 on pass, 1 on failure. */
int Expect(const std::string& label, bool ok) {
  if (ok) {
    HLOG(kSuccess, "PASS: {}", label);
    return 0;
  }
  HLOG(kError, "FAIL: {}", label);
  return 1;
}

}  // namespace

int main(int /*argc*/, char* /*argv*/[]) {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Cloud Assimilator Factory Guard Test");
  HLOG(kInfo, "========================================");

  // The runtime is initialized so logging and worker context are available;
  // the factory itself is exercised with a null CTE client.
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    HLOG(kError, "Failed to initialize Clio");
    return 1;
  }

  clio::cae::core::AssimilatorFactory factory(
      std::shared_ptr<clio::cte::core::Client>(nullptr));

  int failures = 0;

  // Unsupported protocol -> nullptr.
  failures += Expect("unsupported protocol -> nullptr",
                     factory.Get("frobnicate://bucket/key") == nullptr);

  // A bare string with no "://" and no "::" has an empty protocol, which is
  // unsupported -> nullptr. Exercises GetUrlProtocol's final `return ""`.
  failures += Expect("no-delimiter src -> nullptr",
                     factory.Get("just-a-path") == nullptr);

  // Always-available local file backend -> non-null.
  failures += Expect("file:: -> backend",
                     factory.Get("file::/tmp/example.bin") != nullptr);

  // In-memory string backend is always compiled in -> non-null.
  failures += Expect("string:: -> backend",
                     factory.Get("string::my_blob") != nullptr);

  // HDF5 dispatch tracks CLIO_CAE_ENABLE_HDF5.
#ifdef CLIO_CAE_ENABLE_HDF5
  failures += Expect("hdf5:: -> backend (CLIO_CAE_ENABLE_HDF5 on)",
                     factory.Get("hdf5::/tmp/example.h5") != nullptr);
#else
  failures += Expect("hdf5:: -> nullptr (CLIO_CAE_ENABLE_HDF5 off)",
                     factory.Get("hdf5::/tmp/example.h5") == nullptr);
#endif

  // Globus dispatch tracks CAE_ENABLE_GLOBUS, for both the "globus::" custom
  // scheme and the "https://app.globus.org" web-URL fast path.
#ifdef CAE_ENABLE_GLOBUS
  failures += Expect("globus:: -> backend (CAE_ENABLE_GLOBUS on)",
                     factory.Get("globus::/ep/path") != nullptr);
  failures += Expect("globus web URL -> backend (CAE_ENABLE_GLOBUS on)",
                     factory.Get("https://app.globus.org/file-manager") !=
                         nullptr);
#else
  failures += Expect("globus:: -> nullptr (CAE_ENABLE_GLOBUS off)",
                     factory.Get("globus::/ep/path") == nullptr);
  failures += Expect("globus web URL -> nullptr (CAE_ENABLE_GLOBUS off)",
                     factory.Get("https://app.globus.org/file-manager") ==
                         nullptr);
#endif

  // S3 dispatch tracks CAE_ENABLE_S3.
#ifdef CAE_ENABLE_S3
  failures += Expect("s3:// -> backend (CAE_ENABLE_S3 on)",
                     factory.Get("s3://bucket/key") != nullptr);
#else
  failures += Expect("s3:// -> nullptr (CAE_ENABLE_S3 off)",
                     factory.Get("s3://bucket/key") == nullptr);
#endif

  // GCS dispatch tracks CAE_ENABLE_GCS (both gs:// and gcs:// forms).
#ifdef CAE_ENABLE_GCS
  failures += Expect("gs:// -> backend (CAE_ENABLE_GCS on)",
                     factory.Get("gs://bucket/object") != nullptr);
  failures += Expect("gcs:// -> backend (CAE_ENABLE_GCS on)",
                     factory.Get("gcs://bucket/object") != nullptr);
#else
  failures += Expect("gs:// -> nullptr (CAE_ENABLE_GCS off)",
                     factory.Get("gs://bucket/object") == nullptr);
  failures += Expect("gcs:// -> nullptr (CAE_ENABLE_GCS off)",
                     factory.Get("gcs://bucket/object") == nullptr);
#endif

  HLOG(kInfo, "========================================");
  HLOG(kInfo, failures == 0 ? "TEST PASSED" : "TEST FAILED");
  HLOG(kInfo, "========================================");
  return failures == 0 ? 0 : 1;
}
