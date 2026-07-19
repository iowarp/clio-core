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
 * test_s3_assim.cc - End-to-end test for S3 (s3://) assimilation into CTE.
 *
 * The test:
 *   1. Self-skips (exit 0) unless S3_ENDPOINT is set (so default CI, which has
 *      no object store, passes without an endpoint).
 *   2. Seeds a known, patterned object into the S3-compatible store (MinIO)
 *      using the out-of-process cae_s3_tool helper (the AWS SDK must NOT be
 *      linked into this process, which initializes the CLIO runtime).
 *   3. Runs ParseOmni with src="s3://<bucket>/<key>", format="binary".
 *   4. Verifies the object's bytes landed in CTE (tag size == object size).
 *   5. Tears down the seeded object (via cae_s3_tool del).
 *
 * Environment:
 *   S3_ENDPOINT        S3-compatible endpoint (e.g. http://127.0.0.1:9000).
 *                      REQUIRED — the test self-skips when unset.
 *   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   Credentials (read by the helper).
 *   AWS_DEFAULT_REGION Region (default us-east-1 for MinIO).
 *   S3_TEST_BUCKET     Bucket to use (default clio-cae-test).
 *   CAE_S3_TOOL        Path to the cae_s3_tool helper (set by CTest to the
 *                      build-tree binary; defaults to "cae_s3_tool" on PATH).
 */

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>

namespace {

constexpr size_t kObjectSize = 3 * 1024 * 1024;  // 3 MB -> exercises chunking
const std::string kTagName = "test_s3_assim_tag";

/** Build a deterministic, verifiable byte pattern (4-byte block index LE). */
std::string MakePatternedData(size_t size_bytes) {
  std::string data(size_bytes, '\0');
  for (size_t i = 0; i + 4 <= size_bytes; i += 4) {
    uint32_t value = static_cast<uint32_t>(i / 4);
    std::memcpy(&data[i], &value, 4);
  }
  return data;
}

/** Resolve the cae_s3_tool helper (CAE_S3_TOOL override, else PATH lookup). */
std::string ResolveS3Tool() {
  const char* override_path = std::getenv("CAE_S3_TOOL");
  if (override_path && *override_path) {
    return override_path;
  }
  return "cae_s3_tool";
}

/** Fork+exec a process and return its exit status (0 = success, -1 = failure). */
int RunProcess(const std::vector<std::string>& args) {
  pid_t pid = fork();
  if (pid == -1) {
    return -1;
  }
  if (pid == 0) {
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) {
      argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);
    execvp(argv[0], const_cast<char* const*>(argv.data()));
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

}  // namespace

int main(int /*argc*/, char* /*argv*/[]) {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "S3 (s3://) Assimilation Test");
  HLOG(kInfo, "========================================");

  // Self-skip when no S3 endpoint is configured.
  const char* endpoint = std::getenv("S3_ENDPOINT");
  if (!endpoint || !*endpoint) {
    HLOG(kInfo, "S3_ENDPOINT not set -> skipping S3 assimilation test");
    return 0;
  }

  std::string bucket = "clio-cae-test";
  if (const char* b = std::getenv("S3_TEST_BUCKET"); b && *b) {
    bucket = b;
  }
  const std::string key =
      "cae_s3_test/obj_" + std::to_string(static_cast<long>(::getpid()));
  const std::string data = MakePatternedData(kObjectSize);
  const std::string s3_tool = ResolveS3Tool();

  int exit_code = 0;

  // Bring up CLIO + CTE + CAE. The AWS SDK is never linked into this process;
  // all S3 I/O happens out-of-process via the cae_s3_tool helper, so the runtime
  // starts cleanly (linking the AWS SDK here corrupts runtime startup).
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    HLOG(kError, "Failed to initialize Clio");
    return 1;
  }
  clio::cte::core::CLIO_CTE_CLIENT_INIT();
  CLIO_CAE_CLIENT_INIT();
  clio::cae::core::Client cae_client;
  {
    clio::cae::core::CreateParams params;
    auto create_task = cae_client.AsyncCreate(clio::run::PoolQuery::Local(),
                                              "test_cae_pool",
                                              clio::cae::core::kCaePoolId, params);
    create_task.Wait();
  }

  // Seed the object via the helper: write the pattern to a temp file, then
  // `cae_s3_tool put` it (the helper ensures the bucket exists).
  std::string seed_path = "/tmp/cae_s3_seed_" +
                          std::to_string(static_cast<long>(::getpid())) + ".bin";
  {
    std::ofstream seed(seed_path, std::ios::binary | std::ios::trunc);
    seed.write(data.data(), static_cast<std::streamsize>(data.size()));
    seed.close();
    if (!seed) {
      HLOG(kError, "Failed to write seed file '{}'", seed_path);
      return 1;
    }
    int put_rc = RunProcess({s3_tool, "put", bucket, key, seed_path});
    ::unlink(seed_path.c_str());
    if (put_rc != 0) {
      HLOG(kError, "Failed to seed s3://{}/{} via cae_s3_tool (rc={})", bucket,
           key, put_rc);
      return 1;
    }
    HLOG(kSuccess, "Seeded s3://{}/{} ({} bytes)", bucket, key, data.size());
  }

  try {
    // Assimilate the S3 object.
    clio::cae::core::AssimilationCtx ctx;
    ctx.src = "s3://" + bucket + "/" + key;
    ctx.dst = "iowarp::" + kTagName;
    ctx.format = "binary";
    std::vector<clio::cae::core::AssimilationCtx> contexts{ctx};

    HLOG(kInfo, "Calling ParseOmni for {}", ctx.src);
    auto parse_task = cae_client.AsyncParseOmni(contexts);
    parse_task.Wait();
    clio::run::u32 result_code = parse_task->GetReturnCode();
    clio::run::u32 num_scheduled = parse_task->num_tasks_scheduled_;
    HLOG(kInfo, "ParseOmni result_code={} num_tasks_scheduled={}", result_code,
         num_scheduled);
    if (result_code != 0 || num_scheduled == 0) {
      HLOG(kError, "ParseOmni failed for S3 source");
      exit_code = 1;
    }

    // Verify the bytes landed in CTE.
    auto cte_client = CLIO_CTE_CLIENT;
    auto tag_task = cte_client->AsyncGetOrCreateTag(kTagName);
    tag_task.Wait();
    clio::cte::core::TagId tag_id = tag_task->tag_id_;
    if (tag_id.IsNull()) {
      HLOG(kError, "Tag not found in CTE: {}", kTagName);
      exit_code = 1;
    } else {
      auto size_task = cte_client->AsyncGetTagSize(tag_id);
      size_task.Wait();
      size_t tag_size = size_task->tag_size_;
      // The assimilator stores a "description" metadata blob alongside the data
      // chunks (mirrors BinaryFileAssimilator), so the tag holds the object
      // bytes PLUS that blob. Reconstruct the exact description string the
      // assimilator writes (offset 0 for a whole-object, non-range import) to
      // get its byte count, and expect the tag to equal data + metadata.
      std::string description =
          "binary<size=" + std::to_string(kObjectSize) + ", offset=0>";
      size_t expected_size = kObjectSize + description.size();
      HLOG(kInfo, "CTE tag size={} (expected {} = {} object + {} description)",
           tag_size, expected_size, kObjectSize, description.size());
      if (tag_size != expected_size) {
        HLOG(kError, "Tag size mismatch: got {}, expected {}", tag_size,
             expected_size);
        exit_code = 1;
      } else {
        HLOG(kSuccess, "S3 object bytes verified in CTE");
      }
    }
  } catch (const std::exception& e) {
    HLOG(kError, "Exception: {}", e.what());
    exit_code = 1;
  }

  // Teardown the seeded object (best effort).
  RunProcess({s3_tool, "del", bucket, key});

  HLOG(kInfo, "========================================");
  HLOG(kInfo, exit_code == 0 ? "TEST PASSED" : "TEST FAILED");
  HLOG(kInfo, "========================================");
  ctp::SystemInfo::TerminateProcessNow(exit_code);
  return exit_code;
}
