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
 * s3_tool_main.cc - Standalone S3 helper executable (`cae_s3_tool`).
 *
 * This program is the AWS-SDK side of the CAE S3 importer. It is deliberately a
 * SEPARATE process from the CLIO runtime: loading libaws-cpp-sdk-core.so into the
 * CLIO runtime process corrupts runtime startup (the SDK's load-time global
 * constructors / static-baked s2n collide with CLIO's init), so all AWS SDK calls
 * are isolated here and invoked via fork+exec from S3FileAssimilator (mirroring the
 * way the Globus assimilator shells out to curl).
 *
 * Subcommands:
 *   cae_s3_tool get <bucket> <key> <out_path> [range_off] [range_size]
 *       Download an object (optionally a byte range) to <out_path>.
 *   cae_s3_tool put <bucket> <key> <in_path>
 *       Ensure <bucket> exists (best-effort) and upload <in_path> as <key>.
 *   cae_s3_tool del <bucket> <key>
 *       Delete an object (best-effort; missing object is not an error).
 *
 * Configuration is read from the standard AWS environment:
 *   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN (cred chain),
 *   AWS_DEFAULT_REGION (default us-east-1),
 *   S3_ENDPOINT or AWS_ENDPOINT_URL (S3-compatible endpoint -> path-style addressing).
 *
 * Exit status: 0 on success, non-zero on failure (with a message on stderr).
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>

namespace {

/**
 * Build an S3 client from the standard AWS environment.
 *
 * Region comes from AWS_DEFAULT_REGION (default "us-east-1"). When S3_ENDPOINT or
 * AWS_ENDPOINT_URL is set (MinIO and other S3-compatible stores), the endpoint is
 * overridden and path-style addressing is used. Credentials resolve through the
 * SDK's default provider chain.
 *
 * @return A configured Aws::S3::S3Client.
 */
Aws::S3::S3Client MakeS3Client() {
  Aws::Client::ClientConfiguration cfg;
  const char* region_env = std::getenv("AWS_DEFAULT_REGION");
  cfg.region = (region_env && *region_env) ? region_env : "us-east-1";

  const char* endpoint_env = std::getenv("S3_ENDPOINT");
  if (!endpoint_env || !*endpoint_env) {
    endpoint_env = std::getenv("AWS_ENDPOINT_URL");
  }
  bool use_path_style = false;
  if (endpoint_env && *endpoint_env) {
    cfg.endpointOverride = endpoint_env;
    use_path_style = true;
  }
  return Aws::S3::S3Client(
      cfg, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
      /*useVirtualAddressing=*/!use_path_style);
}

/**
 * Download an object (optionally a byte range) to a local file.
 *
 * @param s3 S3 client.
 * @param bucket Bucket name.
 * @param key Object key.
 * @param out_path Destination file path.
 * @param has_range Whether a byte range was requested.
 * @param range_off Range start offset (used when has_range is true).
 * @param range_size Range length in bytes (used when has_range is true).
 * @return 0 on success, non-zero on failure.
 */
int DoGet(Aws::S3::S3Client& s3, const std::string& bucket,
          const std::string& key, const std::string& out_path, bool has_range,
          uint64_t range_off, uint64_t range_size) {
  Aws::S3::Model::GetObjectRequest request;
  request.SetBucket(Aws::String(bucket.c_str()));
  request.SetKey(Aws::String(key.c_str()));
  if (has_range) {
    std::string range = "bytes=" + std::to_string(range_off) + "-" +
                        std::to_string(range_off + range_size - 1);
    request.SetRange(Aws::String(range.c_str()));
  }

  auto outcome = s3.GetObject(request);
  if (!outcome.IsSuccess()) {
    std::cerr << "cae_s3_tool get: GetObject failed for s3://" << bucket << "/"
              << key << ": " << outcome.GetError().GetMessage() << "\n";
    return 1;
  }

  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "cae_s3_tool get: cannot open output file '" << out_path
              << "'\n";
    return 1;
  }
  // Stream the body straight to disk (the SDK owns the body stream).
  out << outcome.GetResult().GetBody().rdbuf();
  out.flush();
  if (!out) {
    std::cerr << "cae_s3_tool get: write failed for '" << out_path << "'\n";
    return 1;
  }
  return 0;
}

/**
 * Ensure a bucket exists (best-effort) and upload a local file as an object.
 *
 * @param s3 S3 client.
 * @param bucket Bucket name.
 * @param key Object key.
 * @param in_path Source file path.
 * @return 0 on success, non-zero on failure.
 */
int DoPut(Aws::S3::S3Client& s3, const std::string& bucket,
          const std::string& key, const std::string& in_path) {
  Aws::S3::Model::CreateBucketRequest create_bucket;
  create_bucket.SetBucket(Aws::String(bucket.c_str()));
  s3.CreateBucket(create_bucket);  // best-effort: already-exists is fine

  auto body = Aws::MakeShared<Aws::FStream>(
      "cae_s3_tool", in_path.c_str(), std::ios::in | std::ios::binary);
  if (!body || !body->good()) {
    std::cerr << "cae_s3_tool put: cannot open input file '" << in_path
              << "'\n";
    return 1;
  }

  Aws::S3::Model::PutObjectRequest put;
  put.SetBucket(Aws::String(bucket.c_str()));
  put.SetKey(Aws::String(key.c_str()));
  put.SetBody(body);
  auto outcome = s3.PutObject(put);
  if (!outcome.IsSuccess()) {
    std::cerr << "cae_s3_tool put: PutObject failed for s3://" << bucket << "/"
              << key << ": " << outcome.GetError().GetMessage() << "\n";
    return 1;
  }
  return 0;
}

/**
 * Delete an object (best-effort).
 *
 * @param s3 S3 client.
 * @param bucket Bucket name.
 * @param key Object key.
 * @return 0 always (a missing object is not treated as an error).
 */
int DoDel(Aws::S3::S3Client& s3, const std::string& bucket,
          const std::string& key) {
  Aws::S3::Model::DeleteObjectRequest del;
  del.SetBucket(Aws::String(bucket.c_str()));
  del.SetKey(Aws::String(key.c_str()));
  s3.DeleteObject(del);
  return 0;
}

void PrintUsage() {
  std::cerr
      << "usage:\n"
      << "  cae_s3_tool get <bucket> <key> <out_path> [range_off range_size]\n"
      << "  cae_s3_tool put <bucket> <key> <in_path>\n"
      << "  cae_s3_tool del <bucket> <key>\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    PrintUsage();
    return 2;
  }
  const std::string cmd = argv[1];

  Aws::SDKOptions options;
  Aws::InitAPI(options);
  int rc = 2;
  {
    Aws::S3::S3Client s3 = MakeS3Client();
    if (cmd == "get" && (argc == 5 || argc == 7)) {
      bool has_range = (argc == 7);
      uint64_t off = has_range ? std::strtoull(argv[5], nullptr, 10) : 0;
      uint64_t size = has_range ? std::strtoull(argv[6], nullptr, 10) : 0;
      rc = DoGet(s3, argv[2], argv[3], argv[4], has_range, off, size);
    } else if (cmd == "put" && argc == 5) {
      rc = DoPut(s3, argv[2], argv[3], argv[4]);
    } else if (cmd == "del" && argc == 4) {
      rc = DoDel(s3, argv[2], argv[3]);
    } else {
      PrintUsage();
      rc = 2;
    }
  }
  Aws::ShutdownAPI(options);
  return rc;
}
