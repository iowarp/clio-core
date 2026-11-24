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

// Self-contained example of the Google Cloud Storage "drive" C++ API, mirroring
// amazon_s3/test_s3_drive.cc. It writes an object and reads it back from a real
// GCS bucket over the JSON/media REST API using Poco for HTTPS. Auth is a bearer
// token obtained with `gcloud auth print-access-token` (no service-account key
// needed). The test self-skips when GCS_ACCESS_TOKEN or GCS_TEST_BUCKET is unset.
//
//   export GCS_ACCESS_TOKEN="$(gcloud auth print-access-token)"
//   export GCS_TEST_BUCKET=<your-bucket>
//   # optional: export GCS_ENDPOINT=https://storage.googleapis.com   (default)

#include <catch2/catch_all.hpp>

#include <Poco/URI.h>
#include <Poco/StreamCopier.h>
#include <Poco/Exception.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPMessage.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/RejectCertificateHandler.h>

#include <cstdlib>
#include <ostream>
#include <istream>
#include <string>

namespace {

// Percent-encode an object key per RFC 3986 ('/' -> %2F so GCS treats it opaque).
std::string UrlEncode(const std::string &value) {
  static const char *hex = "0123456789ABCDEF";
  std::string out;
  out.reserve(value.size() * 3);
  for (unsigned char c : value) {
    bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                      (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                      c == '.' || c == '~';
    if (unreserved) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(hex[(c >> 4) & 0xF]);
      out.push_back(hex[c & 0xF]);
    }
  }
  return out;
}

}  // namespace

TEST_CASE("Google Cloud Storage Read/Write", "[gcs][bdev]") {
  const char *token_env = std::getenv("GCS_ACCESS_TOKEN");
  const char *bucket_env = std::getenv("GCS_TEST_BUCKET");
  if (!token_env || !*token_env || !bucket_env || !*bucket_env) {
    WARN(
        "GCS_ACCESS_TOKEN and/or GCS_TEST_BUCKET not set - skipping GCS drive "
        "test. Export GCS_ACCESS_TOKEN=$(gcloud auth print-access-token) and "
        "GCS_TEST_BUCKET=<bucket> to run it.");
    return;
  }

  const std::string token = token_env;
  const std::string bucket = bucket_env;
  const char *ep_env = std::getenv("GCS_ENDPOINT");
  const std::string endpoint =
      (ep_env && *ep_env) ? ep_env : "https://storage.googleapis.com";

  const std::string object_name = "clio-validation/test_data.txt";
  const std::string data_to_write = "Hello, Google Cloud Storage from Clio Core!";

  // One-time client-side TLS setup: verify the peer against the system CA bundle.
  Poco::Net::initializeSSL();
  Poco::SharedPtr<Poco::Net::InvalidCertificateHandler> cert_handler(
      new Poco::Net::RejectCertificateHandler(false));
  Poco::Net::Context::Ptr context(new Poco::Net::Context(
      Poco::Net::Context::CLIENT_USE, "", "", "",
      Poco::Net::Context::VERIFY_RELAXED, 9, true,
      "ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH"));
  Poco::Net::SSLManager::instance().initializeClient(0, cert_handler, context);

  // POST a media upload: {endpoint}/upload/storage/v1/b/{bucket}/o?uploadType=media&name=
  auto gcs_put = [&](const std::string &key, const std::string &body) -> int {
    Poco::URI uri(endpoint + "/upload/storage/v1/b/" + bucket +
                  "/o?uploadType=media&name=" + UrlEncode(key));
    Poco::Net::HTTPSClientSession session(uri.getHost(), uri.getPort());
    Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_POST,
                               uri.getPathAndQuery(),
                               Poco::Net::HTTPMessage::HTTP_1_1);
    req.set("Authorization", "Bearer " + token);
    req.setContentType("application/octet-stream");
    req.setContentLength(static_cast<std::streamsize>(body.size()));
    std::ostream &os = session.sendRequest(req);
    os << body;
    os.flush();
    Poco::Net::HTTPResponse res;
    std::istream &rs = session.receiveResponse(res);
    std::string drain;
    Poco::StreamCopier::copyToString(rs, drain);
    return res.getStatus();
  };

  // GET the object bytes: {endpoint}/storage/v1/b/{bucket}/o/{key}?alt=media
  // Path is constructed directly (not through Poco::URI) to preserve %2F encoding
  // of slashes in the key; Poco::URI decodes %2F during path normalization, which
  // causes GCS to 404 even when the object exists.
  auto gcs_get = [&](const std::string &key, std::string &out) -> int {
    Poco::URI base_uri(endpoint);
    const std::string path =
        "/storage/v1/b/" + bucket + "/o/" + UrlEncode(key) + "?alt=media";
    Poco::Net::HTTPSClientSession session(base_uri.getHost(), base_uri.getPort());
    Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_GET, path,
                               Poco::Net::HTTPMessage::HTTP_1_1);
    req.set("Authorization", "Bearer " + token);
    session.sendRequest(req);
    Poco::Net::HTTPResponse res;
    std::istream &rs = session.receiveResponse(res);
    out.clear();
    Poco::StreamCopier::copyToString(rs, out);
    return res.getStatus();
  };

  // DELETE for cleanup (best effort). Same %2F-preservation fix as gcs_get.
  auto gcs_delete = [&](const std::string &key) -> int {
    Poco::URI base_uri(endpoint);
    const std::string path = "/storage/v1/b/" + bucket + "/o/" + UrlEncode(key);
    Poco::Net::HTTPSClientSession session(base_uri.getHost(), base_uri.getPort());
    Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_DELETE, path,
                               Poco::Net::HTTPMessage::HTTP_1_1);
    req.set("Authorization", "Bearer " + token);
    session.sendRequest(req);
    Poco::Net::HTTPResponse res;
    std::istream &rs = session.receiveResponse(res);
    std::string drain;
    Poco::StreamCopier::copyToString(rs, drain);
    return res.getStatus();
  };

  SECTION("Write then read data round-trips through GCS") {
    try {
      int put_status = gcs_put(object_name, data_to_write);
      INFO("GCS PUT gs://" << bucket << "/" << object_name
                           << " status=" << put_status);
      REQUIRE(put_status >= 200);
      REQUIRE(put_status < 300);

      std::string read_back;
      int get_status = gcs_get(object_name, read_back);
      INFO("GCS GET gs://" << bucket << "/" << object_name
                           << " status=" << get_status
                           << " bytes=" << read_back.size());
      REQUIRE(get_status >= 200);
      REQUIRE(get_status < 300);
      REQUIRE(read_back == data_to_write);

      gcs_delete(object_name);  // tidy up; ignore status
    } catch (const Poco::Exception &e) {
      FAIL("Poco exception talking to GCS: " << e.displayText());
    }
  }
}
