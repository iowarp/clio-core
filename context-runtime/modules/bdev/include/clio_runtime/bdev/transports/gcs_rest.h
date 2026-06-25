/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_BDEV_GCS_REST_H_
#define CLIO_BDEV_GCS_REST_H_

// Header-only Google Cloud Storage REST client built on Poco::Net (HTTPS) +
// Poco JSON-free string bodies. The data-plane semantics (media upload, alt=media
// download, 404 -> sparse/zero-fill, idempotent bucket create) are ported from the
// hardened libcurl implementation in tag backup/main-pre-rollback-20260624, but the
// transport is Poco instead of libcurl. Auth is a bearer token (GCS_ACCESS_TOKEN,
// typically `gcloud auth print-access-token`); the service-account JWT flow is out
// of scope for now.
//
// This file is only meant to be compiled where CLIO_ENABLE_GOOGLE_CLOUD is defined
// and the Poco::Net / Poco::NetSSL libraries are linked.

#ifdef CLIO_ENABLE_GOOGLE_CLOUD

#include <Poco/URI.h>
#include <Poco/Exception.h>
#include <Poco/StreamCopier.h>
#include <Poco/NullStream.h>
#include <Poco/Timespan.h>
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPMessage.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/RejectCertificateHandler.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <ostream>
#include <istream>
#include <string>

namespace clio::run::bdev::gcs {

/** Connection + addressing config for one GCS bucket/prefix. */
struct GcsConfig {
  std::string endpoint;    ///< e.g. https://storage.googleapis.com
  std::string bucket;      ///< target bucket name
  std::string prefix;      ///< optional key prefix (may be empty)
  std::string project_id;  ///< GCP project (only used when creating the bucket)
  std::string token;       ///< OAuth2 bearer token (GCS_ACCESS_TOKEN)
};

/** Outcome of a single data-plane operation. */
struct GcsResult {
  long http_status = 0;     ///< HTTP status (0 => transport/exception before a response)
  bool not_found = false;   ///< GET returned 404 => caller should zero-fill (sparse read)
  std::string error;        ///< human-readable error, empty on success

  bool ok() const { return error.empty() && http_status >= 200 && http_status < 300; }
};

/** Percent-encode per RFC 3986 (unreserved set kept; '/' becomes %2F). */
inline std::string UrlEncode(const std::string &value) {
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

/** Initialize Poco client-side TLS once per process (system CAs, peer verified). */
inline void EnsureSslInitialized() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    Poco::Net::initializeSSL();
    Poco::SharedPtr<Poco::Net::InvalidCertificateHandler> cert_handler(
        new Poco::Net::RejectCertificateHandler(false));
    // privateKey="", cert="", caLocation="" + loadDefaultCAs=true => use the
    // system CA bundle; VERIFY_RELAXED + RejectCertificateHandler => verify peer.
    Poco::Net::Context::Ptr context(new Poco::Net::Context(
        Poco::Net::Context::CLIENT_USE, "", "", "",
        Poco::Net::Context::VERIFY_RELAXED, 9, true,
        "ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH"));
    Poco::Net::SSLManager::instance().initializeClient(0, cert_handler, context);
  });
}

/** Create an HTTP(S) session for `uri`; HTTPS gets a TLS context, http stays plain. */
inline std::unique_ptr<Poco::Net::HTTPClientSession> MakeSession(const Poco::URI &uri) {
  std::unique_ptr<Poco::Net::HTTPClientSession> session;
  if (uri.getScheme() == "https") {
    EnsureSslInitialized();
    session = std::make_unique<Poco::Net::HTTPSClientSession>(uri.getHost(),
                                                              uri.getPort());
  } else {
    session = std::make_unique<Poco::Net::HTTPClientSession>(uri.getHost(),
                                                             uri.getPort());
  }
  session->setTimeout(Poco::Timespan(30, 0));  // 30s
  return session;
}

/**
 * Minimal GCS JSON-API client. One instance is cheap; it holds no live socket
 * (a fresh session is opened per call, which keeps it usable from multiple
 * workers without sharing a non-thread-safe Poco session).
 */
class GcsRestClient {
 public:
  explicit GcsRestClient(GcsConfig config) : config_(std::move(config)) {}

  const GcsConfig &config() const { return config_; }

  /** Read connection config from the environment for a given bucket/prefix. */
  static GcsConfig ConfigFromEnv(const std::string &bucket,
                                 const std::string &prefix) {
    GcsConfig c;
    const char *ep = std::getenv("GCS_ENDPOINT");
    c.endpoint = (ep && *ep) ? ep : "https://storage.googleapis.com";
    const char *tok = std::getenv("GCS_ACCESS_TOKEN");
    c.token = (tok && *tok) ? tok : "";
    const char *proj = std::getenv("GCS_PROJECT_ID");
    c.project_id = (proj && *proj) ? proj : "clio-prototype";
    c.bucket = bucket;
    c.prefix = prefix;
    return c;
  }

  /** Object key for a block offset: [prefix/]block_<offset> (mirrors the S3 transport). */
  std::string KeyForOffset(uint64_t offset) const {
    std::string key = "block_" + std::to_string(offset);
    return config_.prefix.empty() ? key : (config_.prefix + "/" + key);
  }

  /** Idempotent bucket create. HTTP 200 (created) or 409 (exists) => success. */
  GcsResult EnsureBucket() {
    GcsResult r;
    try {
      std::string url =
          config_.endpoint + "/storage/v1/b?project=" + UrlEncode(config_.project_id);
      Poco::URI uri(url);
      auto session = MakeSession(uri);
      Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_POST,
                                 uri.getPathAndQuery(),
                                 Poco::Net::HTTPMessage::HTTP_1_1);
      ApplyAuth(req);
      std::string body = "{\"name\":\"" + config_.bucket + "\"}";
      req.setContentType("application/json");
      req.setContentLength(static_cast<std::streamsize>(body.size()));
      std::ostream &os = session->sendRequest(req);
      os << body;
      os.flush();
      Poco::Net::HTTPResponse res;
      std::istream &rs = session->receiveResponse(res);
      r.http_status = res.getStatus();
      std::string resp;
      Poco::StreamCopier::copyToString(rs, resp);
      if (r.http_status != 200 && r.http_status != 409) {
        r.error = "GCS bucket ensure failed: HTTP " +
                  std::to_string(r.http_status) + " " + resp;
      }
    } catch (const Poco::Exception &e) {
      r.error = std::string("GCS bucket ensure exception: ") + e.displayText();
    }
    return r;
  }

  /** Upload `len` bytes via a simple media upload. HTTP 2xx => success. */
  GcsResult PutObject(const std::string &key, const char *data, size_t len) {
    GcsResult r;
    try {
      std::string url = config_.endpoint + "/upload/storage/v1/b/" +
                        config_.bucket + "/o?uploadType=media&name=" +
                        UrlEncode(key);
      Poco::URI uri(url);
      auto session = MakeSession(uri);
      Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_POST,
                                 uri.getPathAndQuery(),
                                 Poco::Net::HTTPMessage::HTTP_1_1);
      ApplyAuth(req);
      req.setContentType("application/octet-stream");
      req.setContentLength(static_cast<std::streamsize>(len));
      std::ostream &os = session->sendRequest(req);
      if (len > 0) {
        os.write(data, static_cast<std::streamsize>(len));
      }
      os.flush();
      Poco::Net::HTTPResponse res;
      std::istream &rs = session->receiveResponse(res);
      r.http_status = res.getStatus();
      std::string resp;
      Poco::StreamCopier::copyToString(rs, resp);
      if (!r.ok()) {
        r.error = "GCS PUT failed: HTTP " + std::to_string(r.http_status) + " " +
                  resp;
      }
    } catch (const Poco::Exception &e) {
      r.error = std::string("GCS PUT exception: ") + e.displayText();
    }
    return r;
  }

  /**
   * Download object `key` into `buf` (up to `len` bytes). On HTTP 404 the result
   * has not_found=true and the caller is expected to zero-fill (sparse read). On
   * success `*bytes_read` (if non-null) is set to the number of bytes copied.
   */
  GcsResult GetObject(const std::string &key, char *buf, size_t len,
                      size_t *bytes_read) {
    GcsResult r;
    if (bytes_read) *bytes_read = 0;
    try {
      // Build the path directly (not through Poco::URI) to preserve %2F in the
      // key: Poco::URI decodes %2F during path normalization, making GCS 404 on
      // objects whose names contain '/'. Use Poco::URI only for host/port.
      Poco::URI base_uri(config_.endpoint);
      const std::string path = "/storage/v1/b/" + config_.bucket +
                               "/o/" + UrlEncode(key) + "?alt=media";
      auto session = MakeSession(base_uri);
      Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_GET,
                                 path,
                                 Poco::Net::HTTPMessage::HTTP_1_1);
      ApplyAuth(req);
      session->sendRequest(req);
      Poco::Net::HTTPResponse res;
      std::istream &rs = session->receiveResponse(res);
      r.http_status = res.getStatus();
      if (r.http_status == 404) {
        r.not_found = true;  // sparse: caller zero-fills
        Poco::NullOutputStream null;
        Poco::StreamCopier::copyStream(rs, null);
        return r;
      }
      if (!r.ok()) {
        std::string resp;
        Poco::StreamCopier::copyToString(rs, resp);
        r.error = "GCS GET failed: HTTP " + std::to_string(r.http_status) + " " +
                  resp;
        return r;
      }
      rs.read(buf, static_cast<std::streamsize>(len));
      if (bytes_read) *bytes_read = static_cast<size_t>(rs.gcount());
    } catch (const Poco::Exception &e) {
      r.error = std::string("GCS GET exception: ") + e.displayText();
    }
    return r;
  }

  /** Best-effort object delete (errors are not fatal to the caller). */
  GcsResult DeleteObject(const std::string &key) {
    GcsResult r;
    try {
      // Same %2F-preservation fix as GetObject.
      Poco::URI base_uri(config_.endpoint);
      const std::string path = "/storage/v1/b/" + config_.bucket +
                               "/o/" + UrlEncode(key);
      auto session = MakeSession(base_uri);
      Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_DELETE,
                                 path,
                                 Poco::Net::HTTPMessage::HTTP_1_1);
      ApplyAuth(req);
      session->sendRequest(req);
      Poco::Net::HTTPResponse res;
      std::istream &rs = session->receiveResponse(res);
      r.http_status = res.getStatus();
      Poco::NullOutputStream null;
      Poco::StreamCopier::copyStream(rs, null);
    } catch (const Poco::Exception &e) {
      r.error = std::string("GCS DELETE exception: ") + e.displayText();
    }
    return r;
  }

 private:
  void ApplyAuth(Poco::Net::HTTPRequest &req) const {
    if (!config_.token.empty()) {
      req.set("Authorization", "Bearer " + config_.token);
    }
  }

  GcsConfig config_;
};

}  // namespace clio::run::bdev::gcs

#endif  // CLIO_ENABLE_GOOGLE_CLOUD

#endif  // CLIO_BDEV_GCS_REST_H_
