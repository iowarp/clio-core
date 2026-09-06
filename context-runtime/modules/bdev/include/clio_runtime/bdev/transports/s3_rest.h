/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_BDEV_S3_REST_H_
#define CLIO_BDEV_S3_REST_H_

// Header-only Amazon S3 REST client on Poco::Net (HTTPS) + SigV4 (sigv4.h).
//
// This replaces an implementation that called Aws::S3::S3Client directly.
// Loading aws-cpp-sdk-core into a process that calls CLIO_INIT stack-smashes
// runtime init, and the bdev has no way out of that: WriteBlocks and ReadBlocks
// are per-block coroutine bodies, so the fork+exec isolation the CAE S3
// assimilator uses (one child per whole object) is not viable at this
// granularity. Poco is already a dependency of the working GCS bdev transport,
// so the SDK simply goes away.
//
// Data-plane semantics (PUT / GET with 404 -> sparse zero-fill / best-effort
// DELETE) are deliberately identical to gcs_rest.h. The plumbing below
// duplicates ~60 lines of that file rather than being factored into a shared
// header: the GCS transport works today, and a narrower blast radius is worth
// more here than fewer total lines.
//
// Unlike gcs_rest.h, the S3 data-plane methods take a caller-owned
// S3Connection handle and keep the socket alive across calls (see S3Connection
// below). The client object itself still holds no socket, so one client remains
// usable from many workers at once -- each worker just hands in its own handle.
//
// Only compiled where CLIO_ENABLE_AMAZON_DRIVE is defined and Poco::Net /
// Poco::NetSSL are linked.

#ifdef CLIO_ENABLE_AMAZON_DRIVE

#include <Poco/Exception.h>
#include <Poco/NullStream.h>
#include <Poco/StreamCopier.h>
#include <Poco/Timespan.h>
#include <Poco/URI.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPMessage.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/RejectCertificateHandler.h>
#include <Poco/Net/SSLManager.h>

#include <cstdint>
#include <cstdlib>
#include <istream>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>

#include "clio_runtime/bdev/transports/sigv4.h"

namespace clio::run::bdev::s3 {

/** Connection + addressing config for one S3 bucket/prefix. */
struct S3Config {
  std::string endpoint;    ///< S3_ENDPOINT override; empty => real AWS
  std::string bucket;      ///< target bucket name
  std::string prefix;      ///< optional key prefix (may be empty)
  std::string region;      ///< AWS region, e.g. us-east-2
  std::string access_key;  ///< AWS_ACCESS_KEY_ID
  std::string secret_key;  ///< AWS_SECRET_ACCESS_KEY
  std::string session_token;         ///< AWS_SESSION_TOKEN (optional)
  bool allow_bucket_create = false;  ///< S3_ALLOW_BUCKET_CREATE=1

  /**
   * Path-style addressing whenever an endpoint override is present (MinIO and
   * most S3-compatible servers only speak path-style); real AWS uses
   * virtual-hosted style. The signer must follow whichever form is active --
   * canonical URI and Host header have to match what goes on the wire, or the
   * request comes back 403.
   */
  bool path_style() const { return !endpoint.empty(); }

  /** The subset the signer needs. */
  SigV4Credentials credentials() const {
    return SigV4Credentials{access_key, secret_key, region, session_token};
  }
};

/** Outcome of a single data-plane operation. */
struct S3Result {
  long http_status = 0;    ///< HTTP status (0 => exception before a response)
  bool not_found = false;  ///< GET 404 => caller zero-fills (sparse read)
  std::string error;       ///< human-readable error, empty on success

  bool ok() const {
    return error.empty() && http_status >= 200 && http_status < 300;
  }
};

/** Initialize Poco client-side TLS once per process (system CAs, peer verified). */
inline void EnsureSslInitialized() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    Poco::Net::initializeSSL();
    Poco::SharedPtr<Poco::Net::InvalidCertificateHandler> cert_handler(
        new Poco::Net::RejectCertificateHandler(false));
    // Empty key/cert/caLocation + loadDefaultCAs=true => system CA bundle;
    // VERIFY_RELAXED + RejectCertificateHandler => the peer is verified.
    Poco::Net::Context::Ptr context(new Poco::Net::Context(
        Poco::Net::Context::CLIENT_USE, "", "", "",
        Poco::Net::Context::VERIFY_RELAXED, 9, true,
        "ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH"));
    Poco::Net::SSLManager::instance().initializeClient(0, cert_handler, context);
  });
}

/** Create an HTTP(S) session for `uri`; HTTPS gets a TLS context. */
inline std::unique_ptr<Poco::Net::HTTPClientSession> MakeSession(
    const Poco::URI &uri) {
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
 * A reusable HTTP(S) connection to one S3 endpoint.
 *
 * Not thread-safe by design: each runtime worker owns exactly one and never
 * shares it. That is safe here because S3 ops run synchronously in the worker's
 * task body (blocking Poco I/O, no CO_AWAIT mid-op), so a worker's connection is
 * never touched concurrently -- the property that lets a non-thread-safe Poco
 * session be kept alive without a lock. A null `session` means "connect on next
 * use"; Retire() forces that on the next call after a failure or a server
 * `Connection: close`.
 */
struct S3Connection {
  std::unique_ptr<Poco::Net::HTTPClientSession> session;
  uint64_t connects = 0;  ///< sockets opened (diagnostics / reuse proof)
  uint64_t reuses = 0;    ///< requests served on an already-open socket
  /// How many of `connects` the owner has already reported. Lets a caller log
  /// once per newly-opened socket instead of once per request; see
  /// ReportConnectionChurn in s3_bdev_transport.cc.
  uint64_t logged_connects = 0;
  void Retire() { session.reset(); }
};

/**
 * Minimal S3 REST client. Instances are cheap and hold no live socket -- the
 * caller passes in a per-worker S3Connection, which keeps the client usable
 * from several runtime workers at once without sharing a non-thread-safe Poco
 * session, while still reusing the socket across calls on the same worker.
 */
class S3RestClient {
 public:
  explicit S3RestClient(S3Config config) : config_(std::move(config)) {}

  const S3Config &config() const { return config_; }

  /** Read connection config and credentials from the environment. */
  static S3Config ConfigFromEnv(const std::string &bucket,
                                const std::string &prefix) {
    S3Config c;
    c.endpoint = GetEnv("S3_ENDPOINT");
    c.region = GetEnv("AWS_DEFAULT_REGION");
    if (c.region.empty()) c.region = "us-east-1";
    c.access_key = GetEnv("AWS_ACCESS_KEY_ID");
    c.secret_key = GetEnv("AWS_SECRET_ACCESS_KEY");
    c.session_token = GetEnv("AWS_SESSION_TOKEN");
    c.allow_bucket_create = (GetEnv("S3_ALLOW_BUCKET_CREATE") == "1");
    c.bucket = bucket;
    c.prefix = prefix;
    // Strip trailing '/' so endpoint + path never doubles the separator.
    while (!c.endpoint.empty() && c.endpoint.back() == '/') c.endpoint.pop_back();
    return c;
  }

  /** Object key for a block offset: [prefix/]block_<offset> (as GCS does). */
  std::string KeyForOffset(uint64_t offset) const {
    std::string key = "block_" + std::to_string(offset);
    return config_.prefix.empty() ? key : (config_.prefix + "/" + key);
  }

  /**
   * Confirm the bucket exists, creating it only when explicitly allowed.
   *
   * HEAD first: 200 => ready, 403 => exists but is not ours, 404 => absent. On
   * 404 the bucket is created only under S3_ALLOW_BUCKET_CREATE=1; otherwise
   * this fails loudly naming the bucket, so a typo cannot silently conjure a
   * new billed bucket. (The GCS transport auto-creates unconditionally; on a
   * real AWS account that blast radius makes a poor default.)
   */
  S3Result EnsureBucket() {
    S3Result head = HeadBucket();
    if (head.http_status == 200) {
      head.error.clear();
      return head;
    }
    if (head.http_status == 404 || head.not_found) {
      if (!config_.allow_bucket_create) {
        S3Result r;
        r.http_status = 404;
        r.error = "S3 bucket '" + config_.bucket +
                  "' does not exist. Create it first, or set "
                  "S3_ALLOW_BUCKET_CREATE=1 to have the bdev create it.";
        return r;
      }
      return CreateBucket();
    }
    if (head.http_status == 403) {
      head.error = "S3 bucket '" + config_.bucket +
                   "' is not accessible (HTTP 403): it is owned by another "
                   "account, or these credentials lack s3:ListBucket on it.";
      return head;
    }
    if (head.error.empty()) {
      head.error =
          "S3 bucket head failed: HTTP " + std::to_string(head.http_status);
    }
    return head;
  }

  /** Upload `len` bytes to `key` over `conn`. HTTP 2xx => success. */
  S3Result PutObject(S3Connection &conn, const std::string &key,
                     const char *data, size_t len) {
    S3Result r;
    for (int attempt = 0; attempt < 2; ++attempt) {
      r = S3Result{};
      bool reused = false;
      try {
        Endpoint ep = Resolve(key);
        Poco::Net::HTTPClientSession *session = Connect(conn, ep.base, &reused);
        Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_PUT,
                                   ep.canonical_uri,
                                   Poco::Net::HTTPMessage::HTTP_1_1);
        req.setContentType("application/octet-stream");
        req.setContentLength(static_cast<std::streamsize>(len));
        Sign(req, "PUT", ep);
        std::ostream &os = session->sendRequest(req);
        if (len > 0) os.write(data, static_cast<std::streamsize>(len));
        os.flush();
        Poco::Net::HTTPResponse res;
        std::istream &rs = session->receiveResponse(res);
        r.http_status = res.getStatus();
        std::string resp;
        Poco::StreamCopier::copyToString(rs, resp);  // drain before next use
        if (!r.ok()) {
          r.error = "S3 PUT " + key + " failed: HTTP " +
                    std::to_string(r.http_status) + " " + resp;
        }
        RetireIfClosed(conn, res);
        return r;
      } catch (const Poco::Exception &e) {
        conn.Retire();
        if (reused && attempt == 0) continue;  // stale keep-alive; retry once
        r.error = std::string("S3 PUT exception: ") + e.displayText();
        return r;
      }
    }
    return r;
  }

  /**
   * Download object `key` into `buf` (up to `len` bytes). HTTP 404 sets
   * not_found and the caller zero-fills (sparse read). On success `*bytes_read`
   * (if non-null) receives the byte count copied.
   */
  S3Result GetObject(S3Connection &conn, const std::string &key, char *buf,
                     size_t len, size_t *bytes_read) {
    S3Result r;
    for (int attempt = 0; attempt < 2; ++attempt) {
      r = S3Result{};
      if (bytes_read) *bytes_read = 0;
      bool reused = false;
      try {
        Endpoint ep = Resolve(key);
        Poco::Net::HTTPClientSession *session = Connect(conn, ep.base, &reused);
        Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_GET,
                                   ep.canonical_uri,
                                   Poco::Net::HTTPMessage::HTTP_1_1);
        Sign(req, "GET", ep);
        session->sendRequest(req);
        Poco::Net::HTTPResponse res;
        std::istream &rs = session->receiveResponse(res);
        r.http_status = res.getStatus();
        if (r.http_status == 404) {
          r.not_found = true;  // sparse: caller zero-fills
          DrainAndRetireIfClosed(conn, res, rs);
          return r;
        }
        if (!r.ok()) {
          std::string resp;
          Poco::StreamCopier::copyToString(rs, resp);  // drain before next use
          r.error = "S3 GET " + key + " failed: HTTP " +
                    std::to_string(r.http_status) + " " + resp;
          RetireIfClosed(conn, res);
          return r;
        }
        // A single read() can return short on a chunked or slow response, so
        // loop until the buffer is full or the body ends.
        size_t total = 0;
        while (total < len && rs) {
          rs.read(buf + total, static_cast<std::streamsize>(len - total));
          total += static_cast<size_t>(rs.gcount());
          if (rs.gcount() == 0) break;
        }
        if (bytes_read) *bytes_read = total;
        // Drain any body past the caller buffer so the socket stays reusable.
        DrainAndRetireIfClosed(conn, res, rs);
        return r;
      } catch (const Poco::Exception &e) {
        conn.Retire();
        if (reused && attempt == 0) continue;  // stale keep-alive; retry once
        r.error = std::string("S3 GET exception: ") + e.displayText();
        return r;
      }
    }
    return r;
  }

  /** Best-effort object delete over `conn` (errors are not fatal to caller). */
  S3Result DeleteObject(S3Connection &conn, const std::string &key) {
    S3Result r;
    for (int attempt = 0; attempt < 2; ++attempt) {
      r = S3Result{};
      bool reused = false;
      try {
        Endpoint ep = Resolve(key);
        Poco::Net::HTTPClientSession *session = Connect(conn, ep.base, &reused);
        Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_DELETE,
                                   ep.canonical_uri,
                                   Poco::Net::HTTPMessage::HTTP_1_1);
        Sign(req, "DELETE", ep);
        session->sendRequest(req);
        Poco::Net::HTTPResponse res;
        std::istream &rs = session->receiveResponse(res);
        r.http_status = res.getStatus();
        DrainAndRetireIfClosed(conn, res, rs);
        return r;
      } catch (const Poco::Exception &e) {
        conn.Retire();
        if (reused && attempt == 0) continue;  // stale keep-alive; retry once
        r.error = std::string("S3 DELETE exception: ") + e.displayText();
        return r;
      }
    }
    return r;
  }

 private:
  /** Where a request goes, and what the signer must agree with. */
  struct Endpoint {
    Poco::URI base;             ///< scheme://host[:port] for the session
    std::string host_header;    ///< exact Host value (default port elided)
    std::string canonical_uri;  ///< encoded path, leading '/'
  };

  /** Read an environment variable, returning "" when unset or empty. */
  static std::string GetEnv(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string();
  }

  /**
   * Return an open session for `base`, reusing conn.session when it is still
   * connected. `*reused` reports whether an already-open socket was reused, so
   * the caller knows a mid-flight failure may just be a dropped idle keep-alive
   * (worth one retry) rather than a real error on a fresh socket.
   */
  static Poco::Net::HTTPClientSession *Connect(S3Connection &conn,
                                               const Poco::URI &base,
                                               bool *reused) {
    if (conn.session && conn.session->connected()) {
      *reused = true;
      ++conn.reuses;
      return conn.session.get();
    }
    conn.session = MakeSession(base);
    conn.session->setKeepAlive(true);
    *reused = false;
    ++conn.connects;
    return conn.session.get();
  }

  /** Retire the connection if the server declined to keep it alive. */
  static void RetireIfClosed(S3Connection &conn,
                             const Poco::Net::HTTPResponse &res) {
    if (!res.getKeepAlive()) conn.Retire();
  }

  /** Fully drain the response body (required before the socket is reused),
   *  then retire the connection if the server sent Connection: close. */
  static void DrainAndRetireIfClosed(S3Connection &conn,
                                     const Poco::Net::HTTPResponse &res,
                                     std::istream &rs) {
    Poco::NullOutputStream null;
    Poco::StreamCopier::copyStream(rs, null);
    RetireIfClosed(conn, res);
  }

  /**
   * Resolve host and path for an object key, or the bucket itself when `key`
   * is empty. No endpoint override => virtual-hosted style against real AWS
   * (bucket in the hostname); S3_ENDPOINT set => path-style (bucket as the
   * first path segment), which is what MinIO and friends expect.
   */
  Endpoint Resolve(const std::string &key) const {
    Endpoint ep;
    std::string base_url;
    if (config_.path_style()) {
      base_url = config_.endpoint;
      ep.canonical_uri = "/" + UriEncode(config_.bucket, false);
      if (!key.empty()) ep.canonical_uri += "/" + UriEncode(key, true);
    } else {
      base_url = "https://" + config_.bucket + ".s3." + config_.region +
                 ".amazonaws.com";
      ep.canonical_uri = key.empty() ? "/" : ("/" + UriEncode(key, true));
    }
    ep.base = Poco::URI(base_url);
    const std::string scheme = ep.base.getScheme();
    const int port = ep.base.getPort();
    const bool default_port =
        (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
    ep.host_header = default_port
                         ? ep.base.getHost()
                         : ep.base.getHost() + ":" + std::to_string(port);
    return ep;
  }

  /**
   * Attach the SigV4 headers (host, date, payload hash, Authorization).
   *
   * The payload hash is always UNSIGNED-PAYLOAD, which is legal over HTTPS on
   * real S3 and on S3-compatible servers. Hashing each block instead would run
   * on the runtime worker executing WriteBlocks -- inside the very interval
   * this bdev's throughput is measured over -- so it would surface as CLIO
   * overhead in a benchmark whose purpose is to measure CLIO overhead. The
   * tradeoff is that payload integrity rests on TLS rather than on the
   * signature covering the body.
   */
  void Sign(Poco::Net::HTTPRequest &req, const std::string &method,
            const Endpoint &ep) const {
    const AmzTime t = NowAmzTime();
    req.set("Host", ep.host_header);
    req.set("x-amz-date", t.amz_date);
    req.set("x-amz-content-sha256", kUnsignedPayload);
    if (!config_.session_token.empty()) {
      req.set("x-amz-security-token", config_.session_token);
    }
    SigningInput in;
    in.method = method;
    in.canonical_uri = ep.canonical_uri;
    in.host = ep.host_header;
    in.payload_hash = kUnsignedPayload;
    req.set("Authorization",
            BuildAuthorization(in, config_.credentials(), t));
  }

  /** HEAD the bucket. The status is the answer; 200/403/404 all mean something. */
  S3Result HeadBucket() {
    S3Result r;
    try {
      Endpoint ep = Resolve("");
      auto session = MakeSession(ep.base);
      Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_HEAD,
                                 ep.canonical_uri,
                                 Poco::Net::HTTPMessage::HTTP_1_1);
      Sign(req, "HEAD", ep);
      session->sendRequest(req);
      Poco::Net::HTTPResponse res;
      std::istream &rs = session->receiveResponse(res);
      r.http_status = res.getStatus();
      Poco::NullOutputStream null;
      Poco::StreamCopier::copyStream(rs, null);
      if (r.http_status == 404) r.not_found = true;
    } catch (const Poco::Exception &e) {
      r.error = std::string("S3 bucket head exception: ") + e.displayText();
    }
    return r;
  }

  /**
   * Create the bucket. us-east-1 takes no body; every other region requires a
   * LocationConstraint. 200, and 409 BucketAlreadyOwnedByYou (a concurrent
   * create won), are both success. A bare 409 BucketAlreadyExists means the
   * global name belongs to someone else and is fatal -- unlike GCS, where 409
   * unambiguously means "already yours".
   */
  S3Result CreateBucket() {
    S3Result r;
    try {
      std::string body;
      if (config_.region != "us-east-1") {
        body =
            "<CreateBucketConfiguration "
            "xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
            "<LocationConstraint>" +
            config_.region + "</LocationConstraint></CreateBucketConfiguration>";
      }
      Endpoint ep = Resolve("");
      auto session = MakeSession(ep.base);
      Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_PUT,
                                 ep.canonical_uri,
                                 Poco::Net::HTTPMessage::HTTP_1_1);
      req.setContentLength(static_cast<std::streamsize>(body.size()));
      if (!body.empty()) req.setContentType("application/xml");
      Sign(req, "PUT", ep);
      std::ostream &os = session->sendRequest(req);
      if (!body.empty()) os << body;
      os.flush();
      Poco::Net::HTTPResponse res;
      std::istream &rs = session->receiveResponse(res);
      r.http_status = res.getStatus();
      std::string resp;
      Poco::StreamCopier::copyToString(rs, resp);
      if (r.ok()) return r;
      if (r.http_status == 409 &&
          resp.find("BucketAlreadyOwnedByYou") != std::string::npos) {
        return r;  // concurrent create by another node; the bucket is ours
      }
      r.error = "S3 bucket create failed: HTTP " +
                std::to_string(r.http_status) + " " + resp;
    } catch (const Poco::Exception &e) {
      r.error = std::string("S3 bucket create exception: ") + e.displayText();
    }
    return r;
  }

  S3Config config_;
};

}  // namespace clio::run::bdev::s3

#endif  // CLIO_ENABLE_AMAZON_DRIVE

#endif  // CLIO_BDEV_S3_REST_H_
