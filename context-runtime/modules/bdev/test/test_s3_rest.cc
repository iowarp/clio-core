/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * Round-trip tests for S3RestClient against a local S3 stand-in.
 *
 * s3_stub_server.py starts an in-memory endpoint on an ephemeral port, exports
 * S3_ENDPOINT (which selects path-style addressing) plus credentials, and runs
 * this binary as its child. No AWS account, no network, no Docker.
 *
 * What this covers that test_sigv4.cc cannot: the signer is exercised through
 * the client's own wiring -- the Host header, the canonical URI derived from
 * path-style addressing, a live timestamp -- and the stub verifies the
 * resulting signature independently. A client that signed a virtual-hosted URI
 * while sending a path-style request would pass every frozen vector and fail
 * here, which is the failure this pairing exists to catch.
 *
 * Run standalone (without the wrapper) and every case self-skips.
 */

#include "clio_runtime/bdev/transports/s3_rest.h"

#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPMessage.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/StreamCopier.h>
#include <Poco/URI.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <string>
#include <vector>

#include "simple_test.h"

namespace s3 = clio::run::bdev::s3;

namespace {

/** Read an env var, "" when unset. */
std::string Env(const char *name) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::string(v) : std::string();
}

/** True when s3_stub_server.py is driving us. */
bool StubAvailable() { return !Env("S3_ENDPOINT").empty(); }

/** A client pointed at the stub bucket with the given key prefix. */
s3::S3RestClient MakeClient(const std::string &prefix) {
  return s3::S3RestClient(
      s3::S3RestClient::ConfigFromEnv(Env("S3_STUB_BUCKET"), prefix));
}

/** Deterministic, offset-dependent bytes, so a misplaced block is visible. */
std::vector<char> Pattern(size_t len, unsigned seed) {
  std::vector<char> v(len);
  for (size_t i = 0; i < len; ++i) {
    v[i] = static_cast<char>((i * 31u + seed * 7u + (i >> 8)) & 0xFF);
  }
  return v;
}

/** The stub's connection/request counters (GET /__stats). */
struct StubStats {
  long connections = 0;  ///< accepted TCP sockets so far
  long requests = 0;     ///< handled S3 requests so far (/__stats excluded)
};

/** Pull an integer field out of the stub's tiny {"k": n, ...} JSON by hand. */
long ExtractInt(const std::string &json, const std::string &field) {
  const std::string needle = "\"" + field + "\":";
  size_t p = json.find(needle);
  if (p == std::string::npos) return -1;
  return std::strtol(json.c_str() + p + needle.size(), nullptr, 10);
}

/**
 * Read /__stats over `probe`, a keep-alive session the caller holds open across
 * calls. Because the probe reuses its own single socket, it never perturbs the
 * connection count it is measuring -- so a delta between two reads is entirely
 * the sockets opened by the code under test in between.
 */
StubStats ReadStats(Poco::Net::HTTPClientSession &probe) {
  Poco::URI uri(Env("S3_ENDPOINT"));
  probe.setKeepAlive(true);
  Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_GET, "/__stats",
                             Poco::Net::HTTPMessage::HTTP_1_1);
  req.set("Host", uri.getHost() + ":" + std::to_string(uri.getPort()));
  probe.sendRequest(req);
  Poco::Net::HTTPResponse res;
  std::istream &rs = probe.receiveResponse(res);
  std::string body;
  Poco::StreamCopier::copyToString(rs, body);
  StubStats s;
  s.connections = ExtractInt(body, "connections");
  s.requests = ExtractInt(body, "requests");
  return s;
}

}  // namespace

TEST_CASE("s3_rest_key_for_offset", "[s3_rest]") {
  // Pure addressing: no endpoint needed, so this runs even standalone.
  s3::S3Config bare;
  bare.bucket = "b";
  REQUIRE(s3::S3RestClient(bare).KeyForOffset(0) == "block_0");
  REQUIRE(s3::S3RestClient(bare).KeyForOffset(1048576) == "block_1048576");

  s3::S3Config pre = bare;
  pre.prefix = "clio/run7";
  REQUIRE(s3::S3RestClient(pre).KeyForOffset(4096) == "clio/run7/block_4096");
}

TEST_CASE("s3_rest_addressing_mode_follows_endpoint", "[s3_rest]") {
  s3::S3Config aws;
  aws.bucket = "b";
  REQUIRE_FALSE(aws.path_style());

  s3::S3Config local;
  local.bucket = "b";
  local.endpoint = "http://127.0.0.1:9000";
  REQUIRE(local.path_style());

  // A trailing slash on the endpoint must not survive into the request path.
  s3::S3Config slashed =
      s3::S3RestClient::ConfigFromEnv("b", "p");  // reads S3_ENDPOINT
  REQUIRE(slashed.endpoint.empty() || slashed.endpoint.back() != '/');
}

TEST_CASE("s3_rest_ensure_bucket", "[s3_rest]") {
  if (!StubAvailable()) {
    INFO("S3_ENDPOINT unset; run via s3_stub_server.py. Skipping.");
    return;
  }
  s3::S3RestClient client = MakeClient("ensure");
  s3::S3Result r = client.EnsureBucket();
  REQUIRE(r.error.empty());
  REQUIRE(r.http_status == 200);
}

TEST_CASE("s3_rest_put_get_round_trip", "[s3_rest]") {
  if (!StubAvailable()) {
    INFO("S3_ENDPOINT unset; run via s3_stub_server.py. Skipping.");
    return;
  }
  s3::S3RestClient client = MakeClient("clio/roundtrip");
  s3::S3Connection conn;

  SECTION("a block-sized object survives the round trip byte for byte");
  const size_t kLen = 64 * 1024;
  std::vector<char> out = Pattern(kLen, 3);
  const std::string key = client.KeyForOffset(1048576);
  s3::S3Result put = client.PutObject(conn, key, out.data(), out.size());
  REQUIRE(put.error.empty());
  REQUIRE(put.ok());

  std::vector<char> in(kLen, 0);
  size_t got = 0;
  s3::S3Result get = client.GetObject(conn, key, in.data(), in.size(), &got);
  REQUIRE(get.error.empty());
  REQUIRE(get.ok());
  REQUIRE_FALSE(get.not_found);
  REQUIRE(got == kLen);
  REQUIRE(std::memcmp(out.data(), in.data(), kLen) == 0);

  SECTION("a zero-length object is legal");
  const std::string empty_key = client.KeyForOffset(0);
  REQUIRE(client.PutObject(conn, empty_key, nullptr, 0).ok());
  size_t empty_got = 1;
  s3::S3Result empty = client.GetObject(conn, empty_key, in.data(), 0, &empty_got);
  REQUIRE(empty.ok());
  REQUIRE(empty_got == 0);

  SECTION("delete removes it, and the next read is a sparse miss");
  REQUIRE(client.DeleteObject(conn, key).error.empty());
  size_t after = 1;
  s3::S3Result gone = client.GetObject(conn, key, in.data(), in.size(), &after);
  REQUIRE(gone.not_found);
  REQUIRE(gone.http_status == 404);
  REQUIRE(after == 0);
}

TEST_CASE("s3_rest_missing_object_is_a_sparse_miss", "[s3_rest]") {
  if (!StubAvailable()) {
    INFO("S3_ENDPOINT unset; run via s3_stub_server.py. Skipping.");
    return;
  }
  // The bdev depends on this exact shape: not_found set, error empty, so
  // ReadBlocks zero-fills instead of failing the task.
  s3::S3RestClient client = MakeClient("clio/never-written");
  s3::S3Connection conn;
  char buf[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  size_t got = 99;
  s3::S3Result r = client.GetObject(conn, client.KeyForOffset(8192), buf,
                                    sizeof(buf), &got);
  REQUIRE(r.not_found);
  REQUIRE(r.error.empty());
  REQUIRE(r.http_status == 404);
  REQUIRE(got == 0);
}

TEST_CASE("s3_rest_keys_needing_encoding", "[s3_rest]") {
  if (!StubAvailable()) {
    INFO("S3_ENDPOINT unset; run via s3_stub_server.py. Skipping.");
    return;
  }
  // The canonical URI is percent-encoded but keeps '/', and the signature is
  // computed over that encoded form. If encoding and signing disagreed, the
  // stub would reject the request rather than store it.
  s3::S3RestClient client = MakeClient("clio/odd keys+here");
  s3::S3Connection conn;
  const std::string key = client.KeyForOffset(2097152);
  REQUIRE(key == "clio/odd keys+here/block_2097152");

  std::vector<char> out = Pattern(1024, 9);
  REQUIRE(client.PutObject(conn, key, out.data(), out.size()).ok());

  std::vector<char> in(1024, 0);
  size_t got = 0;
  REQUIRE(client.GetObject(conn, key, in.data(), in.size(), &got).ok());
  REQUIRE(got == out.size());
  REQUIRE(std::memcmp(out.data(), in.data(), out.size()) == 0);
}

TEST_CASE("s3_rest_bad_credentials_are_rejected", "[s3_rest]") {
  if (!StubAvailable()) {
    INFO("S3_ENDPOINT unset; run via s3_stub_server.py. Skipping.");
    return;
  }
  // Proves the stub actually verifies signatures -- without this, every test
  // above would pass against a server that ignored the Authorization header.
  s3::S3Config bad =
      s3::S3RestClient::ConfigFromEnv(Env("S3_STUB_BUCKET"), "clio/bad");
  bad.secret_key = "not-the-right-secret-key-not-even-close";
  s3::S3RestClient client(bad);
  s3::S3Connection conn;

  std::vector<char> out = Pattern(32, 1);
  s3::S3Result r =
      client.PutObject(conn, client.KeyForOffset(0), out.data(), out.size());
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.http_status == 403);
  REQUIRE(r.error.find("403") != std::string::npos);
}

TEST_CASE("s3_rest_ensure_bucket_refuses_to_create_by_default", "[s3_rest]") {
  if (!StubAvailable()) {
    INFO("S3_ENDPOINT unset; run via s3_stub_server.py. Skipping.");
    return;
  }
  // A typo'd bucket must fail loudly rather than silently create a billed one.
  s3::S3Config missing = s3::S3RestClient::ConfigFromEnv("no-such-bucket", "p");
  missing.allow_bucket_create = false;
  s3::S3Result r = s3::S3RestClient(missing).EnsureBucket();
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.error.find("no-such-bucket") != std::string::npos);
  REQUIRE(r.error.find("S3_ALLOW_BUCKET_CREATE") != std::string::npos);
}

TEST_CASE("s3_rest_reuses_one_connection", "[s3_rest]") {
  if (!StubAvailable()) {
    INFO("S3_ENDPOINT unset; run via s3_stub_server.py. Skipping.");
    return;
  }
  // The whole point of the S3Connection handle: a worker reuses one socket for
  // all its ops. Proved two ways -- white-box counters on the handle, and the
  // stub's own accepted-socket count (which cannot be gamed by the client).
  Poco::URI uri(Env("S3_ENDPOINT"));
  Poco::Net::HTTPClientSession probe(uri.getHost(), uri.getPort());
  probe.setKeepAlive(true);  // never adds to the count it measures

  s3::S3RestClient client = MakeClient("clio/reuse");
  std::vector<char> out = Pattern(4096, 5);
  std::vector<char> in(4096, 0);
  StubStats before = ReadStats(probe);

  SECTION("N put+get pairs on one handle open exactly one socket");
  s3::S3Connection conn;
  const int kPairs = 5;
  for (int i = 0; i < kPairs; ++i) {
    const std::string key = client.KeyForOffset(4096 * (i + 1));
    REQUIRE(client.PutObject(conn, key, out.data(), out.size()).ok());
    size_t got = 0;
    REQUIRE(client.GetObject(conn, key, in.data(), in.size(), &got).ok());
    REQUIRE(got == out.size());
  }
  // White-box: one connect, every later request rode the reused socket.
  REQUIRE(conn.connects == 1);
  REQUIRE(conn.reuses == static_cast<uint64_t>(2 * kPairs - 1));

  StubStats after = ReadStats(probe);
  // Server-side: the 2N ops opened exactly one new socket between the reads.
  REQUIRE(after.connections - before.connections == 1);
  REQUIRE(after.requests - before.requests == 2 * kPairs);

  SECTION("two handles open two sockets (per-worker isolation is real)");
  s3::S3Connection a, b;
  const std::string ka = client.KeyForOffset(8192000);
  const std::string kb = client.KeyForOffset(8192001);
  REQUIRE(client.PutObject(a, ka, out.data(), out.size()).ok());
  REQUIRE(client.PutObject(b, kb, out.data(), out.size()).ok());
  REQUIRE(a.connects == 1);
  REQUIRE(b.connects == 1);
  StubStats two = ReadStats(probe);
  REQUIRE(two.connections - after.connections == 2);
}

SIMPLE_TEST_MAIN()
