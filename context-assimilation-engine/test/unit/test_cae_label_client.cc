/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Unit tests for the CAE Ollama label client (label_client.cc).
 *
 * OllamaGenerate talks HTTP to an inference server; these tests cover every
 * branch without a real model: argument validation, connection failure, and
 * — via a tiny in-process HTTP fixture — non-200 status, malformed JSON,
 * JSON missing the 'response' field, and the success path.
 */

#include "simple_test.h"

#include <clio_cae/core/label_client.h>

#ifdef _WIN32
// Including Windows headers from a .cc is allowed (the no-winsock rule only
// applies to headers); keep the macro fallout contained anyway.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cstring>
#include <string>
#include <thread>

using clio::cae::core::OllamaGenerate;

namespace {

#ifdef _WIN32
using sock_t = SOCKET;
const sock_t kBadSock = INVALID_SOCKET;
constexpr int kShutBoth = SD_BOTH;
void CloseSocket(sock_t s) { ::closesocket(s); }
/** WSAStartup must run before any socket call in the fixture. */
struct WinsockSession {
  WinsockSession() {
    WSADATA d;
    (void)::WSAStartup(MAKEWORD(2, 2), &d);
  }
  ~WinsockSession() { ::WSACleanup(); }
};
const WinsockSession kWinsockSession;
#else
using sock_t = int;
constexpr sock_t kBadSock = -1;
constexpr int kShutBoth = SHUT_RDWR;
void CloseSocket(sock_t s) { ::close(s); }
#endif

/**
 * One-shot HTTP server: listens on an ephemeral localhost port and answers
 * every connection with the configured body/status until stopped.
 */
class OneShotHttpServer {
 public:
  explicit OneShotHttpServer(const std::string &status_line,
                             const std::string &body) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char *>(&opt), sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  // ephemeral
    ::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, reinterpret_cast<sockaddr *>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    ::listen(listen_fd_, 4);

    std::string response = status_line +
                           "Content-Type: application/json\r\n"
                           "Content-Length: " +
                           std::to_string(body.size()) +
                           "\r\n"
                           "Connection: close\r\n\r\n" +
                           body;
    server_ = std::thread([this, response]() {
      while (!stop_.load()) {
        sock_t conn = ::accept(listen_fd_, nullptr, nullptr);
        if (conn == kBadSock) {
          break;  // listen_fd_ closed by Stop()
        }
        // Drain the request headers+body (best effort, single read is
        // enough for the small POST bodies the client sends).
        char buf[8192];
        (void)::recv(conn, buf, static_cast<int>(sizeof(buf)), 0);
        (void)::send(conn, response.data(),
                     static_cast<int>(response.size()), 0);
        CloseSocket(conn);
      }
    });
  }

  ~OneShotHttpServer() { Stop(); }

  void Stop() {
    if (!stop_.exchange(true)) {
      ::shutdown(listen_fd_, kShutBoth);
      CloseSocket(listen_fd_);
      if (server_.joinable()) {
        server_.join();
      }
    }
  }

  std::string Endpoint() const {
    return "http://127.0.0.1:" + std::to_string(port_);
  }

 private:
  sock_t listen_fd_ = kBadSock;
  unsigned short port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread server_;
};

}  // namespace

TEST_CASE("LabelClient - argument validation", "[cae][label][args]") {
  std::string out;

  SECTION("Empty endpoint rejected");
  out = "stale";
  REQUIRE_FALSE(OllamaGenerate("", "model", "prompt", 0, 0, out));
  REQUIRE(out.empty());

  SECTION("Empty model rejected");
  REQUIRE_FALSE(OllamaGenerate("http://127.0.0.1:11434", "", "prompt", 0, 0,
                               out));
}

TEST_CASE("LabelClient - transport error on unreachable endpoint",
          "[cae][label][transport]") {
  std::string out;
  // Port 1 on localhost: connection refused almost immediately.
  REQUIRE_FALSE(OllamaGenerate("http://127.0.0.1:1", "m", "p", 128, 16, out));
  REQUIRE(out.empty());
}

TEST_CASE("LabelClient - HTTP error status", "[cae][label][http]") {
  OneShotHttpServer server("HTTP/1.1 500 Internal Server Error\r\n",
                           "{\"error\":\"boom\"}");
  std::string out;
  REQUIRE_FALSE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  server.Stop();
}

TEST_CASE("LabelClient - malformed JSON body", "[cae][label][badjson]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n", "this is not json {{{");
  std::string out;
  REQUIRE_FALSE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  server.Stop();
}

TEST_CASE("LabelClient - JSON missing response field",
          "[cae][label][nofield]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n", "{\"done\":true}");
  std::string out;
  REQUIRE_FALSE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  server.Stop();
}

TEST_CASE("LabelClient - success path", "[cae][label][success]") {
  OneShotHttpServer server("HTTP/1.1 200 OK\r\n",
                           "{\"response\":\"a fine label\",\"done\":true}");
  std::string out;

  SECTION("Plain request succeeds");
  REQUIRE(OllamaGenerate(server.Endpoint(), "m", "p", 0, 0, out));
  REQUIRE(out == "a fine label");

  SECTION("Trailing slash and options (num_ctx/num_predict) accepted");
  REQUIRE(OllamaGenerate(server.Endpoint() + "/", "m", "p", 2048, 64, out));
  REQUIRE(out == "a fine label");

  server.Stop();
}

SIMPLE_TEST_MAIN()
