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

#include "summarizer_http_stub.h"

// Windows API headers are included HERE, in a .cc, and never from the header —
// see the note in summarizer_http_stub.h (issue #476 macro clash).
#ifdef _WIN32
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
#include <cstdlib>
#include <mutex>
#include <thread>

namespace clio_cae_test {

namespace {

#ifdef _WIN32
using sock_t = SOCKET;
const sock_t kBadSock = INVALID_SOCKET;
constexpr int kShutBoth = SD_BOTH;
void CloseSocket(sock_t s) { ::closesocket(s); }

/** WSAStartup must run once before any socket call in the fixture. */
struct WinsockSession {
  WinsockSession() {
    WSADATA d;
    (void)::WSAStartup(MAKEWORD(2, 2), &d);
  }
  ~WinsockSession() { ::WSACleanup(); }
};
void EnsureWinsock() { static const WinsockSession session; }
#else
using sock_t = int;
constexpr sock_t kBadSock = -1;
constexpr int kShutBoth = SHUT_RDWR;
void CloseSocket(sock_t s) { ::close(s); }
void EnsureWinsock() {}
#endif

/**
 * @param request Bytes read from the connection so far.
 * @return true once the headers and the full Content-Length body arrived.
 */
bool RequestComplete(const std::string &request) {
  size_t hdr_end = request.find("\r\n\r\n");
  if (hdr_end == std::string::npos) {
    return false;
  }
  size_t pos = request.find("Content-Length:");
  if (pos == std::string::npos) {
    return true;  // no body expected
  }
  size_t len = static_cast<size_t>(std::strtoul(
      request.c_str() + pos + sizeof("Content-Length:") - 1, nullptr, 10));
  return request.size() >= hdr_end + 4 + len;
}

/**
 * @param request A complete HTTP request.
 * @return Everything after the header terminator, or "" if there is none.
 */
std::string BodyOf(const std::string &request) {
  size_t hdr_end = request.find("\r\n\r\n");
  if (hdr_end == std::string::npos) {
    return std::string();
  }
  return request.substr(hdr_end + 4);
}

}  // namespace

struct OneShotHttpServer::Impl {
  sock_t listen_fd_ = kBadSock;
  unsigned short port_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<size_t> requests_{0};
  mutable std::mutex requests_mu_;
  std::vector<std::string> request_bodies_;
  std::thread server_;
};

OneShotHttpServer::OneShotHttpServer(const std::string &status_line,
                                     const std::string &body,
                                     unsigned short port)
    : impl_(new Impl()) {
  EnsureWinsock();
  Impl *impl = impl_.get();
  impl->listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  ::setsockopt(impl->listen_fd_, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&opt), sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  ::bind(impl->listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  ::getsockname(impl->listen_fd_, reinterpret_cast<sockaddr *>(&addr), &len);
  impl->port_ = ntohs(addr.sin_port);
  ::listen(impl->listen_fd_, 8);

  std::string response = status_line +
                         "Content-Type: application/json\r\n"
                         "Content-Length: " +
                         std::to_string(body.size()) +
                         "\r\n"
                         "Connection: close\r\n\r\n" +
                         body;
  impl->server_ = std::thread([impl, response]() {
    while (!impl->stop_.load()) {
      sock_t conn = ::accept(impl->listen_fd_, nullptr, nullptr);
      if (conn == kBadSock) {
        break;  // listen_fd_ closed by Stop()
      }
      // Read the request headers+body. Keep reading until the peer stops
      // sending so a caller can assert on what was actually POSTed (the
      // prompt), not just that a request arrived.
      std::string request;
      char buf[8192];
      for (;;) {
        int n = ::recv(conn, buf, static_cast<int>(sizeof(buf)), 0);
        if (n <= 0) {
          break;
        }
        request.append(buf, static_cast<size_t>(n));
        if (RequestComplete(request)) {
          break;
        }
      }
      (void)::send(conn, response.data(), static_cast<int>(response.size()), 0);
      CloseSocket(conn);
      {
        std::lock_guard<std::mutex> lock(impl->requests_mu_);
        impl->request_bodies_.push_back(BodyOf(request));
      }
      impl->requests_.fetch_add(1);
    }
  });
}

OneShotHttpServer::~OneShotHttpServer() { Stop(); }

void OneShotHttpServer::Stop() {
  Impl *impl = impl_.get();
  if (!impl->stop_.exchange(true)) {
    ::shutdown(impl->listen_fd_, kShutBoth);
    CloseSocket(impl->listen_fd_);
    if (impl->server_.joinable()) {
      impl->server_.join();
    }
  }
}

std::string OneShotHttpServer::Endpoint() const {
  return "http://127.0.0.1:" + std::to_string(impl_->port_);
}

unsigned short OneShotHttpServer::Port() const { return impl_->port_; }

size_t OneShotHttpServer::RequestCount() const {
  return impl_->requests_.load();
}

std::vector<std::string> OneShotHttpServer::RequestBodies() const {
  std::lock_guard<std::mutex> lock(impl_->requests_mu_);
  return impl_->request_bodies_;
}

}  // namespace clio_cae_test
