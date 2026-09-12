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

#ifndef CLIO_CAE_TEST_SUMMARIZER_HTTP_STUB_H_
#define CLIO_CAE_TEST_SUMMARIZER_HTTP_STUB_H_

/**
 * A minimal in-process HTTP server used by the summarizer tests to stand in
 * for an Ollama endpoint. Shared by:
 *   - test_summarizer_label_client.cc — drives OllamaGenerate's response
 *     handling (status, malformed JSON, missing field, success).
 *   - test_summarizer_store.cc — lets the full PutBlob → summarize → store
 *     path run end-to-end with no model installed.
 *
 * IMPORTANT: this header includes NO OS socket headers. On Windows
 * <winsock2.h> leaks function-like macros (Yield, min, max, ...) into every
 * including translation unit — the issue #476 clash that the rest of the tree
 * is careful to avoid (see clio_ctp/lightbeam/posix_socket.h). All socket work
 * therefore lives behind the pimpl in summarizer_http_stub.cc; a test that
 * includes this header sees only standard library types.
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace clio_cae_test {

/**
 * HTTP server that listens on localhost and answers every connection with the
 * same configured status line and body until stopped.
 */
class OneShotHttpServer {
 public:
  /**
   * Bind, listen, and start answering on a background thread.
   * @param status_line Full status line including trailing CRLF, e.g.
   *        "HTTP/1.1 200 OK\r\n".
   * @param body Response body, served verbatim as application/json.
   * @param port Fixed TCP port to bind, or 0 (default) for an ephemeral one.
   *        A fixed port is needed when a compose YAML has to name the
   *        endpoint before the test runs.
   */
  OneShotHttpServer(const std::string &status_line, const std::string &body,
                    unsigned short port = 0);
  ~OneShotHttpServer();

  OneShotHttpServer(const OneShotHttpServer &) = delete;
  OneShotHttpServer &operator=(const OneShotHttpServer &) = delete;

  /** Close the listener and join the accept thread. Idempotent. */
  void Stop();

  /** Base URL the label client should be pointed at. */
  std::string Endpoint() const;

  /** The bound port (resolved, even when 0 was requested). */
  unsigned short Port() const;

  /** Number of connections answered so far. */
  size_t RequestCount() const;

  /** Bodies of the requests received so far, in arrival order. */
  std::vector<std::string> RequestBodies() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace clio_cae_test

#endif  // CLIO_CAE_TEST_SUMMARIZER_HTTP_STUB_H_
