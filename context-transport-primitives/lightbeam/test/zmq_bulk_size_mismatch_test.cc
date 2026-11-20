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
 * ZeroMqTransport::Recv must reject a message whose bulk frame length differs
 * from the size declared for it, in both receive branches:
 *  - zero-copy: the receiver exposed no buffer, so RecvBulks adopts the
 *    zmq_msg_t; the frame must carry the size the sender declared;
 *  - copy: the receiver's metadata exposed its own buffer, so RecvBulks
 *    memcpys into it; the frame must carry exactly the exposed size.
 * A raw libzmq DEALER forges the messages (Send() can never produce a
 * mismatch). Each case checks that Recv returns EPROTO, that bulks adopted
 * before the bad frame are released, that the caller's buffer for the bad
 * bulk is untouched, and that the next well-formed message on the same
 * socket is received intact (the rest of the bad message was drained).
 */

#include <clio_ctp/lightbeam/zmq_transport.h>
#include <zmq.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "lbm_test_deadline.h"

using ctp::lbm::Bulk;
using ctp::lbm::ClientInfo;
using ctp::lbm::LbmMeta;
using ctp::lbm::TransportMode;
using ctp::lbm::ZeroMqTransport;

/** Check that always runs (unlike assert under NDEBUG); exits on failure. */
#define MISMATCH_CHECK(cond)                                              \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, \
                   #cond);                                                \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

namespace {

/** Byte the receiver pre-fills its exposed buffers with. */
constexpr char kSentinel = '#';

/** Capacity of each receiver-exposed buffer in the copy-branch cases. */
constexpr size_t kExposedCapacity = 256;

/**
 * Receiver-side metadata that exposes caller-owned buffers, as
 * LoadTaskArchive does for a response: on load it turns every declared bulk
 * into a recv entry pointing at `buffers_[i]`, and clears `send` so Recv()
 * adds no zero-copy entries. A non-null recv data pointer is what sends
 * RecvBulks down its copy branch.
 */
class ExposedMeta : public LbmMeta<> {
 public:
  /** Caller-owned destination buffers, one per declared bulk. */
  std::vector<std::vector<char>> *buffers_ = nullptr;

  /**
   * Deserialize the standard fields, then expose one buffer per bulk.
   *
   * @param ar the deserialization archive
   */
  template <typename Ar>
  void serialize(Ar &ar) {
    LbmMeta<>::serialize(ar);
    for (size_t i = 0; i < send.size(); ++i) {
      Bulk bulk;
      bulk.data = ctp::ipc::FullPtr<char>((*buffers_)[i].data());
      bulk.size = send[i].size;
      bulk.flags = send[i].flags;
      recv.push_back(bulk);
    }
    send.clear();
  }
};

/** A raw libzmq DEALER connected to the transport's ROUTER. */
class RawDealer {
 public:
  /**
   * Create a private context and connect a DEALER to 127.0.0.1:port.
   *
   * @param port TCP port the server transport is bound to
   */
  explicit RawDealer(int port) {
    ctx_ = zmq_ctx_new();
    sock_ = zmq_socket(ctx_, ZMQ_DEALER);
    MISMATCH_CHECK(sock_ != nullptr);
    std::string url = "tcp://127.0.0.1:" + std::to_string(port);
    MISMATCH_CHECK(zmq_connect(sock_, url.c_str()) == 0);
  }

  /** Close the socket (no linger) and destroy the context. */
  ~RawDealer() {
    int linger = 0;
    zmq_setsockopt(sock_, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_close(sock_);
    zmq_ctx_destroy(ctx_);
  }

  /**
   * Send one logical message framed like ZeroMqTransport::Send from a
   * DEALER -- [delim, meta, bulk...] -- but with metadata declaring
   * `declared[i]` bytes for bulk i while the frame carries `frames[i]`.
   *
   * @param declared sizes written into the serialized metadata
   * @param frames   bulk frame payloads actually put on the wire
   */
  void SendForged(const std::vector<size_t> &declared,
                  const std::vector<std::string> &frames) {
    LbmMeta<> meta;
    for (size_t size : declared) {
      Bulk bulk;
      bulk.size = size;
      bulk.flags = ctp::bitfield32_t(BULK_XFER);
      meta.send.push_back(bulk);
    }
    meta.send_bulks = declared.size();
    std::vector<char> meta_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(meta_buf);
      ar(meta);
      ar.Finalize();
    }
    MISMATCH_CHECK(zmq_send(sock_, "", 0, ZMQ_SNDMORE) == 0);
    int flags = frames.empty() ? 0 : ZMQ_SNDMORE;
    MISMATCH_CHECK(zmq_send(sock_, meta_buf.data(), meta_buf.size(), flags) ==
                   static_cast<int>(meta_buf.size()));
    for (size_t i = 0; i < frames.size(); ++i) {
      flags = (i + 1 < frames.size()) ? ZMQ_SNDMORE : 0;
      MISMATCH_CHECK(zmq_send(sock_, frames[i].data(), frames[i].size(),
                              flags) == static_cast<int>(frames[i].size()));
    }
  }

  /**
   * Send a well-formed message: each frame's declared size is its length.
   *
   * @param frames bulk frame payloads
   */
  void SendValid(const std::vector<std::string> &frames) {
    std::vector<size_t> declared;
    for (const auto &f : frames) declared.push_back(f.size());
    SendForged(declared, frames);
  }

 private:
  void *ctx_ = nullptr;  /**< Private ZMQ context */
  void *sock_ = nullptr; /**< DEALER socket */
};

/**
 * Poll Recv() until it returns something other than EAGAIN, bounded by a
 * generous wall-clock deadline so a slow runner never fails spuriously.
 *
 * @param server the ROUTER transport
 * @param meta   metadata to receive into
 * @return the ClientInfo of the first non-EAGAIN Recv
 */
template <typename MetaT>
ClientInfo RecvOne(ZeroMqTransport *server, MetaT &meta) {
  lbm_test::Deadline deadline;
  while (true) {
    ClientInfo info = server->Recv(meta);
    if (info.rc != EAGAIN) return info;
    MISMATCH_CHECK(!deadline.Expired());
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

/**
 * Check that a zero-copy receive returned exactly `frames`, then release it.
 *
 * @param server the ROUTER transport that received the message
 * @param meta   the received metadata
 * @param frames the payloads the sender put on the wire
 */
void ExpectZeroCopyPayload(ZeroMqTransport *server, LbmMeta<> &meta,
                           const std::vector<std::string> &frames) {
  MISMATCH_CHECK(meta.recv.size() == frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    MISMATCH_CHECK(meta.recv[i].desc != nullptr);
    MISMATCH_CHECK(meta.recv[i].size == frames[i].size());
    MISMATCH_CHECK(std::string(meta.recv[i].data.ptr_, meta.recv[i].size) ==
                   frames[i]);
  }
  server->ClearRecvHandles(meta);
}

/**
 * Zero-copy branch: a bad frame among several bulks. Bulks before it were
 * adopted and must be released; the rest of the message must be drained.
 *
 * @param server  the ROUTER transport
 * @param dealer  raw DEALER connected to it
 * @param bad_len length of the forged frame for bulk 1 (declared: 16)
 */
void ZeroCopyMismatch(ZeroMqTransport *server, RawDealer *dealer,
                      size_t bad_len) {
  std::printf("zero-copy: bulk 1 declared 16, frame %zu\n", bad_len);
  dealer->SendForged({8, 16, 4},
                     {"AAAAAAAA", std::string(bad_len, 'B'), "CCCC"});
  LbmMeta<> bad;
  ClientInfo info = RecvOne(server, bad);
  MISMATCH_CHECK(info.rc == EPROTO);
  // Nothing adopted survives: bulk 0's zmq_msg was released, the bad one
  // was never stored, and bulk 2 was never reached.
  MISMATCH_CHECK(bad.recv.size() == 3);
  for (const Bulk &bulk : bad.recv) {
    MISMATCH_CHECK(bulk.desc == nullptr);
    MISMATCH_CHECK(bulk.data.ptr_ == nullptr);
  }

  const std::vector<std::string> good = {"next-message-0", "next-1"};
  dealer->SendValid(good);
  LbmMeta<> next;
  info = RecvOne(server, next);
  MISMATCH_CHECK(info.rc == 0);
  ExpectZeroCopyPayload(server, next, good);
}

/**
 * Copy branch: the receiver exposed its own buffers. A frame whose length
 * differs from the exposed size is refused before the copy, leaving the
 * caller's buffer for it untouched; earlier bulks were copied in.
 *
 * @param server  the ROUTER transport
 * @param dealer  raw DEALER connected to it
 * @param bad_len length of the forged frame for bulk 1 (declared: 16)
 */
void CopyMismatch(ZeroMqTransport *server, RawDealer *dealer, size_t bad_len) {
  std::printf("copy: bulk 1 declared 16, frame %zu\n", bad_len);
  std::vector<std::vector<char>> buffers(
      3, std::vector<char>(kExposedCapacity, kSentinel));
  dealer->SendForged({8, 16, 4},
                     {"AAAAAAAA", std::string(bad_len, 'B'), "CCCC"});
  ExposedMeta bad;
  bad.buffers_ = &buffers;
  ClientInfo info = RecvOne(server, bad);
  MISMATCH_CHECK(info.rc == EPROTO);
  MISMATCH_CHECK(bad.recv.size() == 3);
  for (const Bulk &bulk : bad.recv) MISMATCH_CHECK(bulk.desc == nullptr);
  // Bulk 0 went through the copy branch; bulks 1 and 2 were never written.
  MISMATCH_CHECK(std::string(buffers[0].data(), 8) == "AAAAAAAA");
  for (size_t b = 1; b < 3; ++b) {
    for (char c : buffers[b]) MISMATCH_CHECK(c == kSentinel);
  }

  const std::vector<std::string> good = {"next-copy-0", "nc-1"};
  std::vector<std::vector<char>> next_buffers(
      good.size(), std::vector<char>(kExposedCapacity, kSentinel));
  dealer->SendValid(good);
  ExposedMeta next;
  next.buffers_ = &next_buffers;
  info = RecvOne(server, next);
  MISMATCH_CHECK(info.rc == 0);
  MISMATCH_CHECK(next.recv.size() == good.size());
  for (size_t i = 0; i < good.size(); ++i) {
    MISMATCH_CHECK(next.recv[i].desc == nullptr);
    MISMATCH_CHECK(std::string(next_buffers[i].data(), good[i].size()) ==
                   good[i]);
    MISMATCH_CHECK(next_buffers[i][good[i].size()] == kSentinel);
  }
}

/**
 * A bad first bulk (nothing adopted yet) that is also the message's last
 * frame, so RejectBulks has nothing to release and nothing to drain.
 *
 * @param server the ROUTER transport
 * @param dealer raw DEALER connected to it
 */
void SingleBulkMismatch(ZeroMqTransport *server, RawDealer *dealer) {
  std::printf("zero-copy: single bulk declared 32, frame 31\n");
  dealer->SendForged({32}, {std::string(31, 'S')});
  LbmMeta<> bad;
  ClientInfo info = RecvOne(server, bad);
  MISMATCH_CHECK(info.rc == EPROTO);
  MISMATCH_CHECK(bad.recv.size() == 1);
  MISMATCH_CHECK(bad.recv[0].desc == nullptr);

  const std::vector<std::string> good = {"after-single"};
  dealer->SendValid(good);
  LbmMeta<> next;
  info = RecvOne(server, next);
  MISMATCH_CHECK(info.rc == 0);
  ExpectZeroCopyPayload(server, next, good);
}

}  // namespace

/**
 * Run every mismatch case against one server socket, so each case also
 * proves the socket is still on a message boundary after the previous one.
 *
 * @return 0 on success; any failed check exits with status 1
 */
int main() {
  auto server = std::make_unique<ZeroMqTransport>(TransportMode::kServer,
                                                  "127.0.0.1", "tcp", 0);
  int port = server->GetBoundPort();
  MISMATCH_CHECK(port > 0);
  RawDealer dealer(port);

  ZeroCopyMismatch(server.get(), &dealer, 15);  // shorter than declared
  ZeroCopyMismatch(server.get(), &dealer, 17);  // longer than declared
  CopyMismatch(server.get(), &dealer, 15);
  CopyMismatch(server.get(), &dealer, 17);
  SingleBulkMismatch(server.get(), &dealer);

  std::printf("zmq_bulk_size_mismatch_test: all cases passed\n");
  return 0;
}
