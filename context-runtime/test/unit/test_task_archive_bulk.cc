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
 * LoadTaskArchive::bulk copies a received bulk by the bytes the transport
 * actually received, not by the size the task declares, in both directions.
 */

#include "../simple_test.h"

#include <cstring>
#include <thread>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/task_archives.h"

using namespace clio::run;

namespace {

bool InitRuntimeOnce() {
  static const bool ok = [] {
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
      return false;
    }
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return true;
  }();
  return ok;
}

/** A received bulk of `size` bytes of `fill`, in a heap buffer as the ZMQ
 *  and Thallium transports hand them over. */
ctp::lbm::Bulk ReceivedBulk(std::vector<char> &bytes, size_t size, char fill) {
  bytes.assign(size, fill);
  ctp::lbm::Bulk bulk;
  bulk.size = size;
  bulk.flags.SetBits(BULK_XFER);
  bulk.data.ptr_ = bytes.data();
  bulk.data.shm_.alloc_id_ =
      ctp::ipc::AllocatorId(UINT32_MAX - 1, UINT32_MAX - 1);
  bulk.data.shm_.off_ = reinterpret_cast<size_t>(bytes.data());
  return bulk;
}

}  // namespace

TEST_CASE("LoadTaskArchive bulk: input declares more than was received",
          "[task_archive][bulk]") {
  REQUIRE(InitRuntimeOnce());
  std::vector<char> received;
  LoadTaskArchive archive;
  archive.msg_type_ = MsgType::kSerializeIn;
  archive.recv.push_back(ReceivedBulk(received, 16, 'A'));

  // The task declares 4096 bytes; 16 arrived. The task gets a 4096-byte
  // buffer holding those 16 bytes and zeros after them.
  ctp::ipc::ShmPtr<> ptr;
  archive.bulk(ptr, 4096, BULK_XFER);
  ctp::ipc::FullPtr<char> buf = CLIO_IPC->ToFullPtr(ptr).template Cast<char>();
  REQUIRE(buf.ptr_ != nullptr);
  REQUIRE(buf.ptr_ != received.data());
  for (size_t i = 0; i < 16; ++i) REQUIRE(buf.ptr_[i] == 'A');
  for (size_t i = 16; i < 4096; ++i) REQUIRE(buf.ptr_[i] == 0);
  CLIO_IPC->FreeBuffer(buf);
}

TEST_CASE("LoadTaskArchive bulk: input declares no more than was received",
          "[task_archive][bulk]") {
  REQUIRE(InitRuntimeOnce());
  std::vector<char> received;
  LoadTaskArchive archive;
  archive.msg_type_ = MsgType::kSerializeIn;
  archive.recv.push_back(ReceivedBulk(received, 64, 'B'));

  ctp::ipc::ShmPtr<> ptr;
  archive.bulk(ptr, 64, BULK_XFER);
  ctp::ipc::FullPtr<char> buf = CLIO_IPC->ToFullPtr(ptr).template Cast<char>();
  REQUIRE(buf.ptr_ != nullptr);
  for (size_t i = 0; i < 64; ++i) REQUIRE(buf.ptr_[i] == 'B');
  CLIO_IPC->FreeBuffer(buf);
}

TEST_CASE("LoadTaskArchive bulk: output larger than the caller's buffer",
          "[task_archive][bulk]") {
  REQUIRE(InitRuntimeOnce());
  std::vector<char> received;
  LoadTaskArchive archive;
  archive.msg_type_ = MsgType::kSerializeOut;
  archive.recv.push_back(ReceivedBulk(received, 64, 'R'));

  // The caller's buffer is 16 bytes; the response carries 64. Only the
  // caller's 16 bytes change.
  ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(64);
  REQUIRE(buf.ptr_ != nullptr);
  memset(buf.ptr_, 'G', 64);
  ctp::ipc::ShmPtr<> ptr = buf.shm_.template Cast<void>();
  archive.bulk(ptr, 16, BULK_XFER);
  for (size_t i = 0; i < 16; ++i) REQUIRE(buf.ptr_[i] == 'R');
  for (size_t i = 16; i < 64; ++i) REQUIRE(buf.ptr_[i] == 'G');
  CLIO_IPC->FreeBuffer(buf);
}

SIMPLE_TEST_MAIN()
