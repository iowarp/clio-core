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

#pragma once
#if CTP_ENABLE_NIXL

#include "async_io.h"
#include <nixl.h>
#include <atomic>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <memory>
#include <cerrno>

namespace ctp {

/**
 * Pending NIXL transfer operation state.
 * Holds the NIXL request handle and registration descriptors needed for
 * completion polling and cleanup.
 */
struct NixlPendingOp {
  /** NIXL transfer request handle (can be null if submission failed) */
  nixlXferReqH* req_;
  /** Local (DRAM) memory registration descriptor list */
  nixl_reg_dlist_t local_reg_;
  /** Remote (FILE) memory registration descriptor list */
  nixl_reg_dlist_t remote_reg_;
  /** Size of the transfer in bytes (used for result reporting) */
  ssize_t size_;

  /** Constructor initializes all fields with defaults. */
  NixlPendingOp()
      : req_(nullptr),
        local_reg_(DRAM_SEG),
        remote_reg_(FILE_SEG),
        size_(0) {}
};

/**
 * AsyncIO implementation using NIXL (Network Interface eXtension Layer).
 *
 * This implementation uses NIXL's POSIX backend to perform file I/O operations
 * asynchronously. Each operation (Read/Write) creates a transfer request that is
 * polled for completion via IsComplete().
 *
 * The class is thread-safe and manages file descriptors internally. NIXL handles
 * its own queue depth internally, so the io_depth parameter is for reference only.
 */
class NixlAsyncIO : public AsyncIO {
 public:
  /**
   * Construct a NixlAsyncIO instance.
   *
   * @param io_depth Max number of concurrent operations (for reference;
   *                 NIXL manages its own depth internally).
   */
  explicit NixlAsyncIO(uint32_t io_depth)
      : io_depth_(io_depth),
        backend_(nullptr),
        fd_(-1),
        next_token_(1) {}

  /** Destructor cleans up resources. */
  ~NixlAsyncIO() override {
    Close();
  }

  /**
   * Open a file for I/O operations.
   *
   * Creates a NIXL agent and POSIX backend, then opens the file at the given
   * path. The agent name is auto-generated to ensure uniqueness.
   *
   * @param path File path to open.
   * @param flags O_RDWR, O_CREAT, etc. (O_DIRECT is not required for NIXL).
   * @param mode File creation mode (e.g., 0644).
   * @return true on success, false on failure.
   */
  bool Open(const std::string &path, int flags, mode_t mode) override {
    std::lock_guard<std::mutex> lock(mutex_);

    // Open file descriptor
    fd_ = open(path.c_str(), flags, mode);
    if (fd_ < 0) {
      return false;
    }
    path_ = path;

    // Create NIXL agent with unique name
    agent_name_ = MakeAgentName();
    nixlAgentConfig cfg;
    cfg.useProgThread = false;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

    try {
      agent_ = std::make_unique<nixlAgent>(agent_name_, cfg);
    } catch (const std::exception &e) {
      close(fd_);
      fd_ = -1;
      return false;
    }

    // Create POSIX backend
    nixl_b_params_t params;
    nixl_status_t st = agent_->createBackend("POSIX", params, backend_);
    if (st != NIXL_SUCCESS || !backend_) {
      close(fd_);
      fd_ = -1;
      return false;
    }

    return true;
  }

  /**
   * Get the current file size.
   *
   * @return File size in bytes, or -1 on error.
   */
  ssize_t GetFileSize() const override {
    if (fd_ < 0) return -1;
    off_t end = lseek(fd_, 0, SEEK_END);
    if (end < 0) return -1;
    return static_cast<ssize_t>(end);
  }

  /**
   * Truncate or extend the file to the specified size.
   *
   * @param size Target file size in bytes.
   * @return true on success, false on failure.
   */
  bool Truncate(size_t size) override {
    if (fd_ < 0) return false;
    return ftruncate(fd_, static_cast<off_t>(size)) == 0;
  }

  /**
   * Submit an asynchronous write operation.
   *
   * The buffer is transferred to the file at the specified offset. This is a
   * non-blocking call; use IsComplete() to check progress.
   *
   * @param buffer Pointer to data to write.
   * @param size Number of bytes to write.
   * @param offset File offset for the write.
   * @return IoToken for tracking, or kInvalidIoToken on failure.
   */
  IoToken Write(void *buffer, size_t size, int64_t offset) override {
    std::lock_guard<std::mutex> lock(mutex_);
    // Build local (DRAM source) descriptor
    nixl_xfer_dlist_t local(DRAM_SEG);
    local.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer), size, /*devId=*/0));

    // Build remote (FILE destination) descriptor
    nixl_xfer_dlist_t remote(FILE_SEG);
    remote.addDesc(
        nixlBasicDesc(static_cast<uintptr_t>(offset), size,
                      /*fd=*/static_cast<uint64_t>(fd_)));

    return SubmitXfer(NIXL_WRITE, local, remote, static_cast<ssize_t>(size));
  }

  /**
   * Submit an asynchronous read operation.
   *
   * The file at the specified offset is transferred into the buffer. This is a
   * non-blocking call; use IsComplete() to check progress.
   *
   * @param buffer Pointer to destination buffer.
   * @param size Number of bytes to read.
   * @param offset File offset for the read.
   * @return IoToken for tracking, or kInvalidIoToken on failure.
   */
  IoToken Read(void *buffer, size_t size, int64_t offset) override {
    std::lock_guard<std::mutex> lock(mutex_);
    // Build local (DRAM destination) descriptor
    nixl_xfer_dlist_t local(DRAM_SEG);
    local.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer), size, /*devId=*/0));

    // Build remote (FILE source) descriptor
    nixl_xfer_dlist_t remote(FILE_SEG);
    remote.addDesc(
        nixlBasicDesc(static_cast<uintptr_t>(offset), size,
                      /*fd=*/static_cast<uint64_t>(fd_)));

    return SubmitXfer(NIXL_READ, local, remote, static_cast<ssize_t>(size));
  }

  /**
   * Poll for operation completion.
   *
   * Non-blocking check of whether the I/O operation has completed. If complete,
   * fills in the result and cleans up the pending operation.
   *
   * @param token IoToken from Write() or Read().
   * @param result Filled with transfer result (bytes_transferred, error_code).
   * @return true if complete, false if still in progress.
   */
  bool IsComplete(IoToken token, IoResult &result) override {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pending_.find(token);
    if (it == pending_.end()) {
      // Token not found; may have been already reaped
      result.bytes_transferred = -1;
      result.error_code = EINVAL;
      return true;
    }

    NixlPendingOp &op = it->second;

    // Poll the NIXL request status
    nixl_status_t st = agent_->getXferStatus(op.req_);

    if (st == NIXL_IN_PROG) {
      // Still in progress
      return false;
    }

    // Transfer complete (NIXL_SUCCESS or error)
    if (st == NIXL_SUCCESS) {
      result.bytes_transferred = op.size_;
      result.error_code = 0;
    } else {
      result.bytes_transferred = -1;
      result.error_code = static_cast<int>(st);
    }

    // Clean up: release request and deregister memory
    if (op.req_) {
      agent_->releaseXferReq(op.req_);
    }
    agent_->deregisterMem(op.local_reg_);
    agent_->deregisterMem(op.remote_reg_);

    pending_.erase(it);
    return true;
  }

  /**
   * Close the file and clean up all NIXL resources.
   *
   * Waits for any pending operations to complete before closing.
   */
  void Close() override {
    std::lock_guard<std::mutex> lock(mutex_);

    // Complete all pending operations (spin until done)
    while (!pending_.empty()) {
      auto it = pending_.begin();
      NixlPendingOp &op = it->second;
      nixl_status_t st = agent_->getXferStatus(op.req_);
      if (st != NIXL_IN_PROG) {
        if (op.req_) {
          agent_->releaseXferReq(op.req_);
        }
        agent_->deregisterMem(op.local_reg_);
        agent_->deregisterMem(op.remote_reg_);
        pending_.erase(it);
      }
    }

    // Close file descriptor
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }

    // Reset agent (will be destroyed via unique_ptr)
    agent_.reset();
    backend_ = nullptr;
  }

  /**
   * Get the event file descriptor for epoll integration.
   *
   * NIXL does not provide an eventfd, so this returns -1.
   *
   * @return -1 (not supported by NIXL).
   */
  int GetEventFd() const override {
    return -1;
  }

 private:
  uint32_t io_depth_;                  ///< io_depth parameter (for reference)
  std::unique_ptr<nixlAgent> agent_;   ///< NIXL agent instance
  std::string agent_name_;             ///< NIXL agent name
  nixlBackendH* backend_;              ///< POSIX backend handle
  int fd_;                             ///< File descriptor
  std::string path_;                   ///< File path
  std::mutex mutex_;                   ///< Protects all mutable state
  std::unordered_map<IoToken, NixlPendingOp> pending_;  ///< In-flight ops
  std::atomic<uint64_t> next_token_;   ///< Token generator

  /**
   * Generate a unique agent name.
   *
   * Uses a static counter to ensure each instance gets a distinct name.
   *
   * @return Unique agent name string.
   */
  static std::string MakeAgentName() {
    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
    return "nixl_bdev_" + std::to_string(id);
  }

  /**
   * Submit a NIXL transfer request.
   *
   * Registers both local and remote memory, creates a transfer request,
   * posts it, and stores the pending operation for later polling.
   *
   * @param op Transfer operation type (NIXL_READ or NIXL_WRITE).
   * @param local Local (DRAM) descriptor list.
   * @param remote Remote (FILE) descriptor list.
   * @param size Size of transfer in bytes (for result reporting).
   * @return IoToken for tracking, or kInvalidIoToken on error.
   */
  IoToken SubmitXfer(nixl_xfer_op_t op,
                     nixl_xfer_dlist_t &local,
                     nixl_xfer_dlist_t &remote,
                     ssize_t size) {
    if (!agent_) {
      return kInvalidIoToken;
    }

    // Register local memory
    nixl_reg_dlist_t local_reg(DRAM_SEG);
    for (int i = 0; i < local.descCount(); ++i) {
      nixlBlobDesc bd(local[i], "");
      local_reg.addDesc(bd);
    }
    nixl_status_t st = agent_->registerMem(local_reg);
    if (st != NIXL_SUCCESS) {
      return kInvalidIoToken;
    }

    // Register remote memory
    nixl_reg_dlist_t remote_reg(remote.getType());
    for (int i = 0; i < remote.descCount(); ++i) {
      nixlBlobDesc bd(remote[i], "");
      remote_reg.addDesc(bd);
    }
    st = agent_->registerMem(remote_reg);
    if (st != NIXL_SUCCESS) {
      agent_->deregisterMem(local_reg);
      return kInvalidIoToken;
    }

    // Create transfer request
    nixlXferReqH* req = nullptr;
    st = agent_->createXferReq(op, local, remote, agent_name_, req);
    if (st != NIXL_SUCCESS || !req) {
      agent_->deregisterMem(local_reg);
      agent_->deregisterMem(remote_reg);
      return kInvalidIoToken;
    }

    // Post the request
    st = agent_->postXferReq(req);
    if (st != NIXL_SUCCESS && st != NIXL_IN_PROG) {
      agent_->releaseXferReq(req);
      agent_->deregisterMem(local_reg);
      agent_->deregisterMem(remote_reg);
      return kInvalidIoToken;
    }

    // Generate token and store pending operation
    IoToken token = next_token_.fetch_add(1, std::memory_order_relaxed);
    NixlPendingOp op_state;
    op_state.req_ = req;
    op_state.local_reg_ = local_reg;
    op_state.remote_reg_ = remote_reg;
    op_state.size_ = size;

    pending_[token] = op_state;
    return token;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_NIXL
