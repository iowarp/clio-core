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

#include <nixl.h>

#include <atomic>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "clio_ctp/util/logging.h"
#include "lightbeam.h"

namespace ctp::lbm {

/**
 * @brief NIXL-backed lightbeam transport.
 *
 * Supports two data-movement modes:
 *   1. DRAM → DRAM loopback  (same-process copy via NIXL POSIX backend)
 *   2. DRAM → FILE           (CPU-to-storage, dst_fd set in LbmContext)
 *
 * For DRAM→DRAM, the caller must pre-populate meta.recv with destination
 * buffers before calling Send.  For DRAM→FILE, LbmContext::dst_fd_ and
 * LbmContext::dst_offset_ select the target file and byte offset.
 *
 * The underlying NIXL agent is created once per NixlTransport instance and
 * uses the POSIX backend.  A static monotonically-increasing counter ensures
 * that each instance gets a unique agent name.
 */
class NixlTransport : public Transport {
 public:
  /**
   * @brief Construct a NixlTransport with an optional custom agent name.
   *
   * @param mode       TransportMode::kClient or kServer (both are loopback).
   * @param agent_name Unique name for the NIXL agent.  An empty string picks
   *                   an auto-generated name.
   */
  explicit NixlTransport(TransportMode mode,
                         const std::string& agent_name = "")
      : Transport(mode), backend_(nullptr) {
    type_ = TransportType::kNixl;
    agent_name_ = agent_name.empty() ? MakeAgentName() : agent_name;

    nixlAgentConfig cfg;
    cfg.useProgThread = false;
    cfg.useListenThread = false;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    agent_ = std::make_unique<nixlAgent>(agent_name_, cfg);

    nixl_b_params_t params;
    nixl_status_t st = agent_->createBackend("POSIX", params, backend_);
    if (st != NIXL_SUCCESS || !backend_) {
      throw std::runtime_error(
          "NixlTransport: failed to create POSIX backend, status=" +
          std::to_string(st));
    }
    HLOG(kInfo, "[NixlTransport] agent='{}' backend=POSIX", agent_name_);
  }

  ~NixlTransport() = default;

  /**
   * @brief Create a Bulk descriptor for the given buffer.
   *
   * @param ptr       Pointer to the data buffer.
   * @param data_size Size of the buffer in bytes.
   * @param flags     BULK_EXPOSE or BULK_XFER flags.
   * @return Bulk descriptor.
   */
  Bulk Expose(const ctp::ipc::FullPtr<char>& ptr, size_t data_size, u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    return bulk;
  }

  /**
   * @brief Send bulk data to the destination selected by ctx.
   *
   * If ctx.HasFileDst() is true the data is written to ctx.dst_fd_ starting
   * at ctx.dst_offset_.  Otherwise each send bulk is copied into the
   * corresponding pre-allocated recv bulk via NIXL loopback.
   *
   * @tparam MetaT    LbmMeta-compatible metadata type.
   * @param  meta     Transfer metadata; meta.send must be populated.
   * @param  ctx      Transfer context (may carry dst_fd_/dst_offset_).
   * @return 0 on success, non-zero NIXL error code on failure.
   */
  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    if (ctx.HasFileDst()) {
      return SendToFile(meta, ctx);
    }
    return SendToMem(meta, ctx);
  }

  /**
   * @brief Receive bulk data (DRAM mode only).
   *
   * For file-destination mode the recv side is the filesystem; this method
   * is a no-op.  For loopback mode the data was already placed into
   * meta.recv by Send, so this method just builds ClientInfo.
   *
   * @tparam MetaT  LbmMeta-compatible metadata type.
   * @param  meta   Transfer metadata; meta.recv must be pre-allocated.
   * @param  ctx    Transfer context.
   * @return ClientInfo with rc=0 on success.
   */
  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)meta;
    (void)ctx;
    ClientInfo info;
    info.rc = 0;
    return info;
  }

  /** @brief No-op: NIXL does not use per-message recv handles. */
  void ClearRecvHandles(LbmMeta<>& meta) { (void)meta; }

  /** @brief Returns the NIXL agent name used as the logical address. */
  std::string GetAddress() const { return agent_name_; }

  /**
   * @brief NIXL transport is always considered alive.
   *
   * @param ctx Ignored.
   * @return true always.
   */
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const {
    (void)ctx;
    return true;
  }

  /** @brief Event manager registration is not applicable for NIXL. */
  void RegisterEventManager(EventManager& em) { (void)em; }

 private:
  std::unique_ptr<nixlAgent> agent_;  ///< NIXL agent instance
  nixlBackendH* backend_;             ///< POSIX backend handle
  std::string agent_name_;            ///< Agent name (unique per instance)

  /** Generate a unique agent name using a process-wide counter. */
  static std::string MakeAgentName() {
    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
    return "lbm_nixl_agent_" + std::to_string(id);
  }

  /**
   * @brief Register DRAM buffers, perform transfer, then deregister.
   *
   * @param local_descs   Source descriptor list (DRAM_SEG).
   * @param remote_descs  Destination descriptor list (any supported type).
   * @param remote_name   Agent name for remote side (empty = loopback).
   * @return 0 on success, non-zero on error.
   */
  int RunXfer(const nixl_xfer_dlist_t& local_descs,
              const nixl_xfer_dlist_t& remote_descs,
              const std::string& remote_name) {
    // Register local memory
    nixl_reg_dlist_t local_reg(DRAM_SEG);
    for (int i = 0; i < local_descs.descCount(); ++i) {
      nixlBlobDesc bd(local_descs[i], "");
      local_reg.addDesc(bd);
    }
    nixl_status_t st = agent_->registerMem(local_reg);
    if (st != NIXL_SUCCESS) {
      HLOG(kError, "[NixlTransport] registerMem(local) failed: {}", st);
      return static_cast<int>(st);
    }

    // Register remote memory (or file; same type as remote_descs)
    nixl_reg_dlist_t remote_reg(remote_descs.getType());
    for (int i = 0; i < remote_descs.descCount(); ++i) {
      nixlBlobDesc bd(remote_descs[i], "");
      remote_reg.addDesc(bd);
    }
    st = agent_->registerMem(remote_reg);
    if (st != NIXL_SUCCESS) {
      HLOG(kError, "[NixlTransport] registerMem(remote) failed: {}", st);
      agent_->deregisterMem(local_reg);
      return static_cast<int>(st);
    }

    // Create and post transfer request
    nixlXferReqH* req = nullptr;
    std::string peer = remote_name.empty() ? agent_name_ : remote_name;
    st = agent_->createXferReq(NIXL_WRITE, local_descs, remote_descs,
                               peer, req);
    if (st != NIXL_SUCCESS || !req) {
      HLOG(kError, "[NixlTransport] createXferReq failed: {}", st);
      agent_->deregisterMem(local_reg);
      agent_->deregisterMem(remote_reg);
      return static_cast<int>(st);
    }

    st = agent_->postXferReq(req);
    if (st != NIXL_SUCCESS && st != NIXL_IN_PROG) {
      HLOG(kError, "[NixlTransport] postXferReq failed: {}", st);
      agent_->releaseXferReq(req);
      agent_->deregisterMem(local_reg);
      agent_->deregisterMem(remote_reg);
      return static_cast<int>(st);
    }

    // Poll until complete
    while (st == NIXL_IN_PROG) {
      st = agent_->getXferStatus(req);
    }

    int rc = (st == NIXL_SUCCESS) ? 0 : static_cast<int>(st);

    agent_->releaseXferReq(req);
    agent_->deregisterMem(local_reg);
    agent_->deregisterMem(remote_reg);
    return rc;
  }

  /**
   * @brief Send bulk data to a file descriptor (DRAM → FILE).
   *
   * Each BULK_XFER entry in meta.send is written sequentially to
   * ctx.dst_fd_ starting at ctx.dst_offset_.
   *
   * @tparam MetaT  LbmMeta-compatible metadata type.
   * @param  meta   Source metadata.
   * @param  ctx    Context carrying dst_fd_ and dst_offset_.
   * @return 0 on success, non-zero on error.
   */
  template <typename MetaT>
  int SendToFile(MetaT& meta, const LbmContext& ctx) {
    size_t file_offset = ctx.dst_offset_;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      const Bulk& b = meta.send[i];
      if (!b.flags.Any(BULK_XFER)) {
        continue;
      }
      nixl_xfer_dlist_t src(DRAM_SEG);
      src.addDesc(nixlBasicDesc(
          reinterpret_cast<uintptr_t>(b.data.ptr_), b.size, /*devId=*/0));

      nixl_xfer_dlist_t dst(FILE_SEG);
      dst.addDesc(nixlBasicDesc(
          /*addr=file offset*/ static_cast<uintptr_t>(file_offset),
          b.size,
          static_cast<uint64_t>(ctx.dst_fd_)));

      int rc = RunXfer(src, dst, /*loopback*/ "");
      if (rc != 0) {
        HLOG(kError, "[NixlTransport] SendToFile bulk {} failed: {}", i, rc);
        return rc;
      }
      file_offset += b.size;
    }
    return 0;
  }

  /**
   * @brief Send bulk data to pre-allocated DRAM receive buffers (loopback).
   *
   * The NIXL POSIX backend only supports DRAM→FILE transfers, so DRAM→DRAM
   * loopback is performed via direct memcpy.  meta.recv must already contain
   * destination buffers (one per BULK_XFER entry in meta.send).  If meta.recv
   * is empty it is populated with newly heap-allocated buffers.
   *
   * @tparam MetaT  LbmMeta-compatible metadata type.
   * @param  meta   Transfer metadata.
   * @param  ctx    Transfer context (unused in loopback mode).
   * @return 0 on success.
   */
  template <typename MetaT>
  int SendToMem(MetaT& meta, const LbmContext& ctx) {
    (void)ctx;

    // Auto-populate recv entries if not already present
    if (meta.recv.empty()) {
      for (size_t i = 0; i < meta.send.size(); ++i) {
        const Bulk& sb = meta.send[i];
        Bulk rb;
        rb.size = sb.size;
        rb.flags = sb.flags;
        if (sb.flags.Any(BULK_XFER)) {
          char* dst_ptr = new char[sb.size];
          rb.data = ctp::ipc::FullPtr<char>(dst_ptr);
          rb.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
        } else {
          rb.data = ctp::ipc::FullPtr<char>::GetNull();
        }
        meta.recv.push_back(rb);
      }
    }

    // Use memcpy for DRAM→DRAM (NIXL POSIX backend requires FILE_SEG dest)
    for (size_t i = 0; i < meta.send.size(); ++i) {
      const Bulk& sb = meta.send[i];
      if (!sb.flags.Any(BULK_XFER)) {
        continue;
      }
      Bulk& rb = meta.recv[i];
      std::memcpy(rb.data.ptr_, sb.data.ptr_, sb.size);
    }
    return 0;
  }
};

}  // namespace ctp::lbm

#endif  // CTP_ENABLE_NIXL
