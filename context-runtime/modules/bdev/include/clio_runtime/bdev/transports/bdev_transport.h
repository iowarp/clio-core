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

#ifndef CLIO_BDEV_TRANSPORT_H_
#define CLIO_BDEV_TRANSPORT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <vector>
#include <memory>
#include <string>

namespace clio::run::bdev {

class Runtime;

/**
 * Abstract factory class for block storage devices
 */
class BdevTransport {
 public:
  virtual ~BdevTransport() = default;

  /**
   * Initialize the transport
   * @param params Creation parameters
   * @param pool_name Pool name; doubles as the file path (kFile) or S3 bucket
   *   name (kS3). Carried on the create task, not in CreateParams.
   * @param runtime Pointer to the parent bdev runtime
   * @return true if successful
   */
  virtual bool Init(const CreateParams& params, const std::string& pool_name,
                    Runtime* runtime) = 0;

  /**
   * Destroy the transport and release resources
   */
  virtual void Destroy() = 0;

  /**
   * Allocate blocks of a given size
   * @param size Total size to allocate
   * @param worker_id ID of the requesting worker
   * @param blocks Output vector to populate with allocated blocks
   * @return true if successful
   */
  virtual bool AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) = 0;

  /**
   * Free previously allocated blocks
   * @param worker_id ID of the requesting worker
   * @param blocks Blocks to free
   */
  virtual void FreeBlocks(int worker_id, const std::vector<Block>& blocks) = 0;

  /**
   * Write data to the storage blocks
   * @param task The write task containing data and target blocks
   * @param rctx The coroutine context
   * @return TaskResume to yield or complete the task
   */
  virtual clio::run::TaskResume WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, clio::run::RunContext &rctx) = 0;

  /**
   * Read data from the storage blocks
   * @param task The read task containing dest buffer and source blocks
   * @param rctx The coroutine context
   * @return TaskResume to yield or complete the task
   */
  virtual clio::run::TaskResume ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, clio::run::RunContext &rctx) = 0;

  /**
   * Get the total capacity of the storage device
   */
  virtual clio::run::u64 GetCapacity() const = 0;

  /**
   * Get the remaining allocatable size
   */
  virtual clio::run::u64 GetRemainingSize() const = 0;
};

/**
 * Factory for creating BdevTransport instances
 */
class BdevTransportFactory {
 public:
  static std::unique_ptr<BdevTransport> Create(BdevType type);
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_TRANSPORT_H_
