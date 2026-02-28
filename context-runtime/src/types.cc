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
 * Type implementations
 */

#include "chimaera/types.h"
#include "chimaera/worker.h"
#include <sstream>
#include <stdexcept>

namespace chi {

UniqueId UniqueId::FromString(const std::string& str) {
  // Parse format "major.minor"
  size_t dot_pos = str.find('.');
  if (dot_pos == std::string::npos) {
    throw std::invalid_argument("Invalid UniqueId format, expected 'major.minor'");
  }

  try {
    u32 major = std::stoul(str.substr(0, dot_pos));
    u32 minor = std::stoul(str.substr(dot_pos + 1));
    return UniqueId(major, minor);
  } catch (const std::exception& e) {
    throw std::invalid_argument("Failed to parse UniqueId: " + std::string(e.what()));
  }
}

std::string UniqueId::ToString() const {
  return std::to_string(major_) + "." + std::to_string(minor_);
}

LockOwnerId GetCurrentLockOwnerId() {
  LockOwnerId id;
  Worker *worker = CHI_CUR_WORKER;
  if (!worker) return id;
  FullPtr<Task> task = worker->GetCurrentTask();
  if (task.ptr_ == nullptr) return id;
  id.worker_id_ = worker->GetId();
  id.pid_ = task->task_id_.pid_;
  id.tid_ = task->task_id_.tid_;
  id.major_ = task->task_id_.major_;
  id.node_id_ = task->task_id_.node_id_;
  return id;
}

}  // namespace chi
