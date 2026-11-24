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

#ifndef CLIO_CAE_CORE_PATH_POLICY_H_
#define CLIO_CAE_CORE_PATH_POLICY_H_

#include <string>
#include <vector>

namespace clio::cae::core {

/**
 * Which local paths CAE may read (file:: and hdf5:: sources) and write
 * (ExportData output).
 *
 * CAE opens these as the runtime user, at paths taken from the client's task.
 * The policy comes from the runtime's own configuration, never from a task:
 * the `allowed_roots` list on the `clio_cae_core` compose entry.
 *
 * - With `allowed_roots` set, a path must resolve to somewhere under one of
 *   them. Without it, any location passes this step.
 * - In both cases, credential and system locations are refused: any path
 *   through a directory such as `.ssh`, `.aws`, `.gnupg`, `.config` or
 *   `.clio`; files such as `.netrc`; and `/proc`, `/sys`, `/dev` and
 *   `/etc/shadow`. Writes are also refused for any hidden file or directory
 *   and for system directories, so an export cannot replace a dotfile. This
 *   denylist is a backstop; `allowed_roots` is the control.
 *
 * Paths are made absolute and resolved through symlinks (as far as they
 * exist) before the check, so `..` and links cannot step outside a root.
 */
class PathPolicy {
 public:
  /** The policy loaded from the runtime's compose configuration. */
  static const PathPolicy &Get();

  /** A policy with these allowed roots (empty: the default denylist). */
  explicit PathPolicy(const std::vector<std::string> &allowed_roots);

  /**
   * Whether a path may be read or written.
   * @param path The path as the task gave it
   * @param write True for an output path
   * @param reason Set to why the path was refused
   * @return true if allowed
   */
  bool Allows(const std::string &path, bool write, std::string &reason) const;

  /** The configured roots, resolved (empty: the default denylist applies). */
  const std::vector<std::string> &Roots() const { return roots_; }

 private:
  std::vector<std::string> roots_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_PATH_POLICY_H_
