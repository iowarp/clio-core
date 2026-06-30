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

#ifndef CLIO_RUNTIME_INCLUDE_DYNAMIC_CONTAINER_H_
#define CLIO_RUNTIME_INCLUDE_DYNAMIC_CONTAINER_H_

#include <cstddef>
#include <string>
#include <utility>

#include "clio_runtime/types.h"

namespace clio::run {

// Forward declarations: defined in container.h / task.h. The container methods
// and the TaskResume coroutine bodies that need the full definitions live in
// dynamic_container.cc; everything else is header-inline.
class Container;
class TaskResume;

/**
 * ContainerHold - the ONLY handle through which a Container is touched.
 *
 * Wraps a raw Container* but never hands it out: the only access is the arrow
 * operator (call a method on the container). A container is built in-place via
 * the ModuleManager. ContainerHold is a cheap by-value handle (copying it copies
 * the underlying pointer); the Container object itself is owned by the
 * ModuleManager, so copies all observe the same container.
 */
class ContainerHold {
 public:
  ContainerHold() = default;

  /** Build a container in-place via the ModuleManager (defined in
   *  dynamic_container.cc). ptr_ is null if creation fails. */
  ContainerHold(const std::string &chimod_name, const PoolId &pool_id,
                const std::string &pool_name);

  /** Arrow / deref operators: the way to reach the container (call its methods).
   *  operator* yields a Container& (not a raw Container*), so no pointer leaks. */
  Container *operator->() const { return ptr_; }
  Container &operator*() const { return *ptr_; }

  explicit operator bool() const { return ptr_ != nullptr; }
  bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return ptr_ != nullptr; }
  /** Identity comparison (same underlying container). Does not expose ptr_. */
  bool operator==(const ContainerHold &o) const { return ptr_ == o.ptr_; }
  bool operator!=(const ContainerHold &o) const { return ptr_ != o.ptr_; }

  /** Destroy the held container via the ModuleManager (defined in
   *  dynamic_container.cc). The raw Container* is never exposed; this is the
   *  encapsulated teardown. */
  void Destroy(const std::string &chimod_name);

 private:
  // NOTE: deliberately NO getter/setter for ptr_. Callers reach the container
  // through operator-> (method calls) or Destroy() (teardown) — no raw
  // Container* ever leaks out of this class.
  Container *ptr_ = nullptr;
};

/**
 * DynamicContainer - the by-value handle stored in PoolInfo and RunContext.
 *
 * Holds a single ContainerHold by value. Copying a DynamicContainer (e.g. out of
 * the PoolManager into a RunContext) copies the ContainerHold, so every copy
 * observes the same ModuleManager-owned container. The per-execution PoolManager
 * map lookup ExecTask used to do is replaced by caching this handle once at
 * routing time.
 *
 * NOTE: online Upgrade/Migrate are currently dormant — we assume the container
 * is never swapped or migrated while handles are live, so no shared_ptr / lock
 * indirection is needed. The Migrate/Recover coroutines are kept (so re-enabling
 * online migration later is a localized change) but do not synchronize.
 */
class DynamicContainer {
 public:
  DynamicContainer() = default;

  /** Build a fresh handle, constructing the container in-place via the
   *  ModuleManager. (Defined in dynamic_container.cc.) */
  DynamicContainer(const std::string &chimod_name, const PoolId &pool_id,
                   const std::string &pool_name);

  /** @return true if this handle refers to a container. */
  bool IsValid() const { return static_cast<bool>(hold_); }
  explicit operator bool() const { return IsValid(); }
  bool operator==(std::nullptr_t) const { return !IsValid(); }
  bool operator!=(std::nullptr_t) const { return IsValid(); }

  /** A handle to the container for normal (read) use: dc.get()->Method(). */
  ContainerHold get() const { return hold_; }
  /** A handle to the container for the upgrade/migration path. Identical to
   *  get() while online migration is dormant; kept as a distinct entry point so
   *  the write side can be reintroduced here without touching call sites. */
  ContainerHold get_update() const { return hold_; }

  /** @return true if a migration/upgrade is in progress (always false while the
   *  migration machinery is dormant). */
  bool IsPlugged() const { return false; }

  /** Migrate this container's state to `dest_node`. Dormant: drives the
   *  container's own Migrate without draining (no concurrent migration assumed).
   *  Returns a TaskResume so callers co_await it like any runtime coroutine. */
  TaskResume Migrate(u32 dest_node);

  /** Online-upgrade / recover entry point (dormant; kept for future use). */
  TaskResume Recover();

 private:
  ContainerHold hold_;
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_DYNAMIC_CONTAINER_H_
