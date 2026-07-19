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

#ifndef WRPCTE_CORE_DATA_ORGANIZER_DATA_ORGANIZER_H_
#define WRPCTE_CORE_DATA_ORGANIZER_DATA_ORGANIZER_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_tasks.h>

#include <memory>
#include <string>
#include <vector>

// Data organizer interface + factory (issue #738). Each organization POLICY
// lives in its own header/source pair in this directory (e.g.
// frecency_organizer.h) — add a new policy by adding a new pair and
// registering it in DataOrganizerFactory::Get
// (src/data_organizer/data_organizer.cc).

namespace clio::cte::core {

// Forward declaration — organizers hold only a Runtime* in their interface;
// the implementations (src/data_organizer/*.cc) include core_runtime.h.
class Runtime;

/**
 * A read-only snapshot of one blob's organizer-relevant metadata, taken by
 * Runtime::CollectOrganizerBlobStats. Organizers score blobs from this
 * snapshot instead of holding references into the live blob map, so the
 * (potentially long) rescoring loop never pins map entries across co_awaits.
 */
struct OrganizerBlobStat {
  TagId tag_id_;               // Owning tag
  std::string blob_name_;      // Blob name within the tag
  float score_;                // Current placement score (0-1)
  Timestamp last_modified_;    // Last write, steady-clock ns
  Timestamp last_read_;        // Last read, steady-clock ns
  clio::run::u64 access_count_;  // Put+Get data ops since creation
  clio::run::u64 size_;          // Logical size in bytes

  OrganizerBlobStat()
      : tag_id_(TagId::GetNull()),
        score_(0.0f),
        last_modified_(0),
        last_read_(0),
        access_count_(0),
        size_(0) {}
};

/**
 * Abstract data organizer (issue #738).
 *
 * A DataOrganizer is the CTE's internal, periodically-driven reorganization
 * engine. The periodic DynamicReorganize task delegates each firing to
 * Reorganize(), which owns ALL organization logic: deciding which blobs to
 * rescore (event queues, access statistics, model-driven prefetching, ...)
 * and applying the new scores through the server's ReorganizeBlob logic
 * directly — it must NOT spawn per-blob ReorganizeBlobTasks.
 *
 * Reorganize is a coroutine (returns TaskResume) so implementations can
 * CLIO_CO_AWAIT the server's data-movement helpers without blocking the
 * worker thread.
 */
class DataOrganizer {
 public:
  virtual ~DataOrganizer() = default;

  /**
   * Run one round of reorganization.
   * @param server The CTE runtime container to organize
   * @param replica_id Which of the `organizer_tasks` periodic replicas is
   *        invoking us (0-based); implementations use it to partition the
   *        blob space so replicas parallelize instead of duplicating work
   */
  virtual clio::run::TaskResume Reorganize(Runtime *server,
                                           clio::run::u32 replica_id) = 0;

  /** Organizer name, as it appears in the CTE config. */
  virtual std::string GetName() const = 0;
};

/**
 * Data organizer factory: maps the `organizer:` config name to an instance.
 */
class DataOrganizerFactory {
 public:
  /**
   * Create the organizer registered under `name`.
   * @param name Organizer name from the CTE config ("frecency", ...)
   * @return New organizer instance, or nullptr for "none"/""/unknown names
   *         (unknown names are logged)
   */
  static std::unique_ptr<DataOrganizer> Get(const std::string &name);
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_DATA_ORGANIZER_DATA_ORGANIZER_H_
