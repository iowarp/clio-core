/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file path_tracer.h
 * @brief Chronological record of where a chunk's bytes actually travel.
 *
 * The sibling neuropress_gpu_chunk_equivalence harness answers "do native and
 * Clio agree, callback by callback". This answers a different question, about
 * Clio alone:
 *
 *   A chunk is generated in GPU memory and handed to Clio. What moves, in what
 *   order, in which direction, on which thread -- and does the payload stay on
 *   the device until it has been compressed?
 *
 * So this is a TIMELINE, not a comparison. Every cudaMemcpy and kernel launch
 * is recorded in the order it happened, tagged with the phase that was open and
 * with which named memory region each end of the copy falls in, so a line reads
 *
 *   D2D 16.0 MiB  [chunk0.src] -> [chunk0.ipc]        during stage-to-ipc
 *
 * rather than just "a copy happened". Region tagging is what turns a list of
 * transfers into a data path.
 *
 * Instrumentation is CUPTI's runtime-API callbacks, so this observes the REAL
 * calls the runtime makes with no change to it. Clio's runtime is started
 * in-process (CLIO_INIT with default_with_runtime = true), which is what makes
 * its worker threads visible here; transfers are captured from every thread and
 * the thread id is recorded.
 */
#ifndef CLIO_CTE_COMPRESSOR_EXAMPLE_NP_PATH_TRACER_H_
#define CLIO_CTE_COMPRESSOR_EXAMPLE_NP_PATH_TRACER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nppath {

/** @brief What kind of event a timeline entry is. */
enum class EventKind { kPhase, kTransfer, kKernel, kAlloc, kFree, kNote };

/** @brief Where a pointer lives, as the driver reports it. */
enum class Loc { kUnknown, kDevice, kHost, kManaged };
const char *LocName(Loc loc);

/**
 * @brief A named span of memory the driver told us about.
 *
 * Without this a trace can say "a 16 MiB D2H happened" but not "the chunk left
 * the device", which is the only form of the statement that answers the
 * question. Regions are registered by the driver as it allocates them.
 */
struct Region {
  std::string name;
  const void *base = nullptr;
  size_t bytes = 0;
  Loc loc = Loc::kUnknown;
};

/** @brief One thing that happened, in order. */
struct Event {
  uint64_t seq = 0;
  double t_ms = 0.0;
  EventKind kind = EventKind::kNote;
  std::string phase;
  unsigned long thread = 0;

  // Transfers
  std::string direction;   /**< H2D / D2H / D2D / H2H */
  size_t bytes = 0;
  bool async = false;
  const void *src = nullptr;
  const void *dst = nullptr;
  std::string src_region;  /**< named region, or "" */
  std::string dst_region;
  Loc src_loc = Loc::kUnknown;
  Loc dst_loc = Loc::kUnknown;

  // Kernels
  std::string symbol;

  // Alloc/free
  size_t alloc_bytes = 0;

  /** Set for copies this harness itself made, never for the pipeline's. */
  bool harness = false;
  std::string text;
};

/**
 * @brief Collects the timeline and owns the CUPTI subscription.
 *
 * Single global instance: CUPTI's subscription is process-wide and the point is
 * to see everything, including work the runtime does on its own threads.
 */
class Tracer {
 public:
  static Tracer &Instance();

  bool Start(std::string *error);
  void Stop();
  bool Active() const { return active_; }

  /** @brief Name a span of memory so transfers touching it are identifiable. */
  void AddRegion(const std::string &name, const void *base, size_t bytes);
  void DropRegion(const void *base);

  /** @brief Open a named phase. Everything after this is tagged with it. */
  void BeginPhase(const std::string &name);
  void EndPhase();
  void Note(const std::string &text);

  /** @brief Bracket copies this harness makes, so they are never miscounted. */
  void PushHarness();
  void PopHarness();

  /** @brief Called from the CUPTI callback. */
  void OnTransfer(int kind, size_t bytes, bool async, const void *dst,
                  const void *src);
  void OnKernel(const char *symbol);
  void OnAlloc(size_t bytes);
  void OnFree();

  const std::vector<Event> &Events() const { return events_; }
  const std::vector<Region> &Regions() const { return regions_; }

  /** @brief Bytes moved in a direction, production only. */
  size_t Bytes(const std::string &direction) const;
  int Count(const std::string &direction) const;

 private:
  Tracer() = default;
  Event *Append(EventKind kind);

  std::vector<Event> events_;
  std::vector<Region> regions_;
  std::vector<std::string> phase_stack_;
  uint64_t seq_ = 0;
  bool active_ = false;
  void *subscriber_ = nullptr;
};

/** @brief RAII phase marker. */
class Phase {
 public:
  explicit Phase(const std::string &name) { Tracer::Instance().BeginPhase(name); }
  ~Phase() { Tracer::Instance().EndPhase(); }
  Phase(const Phase &) = delete;
  Phase &operator=(const Phase &) = delete;
};

/**
 * @brief Write the timeline and the residency verdict.
 *
 * @param payload_bytes One chunk's size, used to decide which transfers are
 *   big enough to BE the payload rather than metadata.
 */
bool WriteReport(const std::string &path, size_t payload_bytes,
                 size_t num_chunks);
void PrintReport(std::ostream &os, size_t payload_bytes, size_t num_chunks);
bool WriteTimelineJson(const std::string &path);

}  // namespace nppath

#endif  // CLIO_CTE_COMPRESSOR_EXAMPLE_NP_PATH_TRACER_H_
