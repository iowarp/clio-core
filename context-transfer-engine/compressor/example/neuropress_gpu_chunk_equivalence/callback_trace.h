/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file callback_trace.h
 * @brief The callback trace: what ran, in what order, on which GPU buffer.
 *
 * WHY THIS SHAPE -- read before adding a field.
 *
 * Native NeuroPress has NO callback, observer or hook API. That was checked
 * first, as the harness requires: `gpucompress.h` exposes no registration
 * function of any kind. What it DOES expose is a per-chunk diagnostics record
 * (`gpucompress_chunk_diag_t`, plus reset/count/get), written by
 * `recordChunkDiagnostic()` at the end of `gpucompress_compress_with_action_gpu`
 * -- so it is populated on the direct GPU compress path, not only under the
 * HDF5 VOL. Alongside it, every pipeline stage is a separately exported entry
 * point of the shared library (`runStatsKernelsNoSync`, `gpucompress_infer_gpu`,
 * `quantize_simple`, `byte_shuffle_simple`,
 * `gpucompress_compress_with_action_gpu`).
 *
 * So this harness does NOT invent a parallel callback system. It uses:
 *
 *   1. NATIVE'S OWN STAGE ENTRY POINTS as the callback boundaries. The stages
 *      below are exactly the functions upstream's own AUTO path calls, in the
 *      order it calls them (gpucompress_compress.cpp).
 *   2. NATIVE'S OWN DIAGNOSTICS RECORD for the diagnostics stage, read through
 *      the public `gpucompress_get_chunk_diag`, and NATIVE'S OWN ranking
 *      outputs (`out_top_actions`, `out_predicted_costs`, `NNDebugPerConfig`)
 *      for inference and ranking. No approximation of either is constructed.
 *   3. CUPTI's runtime-API callbacks for the transfer and kernel record. This
 *      is the smallest possible instrumentation: it observes the REAL CUDA
 *      calls both implementations make, in-process, without a line of either
 *      being modified. It is what makes "did this stage actually run on the
 *      GPU" and "did anything sneak a D->H copy in here" answerable rather
 *      than assumed.
 *
 * Clio's side is traced through the same stage taxonomy, calling the same
 * production functions its compressor runtime calls (compressor_runtime.cc:
 * 641-725 for statistics/inference/ranking, :2346-2508 for quantize, shuffle
 * and the codec).
 */
#ifndef CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_CALLBACK_TRACE_H_
#define CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_CALLBACK_TRACE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace npeq {

/** @brief Which implementation produced an entry. */
enum class Side { kNative, kClio };
const char *SideName(Side side);

/**
 * @brief Whether a stage is part of NeuroPress's chunk processing, or one of
 *        Clio's own architectural operations.
 *
 * Only kClioArchitectural entries are removed by normalization, and the list
 * of stages that may carry that classification is closed (see KindOf). No
 * NeuroPress functional stage can ever be normalized away -- that would be
 * masking a real difference, which the comparison explicitly refuses to do.
 */
enum class StageKind { kNeuroPressFunctional, kClioArchitectural };

/**
 * @brief The callback (stage) boundaries.
 *
 * Order of the enumerators is the order the pipeline runs in. Note QUANTIZE
 * BEFORE SHUFFLE: that is the real order on both sides
 * (gpucompress_compress.cpp and compressor_runtime.cc:2346-2465), and
 * writing the taxonomy in the order the code actually executes is the whole
 * point -- an idealized order would hide an ordering divergence rather than
 * detect one.
 */
enum class Stage {
  kStatistics,
  kDiagnostics,
  kInference,
  kRanking,
  kSelection,
  kQuantization,
  kShuffle,
  kCompression,
  // Clio architectural operations. These have no NeuroPress counterpart.
  kClioAllocateDeviceBuffer,
  kClioAttachIpcMetadata,
  kClioRouteChunk,
  kClioReleaseDeviceBuffer,
};

const char *StageName(Stage stage);
StageKind KindOf(Stage stage);

/** @brief Whether a stage ran, was legitimately skipped, or failed. */
enum class Status { kExecuted, kSkippedNotSelected, kFailed };
const char *StatusName(Status status);

/** @brief Where a buffer lives, as verified by the driver (not assumed). */
enum class MemLoc { kUnknown, kDevice, kHost, kNone };
const char *MemLocName(MemLoc loc);

/**
 * @brief One CUDA transfer, as CUPTI saw it.
 *
 * `harness` is the field requirement 21 turns on: a copy this test made to
 * report a result is not a copy the production pipeline made, and conflating
 * the two would turn the harness into the very thing it is measuring.
 */
struct TransferRecord {
  std::string direction;  /**< H2D / D2H / D2D / H2H / DEFAULT */
  size_t bytes = 0;
  bool async = false;
  bool harness = false;
};

/** @brief A kernel launch, by symbol name, as CUPTI saw it. */
struct KernelRecord {
  std::string name;
  int count = 0;
};

/** @brief The three NN selection features. */
struct StatsPayload {
  bool valid = false;
  double entropy = 0.0;
  double mad = 0.0;
  double second_derivative = 0.0;
};

/** @brief Per-candidate network outputs, indexed by upstream ACTION id. */
struct InferencePayload {
  bool valid = false;
  int num_candidates = 0;
  std::array<bool, 32> present{};
  std::array<float, 32> ratio{};
  std::array<float, 32> comp_ms{};
  std::array<float, 32> decomp_ms{};
  std::array<float, 32> psnr_db{};
};

/**
 * @brief The ranked action order and the per-action cost it came from.
 *
 * `order` holds ACTION IDS best-first. Both sides can express this natively:
 * upstream returns `out_top_actions[32]`, and Clio's candidate slot index IS
 * the action id by construction (neuropress_bridge.cc:224-265 sorts the
 * candidate list into action order precisely so that slot == action).
 */
struct RankingPayload {
  bool valid = false;
  std::vector<int> order;
  std::array<bool, 32> cost_present{};
  std::array<double, 32> cost{};
};

/** @brief The winning configuration, decoded. */
struct SelectionPayload {
  bool valid = false;
  int action = -1;
  int algo = -1;
  bool quantize = false;
  bool shuffle = false;
  std::string algo_name;
};

/** @brief Everything a reader needs to invert a quantization. */
struct QuantPayload {
  bool valid = false;
  int precision = 0;
  double error_bound = 0.0;
  double effective_error_bound = 0.0;
  double scale = 0.0;
  double data_min = 0.0;
  double data_max = 0.0;
  bool bound_achievable = true;
};

/** @brief The codec's own result. */
struct CompressionPayload {
  bool valid = false;
  std::string algo_name;
  size_t compressed_bytes = 0;
};

/**
 * @brief Native's per-chunk diagnostics record, in the fields both sides can
 *        be compared on.
 *
 * Deliberately NOT a copy of all ~80 fields of `gpucompress_chunk_diag_t`.
 * Most of those are timings, which are not an equivalence criterion, and
 * exploration/SGD state, which is off in this harness. The fields kept are the
 * ones that describe WHAT WAS DECIDED, which is what has to match.
 */
struct DiagnosticsPayload {
  bool valid = false;
  bool native_record_present = false;  /**< A real gpucompress_chunk_diag_t */
  int nn_action = -1;
  int nn_original_action = -1;
  int exploration_triggered = 0;
  int sgd_fired = 0;
  float feat_entropy = 0.0f;
  float feat_mad = 0.0f;
  float feat_deriv = 0.0f;
  float predicted_ratio = 0.0f;
  float predicted_comp_time = 0.0f;
  float predicted_decomp_time = 0.0f;
  float predicted_psnr = 0.0f;
  int predicted_ranking_count = 0;
  std::array<int, 32> predicted_ranking{};
};

/**
 * @brief One traced callback.
 *
 * Every field here is either populated from a real source on at least one side
 * or left explicitly invalid. Nothing is invented to fill the struct out --
 * a field that cannot be sourced is absent from the schema rather than
 * fabricated (requirement 5).
 */
struct TraceEntry {
  uint64_t sequence = 0;
  int chunk_id = -1;
  Side side = Side::kNative;
  Stage stage = Stage::kStatistics;
  StageKind kind = StageKind::kNeuroPressFunctional;
  Status status = Status::kExecuted;

  const void *input_ptr = nullptr;
  const void *output_ptr = nullptr;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  MemLoc input_loc = MemLoc::kUnknown;
  MemLoc output_loc = MemLoc::kUnknown;

  bool input_hash_valid = false;
  bool output_hash_valid = false;
  uint64_t input_hash = 0;
  uint64_t output_hash = 0;

  std::string datatype = "float32";
  int element_size = 4;
  int cuda_device = -1;
  const void *stream = nullptr;

  double entry_time_ms = 0.0;
  double exit_time_ms = 0.0;

  StatsPayload stats;
  DiagnosticsPayload diagnostics;
  InferencePayload inference;
  RankingPayload ranking;
  SelectionPayload selection;
  QuantPayload quant;
  CompressionPayload compression;

  std::vector<TransferRecord> transfers;
  std::vector<KernelRecord> kernels;

  std::string note;

  /** @brief Total production (non-harness) bytes moved in a direction. */
  size_t TransferBytes(const char *direction, bool harness) const;
  /** @brief Count of production (non-harness) transfers in a direction. */
  int TransferCount(const char *direction, bool harness) const;
  int KernelLaunches() const;
};

/** @brief One side's trace for one chunk. */
struct ChunkTrace {
  int chunk_id = -1;
  Side side = Side::kNative;
  std::string regime;
  size_t chunk_bytes = 0;
  double error_bound = 0.0;
  bool input_device_verified = false;
  uint64_t input_hash = 0;
  /**
   * A deque, not a vector, and that is load-bearing: StageScope holds a
   * TraceEntry* across later Begin() calls (the CUPTI callback writes through
   * it), and vector reallocation would leave that pointer dangling.
   */
  std::deque<TraceEntry> entries;

  /** @brief Entries with Clio's architectural stages removed. */
  std::vector<const TraceEntry *> Normalized() const;
  const TraceEntry *Find(Stage stage) const;
};

/**
 * @brief Collects entries and owns the CUPTI subscription.
 *
 * Attribution rule: a CUDA call is attributed to the stage open ON THE CALLING
 * THREAD. Clio's runtime runs in-process here (CLIO_INIT with
 * default_with_runtime = true), so its worker threads would otherwise dump
 * unrelated CUDA activity into whatever stage the main thread had open. When a
 * stage is opened in "global" mode -- used to bracket the end-to-end
 * production run, whose work genuinely happens on other threads -- attribution
 * falls back to that stage for threads with nothing open of their own.
 */
class Recorder {
 public:
  static Recorder &Instance();

  /** @brief Start CUPTI. Returns false (and traces without transfer/kernel
   *  records) if CUPTI is unavailable, rather than failing the run. */
  bool StartInstrumentation();
  void StopInstrumentation();
  bool InstrumentationActive() const { return cupti_active_; }
  const std::string &InstrumentationError() const { return cupti_error_; }

  void BeginChunk(int chunk_id, Side side, const std::string &regime,
                  size_t chunk_bytes, double error_bound);
  ChunkTrace EndChunk();

  /** @brief Open a stage on this thread. */
  TraceEntry *Begin(Stage stage);
  /** @brief Open a stage that also captures other threads' CUDA activity. */
  TraceEntry *BeginGlobal(Stage stage);
  void End(TraceEntry *entry);

  /** @brief Record a stage that did not run, with the reason. */
  void RecordSkipped(Stage stage, const std::string &why);

  /**
   * @brief Mark subsequent CUDA activity on this thread as harness-owned.
   *
   * Wrapped around the hashing and comparison probes so their copies land in
   * the trace as TEST_HARNESS_TRANSFER rather than being mistaken for
   * something the pipeline did -- or, worse, being hidden.
   */
  void PushHarness();
  void PopHarness();

  /** @brief Called from the CUPTI callback. */
  void NoteTransfer(int cuda_memcpy_kind, size_t bytes, bool async,
                    const void *dst, const void *src);
  void NoteKernel(const char *symbol);

  size_t HarnessTransferBytes() const { return harness_bytes_; }
  int HarnessTransferCount() const { return harness_count_; }

 private:
  Recorder() = default;

  ChunkTrace current_;
  uint64_t sequence_ = 0;
  bool cupti_active_ = false;
  std::string cupti_error_;
  void *subscriber_ = nullptr;
  size_t harness_bytes_ = 0;
  int harness_count_ = 0;
};

/** @brief RAII wrapper around Recorder::Begin/End. */
class StageScope {
 public:
  StageScope(Stage stage, const void *input_ptr, size_t input_bytes);
  /** @brief Bracket work that happens on other threads too. */
  StageScope(Stage stage, const void *input_ptr, size_t input_bytes,
             bool global);
  ~StageScope();

  StageScope(const StageScope &) = delete;
  StageScope &operator=(const StageScope &) = delete;

  TraceEntry *operator->() { return entry_; }
  TraceEntry *get() { return entry_; }

  /** @brief Set the stage's output and hash it (as a harness operation). */
  void SetOutput(const void *ptr, size_t bytes);
  void Fail(const std::string &why);

 private:
  TraceEntry *entry_ = nullptr;
};

/** @brief Hash a device buffer, accounting the probe as harness activity. */
bool HashAsHarness(const void *device_ptr, size_t bytes, uint64_t *out);

/** @brief Serialize one side's traces to JSON. */
bool WriteTraceJson(const std::string &path,
                    const std::vector<ChunkTrace> &traces);

}  // namespace npeq

#endif  // CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_CALLBACK_TRACE_H_
