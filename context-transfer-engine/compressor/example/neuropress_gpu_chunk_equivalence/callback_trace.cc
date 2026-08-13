/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file callback_trace.cc
 * @brief Trace recording and the CUPTI subscription behind it.
 */

#include "callback_trace.h"

#include <cuda_runtime.h>
#include <cupti.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>

#include "device_probe.h"

namespace npeq {

namespace {

/**
 * The stage open on THIS thread, if any. Thread-local because Clio's runtime
 * runs in-process and its workers would otherwise attribute their CUDA calls
 * to whatever the main thread happened to have open.
 */
thread_local TraceEntry *t_current = nullptr;
/** Nesting depth of PushHarness on this thread. */
thread_local int t_harness_depth = 0;
/** Re-entry guard: CUDA calls made from inside a CUPTI callback would recurse. */
thread_local bool t_in_callback = false;

/** Stage that also captures other threads. Guarded by g_mutex. */
TraceEntry *g_global = nullptr;
std::mutex g_mutex;

double NowMs() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point origin = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - origin)
      .count();
}

const char *DirectionName(int kind) {
  switch (kind) {
    case cudaMemcpyHostToHost: return "H2H";
    case cudaMemcpyHostToDevice: return "H2D";
    case cudaMemcpyDeviceToHost: return "D2H";
    case cudaMemcpyDeviceToDevice: return "D2D";
    default: return "DEFAULT";
  }
}

/**
 * Resolve cudaMemcpyDefault into a real direction.
 *
 * A `DEFAULT` copy that is really a D->H is exactly the thing this harness
 * exists to catch, so leaving it unclassified would defeat the purpose. Safe
 * to call CUDA here only because the caller holds the re-entry guard.
 */
const char *ResolveDefault(const void *dst, const void *src) {
  const PointerKind d = ClassifyPointer(dst);
  const PointerKind s = ClassifyPointer(src);
  const bool dd = (d == PointerKind::kDevice);
  const bool sd = (s == PointerKind::kDevice);
  if (dd && sd) return "D2D";
  if (dd && !sd) return "H2D";
  if (!dd && sd) return "D2H";
  return "H2H";
}

void CUPTIAPI TraceCallback(void * /*userdata*/, CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid, const void *cbdata) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
  if (t_in_callback) return;
  const auto *cb = static_cast<const CUpti_CallbackData *>(cbdata);
  if (cb->callbackSite != CUPTI_API_ENTER) return;

  t_in_callback = true;
  switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020: {
      const auto *p =
          static_cast<const cudaMemcpy_v3020_params *>(cb->functionParams);
      Recorder::Instance().NoteTransfer(p->kind, p->count, false, p->dst,
                                        p->src);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
      const auto *p =
          static_cast<const cudaMemcpyAsync_v3020_params *>(cb->functionParams);
      Recorder::Instance().NoteTransfer(p->kind, p->count, true, p->dst,
                                        p->src);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000: {
      Recorder::Instance().NoteKernel(cb->symbolName);
      break;
    }
    default:
      break;
  }
  t_in_callback = false;
}

}  // namespace

const char *SideName(Side side) {
  return side == Side::kNative ? "native" : "clio";
}

const char *StageName(Stage stage) {
  switch (stage) {
    case Stage::kStatistics: return "statistics";
    case Stage::kDiagnostics: return "diagnostics";
    case Stage::kInference: return "inference";
    case Stage::kRanking: return "ranking";
    case Stage::kSelection: return "selection";
    case Stage::kQuantization: return "quantization";
    case Stage::kShuffle: return "shuffle";
    case Stage::kCompression: return "compression";
    case Stage::kClioAllocateDeviceBuffer: return "clio_allocate_device_buffer";
    case Stage::kClioAttachIpcMetadata: return "clio_attach_ipc_metadata";
    case Stage::kClioRouteChunk: return "clio_route_chunk";
    case Stage::kClioReleaseDeviceBuffer: return "clio_release_device_buffer";
  }
  return "unknown";
}

StageKind KindOf(Stage stage) {
  switch (stage) {
    case Stage::kClioAllocateDeviceBuffer:
    case Stage::kClioAttachIpcMetadata:
    case Stage::kClioRouteChunk:
    case Stage::kClioReleaseDeviceBuffer:
      return StageKind::kClioArchitectural;
    default:
      // Everything else is NeuroPress functional and can never be normalized
      // away. Listed as the default on purpose: a stage added later is
      // functional until someone deliberately classifies it otherwise.
      return StageKind::kNeuroPressFunctional;
  }
}

const char *StatusName(Status status) {
  switch (status) {
    case Status::kExecuted: return "executed";
    case Status::kSkippedNotSelected: return "skipped-not-selected";
    case Status::kFailed: return "failed";
  }
  return "unknown";
}

const char *MemLocName(MemLoc loc) {
  switch (loc) {
    case MemLoc::kDevice: return "DEVICE";
    case MemLoc::kHost: return "HOST";
    case MemLoc::kNone: return "NONE";
    default: return "UNKNOWN";
  }
}

size_t TraceEntry::TransferBytes(const char *direction, bool harness) const {
  size_t total = 0;
  for (const auto &t : transfers) {
    if (t.harness == harness && t.direction == direction) total += t.bytes;
  }
  return total;
}

int TraceEntry::TransferCount(const char *direction, bool harness) const {
  int n = 0;
  for (const auto &t : transfers) {
    if (t.harness == harness && t.direction == direction) ++n;
  }
  return n;
}

int TraceEntry::KernelLaunches() const {
  int n = 0;
  for (const auto &k : kernels) n += k.count;
  return n;
}

std::vector<const TraceEntry *> ChunkTrace::Normalized() const {
  std::vector<const TraceEntry *> out;
  for (const auto &e : entries) {
    if (e.kind == StageKind::kClioArchitectural) continue;
    out.push_back(&e);
  }
  return out;
}

const TraceEntry *ChunkTrace::Find(Stage stage) const {
  for (const auto &e : entries) {
    if (e.stage == stage) return &e;
  }
  return nullptr;
}

Recorder &Recorder::Instance() {
  static Recorder instance;
  return instance;
}

bool Recorder::StartInstrumentation() {
  if (cupti_active_) return true;
  CUpti_SubscriberHandle handle{};
  CUptiResult rc = cuptiSubscribe(&handle, TraceCallback, nullptr);
  if (rc != CUPTI_SUCCESS) {
    const char *msg = nullptr;
    cuptiGetResultString(rc, &msg);
    cupti_error_ = msg ? msg : "cuptiSubscribe failed";
    return false;
  }
  rc = cuptiEnableDomain(1, handle, CUPTI_CB_DOMAIN_RUNTIME_API);
  if (rc != CUPTI_SUCCESS) {
    const char *msg = nullptr;
    cuptiGetResultString(rc, &msg);
    cupti_error_ = msg ? msg : "cuptiEnableDomain failed";
    cuptiUnsubscribe(handle);
    return false;
  }
  subscriber_ = handle;
  cupti_active_ = true;
  return true;
}

void Recorder::StopInstrumentation() {
  if (!cupti_active_) return;
  cuptiUnsubscribe(static_cast<CUpti_SubscriberHandle>(subscriber_));
  subscriber_ = nullptr;
  cupti_active_ = false;
}

void Recorder::BeginChunk(int chunk_id, Side side, const std::string &regime,
                          size_t chunk_bytes, double error_bound) {
  current_ = ChunkTrace{};
  current_.chunk_id = chunk_id;
  current_.side = side;
  current_.regime = regime;
  current_.chunk_bytes = chunk_bytes;
  current_.error_bound = error_bound;
  sequence_ = 0;
}

ChunkTrace Recorder::EndChunk() {
  ChunkTrace done = std::move(current_);
  current_ = ChunkTrace{};
  return done;
}

TraceEntry *Recorder::Begin(Stage stage) {
  TraceEntry entry;
  entry.sequence = ++sequence_;
  entry.chunk_id = current_.chunk_id;
  entry.side = current_.side;
  entry.stage = stage;
  entry.kind = KindOf(stage);
  entry.entry_time_ms = NowMs();
  cudaGetDevice(&entry.cuda_device);
  current_.entries.push_back(std::move(entry));
  TraceEntry *ptr = &current_.entries.back();
  t_current = ptr;
  return ptr;
}

TraceEntry *Recorder::BeginGlobal(Stage stage) {
  TraceEntry *ptr = Begin(stage);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_global = ptr;
  return ptr;
}

void Recorder::End(TraceEntry *entry) {
  if (entry == nullptr) return;
  entry->exit_time_ms = NowMs();
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_global == entry) g_global = nullptr;
  }
  if (t_current == entry) t_current = nullptr;
}

void Recorder::RecordSkipped(Stage stage, const std::string &why) {
  TraceEntry *entry = Begin(stage);
  entry->status = Status::kSkippedNotSelected;
  entry->note = why;
  entry->input_loc = MemLoc::kNone;
  entry->output_loc = MemLoc::kNone;
  End(entry);
}

void Recorder::PushHarness() { ++t_harness_depth; }

void Recorder::PopHarness() {
  if (t_harness_depth > 0) --t_harness_depth;
}

void Recorder::NoteTransfer(int cuda_memcpy_kind, size_t bytes, bool async,
                            const void *dst, const void *src) {
  TransferRecord rec;
  rec.direction = DirectionName(cuda_memcpy_kind);
  if (rec.direction == "DEFAULT") rec.direction = ResolveDefault(dst, src);
  rec.bytes = bytes;
  rec.async = async;
  rec.harness = (t_harness_depth > 0);

  if (rec.harness) {
    harness_bytes_ += bytes;
    ++harness_count_;
  }

  TraceEntry *target = t_current;
  if (target == nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    target = g_global;
    if (target != nullptr) {
      target->transfers.push_back(std::move(rec));
    }
    return;
  }
  target->transfers.push_back(std::move(rec));
}

void Recorder::NoteKernel(const char *symbol) {
  const char *name = (symbol != nullptr && symbol[0] != '\0') ? symbol
                                                              : "<anonymous>";
  auto append = [name](TraceEntry *target) {
    for (auto &k : target->kernels) {
      if (k.name == name) {
        ++k.count;
        return;
      }
    }
    target->kernels.push_back(KernelRecord{name, 1});
  };

  TraceEntry *target = t_current;
  if (target == nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_global != nullptr) append(g_global);
    return;
  }
  append(target);
}

StageScope::StageScope(Stage stage, const void *input_ptr, size_t input_bytes)
    : StageScope(stage, input_ptr, input_bytes, false) {}

StageScope::StageScope(Stage stage, const void *input_ptr, size_t input_bytes,
                       bool global) {
  // Classify the input BEFORE opening the stage: cudaPointerGetAttributes is
  // itself a CUDA call, and attributing it to the stage under test would put a
  // harness operation inside a production measurement.
  MemLoc loc = MemLoc::kNone;
  if (input_ptr != nullptr) {
    switch (ClassifyPointer(input_ptr)) {
      case PointerKind::kDevice: loc = MemLoc::kDevice; break;
      case PointerKind::kManaged: loc = MemLoc::kDevice; break;
      case PointerKind::kHost: loc = MemLoc::kHost; break;
      default: loc = MemLoc::kUnknown; break;
    }
  }
  uint64_t hash = 0;
  bool hash_ok = false;
  if (loc == MemLoc::kDevice && input_bytes > 0) {
    hash_ok = HashAsHarness(input_ptr, input_bytes, &hash);
  }

  entry_ = global ? Recorder::Instance().BeginGlobal(stage)
                  : Recorder::Instance().Begin(stage);
  entry_->input_ptr = input_ptr;
  entry_->input_bytes = input_bytes;
  entry_->input_loc = loc;
  entry_->input_hash = hash;
  entry_->input_hash_valid = hash_ok;
}

StageScope::~StageScope() {
  if (entry_ != nullptr) Recorder::Instance().End(entry_);
}

void StageScope::SetOutput(const void *ptr, size_t bytes) {
  if (entry_ == nullptr) return;
  // Close the stage first so the hashing probe cannot land inside it.
  Recorder::Instance().End(entry_);
  entry_->output_ptr = ptr;
  entry_->output_bytes = bytes;
  MemLoc loc = MemLoc::kNone;
  if (ptr != nullptr) {
    switch (ClassifyPointer(ptr)) {
      case PointerKind::kDevice:
      case PointerKind::kManaged: loc = MemLoc::kDevice; break;
      case PointerKind::kHost: loc = MemLoc::kHost; break;
      default: loc = MemLoc::kUnknown; break;
    }
  }
  entry_->output_loc = loc;
  if (loc == MemLoc::kDevice && bytes > 0) {
    entry_->output_hash_valid =
        HashAsHarness(ptr, bytes, &entry_->output_hash);
  }
  entry_ = nullptr;
}

void StageScope::Fail(const std::string &why) {
  if (entry_ == nullptr) return;
  entry_->status = Status::kFailed;
  entry_->note = why;
}

bool HashAsHarness(const void *device_ptr, size_t bytes, uint64_t *out) {
  Recorder::Instance().PushHarness();
  const bool ok = HashDeviceBuffer(device_ptr, bytes, out);
  Recorder::Instance().PopHarness();
  return ok;
}

namespace {

void JsonString(std::ostream &os, const std::string &s) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\t': os << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          os << buf;
        } else {
          os << c;
        }
    }
  }
  os << '"';
}

void JsonPtr(std::ostream &os, const void *p) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "\"0x%llx\"",
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(p)));
  os << buf;
}

void WriteEntry(std::ostream &os, const TraceEntry &e) {
  os << "      {\n";
  os << "        \"sequence\": " << e.sequence << ",\n";
  os << "        \"callback\": ";
  JsonString(os, StageName(e.stage));
  os << ",\n";
  os << "        \"classification\": \""
     << (e.kind == StageKind::kClioArchitectural ? "CLIO_ARCHITECTURAL"
                                                 : "NEUROPRESS_FUNCTIONAL")
     << "\",\n";
  os << "        \"status\": \"" << StatusName(e.status) << "\",\n";
  os << "        \"chunk_id\": " << e.chunk_id << ",\n";
  os << "        \"cuda_device\": " << e.cuda_device << ",\n";
  os << "        \"datatype\": ";
  JsonString(os, e.datatype);
  os << ",\n";
  os << "        \"input\": {\"ptr\": ";
  JsonPtr(os, e.input_ptr);
  os << ", \"bytes\": " << e.input_bytes << ", \"location\": \""
     << MemLocName(e.input_loc) << "\"";
  if (e.input_hash_valid) os << ", \"hash\": " << e.input_hash;
  os << "},\n";
  os << "        \"output\": {\"ptr\": ";
  JsonPtr(os, e.output_ptr);
  os << ", \"bytes\": " << e.output_bytes << ", \"location\": \""
     << MemLocName(e.output_loc) << "\"";
  if (e.output_hash_valid) os << ", \"hash\": " << e.output_hash;
  os << "},\n";
  os << "        \"entry_time_ms\": " << e.entry_time_ms
     << ", \"exit_time_ms\": " << e.exit_time_ms << ",\n";

  if (e.stats.valid) {
    os << "        \"statistics\": {\"entropy\": " << e.stats.entropy
       << ", \"mad\": " << e.stats.mad
       << ", \"second_derivative\": " << e.stats.second_derivative << "},\n";
  }
  if (e.diagnostics.valid) {
    const auto &d = e.diagnostics;
    os << "        \"diagnostics\": {\"native_record_present\": "
       << (d.native_record_present ? "true" : "false")
       << ", \"nn_action\": " << d.nn_action
       << ", \"nn_original_action\": " << d.nn_original_action
       << ", \"exploration_triggered\": " << d.exploration_triggered
       << ", \"sgd_fired\": " << d.sgd_fired
       << ", \"feat_entropy\": " << d.feat_entropy
       << ", \"feat_mad\": " << d.feat_mad
       << ", \"feat_deriv\": " << d.feat_deriv
       << ", \"predicted_ratio\": " << d.predicted_ratio
       << ", \"predicted_comp_time\": " << d.predicted_comp_time
       << ", \"predicted_decomp_time\": " << d.predicted_decomp_time
       << ", \"predicted_psnr\": " << d.predicted_psnr
       << ", \"predicted_ranking_count\": " << d.predicted_ranking_count
       << "},\n";
  }
  if (e.inference.valid) {
    os << "        \"inference\": {\"num_candidates\": "
       << e.inference.num_candidates << ", \"per_action\": [";
    bool first = true;
    for (int a = 0; a < 32; ++a) {
      if (!e.inference.present[a]) continue;
      if (!first) os << ", ";
      first = false;
      os << "{\"action\": " << a << ", \"ratio\": " << e.inference.ratio[a]
         << ", \"comp_ms\": " << e.inference.comp_ms[a]
         << ", \"decomp_ms\": " << e.inference.decomp_ms[a]
         << ", \"psnr_db\": " << e.inference.psnr_db[a] << "}";
    }
    os << "]},\n";
  }
  if (e.ranking.valid) {
    os << "        \"ranking\": {\"order\": [";
    for (size_t i = 0; i < e.ranking.order.size(); ++i) {
      if (i != 0) os << ", ";
      os << e.ranking.order[i];
    }
    os << "], \"costs\": [";
    bool first = true;
    for (int a = 0; a < 32; ++a) {
      if (!e.ranking.cost_present[a]) continue;
      if (!first) os << ", ";
      first = false;
      os << "{\"action\": " << a << ", \"cost\": " << e.ranking.cost[a] << "}";
    }
    os << "]},\n";
  }
  if (e.selection.valid) {
    os << "        \"selection\": {\"action\": " << e.selection.action
       << ", \"algo\": " << e.selection.algo << ", \"algo_name\": ";
    JsonString(os, e.selection.algo_name);
    os << ", \"quantize\": " << (e.selection.quantize ? "true" : "false")
       << ", \"shuffle\": " << (e.selection.shuffle ? "true" : "false")
       << "},\n";
  }
  if (e.quant.valid) {
    os << "        \"quantization\": {\"precision\": " << e.quant.precision
       << ", \"error_bound\": " << e.quant.error_bound
       << ", \"effective_error_bound\": " << e.quant.effective_error_bound
       << ", \"scale\": " << e.quant.scale
       << ", \"data_min\": " << e.quant.data_min
       << ", \"data_max\": " << e.quant.data_max
       << ", \"bound_achievable\": "
       << (e.quant.bound_achievable ? "true" : "false") << "},\n";
  }
  if (e.compression.valid) {
    os << "        \"compression\": {\"algo_name\": ";
    JsonString(os, e.compression.algo_name);
    os << ", \"compressed_bytes\": " << e.compression.compressed_bytes
       << "},\n";
  }

  os << "        \"transfers\": [";
  for (size_t i = 0; i < e.transfers.size(); ++i) {
    const auto &t = e.transfers[i];
    if (i != 0) os << ", ";
    os << "{\"direction\": \"" << t.direction << "\", \"bytes\": " << t.bytes
       << ", \"async\": " << (t.async ? "true" : "false") << ", \"origin\": \""
       << (t.harness ? "TEST_HARNESS_TRANSFER" : "PRODUCTION") << "\"}";
  }
  os << "],\n";

  os << "        \"kernels\": [";
  for (size_t i = 0; i < e.kernels.size(); ++i) {
    if (i != 0) os << ", ";
    os << "{\"name\": ";
    JsonString(os, e.kernels[i].name);
    os << ", \"count\": " << e.kernels[i].count << "}";
  }
  os << "],\n";
  os << "        \"note\": ";
  JsonString(os, e.note);
  os << "\n      }";
}

}  // namespace

bool WriteTraceJson(const std::string &path,
                    const std::vector<ChunkTrace> &traces) {
  std::ofstream os(path);
  if (!os) return false;
  os.precision(10);
  os << "{\n  \"chunks\": [\n";
  for (size_t c = 0; c < traces.size(); ++c) {
    const auto &t = traces[c];
    os << "    {\n";
    os << "      \"chunk_id\": " << t.chunk_id << ",\n";
    os << "      \"side\": \"" << SideName(t.side) << "\",\n";
    os << "      \"regime\": ";
    JsonString(os, t.regime);
    os << ",\n";
    os << "      \"chunk_bytes\": " << t.chunk_bytes << ",\n";
    os << "      \"error_bound\": " << t.error_bound << ",\n";
    os << "      \"input_device_verified\": "
       << (t.input_device_verified ? "true" : "false") << ",\n";
    os << "      \"input_hash\": " << t.input_hash << ",\n";
    os << "      \"callbacks\": [\n";
    for (size_t i = 0; i < t.entries.size(); ++i) {
      WriteEntry(os, t.entries[i]);
      if (i + 1 < t.entries.size()) os << ",";
      os << "\n";
    }
    os << "      ]\n    }";
    if (c + 1 < traces.size()) os << ",";
    os << "\n";
  }
  os << "  ]\n}\n";
  return true;
}

}  // namespace npeq
