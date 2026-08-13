/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file path_tracer.cc
 * @brief CUPTI subscription, region tagging and the timeline report.
 */

#include "path_tracer.h"

#include <cuda_runtime.h>
#include <cupti.h>
#include <cxxabi.h>

#include <cctype>
#include <cstring>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace nppath {

namespace {

std::mutex g_mutex;
/** Re-entry guard: CUDA calls made inside a CUPTI callback would recurse. */
thread_local bool t_in_cb = false;
thread_local int t_harness = 0;

double NowMs() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point origin = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - origin).count();
}

unsigned long ThreadId() {
  return static_cast<unsigned long>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFu);
}

Loc Classify(const void *p) {
  if (p == nullptr) return Loc::kUnknown;
  cudaPointerAttributes a{};
  if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
    cudaGetLastError();
    return Loc::kHost;
  }
  switch (a.type) {
    case cudaMemoryTypeDevice: return Loc::kDevice;
    case cudaMemoryTypeManaged: return Loc::kManaged;
    case cudaMemoryTypeHost: return Loc::kHost;
    default: return Loc::kHost;
  }
}

const char *DirName(int kind) {
  switch (kind) {
    case cudaMemcpyHostToHost: return "H2H";
    case cudaMemcpyHostToDevice: return "H2D";
    case cudaMemcpyDeviceToHost: return "D2H";
    case cudaMemcpyDeviceToDevice: return "D2D";
    default: return "DEFAULT";
  }
}

void CUPTIAPI Callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                       const void *cbdata) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || t_in_cb) return;
  const auto *cb = static_cast<const CUpti_CallbackData *>(cbdata);
  if (cb->callbackSite != CUPTI_API_ENTER) return;
  t_in_cb = true;
  switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020: {
      const auto *p = static_cast<const cudaMemcpy_v3020_params *>(cb->functionParams);
      Tracer::Instance().OnTransfer(p->kind, p->count, false, p->dst, p->src);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
      const auto *p =
          static_cast<const cudaMemcpyAsync_v3020_params *>(cb->functionParams);
      Tracer::Instance().OnTransfer(p->kind, p->count, true, p->dst, p->src);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
      Tracer::Instance().OnKernel(cb->symbolName);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: {
      const auto *p = static_cast<const cudaMalloc_v3020_params *>(cb->functionParams);
      Tracer::Instance().OnAlloc(p->size);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
      Tracer::Instance().OnFree();
      break;
    default:
      break;
  }
  t_in_cb = false;
}

/**
 * Turn a CUDA kernel symbol into something a person can read.
 *
 * Two obstacles. nvcc wraps kernels from anonymous namespaces in a
 * `__nv_static_<hash>_..._<hash>__ZN...` prefix, which defeats a plain
 * demangle; and even demangled, a name carries its full namespace path,
 * template arguments and parameter list, which is ~300 characters of noise per
 * timeline row. So: demangle from the embedded `_ZN` if there is one, then keep
 * only the final identifier.
 */
std::string ShortKernel(const std::string &raw) {
  const size_t zn = raw.find("_ZN");
  const std::string mangled = (zn == std::string::npos) ? raw : raw.substr(zn);

  int status = 0;
  char *dem = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
  const bool demangled = (status == 0 && dem != nullptr);
  std::string name = demangled ? std::string(dem) : std::string();
  std::free(dem);

  if (!demangled) {
    // The wrapper's embedded name often will not demangle: nvcc emits an
    // anonymous-namespace component whose length prefix the ABI decoder
    // rejects. Parse the nested-name SEQUENTIALLY instead -- `_ZN` followed by
    // <len><chars> components -- and keep the last, which is the function
    // identifier. Sequential parsing matters: the anonymous-namespace component
    // contains its own digit runs ("..._25_data_stats_gpu_kernels_cu_1d857366"),
    // so scanning for digit runs anywhere in the string wanders inside it and
    // picks up fragments of a hash.
    size_t pos = mangled.rfind("_ZN", 0) == 0 ? 3 : 0;
    while (pos < mangled.size() &&
           std::isdigit(static_cast<unsigned char>(mangled[pos]))) {
      size_t len = 0;
      while (pos < mangled.size() &&
             std::isdigit(static_cast<unsigned char>(mangled[pos]))) {
        len = len * 10 + static_cast<size_t>(mangled[pos] - '0');
        ++pos;
      }
      if (len == 0 || pos + len > mangled.size()) break;
      name = mangled.substr(pos, len);
      pos += len;
    }
    if (name.empty() || name.rfind("_GLOBAL__N_", 0) == 0) return raw;
    return name;
  }

  // A demangled kernel looks like
  //   void ctp::(anonymous namespace)::StatsPass1Kernel<float>(float const*, ...)
  // ORDER MATTERS HERE. "(anonymous namespace)" contains the first '(', so
  // cutting at that paren first leaves "void ctp::" and then trimming to the
  // last "::" yields an empty string. Remove the anonymous-namespace marker
  // before touching the parentheses.
  for (size_t p = name.find("(anonymous namespace)::");
       p != std::string::npos; p = name.find("(anonymous namespace)::")) {
    name.erase(p, std::strlen("(anonymous namespace)::"));
  }
  const size_t paren = name.find('(');            // now the parameter list
  if (paren != std::string::npos) name = name.substr(0, paren);
  const size_t angle = name.find('<');            // template arguments
  if (angle != std::string::npos) name = name.substr(0, angle);
  const size_t colons = name.rfind("::");         // namespaces
  if (colons != std::string::npos) name = name.substr(colons + 2);
  const size_t space = name.rfind(' ');           // leading return type
  if (space != std::string::npos) name = name.substr(space + 1);

  if (name.empty()) return raw;
  return name;
}

std::string Human(size_t bytes) {
  char buf[48];
  if (bytes >= (1u << 20)) {
    std::snprintf(buf, sizeof(buf), "%.2f MiB", bytes / 1048576.0);
  } else if (bytes >= 1024) {
    std::snprintf(buf, sizeof(buf), "%.1f KiB", bytes / 1024.0);
  } else {
    std::snprintf(buf, sizeof(buf), "%zu B", bytes);
  }
  return buf;
}

}  // namespace

const char *LocName(Loc loc) {
  switch (loc) {
    case Loc::kDevice: return "DEVICE";
    case Loc::kHost: return "HOST";
    case Loc::kManaged: return "MANAGED";
    default: return "?";
  }
}

Tracer &Tracer::Instance() {
  static Tracer t;
  return t;
}

bool Tracer::Start(std::string *error) {
  if (active_) return true;
  CUpti_SubscriberHandle h{};
  CUptiResult rc = cuptiSubscribe(&h, Callback, nullptr);
  if (rc == CUPTI_SUCCESS) rc = cuptiEnableDomain(1, h, CUPTI_CB_DOMAIN_RUNTIME_API);
  if (rc != CUPTI_SUCCESS) {
    const char *msg = nullptr;
    cuptiGetResultString(rc, &msg);
    if (error) *error = msg ? msg : "CUPTI subscription failed";
    return false;
  }
  subscriber_ = h;
  active_ = true;
  events_.reserve(4096);
  return true;
}

void Tracer::Stop() {
  if (!active_) return;
  cuptiUnsubscribe(static_cast<CUpti_SubscriberHandle>(subscriber_));
  active_ = false;
}

Event *Tracer::Append(EventKind kind) {
  Event e;
  e.seq = ++seq_;
  e.t_ms = NowMs();
  e.kind = kind;
  e.thread = ThreadId();
  e.phase = phase_stack_.empty() ? std::string("-") : phase_stack_.back();
  e.harness = (t_harness > 0);
  events_.push_back(std::move(e));
  return &events_.back();
}

void Tracer::AddRegion(const std::string &name, const void *base, size_t bytes) {
  std::lock_guard<std::mutex> lock(g_mutex);
  regions_.push_back(Region{name, base, bytes, Classify(base)});
  Event *e = Append(EventKind::kNote);
  e->text = "region " + name + " = " + Human(bytes) + " at " +
            LocName(regions_.back().loc);
  e->bytes = bytes;
}

void Tracer::DropRegion(const void *base) {
  std::lock_guard<std::mutex> lock(g_mutex);
  regions_.erase(std::remove_if(regions_.begin(), regions_.end(),
                                [base](const Region &r) { return r.base == base; }),
                 regions_.end());
}

void Tracer::BeginPhase(const std::string &name) {
  std::lock_guard<std::mutex> lock(g_mutex);
  phase_stack_.push_back(name);
  Append(EventKind::kPhase)->text = "BEGIN " + name;
}

void Tracer::EndPhase() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (phase_stack_.empty()) return;
  const std::string name = phase_stack_.back();
  Append(EventKind::kPhase)->text = "END " + name;
  phase_stack_.pop_back();
}

void Tracer::Note(const std::string &text) {
  std::lock_guard<std::mutex> lock(g_mutex);
  Append(EventKind::kNote)->text = text;
}

void Tracer::PushHarness() { ++t_harness; }
void Tracer::PopHarness() { if (t_harness > 0) --t_harness; }

void Tracer::OnTransfer(int kind, size_t bytes, bool async, const void *dst,
                        const void *src) {
  // Pointer classification calls CUDA, which is safe here only because the
  // callback holds the re-entry guard.
  const Loc dl = Classify(dst);
  const Loc sl = Classify(src);
  std::string dir = DirName(kind);
  if (dir == "DEFAULT") {
    const bool dd = dl == Loc::kDevice, sd = sl == Loc::kDevice;
    dir = dd && sd ? "D2D" : dd ? "H2D" : sd ? "D2H" : "H2H";
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  Event *e = Append(EventKind::kTransfer);
  e->direction = dir;
  e->bytes = bytes;
  e->async = async;
  e->src = src;
  e->dst = dst;
  e->src_loc = sl;
  e->dst_loc = dl;
  for (const auto &r : regions_) {
    const auto *b = static_cast<const char *>(r.base);
    if (src >= r.base && static_cast<const char *>(src) < b + r.bytes) {
      e->src_region = r.name;
    }
    if (dst >= r.base && static_cast<const char *>(dst) < b + r.bytes) {
      e->dst_region = r.name;
    }
  }
}

void Tracer::OnKernel(const char *symbol) {
  std::lock_guard<std::mutex> lock(g_mutex);
  Append(EventKind::kKernel)->symbol =
      (symbol && symbol[0]) ? ShortKernel(symbol) : "<anonymous>";
}

void Tracer::OnAlloc(size_t bytes) {
  std::lock_guard<std::mutex> lock(g_mutex);
  Append(EventKind::kAlloc)->alloc_bytes = bytes;
}

void Tracer::OnFree() {
  std::lock_guard<std::mutex> lock(g_mutex);
  Append(EventKind::kFree);
}

size_t Tracer::Bytes(const std::string &direction) const {
  size_t total = 0;
  for (const auto &e : events_) {
    if (e.kind == EventKind::kTransfer && !e.harness && e.direction == direction) {
      total += e.bytes;
    }
  }
  return total;
}

int Tracer::Count(const std::string &direction) const {
  int n = 0;
  for (const auto &e : events_) {
    if (e.kind == EventKind::kTransfer && !e.harness && e.direction == direction) ++n;
  }
  return n;
}

void PrintReport(std::ostream &os, size_t payload_bytes, size_t num_chunks) {
  const auto &tr = Tracer::Instance();
  const auto &ev = tr.Events();
  const std::string rule(100, '=');
  const std::string thin(100, '-');

  // "Payload-sized" means at least a quarter of a chunk: quantization can
  // legitimately narrow a chunk to a quarter of its width before the codec
  // sees it, so a fixed byte threshold would either miss a real payload move
  // or flag a header.
  const size_t payload_threshold = std::max<size_t>(payload_bytes / 4, 4096);

  os << rule << "\nCLIO-NEUROPRESS DATA PATH TRACE\n" << rule << "\n\n"
     << "Chunks: " << num_chunks << " x " << Human(payload_bytes)
     << "    payload-sized transfer threshold: " << Human(payload_threshold)
     << "\n"
     << "Instrumentation: CUPTI runtime-API callbacks -- every cudaMemcpy and\n"
        "kernel launch below was observed, on whichever thread made it.\n"
        "Clio's runtime runs in-process, so its worker threads are visible here.\n\n";

  os << "Named memory regions:\n";
  for (const auto &r : tr.Regions()) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "    %-22s 0x%-14llx %-8s %s\n", r.name.c_str(),
                  static_cast<unsigned long long>(
                      reinterpret_cast<uintptr_t>(r.base)),
                  LocName(r.loc), Human(r.bytes).c_str());
    os << buf;
  }

  os << "\n" << thin << "\nTIMELINE\n" << thin << "\n";
  os << "    seq       t(ms) thr  phase                event\n";
  for (const auto &e : ev) {
    char head[96];
    std::snprintf(head, sizeof(head), "  %6llu %10.3f %4lu  %-20s ",
                  static_cast<unsigned long long>(e.seq), e.t_ms, e.thread,
                  e.phase.c_str());
    os << head;
    switch (e.kind) {
      case EventKind::kPhase:
        os << "== " << e.text << " ==\n";
        break;
      case EventKind::kNote:
        os << e.text << "\n";
        break;
      case EventKind::kKernel:
        os << "kernel  " << e.symbol << "\n";
        break;
      case EventKind::kAlloc:
        os << "cudaMalloc " << Human(e.alloc_bytes) << "\n";
        break;
      case EventKind::kFree:
        os << "cudaFree\n";
        break;
      case EventKind::kTransfer: {
        os << e.direction << " " << std::setw(10) << Human(e.bytes)
           << (e.async ? " async" : " sync ") << "  ["
           << (e.src_region.empty() ? std::string(LocName(e.src_loc))
                                    : e.src_region)
           << " -> "
           << (e.dst_region.empty() ? std::string(LocName(e.dst_loc))
                                    : e.dst_region)
           << "]";
        if (e.harness) os << "  TEST_HARNESS";
        if (!e.harness && e.bytes >= payload_threshold) os << "   <== PAYLOAD-SIZED";
        os << "\n";
        break;
      }
    }
  }

  // ---- the question this tool exists to answer ----
  os << "\n" << thin << "\nPAYLOAD-SIZED PRODUCTION TRANSFERS\n" << thin << "\n";
  int payload_d2h = 0, payload_h2d = 0, payload_d2d = 0;
  for (const auto &e : ev) {
    if (e.kind != EventKind::kTransfer || e.harness) continue;
    if (e.bytes < payload_threshold) continue;
    if (e.direction == "D2H") ++payload_d2h;
    if (e.direction == "H2D") ++payload_h2d;
    if (e.direction == "D2D") ++payload_d2d;
    char buf[220];
    std::snprintf(buf, sizeof(buf), "    seq %-6llu %-4s %-11s %-20s [%s -> %s]\n",
                  static_cast<unsigned long long>(e.seq), e.direction.c_str(),
                  Human(e.bytes).c_str(), e.phase.c_str(),
                  e.src_region.empty() ? LocName(e.src_loc) : e.src_region.c_str(),
                  e.dst_region.empty() ? LocName(e.dst_loc) : e.dst_region.c_str());
    os << buf;
  }
  if (payload_d2h + payload_h2d + payload_d2d == 0) {
    os << "    (none)\n";
  }

  os << "\n" << thin << "\nTOTALS (production only; harness copies excluded)\n"
     << thin << "\n";
  for (const char *d : {"H2D", "D2H", "D2D", "H2H"}) {
    os << "    " << d << ": " << std::setw(4) << tr.Count(d) << " copies, "
       << Human(tr.Bytes(d)) << "\n";
  }

  os << "\n" << thin << "\nRESIDENCY VERDICT\n" << thin << "\n";
  os << "    Payload-sized D->H : " << payload_d2h << "\n";
  os << "    Payload-sized H->D : " << payload_h2d << "\n";
  os << "    Payload-sized D->D : " << payload_d2d << "\n\n";
  if (payload_d2h == 0) {
    os << "    The chunk's bytes NEVER crossed to the host at payload size.\n"
          "    Compression consumed the data where it was generated; only the\n"
          "    compressed result and small metadata left the device.\n";
  } else {
    os << "    The payload DID cross to the host " << payload_d2h
       << " time(s). Read the timeline above at the\n"
          "    listed seq numbers to see in which phase, and whether it happened\n"
          "    before or after the compression kernels.\n";
  }
}

bool WriteReport(const std::string &path, size_t payload_bytes,
                 size_t num_chunks) {
  std::ofstream os(path);
  if (!os) return false;
  PrintReport(os, payload_bytes, num_chunks);
  return true;
}

bool WriteTimelineJson(const std::string &path) {
  std::ofstream os(path);
  if (!os) return false;
  const auto &ev = Tracer::Instance().Events();
  os << "{\n  \"events\": [\n";
  for (size_t i = 0; i < ev.size(); ++i) {
    const auto &e = ev[i];
    os << "    {\"seq\": " << e.seq << ", \"t_ms\": " << e.t_ms
       << ", \"thread\": " << e.thread << ", \"phase\": \"" << e.phase << "\"";
    switch (e.kind) {
      case EventKind::kTransfer:
        os << ", \"kind\": \"transfer\", \"direction\": \"" << e.direction
           << "\", \"bytes\": " << e.bytes
           << ", \"async\": " << (e.async ? "true" : "false")
           << ", \"src_region\": \"" << e.src_region << "\", \"dst_region\": \""
           << e.dst_region << "\", \"src_loc\": \"" << LocName(e.src_loc)
           << "\", \"dst_loc\": \"" << LocName(e.dst_loc)
           << "\", \"origin\": \""
           << (e.harness ? "TEST_HARNESS_TRANSFER" : "PRODUCTION") << "\"";
        break;
      case EventKind::kKernel:
        os << ", \"kind\": \"kernel\", \"symbol\": \"" << e.symbol << "\"";
        break;
      case EventKind::kAlloc:
        os << ", \"kind\": \"alloc\", \"bytes\": " << e.alloc_bytes;
        break;
      case EventKind::kFree:
        os << ", \"kind\": \"free\"";
        break;
      default:
        os << ", \"kind\": \"note\", \"text\": \"" << e.text << "\"";
        break;
    }
    os << "}";
    if (i + 1 < ev.size()) os << ",";
    os << "\n";
  }
  os << "  ]\n}\n";
  return true;
}

}  // namespace nppath
