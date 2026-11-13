/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_EVLAT_H_
#define CLIO_RUNTIME_EVLAT_H_

// Latency-report (CLIO_EVLAT) channels for the cross-node task path.
//
// rtt_remote (channel 10) times a remote round trip end to end on the ORIGIN.
// These channels split it into stages. Each stage is timed on the node it runs
// on (rdtsc is not comparable across nodes), so with a symmetric workload --
// every node both originates and serves -- a node's report covers both halves:
//
//   origin  o_netq      worker EndTask/route -> net send worker pops SendIn
//           o_send      SendIn(): plan + serialize + transport Send
//   ......  wire + ZMQ I/O thread + wait in socket until chi-peer-recv reads
//   remote  r_read      chi-peer-recv: transport Recv() of a request
//           r_recvin    RecvIn(): deserialize + push to the ingress lane
//           r_lane      ingress lane push -> a worker first runs the task
//           r_exec      first run -> EndTask (incl. any subtask awaits)
//           r_netq      EndTask -> net send worker pops SendOut
//           r_send      SendOut(): serialize + transport Send
//           r_resid     RecvIn start -> SendOut done (the whole remote stay)
//   ......  wire + wait in socket on the way back
//   origin  o_read      chi-peer-recv: transport Recv() of a reply
//           o_recvout   RecvOut(): deserialize + aggregate + complete origin
//           peer_poll   chi-peer-recv blocked in PollRecv (idle)
//
// rtt_remote - o_send - r_resid - o_read - o_recvout ~= wire + socket waits.
//
// The two single threads on this path also report how busy they were per
// ~1 s window (clio_evlat_busy, printed as clio-evbusy): chi-peer-recv
// (read + handle) and the net send worker (SendIn + SendOut).

#include <cstdlib>

#include "clio_runtime/cycle_counter.h"

/** Defined in ipc_gpu2cpu.cc. */
extern "C" void clio_evlat_add(int which, unsigned long long cycles);
extern "C" void clio_evlat_busy(int which, unsigned long long start,
                                unsigned long long end);

namespace clio::run::evlat {

enum Chan : int {
  kRttRemote = 10,
  kEvqWait = 11,
  kOriginNetQ = 19,
  kOriginSend = 20,
  kRemoteRead = 21,
  kRemoteRecvIn = 22,
  kRemoteLane = 23,
  kRemoteExec = 24,
  kRemoteNetQ = 25,
  kRemoteSend = 26,
  kRemoteResid = 27,
  kOriginRead = 28,
  kOriginRecvOut = 29,
  kPeerRecvPoll = 30,
  // Client <-> runtime on one node (the part of a request no channel above
  // covers): c_e2e is timed IN THE CLIENT (CLIO_EVLAT in its env), from
  // registering the pending future at send to its receive thread demuxing the
  // reply; o_res_loc / o_res_rem are timed in the runtime, from a worker
  // popping the request off the client SHM shard to the response being
  // copied into the client's ring, for requests served here vs sent to a peer.
  kClientE2E = 31,
  kOriginResLocal = 32,
  kOriginResRemote = 33,
  kNumChan = 34,
};

enum Busy : int { kBusyPeerRecv = 0, kBusyNetSend = 1, kNumBusy = 2 };

/** Stage stamps are taken only under CLIO_EVLAT (set, non-empty, not "0"). */
inline bool On() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_EVLAT");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return on;
}

/** A cycle stamp when enabled, else 0 (which every Span below ignores). */
inline unsigned long long Now() { return On() ? CycleNow() : 0; }

/** Record t1 - t0 on channel `ch`; no-op if either stamp is unset. */
inline void Span(int ch, unsigned long long t0, unsigned long long t1) {
  if (t0 != 0 && t1 > t0) clio_evlat_add(ch, t1 - t0);
}

/** Record [t0, now) on channel `ch` (and optionally a busy kind) on scope
 *  exit, so functions with many early returns are timed in one place. */
struct ScopedSpan {
  int ch;
  int busy;
  unsigned long long t0;
  explicit ScopedSpan(int c, int b = -1) : ch(c), busy(b), t0(Now()) {}
  ~ScopedSpan() {
    if (t0 == 0) return;
    const unsigned long long t1 = CycleNow();
    Span(ch, t0, t1);
    if (busy >= 0) clio_evlat_busy(busy, t0, t1);
  }
  ScopedSpan(const ScopedSpan &) = delete;
  ScopedSpan &operator=(const ScopedSpan &) = delete;
};

}  // namespace clio::run::evlat

#endif  // CLIO_RUNTIME_EVLAT_H_
