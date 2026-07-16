// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// C ABI shim over thallium (Mochi: thallium -> margo -> mercury/argobots) for
// the ctp-net `thallium` feature. thallium is a header-only C++14/17 template
// library with no stable C ABI, so Rust cannot bind it directly; this shim is
// the ABI. It mirrors the surface that
// clio_ctp/lightbeam/thallium_transport.h actually uses:
//
//   tl::engine(proto, THALLIUM_SERVER_MODE, use_progress_thread, rpc_threads)
//   engine.self() / engine.lookup(addr) / engine.finalize()
//   engine.define(name[, handler]) -> tl::remote_procedure
//   engine.expose(segs, bulk_mode) -> tl::bulk
//   rpc.on(endpoint)(meta_blob, bulks) -> int
//   req.get_endpoint() / req.respond(int)
//   remote.on(sender_ep) >> local        (synchronous RDMA pull)
//
// ABI rules (the Rust side depends on all of these):
//  - No C++ exception may cross this boundary. Every entry point is wrapped in
//    try/catch and reports failure as a negative status plus a message written
//    into the caller's `errbuf` (NUL-terminated, truncated to `errlen`).
//  - No thread_local anywhere (project rule; header thread_locals also
//    duplicate per-DLL on Windows). Error text travels through the caller's
//    buffer instead of a per-thread "last error" slot, so the shim carries no
//    hidden per-thread state.
//  - The engine is a process singleton, matching ThalliumEngine::Get()'s
//    call_once/unique_ptr. It is never destroyed before exit, so once
//    ctp_th_engine_init() succeeds the tl::engine* stays valid and callers may
//    use handles without holding the init mutex.

#include <thallium.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <thallium/serialization/stl/vector.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tl = thallium;

// Status codes — keep in sync with the constants in src/thallium.rs.
#define CTP_TH_OK 0
#define CTP_TH_ALREADY_INIT 1
#define CTP_TH_ERR (-1)
#define CTP_TH_ERR_INVAL (-2)
#define CTP_TH_ERR_NOT_INIT (-3)

// Bulk permission modes — keep in sync with src/thallium.rs.
#define CTP_TH_BULK_READ_ONLY 0
#define CTP_TH_BULK_WRITE_ONLY 1
#define CTP_TH_BULK_READ_WRITE 2

namespace {

/** Copy `msg` into the caller's error buffer, always NUL-terminating. */
void SetErr(char *errbuf, size_t errlen, const char *msg) {
  if (errbuf == nullptr || errlen == 0) return;
  if (msg == nullptr) msg = "";
  std::strncpy(errbuf, msg, errlen - 1);
  errbuf[errlen - 1] = '\0';
}

/**
 * Process-singleton engine, mirroring ThalliumEngine::Get(). The mutex guards
 * only construction and the finalized flag; the engine pointer itself is
 * stable for the life of the process once set (we never reset the unique_ptr,
 * because outstanding endpoint/rpc/bulk handles reference it).
 */
struct EngineHolder {
  std::mutex mtx;
  std::unique_ptr<tl::engine> engine;
  bool finalized = false;
};

EngineHolder &Holder() {
  // Function-local static: initialized on first use, destroyed at exit.
  static EngineHolder holder;
  return holder;
}

/** The live engine, or nullptr if not initialized (or already finalized). */
tl::engine *LiveEngine() {
  EngineHolder &h = Holder();
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.engine || h.finalized) return nullptr;
  return h.engine.get();
}

tl::bulk_mode ToBulkMode(int mode) {
  switch (mode) {
    case CTP_TH_BULK_WRITE_ONLY:
      return tl::bulk_mode::write_only;
    case CTP_TH_BULK_READ_WRITE:
      return tl::bulk_mode::read_write;
    case CTP_TH_BULK_READ_ONLY:
    default:
      return tl::bulk_mode::read_only;
  }
}

}  // namespace

extern "C" {

/**
 * RPC handler trampoline signature. Called on one of the engine's
 * rpc_thread_count Argobots execution streams.
 *
 * `sender_ep` and each element of `remote_bulks` are BORROWED for the duration
 * of the call only — the callee must not free them, and must not stash them.
 * The returned int becomes the RPC response (req.respond(rc)); the shim
 * responds exactly once, on every path.
 */
typedef int (*ctp_th_handler_fn)(void *user, void *sender_ep, const char *meta,
                                 size_t meta_len, void *const *remote_bulks,
                                 size_t n_bulks);

/**
 * Initialize the process-singleton engine. First call wins: a later call with
 * different parameters leaves the existing engine untouched and returns
 * CTP_TH_ALREADY_INIT. This mirrors ThalliumEngine::Get()'s std::call_once.
 *
 * @param protocol Mercury protocol, e.g. "ofi+tcp;ofi_rxm" or "na+sm".
 * @param server_mode Nonzero -> THALLIUM_SERVER_MODE (can serve RPCs).
 * @param use_progress_thread Nonzero -> dedicated Mercury progress thread.
 *        Without it the caller must drive progress and blocking RPCs deadlock.
 * @param rpc_thread_count Argobots ES for RPC handlers (clamped to >= 1).
 * @return CTP_TH_OK on fresh init, CTP_TH_ALREADY_INIT if it already existed,
 *         CTP_TH_ERR_INVAL / CTP_TH_ERR on failure.
 */
int ctp_th_engine_init(const char *protocol, int server_mode,
                       int use_progress_thread, int rpc_thread_count,
                       char *errbuf, size_t errlen) {
  if (protocol == nullptr || *protocol == '\0') {
    SetErr(errbuf, errlen, "protocol must be a non-empty string");
    return CTP_TH_ERR_INVAL;
  }
  EngineHolder &h = Holder();
  std::lock_guard<std::mutex> lock(h.mtx);
  if (h.engine) return CTP_TH_ALREADY_INIT;
  if (rpc_thread_count < 1) rpc_thread_count = 1;
  try {
    h.engine = std::unique_ptr<tl::engine>(new tl::engine(
        std::string(protocol),
        server_mode != 0 ? THALLIUM_SERVER_MODE : THALLIUM_CLIENT_MODE,
        use_progress_thread != 0,
        static_cast<std::int32_t>(rpc_thread_count)));
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception constructing tl::engine");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/** @return nonzero if the engine exists and has not been finalized. */
int ctp_th_engine_is_initialized(void) { return LiveEngine() != nullptr ? 1 : 0; }

/**
 * Write the engine's own address (engine.self()) into `buf`.
 *
 * `*needed` is always set to the full length INCLUDING the NUL terminator. If
 * `needed > buflen` the buffer was too small and `buf` is left untouched; the
 * call still returns CTP_TH_OK so the caller can resize and retry.
 */
int ctp_th_engine_self(char *buf, size_t buflen, size_t *needed, char *errbuf,
                       size_t errlen) {
  if (needed == nullptr) {
    SetErr(errbuf, errlen, "needed out-parameter must not be null");
    return CTP_TH_ERR_INVAL;
  }
  tl::engine *eng = LiveEngine();
  if (eng == nullptr) {
    SetErr(errbuf, errlen, "engine not initialized");
    return CTP_TH_ERR_NOT_INIT;
  }
  try {
    std::string self = std::string(eng->self());
    *needed = self.size() + 1;
    if (buf != nullptr && buflen >= *needed) {
      std::memcpy(buf, self.c_str(), *needed);
    }
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception in engine.self()");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/**
 * Stop the progress thread / RPC ES pool. A THALLIUM_SERVER_MODE engine parks
 * in its destructor until finalized, so this must be called before exit.
 * Idempotent. After this, handles must not be used.
 */
void ctp_th_engine_finalize(void) {
  EngineHolder &h = Holder();
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.engine || h.finalized) return;
  try {
    h.engine->finalize();
  } catch (...) {
    // Nothing actionable at teardown; swallow rather than escape the ABI.
  }
  h.finalized = true;
}

/** Resolve `addr` to an endpoint. On success *out_ep owns a tl::endpoint. */
int ctp_th_endpoint_lookup(const char *addr, void **out_ep, char *errbuf,
                           size_t errlen) {
  if (addr == nullptr || out_ep == nullptr) {
    SetErr(errbuf, errlen, "addr and out_ep must not be null");
    return CTP_TH_ERR_INVAL;
  }
  *out_ep = nullptr;
  tl::engine *eng = LiveEngine();
  if (eng == nullptr) {
    SetErr(errbuf, errlen, "engine not initialized");
    return CTP_TH_ERR_NOT_INIT;
  }
  try {
    *out_ep = static_cast<void *>(new tl::endpoint(eng->lookup(addr)));
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception in engine.lookup()");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/** Free an endpoint from ctp_th_endpoint_lookup. Null-safe. */
void ctp_th_endpoint_free(void *ep) { delete static_cast<tl::endpoint *>(ep); }

/**
 * Stringify an endpoint. Same `needed` contract as ctp_th_engine_self.
 * `ep` may be a borrowed handler-side endpoint.
 */
int ctp_th_endpoint_to_string(void *ep, char *buf, size_t buflen,
                              size_t *needed, char *errbuf, size_t errlen) {
  if (ep == nullptr || needed == nullptr) {
    SetErr(errbuf, errlen, "ep and needed must not be null");
    return CTP_TH_ERR_INVAL;
  }
  try {
    std::string s = static_cast<std::string>(*static_cast<tl::endpoint *>(ep));
    *needed = s.size() + 1;
    if (buf != nullptr && buflen >= *needed) {
      std::memcpy(buf, s.c_str(), *needed);
    }
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception stringifying endpoint");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/**
 * Declare an RPC by name WITHOUT a handler (client side): engine.define(name).
 */
int ctp_th_rpc_define(const char *name, void **out_rpc, char *errbuf,
                      size_t errlen) {
  if (name == nullptr || out_rpc == nullptr) {
    SetErr(errbuf, errlen, "name and out_rpc must not be null");
    return CTP_TH_ERR_INVAL;
  }
  *out_rpc = nullptr;
  tl::engine *eng = LiveEngine();
  if (eng == nullptr) {
    SetErr(errbuf, errlen, "engine not initialized");
    return CTP_TH_ERR_NOT_INIT;
  }
  try {
    *out_rpc =
        static_cast<void *>(new tl::remote_procedure(eng->define(name)));
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception in engine.define()");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/**
 * Define an RPC WITH a handler (server side). The handler receives the
 * serialized metadata blob and the sender's remote bulk handles, exactly like
 * ThalliumTransport's lambda, and must pull any bulk it wants INSIDE the call:
 * the sender blocks on the response, which we do not issue until the callback
 * returns, so its exposed memory stays alive for the duration.
 *
 * `user` must outlive every RPC delivery. margo keeps the registration for the
 * engine's lifetime and deleting the returned handle does NOT deregister it,
 * so the Rust side intentionally leaks the boxed closure.
 */
int ctp_th_rpc_define_handler(const char *name, ctp_th_handler_fn cb,
                              void *user, void **out_rpc, char *errbuf,
                              size_t errlen) {
  if (name == nullptr || cb == nullptr || out_rpc == nullptr) {
    SetErr(errbuf, errlen, "name, cb and out_rpc must not be null");
    return CTP_TH_ERR_INVAL;
  }
  *out_rpc = nullptr;
  tl::engine *eng = LiveEngine();
  if (eng == nullptr) {
    SetErr(errbuf, errlen, "engine not initialized");
    return CTP_TH_ERR_NOT_INIT;
  }
  try {
    auto handler = [cb, user](const tl::request &req, std::string &meta_blob,
                              std::vector<tl::bulk> &remote_bulks) {
      int rc;
      try {
        tl::endpoint sender = req.get_endpoint();
        std::vector<void *> bulk_ptrs;
        bulk_ptrs.reserve(remote_bulks.size());
        for (tl::bulk &b : remote_bulks) {
          bulk_ptrs.push_back(static_cast<void *>(&b));
        }
        rc = cb(user, static_cast<void *>(&sender), meta_blob.data(),
                meta_blob.size(), bulk_ptrs.data(), bulk_ptrs.size());
      } catch (const std::exception &) {
        rc = CTP_TH_ERR;
      } catch (...) {
        rc = CTP_TH_ERR;
      }
      // Respond exactly once, on every path: the sender is blocked on this.
      try {
        req.respond(rc);
      } catch (...) {
        // Peer vanished; nothing else to do.
      }
    };
    *out_rpc = static_cast<void *>(
        new tl::remote_procedure(eng->define(name, handler)));
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception in engine.define(name, handler)");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/** Free a remote_procedure handle. Does NOT deregister a handler. Null-safe. */
void ctp_th_rpc_free(void *rpc) {
  delete static_cast<tl::remote_procedure *>(rpc);
}

/**
 * Blocking call: rpc.on(ep)(meta_blob, bulks). Mirrors ThalliumTransport::Send.
 *
 * @param bulks Array of n_bulks tl::bulk* (from ctp_th_bulk_expose). They are
 *              copied into the argument vector; ownership stays with caller.
 * @param out_rc Receives the peer's response code (the handler's return).
 * @return CTP_TH_OK if the RPC completed (check *out_rc for the peer's answer).
 */
int ctp_th_rpc_call(void *rpc, void *ep, const char *meta, size_t meta_len,
                    void *const *bulks, size_t n_bulks, int *out_rc,
                    char *errbuf, size_t errlen) {
  if (rpc == nullptr || ep == nullptr || out_rc == nullptr ||
      (meta == nullptr && meta_len != 0) || (bulks == nullptr && n_bulks != 0)) {
    SetErr(errbuf, errlen, "invalid null argument to ctp_th_rpc_call");
    return CTP_TH_ERR_INVAL;
  }
  if (LiveEngine() == nullptr) {
    SetErr(errbuf, errlen, "engine not initialized");
    return CTP_TH_ERR_NOT_INIT;
  }
  try {
    tl::remote_procedure *r = static_cast<tl::remote_procedure *>(rpc);
    tl::endpoint *e = static_cast<tl::endpoint *>(ep);
    std::string meta_blob(meta, meta_len);
    std::vector<tl::bulk> bulk_vec;
    bulk_vec.reserve(n_bulks);
    for (size_t i = 0; i < n_bulks; ++i) {
      if (bulks[i] == nullptr) {
        SetErr(errbuf, errlen, "null tl::bulk handle in bulks[]");
        return CTP_TH_ERR_INVAL;
      }
      bulk_vec.push_back(*static_cast<tl::bulk *>(bulks[i]));
    }
    *out_rc = r->on(*e)(meta_blob, bulk_vec);
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception in rpc.on(ep)(...)");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/**
 * Register [ptr, ptr+size) for RDMA: engine.expose(segs, mode). Mirrors
 * ThalliumTransport::Expose (which registers read_only for the send side).
 * The memory must stay put and valid until ctp_th_bulk_free.
 */
int ctp_th_bulk_expose(void *ptr, size_t size, int mode, void **out_bulk,
                       char *errbuf, size_t errlen) {
  if (out_bulk == nullptr) {
    SetErr(errbuf, errlen, "out_bulk must not be null");
    return CTP_TH_ERR_INVAL;
  }
  *out_bulk = nullptr;
  if (ptr == nullptr || size == 0) {
    SetErr(errbuf, errlen, "cannot expose an empty region");
    return CTP_TH_ERR_INVAL;
  }
  tl::engine *eng = LiveEngine();
  if (eng == nullptr) {
    SetErr(errbuf, errlen, "engine not initialized");
    return CTP_TH_ERR_NOT_INIT;
  }
  try {
    std::vector<std::pair<void *, size_t>> segs{{ptr, size}};
    *out_bulk = static_cast<void *>(
        new tl::bulk(eng->expose(segs, ToBulkMode(mode))));
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception in engine.expose()");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

/** Deregister/free a bulk handle. Null-safe. */
void ctp_th_bulk_free(void *bulk) { delete static_cast<tl::bulk *>(bulk); }

/** Byte length of a bulk region (tl::bulk::size()). 0 if `bulk` is null. */
size_t ctp_th_bulk_size(void *bulk) {
  if (bulk == nullptr) return 0;
  return static_cast<tl::bulk *>(bulk)->size();
}

/**
 * Synchronous RDMA pull of an entire remote bulk into `dst`:
 *   remote.on(sender_ep) >> local
 * where `local` is a write_only exposure of exactly remote.size() bytes of
 * `dst`, created and released inside this call. This is the whole of
 * ThalliumTransport's handler-side pull, condensed into one primitive so the
 * transient local registration never has to cross the ABI.
 *
 * @return CTP_TH_ERR_INVAL if dst_len < remote.size().
 */
int ctp_th_bulk_pull_into(void *remote_bulk, void *sender_ep, void *dst,
                          size_t dst_len, char *errbuf, size_t errlen) {
  if (remote_bulk == nullptr || sender_ep == nullptr) {
    SetErr(errbuf, errlen, "remote_bulk and sender_ep must not be null");
    return CTP_TH_ERR_INVAL;
  }
  tl::engine *eng = LiveEngine();
  if (eng == nullptr) {
    SetErr(errbuf, errlen, "engine not initialized");
    return CTP_TH_ERR_NOT_INIT;
  }
  try {
    tl::bulk *remote = static_cast<tl::bulk *>(remote_bulk);
    tl::endpoint *ep = static_cast<tl::endpoint *>(sender_ep);
    size_t sz = remote->size();
    if (sz == 0) return CTP_TH_OK;
    if (dst == nullptr || dst_len < sz) {
      SetErr(errbuf, errlen, "destination smaller than the remote bulk");
      return CTP_TH_ERR_INVAL;
    }
    // Expose exactly sz bytes: `>>` moves min(remote.size(), local.size()),
    // so sizing the local view to the remote makes the transfer exact.
    std::vector<std::pair<void *, size_t>> segs{{dst, sz}};
    tl::bulk local = eng->expose(segs, tl::bulk_mode::write_only);
    remote->on(*ep) >> local;
  } catch (const std::exception &e) {
    SetErr(errbuf, errlen, e.what());
    return CTP_TH_ERR;
  } catch (...) {
    SetErr(errbuf, errlen, "unknown exception during RDMA pull");
    return CTP_TH_ERR;
  }
  return CTP_TH_OK;
}

}  // extern "C"
