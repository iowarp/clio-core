/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Clio HDF5 VOL Connector — pass-through, native-authoritative.
 *
 * EVERY operation is delegated to the native VOL, which owns the file: the
 * on-disk HDF5 image is written synchronously and is the source of truth, so
 * standard tools read it live and CLIO can be removed without losing data.
 * On top of that, whole-dataset H5Dwrite/H5Dread transfers of flat datatypes
 * are additionally MIRRORED into the CTE tier as chunked blobs, and hyperslab /
 * point reads can be served from that mirror (serve-only; a miss falls back to
 * native). The cache is an accelerator, never an authority: any doubt about
 * whether it matches the file resolves by invalidating it and re-reading native.
 *
 * If the CLIO runtime is unreachable, or CLIO_VOL_CACHE=0 is set, the connector
 * degrades to a pure pass-through and remains correct.
 *
 * See README.md for configuration and translation/VOL_AUDIT.md for the current
 * gap list.
 */

#include "clio_vol.h"

#include <H5PLextern.h>     /* H5PLget_plugin_type / H5PLget_plugin_info */
#ifdef H5_HAVE_PARALLEL
#include <H5FDmpio.h>       /* H5Pget_dxpl_mpio (collective-IO detection) */
#endif

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <chrono>

#include "clio_vol_trace.h"

#include <clio_runtime/clio_runtime.h>
/* transport_factory_impl.h provides the inline definitions of
   ctp::lbm::Transport::ClearRecvHandles / Send<...>. They are required
   when AsyncPutBlob template-instantiates here (it doesn't fire from
   clio.h alone). Without this include the linker leaves the
   templated symbols undefined in our .so. */
#include <clio_ctp/lightbeam/transport_factory_impl.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>

/* The connector's capability set. Defined once because it is reported from two
   places (the class literal and introspect_get_cap_flags) that must agree. */
#define CLIO_VOL_CAP_FLAGS                                                 \
  (H5VL_CAP_FLAG_FILE_BASIC | H5VL_CAP_FLAG_DATASET_BASIC |                \
   H5VL_CAP_FLAG_GROUP_BASIC | H5VL_CAP_FLAG_ATTR_BASIC |                  \
   H5VL_CAP_FLAG_LINK_BASIC | H5VL_CAP_FLAG_OBJECT_BASIC)

/* ========================================================================
 * Internal state structures
 * ======================================================================== */

struct clio_file_t;     /* fwd */
struct clio_dataset_t;  /* fwd */

/* parent_file points to the clio_file_t this object belongs to. For
   files it is a self-pointer; for groups, datasets, attributes it is
   inherited from the containing file/group at create-time. The
   dataset_t branch reads it to find the CTE tag for AsyncPutBlob /
   AsyncGetBlob. */
/* Which wrapper struct a void* actually points at. clio_dataset_t and
   clio_file_t are NOT derived from clio_obj_t -- they CONTAIN one as their first
   member -- so a `delete (clio_obj_t *)p` on a dataset is undefined behaviour
   and leaks its std::string/vectors. Every wrapper records its kind here so the
   generic paths (notably clio_unwrap_object) can destroy the right type. */
enum class clio_kind_t : int { kObj = 0, kFile, kDataset };

struct clio_obj_t {
  void           *under_object;
  hid_t           under_vol_id;
  clio_file_t  *parent_file;
  clio_kind_t     kind = clio_kind_t::kObj;
};

struct clio_file_t {
  clio_obj_t obj;
  clio::cte::core::TagId tag_id;
  std::string file_name;
  size_t chunk_size;
  /* False when the CTE tier is bypassed entirely for this file: either the user
     disabled it (CLIO_VOL_CACHE=0) or the CLIO runtime was unreachable at open.
     The connector then behaves as a pure pass-through to the native VOL, which
     is always a correct (if unaccelerated) mode -- the native file is
     authoritative regardless. */
  bool cache_enabled = true;
  /* Safe mode: cacheable datasets currently open in this file. H5Fflush and
     H5Fclose drain their pending CTE puts so no async write outlives a
     successful flush/close (the native file is already written synchronously and
     stays authoritative; this keeps the CTE cache coherent with it). */
  std::mutex ds_mtx;
  std::unordered_set<clio_dataset_t *> open_datasets;
  /* Access telemetry (observe-only); non-null only when CLIO_VOL_TRACE is set.
     Finalized to a summary JSON at file close. */
  clio::trace::FileTrace *trace = nullptr;
};

struct clio_dataset_t {
  clio_obj_t obj;
  clio_file_t *file;
  std::string dataset_path;
  /* The dataset's FILE datatype, resolved lazily on first cache-eligible
     transfer. The blob cache is a flat byte image sized by the transfer's
     datatype, so it is only coherent when the caller's memory type is the same
     size as what HDF5 actually stores. Without this check, writing
     H5T_NATIVE_INT into an H5T_STD_I64LE dataset caches 4n bytes and a later
     read with H5T_NATIVE_LONG asks for 8n, hits, and reassembles garbage. */
  hid_t file_type = H5I_INVALID_HID;
  size_t file_type_size = 0;
  int file_type_probed = 0;
  /* When false the CTE cache is bypassed and every transfer goes to the native
     VOL. Set for datasets whose stable path is unknown (opened via the generic
     object-open / wrap paths), so we never key a blob by an empty/ambiguous
     name. */
  bool cacheable;
  /* Pending async writes flushed on close */
  std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> pending_puts;
  std::vector<ctp::ipc::FullPtr<char>> pending_buffers;
  /* Telemetry-only storage-layout probe, filled lazily on first traced access
     (never touched unless CLIO_VOL_TRACE is set). */
  int layout_probed = 0;         /* 0 = not yet probed, 1 = probed */
  bool chunked = false;
  int chunk_rank = 0;
  hsize_t chunk_dims[H5S_MAX_RANK];
};

/* Build a dataset wrapper. Centralised so every code path that can produce a
   dataset object (dataset_open/create, object_open, wrap_object) yields the
   same fully-formed clio_dataset_t — otherwise a dataset returned as a bare
   clio_obj_t would be fatally mis-cast when HDF5 later routes
   dataset_read/close to it. */
static clio_dataset_t *make_dataset_wrapper(void *under, hid_t under_vol_id,
                                              clio_file_t *parent_file,
                                              const char *path) {
  auto *dset = new clio_dataset_t;
  dset->obj.under_object = under;
  dset->obj.under_vol_id = under_vol_id;
  dset->obj.kind = clio_kind_t::kDataset;
  dset->file = parent_file;
  dset->obj.parent_file = parent_file;
  dset->dataset_path = path ? path : "";
  dset->cacheable = (parent_file != nullptr) && parent_file->cache_enabled &&
                    path && path[0] != '\0';
  /* Register with the file so Safe-mode flush/close can drain this dataset's
     pending puts. Only cacheable datasets accumulate CTE puts. */
  if (dset->cacheable) {
    std::lock_guard<std::mutex> lk(parent_file->ds_mtx);
    parent_file->open_datasets.insert(dset);
  }
  return dset;
}

/* Block until every async CTE put for this dataset has completed, then release
   the shared-memory buffers. Shared by dataset_close and Safe-mode flush/close.

   Returns false if ANY put reported a failure. Waiting is not the same as
   succeeding: a put that landed with a non-zero return code left the cache
   holding less than it believes it does, so the caller invalidates the
   dataset's cache rather than leave a partially-staged image that a later read
   would treat as a hit. The native file is authoritative and unaffected. */
static bool drain_dataset_puts(clio_dataset_t *dset) {
  bool ok = true;
  for (auto &future : dset->pending_puts) {
    future.Wait();
    if (future->GetReturnCode() != 0) {
      ok = false;
    }
  }
  dset->pending_puts.clear();
  dset->pending_buffers.clear();
  return ok;
}

/* VOL object-wrap context. HDF5 uses this during link/object iteration
   (H5Literate2 / H5Lvisit2 / H5Ovisit2): before invoking the user's operator it
   saves a wrap context from this connector, then for every iterated object it
   calls clio_wrap_object() to re-wrap the native object back into our VOL so
   the operator's hid_t routes through this connector. Mirrors the reference
   pass-through connector (H5VLpassthru). The previous implementation returned
   the native wrap context directly and had clio_wrap_object() skip
   H5VLwrap_object(), which left the iterated object only half-wrapped and made
   H5Literate2 abort with "... is not a VOL connector ID" — the deeper blocker
   that kept neuroh5's group enumeration from completing. */
struct clio_wrap_ctx_t {
  hid_t under_vol_id;
  void *under_wrap_ctx;
  /* The clio_file_t the wrapped-over container belongs to, so datasets
     re-wrapped during H5Literate/H5Ovisit iteration inherit their file (and
     thus its CLIO_VOL_TRACE FileTrace). Without it, iterated datasets get a
     null file and their reads/writes go untraced. */
  clio_file_t *parent_file;
};

/* ========================================================================
 * Helper: Get CTE client
 * ======================================================================== */

static clio::cte::core::Client *get_cte_client() {
  /* Lazily attach this process to the running clio/CTE runtime on first
     use. When HDF5 dlopen()s the connector via HDF5_VOL_CONNECTOR there is no
     LD_PRELOAD constructor to do it (the POSIX adapter inits in
     Filesystem::Filesystem -> CLIO_CTE_CLIENT_INIT()); without this the CTE
     client singleton is unbound and the first AsyncGetOrCreateTag segfaults.
     Config comes from CLIO_SERVER_CONF, same as the runtime.

     Returns nullptr when the runtime is unreachable. Callers MUST check: the
     attach result was previously discarded, so an absent runtime faulted inside
     H5Fcreate. There is always a correct fallback -- pass everything through to
     the native VOL -- because the native file is authoritative and the CTE tier
     is a performance layer, not a correctness one. */
  static std::once_flag once;
  static bool attached = false;
  std::call_once(once, []() {
    attached = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    if (!attached) {
      fprintf(stderr,
              "[clio-vol] CLIO runtime unavailable; running as a pure "
              "pass-through to the native VOL (cache disabled)\n");
    }
  });
  return attached ? CLIO_CTE_CLIENT : nullptr;
}

/* Is the CTE cache path usable for this file at all? One check for the two
   independent reasons it may not be: the user turned it off, or the runtime is
   not there. */
static bool clio_cache_usable(clio_file_t *file) {
  return file && file->cache_enabled && get_cte_client() != nullptr;
}

/* Read the CLIO_VOL_CACHE opt-out. Default on; "0"/"off"/"false"/"no" disable
   the CTE tier and make the connector a pure pass-through. Without this there
   was NO configuration in which the connector could be loaded and not require a
   CLIO runtime, which is what Safe mode's environment-robustness clause needs. */
static bool clio_cache_env_enabled() {
  const char *v = std::getenv("CLIO_VOL_CACHE");
  if (!v || !*v) return true;
  return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
           std::strcmp(v, "false") == 0 || std::strcmp(v, "no") == 0);
}

/* Drop this dataset's cached image so later reads miss and re-stage from the
   authoritative native file. Used whenever the cache may no longer mirror the
   file: a partial write, or a put that failed to land. Deleting chunk_0 (the
   hit-test key) is sufficient -- a miss re-stages every chunk -- but the delete
   itself must be CHECKED, because a failed invalidation leaves the cache
   claiming to hold pre-write data. On failure the dataset is marked
   uncacheable for the rest of the session, which is the fail-closed choice. */
static void clio_invalidate_dataset(clio_dataset_t *dset) {
  if (!dset || !dset->file || !dset->cacheable) return;
  auto *cte_client = get_cte_client();
  if (!cte_client) {
    dset->cacheable = false;
    return;
  }
  auto del = cte_client->AsyncDelBlob(dset->file->tag_id,
                                      dset->dataset_path + "/chunk_0");
  del.Wait();
  if (del->GetReturnCode() != 0) {
    fprintf(stderr,
            "[clio-vol] cache invalidation failed for %s; bypassing the cache "
            "for this dataset (native file is authoritative)\n",
            dset->dataset_path.c_str());
    dset->cacheable = false;
  }
}

/* ========================================================================
 * Info callbacks
 * ======================================================================== */

static void *clio_info_copy(const void *_info) {
  const auto *info = static_cast<const clio_vol_info_t *>(_info);
  auto *new_info = new clio_vol_info_t(*info);
  if (info->under_vol_info) {
    H5VLcopy_connector_info(info->under_vol_id, &new_info->under_vol_info,
                            info->under_vol_info);
  }
  return new_info;
}

static herr_t clio_info_free(void *_info) {
  auto *info = static_cast<clio_vol_info_t *>(_info);
  if (info->under_vol_info) {
    H5VLfree_connector_info(info->under_vol_id, info->under_vol_info);
  }
  delete info;
  return 0;
}

/* ========================================================================
 * Wrap / unwrap callbacks
 * ======================================================================== */

static void *clio_wrap_get_object(const void *obj) {
  auto *o = static_cast<const clio_obj_t *>(obj);
  return H5VLget_object(o->under_object, o->under_vol_id);
}

static herr_t clio_get_wrap_ctx(const void *obj, void **wrap_ctx) {
  auto *o = static_cast<const clio_obj_t *>(obj);
  /* Carry the under VOL id alongside its wrap context so clio_wrap_object()
     can call H5VLwrap_object() against the right connector. */
  auto *ctx = new clio_wrap_ctx_t;
  ctx->under_vol_id = o->under_vol_id;
  H5Iinc_ref(ctx->under_vol_id);
  ctx->under_wrap_ctx = nullptr;
  ctx->parent_file = o->parent_file;
  H5VLget_wrap_ctx(o->under_object, o->under_vol_id, &ctx->under_wrap_ctx);
  *wrap_ctx = ctx;
  return 0;
}

static void *clio_wrap_object(void *under_obj, H5I_type_t obj_type,
                                void *_wrap_ctx) {
  auto *ctx = static_cast<clio_wrap_ctx_t *>(_wrap_ctx);
  hid_t under_vol_id = ctx ? ctx->under_vol_id : H5VL_NATIVE;
  void *under_wrap_ctx = ctx ? ctx->under_wrap_ctx : nullptr;

  /* Let the underlying (native) VOL wrap the raw iteration object first.
     Skipping this step — as the old code did — left the object unusable by the
     native VOL and made link/object iteration abort. */
  void *under = H5VLwrap_object(under_obj, obj_type, under_vol_id,
                                under_wrap_ctx);
  if (!under) return nullptr;

  /* The stable dataset *path* is still unknown during iteration, so wrapped
     datasets stay non-cacheable (make_dataset_wrapper keys cacheability on a
     non-empty path) and their transfers fall back to the native VOL. But the
     parent file IS known via the wrap context, so thread it through: a wrapped
     dataset then inherits file->trace and its reads/writes are recorded by
     CLIO_VOL_TRACE (previously they were silently dropped). */
  clio_file_t *parent_file = ctx ? ctx->parent_file : nullptr;
  if (obj_type == H5I_DATASET) {
    return make_dataset_wrapper(under, under_vol_id, parent_file, nullptr);
  }
  auto *o = new clio_obj_t;
  o->under_object = under;
  o->under_vol_id = under_vol_id;
  o->parent_file = parent_file;
  return o;
}

/* Destroy a wrapper through its ACTUAL type. clio_dataset_t contains (does not
   inherit) a clio_obj_t, so deleting one as a clio_obj_t* is undefined
   behaviour and leaks its std::string, its two vectors, and the SHM buffers
   they own. The kind tag exists to make this dispatch possible. */
static void clio_destroy_wrapper(clio_obj_t *o) {
  if (!o) return;
  switch (o->kind) {
    case clio_kind_t::kDataset: {
      auto *dset = reinterpret_cast<clio_dataset_t *>(o);
      if (dset->file && dset->cacheable) {
        std::lock_guard<std::mutex> lk(dset->file->ds_mtx);
        dset->file->open_datasets.erase(dset);
      }
      if (dset->file_type >= 0) H5Tclose(dset->file_type);
      delete dset;
      break;
    }
    case clio_kind_t::kFile:
      delete reinterpret_cast<clio_file_t *>(o);
      break;
    case clio_kind_t::kObj:
    default:
      delete o;
      break;
  }
}

static void *clio_unwrap_object(void *obj) {
  auto *o = static_cast<clio_obj_t *>(obj);
  /* Symmetric with clio_wrap_object()'s H5VLwrap_object(): peel the native
     wrapper back off before discarding our wrapper. */
  void *under = H5VLunwrap_object(o->under_object, o->under_vol_id);
  /* Free the wrapper either way -- returning NULL previously leaked it. */
  clio_destroy_wrapper(o);
  return under;
}

static herr_t clio_free_wrap_ctx(void *_wrap_ctx) {
  auto *ctx = static_cast<clio_wrap_ctx_t *>(_wrap_ctx);
  if (!ctx) return 0;
  /* Preserve the active HDF5 error stack across the cleanup calls, per the
     reference connector. */
  hid_t err_id = H5Eget_current_stack();
  if (ctx->under_wrap_ctx) {
    H5VLfree_wrap_ctx(ctx->under_wrap_ctx, ctx->under_vol_id);
  }
  H5Idec_ref(ctx->under_vol_id);
  H5Eset_current_stack(err_id);
  delete ctx;
  return 0;
}

/* ========================================================================
 * File callbacks
 * ======================================================================== */

/* Resolve the CTE chunk size for a file: connector info, then CLIO_VOL_CHUNK_SIZE,
   then the built-in default. Shared so create and open cannot disagree -- open
   previously never read the info struct at all, so a chunk_size set through
   H5Pset_vol was silently honoured on create and ignored on open. */
static size_t clio_resolve_chunk_size(hid_t fapl_id) {
  size_t chunk_size = CLIO_VOL_DEFAULT_CHUNK_SIZE;
  clio_vol_info_t *vol_info = nullptr;
  H5Pget_vol_info(fapl_id, reinterpret_cast<void **>(&vol_info));
  if (vol_info && vol_info->chunk_size > 0) {
    chunk_size = vol_info->chunk_size;
  }
  const char *env_chunk = std::getenv("CLIO_VOL_CHUNK_SIZE");
  if (env_chunk) {
    chunk_size = std::strtoul(env_chunk, nullptr, 10);
    if (chunk_size == 0) chunk_size = CLIO_VOL_DEFAULT_CHUNK_SIZE;
  }
  return chunk_size;
}

/* Build the connector's file wrapper around an already-opened native file, and
   bind its CTE tag if the cache is usable. `truncated` means the native file was
   just emptied (H5F_ACC_TRUNC), so any tag left over from a PREVIOUS file of the
   same name describes data that no longer exists and must be dropped -- the tag
   is keyed on the filename, which a truncate does not change. Without this,
   creating a file over an old one and then reading a dataset the new file never
   wrote would hit the old file's cached blobs. */
static clio_file_t *clio_make_file(void *under_file, const char *name,
                                     size_t chunk_size, bool truncated) {
  auto *file = new clio_file_t;
  file->obj.under_object = under_file;
  file->obj.under_vol_id = H5VL_NATIVE;
  file->obj.parent_file = file;          /* self-pointer */
  file->obj.kind = clio_kind_t::kFile;
  file->file_name = name;
  file->chunk_size = chunk_size;
  file->cache_enabled = clio_cache_env_enabled();
  file->trace = clio::trace::open_file(name);

  auto *cte_client = file->cache_enabled ? get_cte_client() : nullptr;
  if (!cte_client) {
    /* Pure pass-through: no tag, no cache. Correct, just unaccelerated. */
    file->cache_enabled = false;
    return file;
  }

  const std::string tag_name = std::string("hdf5:") + name;
  if (truncated) {
    auto del = cte_client->AsyncDelTag(tag_name);
    del.Wait();  /* absent tag is a harmless no-op */
  }
  auto tag_task = cte_client->AsyncGetOrCreateTag(tag_name);
  tag_task.Wait();
  if (tag_task->GetReturnCode() != 0) {
    file->cache_enabled = false;
    return file;
  }
  file->tag_id = tag_task->tag_id_;
  return file;
}

static void *clio_file_create(const char *name, unsigned flags,
                                hid_t fcpl_id, hid_t fapl_id,
                                hid_t dxpl_id, void **req) {
  size_t chunk_size = clio_resolve_chunk_size(fapl_id);

  /* Create file via native VOL */
  hid_t native_fapl = H5Pcopy(fapl_id);
  H5Pset_vol(native_fapl, H5VL_NATIVE, nullptr);
  void *under_file = H5VLfile_create(name, flags, fcpl_id, native_fapl,
                                      dxpl_id, req);
  H5Pclose(native_fapl);
  if (!under_file) return nullptr;

  /* H5Fcreate always yields an empty file (TRUNC replaces an existing one,
     EXCL/CREAT means there was none), so any pre-existing tag is stale. */
  return clio_make_file(under_file, name, chunk_size, /*truncated*/ true);
}

static void *clio_file_open(const char *name, unsigned flags,
                              hid_t fapl_id, hid_t dxpl_id, void **req) {
  size_t chunk_size = clio_resolve_chunk_size(fapl_id);

  /* Open file via native VOL */
  hid_t native_fapl = H5Pcopy(fapl_id);
  H5Pset_vol(native_fapl, H5VL_NATIVE, nullptr);
  void *under_file = H5VLfile_open(name, flags, native_fapl, dxpl_id, req);
  H5Pclose(native_fapl);
  if (!under_file) return nullptr;

  return clio_make_file(under_file, name, chunk_size, /*truncated*/ false);
}

static herr_t clio_file_get(void *obj, H5VL_file_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  auto *file = static_cast<clio_file_t *>(obj);
  return H5VLfile_get(file->obj.under_object, file->obj.under_vol_id,
                      args, dxpl_id, req);
}

static herr_t clio_file_specific(void *obj,
                                   H5VL_file_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  /* OBJECT-LESS OPERATIONS COME FIRST -- they are called with obj == NULL.
     H5VL_FILE_IS_ACCESSIBLE and H5VL_FILE_DELETE act on a FILENAME, not on an
     open file, so there is no object to unwrap and the cast below would be a
     NULL dereference.

     This is not a theoretical path. HDF5 walks it whenever an H5Fopen has to
     work out WHICH connector can open a file: H5VL__file_open_find_connector_cb
     iterates every plugin on HDF5_PLUGIN_PATH and asks each one
     IS_ACCESSIBLE. So merely having this .so sitting in the plugin directory
     was enough to turn a plain `H5Fopen` of a missing (or non-HDF5) file into a
     SEGFAULT -- with the clio connector not selected, not requested, and
     nothing in the application referring to it. Selecting it explicitly hid the
     bug, because that path opens directly and never probes.

     Delegation mirrors clio_file_open/create: copy the caller's FAPL, point it
     at the native connector, and hand the operation to native with a NULL
     object. args->args.*.fapl_id must be swapped too -- it is the FAPL native
     will re-read, and leaving it pointing at a clio-VOL FAPL sends HDF5 straight
     back into this connector (infinite recursion instead of an answer). */
  if (args && (args->op_type == H5VL_FILE_IS_ACCESSIBLE ||
               args->op_type == H5VL_FILE_DELETE)) {
    hid_t *fapl_slot = (args->op_type == H5VL_FILE_IS_ACCESSIBLE)
                           ? &args->args.is_accessible.fapl_id
                           : &args->args.del.fapl_id;
    hid_t saved_fapl = *fapl_slot;
    hid_t native_fapl = H5Pcopy(saved_fapl);
    if (native_fapl < 0) return -1;
    H5Pset_vol(native_fapl, H5VL_NATIVE, nullptr);
    *fapl_slot = native_fapl;
    herr_t ret = H5VLfile_specific(nullptr, H5VL_NATIVE, args, dxpl_id, req);
    *fapl_slot = saved_fapl; /* the caller owns theirs; give it back intact */
    H5Pclose(native_fapl);
    return ret;
  }

  auto *file = static_cast<clio_file_t *>(obj);
  if (!file) return -1; /* every remaining op requires an open file */
  /* Safe mode: on H5Fflush, drain every open cacheable dataset's pending CTE
     puts BEFORE delegating the flush to native, so no async write outlives a
     successful flush. A put that failed invalidates that dataset's cache rather
     than leaving a half-staged image a later read would treat as a hit. The
     native file is written synchronously and stays authoritative throughout. */
  if (args && args->op_type == H5VL_FILE_FLUSH) {
    /* Snapshot under the lock, drain outside it: draining blocks on every
       in-flight put, and holding ds_mtx across that serialises flush against
       every concurrent dataset open/close. */
    std::vector<clio_dataset_t *> to_drain;
    {
      std::lock_guard<std::mutex> lk(file->ds_mtx);
      to_drain.assign(file->open_datasets.begin(), file->open_datasets.end());
    }
    for (auto *dset : to_drain) {
      if (!drain_dataset_puts(dset)) {
        clio_invalidate_dataset(dset);
      }
    }
  }
  return H5VLfile_specific(file->obj.under_object, file->obj.under_vol_id,
                           args, dxpl_id, req);
}

/* Pass-through file "optional" — MUST forward to the under-VOL. HDF5 finalizes
   every file open/create by invoking the native-specific optional op
   H5VL_NATIVE_FILE_POST_OPEN, which runs H5F__post_open() to populate the file's
   VOL object (H5F_VOL_OBJ). With this callback left null the op never reached
   native, so H5F_VOL_OBJ stayed NULL and any variable-length datatype crashed at
   create time in H5T__vlen_set_loc -> H5VL_file_get(NULL). Mirrors H5VLpassthru. */
static herr_t clio_file_optional(void *obj, H5VL_optional_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *file = static_cast<clio_file_t *>(obj);
  return H5VLfile_optional(file->obj.under_object, file->obj.under_vol_id,
                           args, dxpl_id, req);
}

static herr_t clio_file_close(void *obj, hid_t dxpl_id, void **req) {
  auto *file = static_cast<clio_file_t *>(obj);
  /* Safe mode: drain any datasets still open at file-close time (e.g. weak
     close degree) so no async CTE put outlives the file. Datasets closed the
     normal way have already drained and unregistered themselves. */
  {
    std::vector<clio_dataset_t *> to_drain;
    {
      std::lock_guard<std::mutex> lk(file->ds_mtx);
      to_drain.assign(file->open_datasets.begin(), file->open_datasets.end());
    }
    for (auto *dset : to_drain) {
      if (!drain_dataset_puts(dset)) {
        clio_invalidate_dataset(dset);
      }
    }
  }
  herr_t ret = H5VLfile_close(file->obj.under_object, file->obj.under_vol_id,
                               dxpl_id, req);
  clio::trace::close_file(file->trace);  /* writes the summary; no-op if null */
  delete file;
  return ret;
}

/* ========================================================================
 * Dataset callbacks
 * ======================================================================== */

/**
 * Helper: extract the clio_file_t from an obj pointer.
 *
 * Every wrapper (file, group, dataset, attr) has an clio_obj_t as its
 * first member with a parent_file pointer. Files set it to themselves;
 * groups inherit it from their parent; datasets/attrs inherit it from
 * their containing file or group.
 *
 * If obj came in without parent_file populated (e.g. from
 * clio_wrap_object where the wrap context didn't include a file
 * reference), this returns nullptr and the caller falls back to pure
 * native-VOL passthrough.
 */
static clio_file_t *find_parent_file(void *obj) {
  if (!obj) return nullptr;
  /* All clio wrappers (clio_obj_t, clio_file_t, clio_dataset_t)
     start with clio_obj_t as their first member, so reading
     parent_file via this cast is safe. */
  return static_cast<clio_obj_t *>(obj)->parent_file;
}

/**
 * Helper: is this a whole-dataset transfer we can represent as linear CTE
 * chunks? The blob cache is keyed by linear byte offset over the full dataset
 * extent, so it can only correctly serve transfers covering the entire dataset
 * contiguously (H5S_ALL on both mem and file spaces). Any hyperslab / point
 * selection must go to the native VOL.
 */
static bool clio_is_whole_read(hid_t mem_space_id, hid_t file_space_id) {
  return mem_space_id == H5S_ALL && file_space_id == H5S_ALL;
}

/**
 * Helper: true when the transfer plist requests collective MPI-IO. Collective
 * transfers must stay on the native VOL — serving some ranks from cache while
 * others miss would desynchronise the collective call and deadlock.
 */
static bool clio_is_collective(hid_t dxpl_id) {
#ifdef H5_HAVE_PARALLEL
  if (dxpl_id == H5P_DEFAULT) return false;
  H5FD_mpio_xfer_t xfer = H5FD_MPIO_INDEPENDENT;
  if (H5Pget_dxpl_mpio(dxpl_id, &xfer) >= 0 && xfer == H5FD_MPIO_COLLECTIVE) {
    return true;
  }
#else
  (void)dxpl_id;
#endif
  return false;
}

/**
 * Helper: true when the datatype's in-memory transfer buffer is a flat byte image
 * IDENTICAL to what HDF5 stores on disk, so a raw memcpy into the linear CTE chunk
 * cache faithfully mirrors the file and can be served back (whole or via
 * selection) without diverging from native.
 *
 * Restricted to ATOMIC classes (integer, float, enum, bitfield, fixed-length
 * string). Excluded:
 *   - vlen strings / vlen sequences (hvl_t) / references — the buffer holds
 *     POINTERS, not content; caching them would store pointer values.
 *   - compound and array datatypes — their memory image can differ from the
 *     on-disk element image (member padding, subarray semantics), so the cached
 *     write buffer may diverge from what HDF5 actually stores. Observed with a
 *     subarray datatype whose file image was zero-filled while the write buffer
 *     was not: serving that from cache returned data native never wrote.
 * Excluded types are delegated to the native VOL only (the native file still holds
 * the real data; it is simply not mirrored into CTE).
 */
static bool clio_type_is_cacheable(hid_t type_id) {
  if (type_id < 0) return false;
  if (H5Tis_variable_str(type_id) > 0) return false;
  switch (H5Tget_class(type_id)) {
    case H5T_INTEGER:
    case H5T_FLOAT:
    case H5T_ENUM:
    case H5T_BITFIELD:
    case H5T_STRING:   /* vlen strings already excluded above */
      return true;
    default:           /* compound, array, vlen, reference, opaque, time */
      return false;
  }
}

/**
 * Recursively true when a datatype is a fixed-size, self-contained POD image —
 * no embedded pointers — so its in-memory transfer buffer can be byte-copied
 * into / out of the linear CTE cache. Extends clio_type_is_cacheable to fixed
 * COMPOUND types (whose members are themselves flat), which the READ cache can
 * safely mirror: on a miss we cache exactly what the native VOL produced and
 * serve those same bytes on a hit, so no divergence is possible.
 * Excluded (return false):
 *   - vlen sequences, variable-length strings, object/region references — the
 *     buffer holds POINTERS, so a raw copy caches addresses, not content.
 *   - ARRAY datatypes — their element image round-trips incorrectly through the
 *     byte cache (the VOL compat suite's array_dtype roundtrip regressed), so
 *     they stay on the native path with the atomic/compound scalars below. Also
 *     excludes any compound with a nested array member (recursion returns false).
 *
 * NOTE: read-only. The write path keeps clio_type_is_cacheable (atomic-only),
 * because a *written* compound buffer can differ from the element image HDF5
 * actually persists (member padding), which would let a cached read return
 * bytes native never wrote.
 */
static bool clio_type_is_read_cacheable(hid_t t) {
  if (t < 0) return false;
  switch (H5Tget_class(t)) {
    case H5T_INTEGER:
    case H5T_FLOAT:
    case H5T_ENUM:
    case H5T_BITFIELD:
    case H5T_OPAQUE:
    case H5T_TIME:
      return true;
    case H5T_STRING:
      return H5Tis_variable_str(t) <= 0;  /* fixed-length only */
    case H5T_COMPOUND: {
      int n = H5Tget_nmembers(t);
      if (n < 0) return false;
      for (int i = 0; i < n; ++i) {
        hid_t m = H5Tget_member_type(t, i);
        bool ok = clio_type_is_read_cacheable(m);
        if (m >= 0) H5Tclose(m);
        if (!ok) return false;
      }
      return true;
    }
    default:  /* H5T_ARRAY, H5T_VLEN, H5T_REFERENCE, ... — unsafe to byte-cache */
      return false;
  }
}

/**
 * True when a dataspace selection is the ENTIRE extent in natural, contiguous
 * order — i.e. equivalent to a whole read for linear-cache purposes. Accepts
 * H5S_ALL, H5S_SEL_ALL, and a single regular hyperslab that starts at the origin
 * and covers every dimension with unit stride (start=0, stride=block,
 * count*block=dims). This is what h5dump / H5Dread of a full dataset issue —
 * often as a full hyperslab rather than H5S_ALL — so treating it as whole lets
 * those reads populate and be served from the CTE tier instead of falling into
 * the serve-only partial path (which never stages).
 */
static bool clio_space_is_natural_full(hid_t s) {
  if (s == H5S_ALL) return true;
  int rank = H5Sget_simple_extent_ndims(s);
  if (rank < 0) return false;
  hssize_t sel = H5Sget_select_npoints(s);
  hssize_t ext = H5Sget_simple_extent_npoints(s);
  if (sel <= 0 || sel != ext) return false;
  H5S_sel_type st = H5Sget_select_type(s);
  if (st == H5S_SEL_ALL) return true;
  if (st != H5S_SEL_HYPERSLABS) return false;
  if (H5Sis_regular_hyperslab(s) <= 0) return false;
  hsize_t start[H5S_MAX_RANK], stride[H5S_MAX_RANK], cnt[H5S_MAX_RANK],
      blk[H5S_MAX_RANK], dims[H5S_MAX_RANK];
  if (H5Sget_regular_hyperslab(s, start, stride, cnt, blk) < 0) return false;
  if (H5Sget_simple_extent_dims(s, dims, nullptr) < 0) return false;
  for (int i = 0; i < rank; ++i) {
    if (start[i] != 0) return false;
    if (stride[i] != blk[i]) return false;      /* contiguous, no gaps */
    if (cnt[i] * blk[i] != dims[i]) return false; /* covers the full extent */
  }
  return true;
}

/** Read counterpart to clio_is_whole_read that also accepts a full-extent
 *  selection (see clio_space_is_natural_full). */
static bool clio_read_is_whole(hid_t mem_space_id, hid_t file_space_id) {
  return clio_space_is_natural_full(mem_space_id) &&
         clio_space_is_natural_full(file_space_id);
}

/**
 * True when the transfer's MEMORY datatype has the same element size as what
 * HDF5 stores on disk for this dataset.
 *
 * The blob cache is a flat byte image whose length is derived from the
 * transfer's datatype size (nelem * H5Tget_size(mem_type)). That is only a
 * coherent key when the memory image and the file image agree on element size;
 * otherwise the SAME dataset has different cached lengths depending on which
 * type the caller happened to use, and a hit computed with one size reassembles
 * blobs written with another. Concretely, without this check:
 *
 *   H5Dwrite(d, H5T_NATIVE_INT,  ...)   on an H5T_STD_I64LE dataset -> caches 4n
 *   H5Dread (d, H5T_NATIVE_LONG, ...)   asks for 8n, hit-tests chunk_0 (present),
 *                                       and reassembles 8n bytes from 4n -> the
 *                                       tail is short/zero-filled garbage,
 *                                       returned with a success status.
 *
 * HDF5 converts between memory and file types freely, so this is a legal and
 * unremarkable thing for an application to do. The dataset's file type is
 * fetched once and cached on the wrapper.
 */
static bool clio_type_matches_file(clio_dataset_t *dataset, hid_t mem_type_id,
                                     hid_t dxpl_id) {
  if (mem_type_id < 0) return false;
  if (!dataset->file_type_probed) {
    dataset->file_type_probed = 1;
    H5VL_dataset_get_args_t ga;
    ga.op_type = H5VL_DATASET_GET_TYPE;
    ga.args.get_type.type_id = H5I_INVALID_HID;
    if (H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id,
                        &ga, dxpl_id, nullptr) >= 0 &&
        ga.args.get_type.type_id >= 0) {
      dataset->file_type = ga.args.get_type.type_id;
      dataset->file_type_size = H5Tget_size(dataset->file_type);
    }
  }
  if (dataset->file_type_size == 0) return false;  /* unknown -> don't cache */
  return H5Tget_size(mem_type_id) == dataset->file_type_size;
}

static void *clio_dataset_create(void *obj,
                                   const H5VL_loc_params_t *loc_params,
                                   const char *name, hid_t lcpl_id,
                                   hid_t type_id, hid_t space_id,
                                   hid_t dcpl_id, hid_t dapl_id,
                                   hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);

  /* Create dataset via native VOL */
  void *under_dset = H5VLdataset_create(
      o->under_object, loc_params, o->under_vol_id, name,
      lcpl_id, type_id, space_id, dcpl_id, dapl_id, dxpl_id, req);
  if (!under_dset) return nullptr;

  return make_dataset_wrapper(under_dset, o->under_vol_id,
                              find_parent_file(obj), name);
}

static void *clio_dataset_open(void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 const char *name, hid_t dapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);

  void *under_dset = H5VLdataset_open(
      o->under_object, loc_params, o->under_vol_id, name,
      dapl_id, dxpl_id, req);
  if (!under_dset) return nullptr;

  return make_dataset_wrapper(under_dset, o->under_vol_id,
                              find_parent_file(obj), name);
}

/* ---- Access telemetry helpers (observe-only; active only under CLIO_VOL_TRACE) */

static const char *clio_dtype_class_name(hid_t t) {
  if (t < 0) return "unknown";
  if (H5Tis_variable_str(t) > 0) return "vlen_string";
  switch (H5Tget_class(t)) {
    case H5T_INTEGER: return "integer";
    case H5T_FLOAT: return "float";
    case H5T_STRING: return "string";
    case H5T_COMPOUND: return "compound";
    case H5T_ARRAY: return "array";
    case H5T_ENUM: return "enum";
    case H5T_VLEN: return "vlen";
    case H5T_REFERENCE: return "reference";
    case H5T_BITFIELD: return "bitfield";
    case H5T_OPAQUE: return "opaque";
    default: return "other";
  }
}

/* Element count a transfer touches: the file-space selection if given, else the
   mem-space selection, else the whole dataset (needs one GET_SPACE). Only called
   when telemetry is enabled. */
static long long clio_sel_nelem(clio_dataset_t *dataset, hid_t mem_space,
                                  hid_t file_space, hid_t dxpl) {
  if (file_space != H5S_ALL) return H5Sget_select_npoints(file_space);
  if (mem_space != H5S_ALL) return H5Sget_select_npoints(mem_space);
  H5VL_dataset_get_args_t ga;
  ga.op_type = H5VL_DATASET_GET_SPACE;
  ga.args.get_space.space_id = H5I_INVALID_HID;
  if (H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id, &ga,
                      dxpl, nullptr) < 0)
    return -1;
  hid_t sp = ga.args.get_space.space_id;
  if (sp < 0) return -1;
  hssize_t n = H5Sget_simple_extent_npoints(sp);
  H5Sclose(sp);
  return n;
}

/* Probe the dataset's storage layout once (chunked? chunk dims), caching the
   result on the dataset. Only reached when telemetry is enabled. */
static void clio_probe_layout(clio_dataset_t *dataset, hid_t dxpl_id) {
  if (dataset->layout_probed) return;
  dataset->layout_probed = 1;
  H5VL_dataset_get_args_t ga;
  ga.op_type = H5VL_DATASET_GET_DCPL;
  ga.args.get_dcpl.dcpl_id = H5I_INVALID_HID;
  if (H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id, &ga,
                      dxpl_id, nullptr) < 0)
    return;
  hid_t dcpl = ga.args.get_dcpl.dcpl_id;
  if (dcpl < 0) return;
  if (H5Pget_layout(dcpl) == H5D_CHUNKED) {
    int r = H5Pget_chunk(dcpl, H5S_MAX_RANK, dataset->chunk_dims);
    if (r > 0) { dataset->chunked = true; dataset->chunk_rank = r; }
  }
  H5Pclose(dcpl);
}

/* Classify a file-space selection's alignment to the chunk grid: 1 aligned,
   0 misaligned, -1 n/a (contiguous dataset, or a non-hyperslab selection).
   Whole reads of a chunked dataset are aligned by definition. For a hyperslab,
   the bounding box must snap to chunk boundaries (start on a chunk edge and end
   on a chunk edge or the dataset extent) — misalignment is the read-amplification
   / rechunk signal. */
static int clio_chunk_alignment(clio_dataset_t *dataset, hid_t file_space_id) {
  if (!dataset->chunked) return -1;
  if (file_space_id == H5S_ALL) return 1;
  H5S_sel_type st = H5Sget_select_type(file_space_id);
  if (st == H5S_SEL_ALL) return 1;
  if (st != H5S_SEL_HYPERSLABS) return -1;  /* points: alignment not meaningful */
  int rank = H5Sget_simple_extent_ndims(file_space_id);
  if (rank <= 0 || rank > H5S_MAX_RANK || rank != dataset->chunk_rank) return -1;
  hsize_t start[H5S_MAX_RANK], end[H5S_MAX_RANK], ext[H5S_MAX_RANK];
  if (H5Sget_select_bounds(file_space_id, start, end) < 0) return -1;
  if (H5Sget_simple_extent_dims(file_space_id, ext, nullptr) < 0) return -1;
  for (int i = 0; i < rank; ++i) {
    hsize_t c = dataset->chunk_dims[i];
    if (c == 0) return -1;
    if (start[i] % c != 0) return 0;                 /* start not on a chunk edge */
    hsize_t past = end[i] + 1;                        /* one past the last element */
    if (past % c != 0 && past != ext[i]) return 0;    /* end not on a chunk/extent edge */
  }
  return 1;
}

/* Record one read/write access. No-op unless the file has telemetry enabled. */
static void clio_trace_access(clio_dataset_t *dataset, clio::trace::Op op,
                                hid_t mem_type_id, hid_t mem_space_id,
                                hid_t file_space_id, hid_t dxpl_id,
                                clio::trace::Served served, double dur_us) {
  if (!dataset || !dataset->file || !dataset->file->trace) return;
  clio::trace::Access a;
  a.op = op;
  a.served = served;
  a.dataset = dataset->dataset_path;
  a.sel = clio::trace::classify(file_space_id, &a.sel_sig);
  a.dtype = clio_dtype_class_name(mem_type_id);
  a.elem_size = (mem_type_id >= 0) ? H5Tget_size(mem_type_id) : 0;
  a.ndims = (file_space_id != H5S_ALL)
                ? H5Sget_simple_extent_ndims(file_space_id) : 0;
  long long n = clio_sel_nelem(dataset, mem_space_id, file_space_id, dxpl_id);
  a.nelem_sel = n;
  a.bytes = (n > 0) ? static_cast<size_t>(n) * a.elem_size : 0;
  a.dur_us = dur_us;
  clio_probe_layout(dataset, dxpl_id);
  a.chunked = dataset->chunked;
  a.chunk_aligned = clio_chunk_alignment(dataset, file_space_id);
  if (dataset->chunked) {
    a.chunk_dims = "[";
    for (int i = 0; i < dataset->chunk_rank; ++i)
      a.chunk_dims += (i ? "," : "") + std::to_string(dataset->chunk_dims[i]);
    a.chunk_dims += "]";
  }
  clio::trace::record(dataset->file->trace, a);
}

static inline double clio_since_us(std::chrono::steady_clock::time_point s) {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - s).count();
}

/**
 * Dataset write: chunk data into async PutBlob calls.
 *
 * Each chunk is copied to shared memory and submitted via AsyncPutBlob.
 * Futures are collected in the dataset state and flushed on close.
 */
static herr_t clio_dataset_write(size_t count, void *dset[],
                                   hid_t mem_type_id[],
                                   hid_t mem_space_id[],
                                   hid_t file_space_id[],
                                   hid_t dxpl_id, const void *buf[],
                                   void **req) {
  const bool tracing = clio::trace::enabled();
  herr_t ret_value = 0;
  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<clio_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;
    auto t0 = std::chrono::steady_clock::now();

    /* Only whole-dataset, independent writes can be represented in the linear
       CTE chunk cache. For no-file-reference, partial (hyperslab), or
       collective writes, persist to the native VOL only — caching a partial
       write under a whole-dataset key would poison a later whole read. */
    if (!clio_cache_usable(dataset->file) || !dataset->cacheable ||
        !clio_is_whole_read(mem_space_id[d], file_space_id[d]) ||
        !clio_type_is_cacheable(mem_type_id[d]) ||
        !clio_type_matches_file(dataset, mem_type_id[d], dxpl_id) ||
        clio_is_collective(dxpl_id)) {
      /* The native VOL is the authoritative store: its status IS this call's
         status. Discarding it (as this did) reported success for writes that
         never reached the file -- ENOSPC, filter failure, a bad selection. */
      herr_t rc = H5VLdataset_write(1, &dataset->obj.under_object,
                         dataset->obj.under_vol_id,
                         &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                         dxpl_id, &buf[d], req);
      if (rc < 0) ret_value = rc;
      /* This write did NOT refresh the whole linear image, so any cached image
         is now stale. Invalidate it (drop the chunk_0 hit-test key) so later
         whole/selection reads miss and re-stage fresh data from native. Without
         this, a partial write followed by a cached read returns pre-write data.
         Only meaningful if the write actually landed. */
      if (rc >= 0) {
        clio_invalidate_dataset(dataset);
      }
      if (tracing)
        clio_trace_access(dataset, clio::trace::Op::kWrite, mem_type_id[d],
                            mem_space_id[d], file_space_id[d], dxpl_id,
                            clio::trace::Served::kUncacheable,
                            clio_since_us(t0));
      continue;
    }
    auto *cte_client = get_cte_client();

    /* Compute total data size from memory dataspace. When we had to ask the
       native dataset for its space we own that hid_t and must close it -- the
       previous code leaked one per whole-dataset write. */
    hid_t space = mem_space_id[d];
    hid_t owned_space = H5I_INVALID_HID;
    if (space == H5S_ALL) {
      /* Get dataspace from the native dataset */
      H5VL_dataset_get_args_t get_args;
      get_args.op_type = H5VL_DATASET_GET_SPACE;
      get_args.args.get_space.space_id = H5I_INVALID_HID;
      H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id,
                       &get_args, dxpl_id, nullptr);
      space = get_args.args.get_space.space_id;
      owned_space = space;
    }
    hssize_t nelem = (space >= 0) ? H5Sget_simple_extent_npoints(space) : -1;
    if (owned_space >= 0) H5Sclose(owned_space);
    if (nelem <= 0) continue;

    size_t type_size = H5Tget_size(mem_type_id[d]);
    size_t total_size = static_cast<size_t>(nelem) * type_size;
    size_t chunk_size = dataset->file->chunk_size;
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    const char *src = static_cast<const char *>(buf[d]);

    for (size_t i = 0; i < num_chunks; ++i) {
      size_t offset = i * chunk_size;
      size_t this_size = std::min(chunk_size, total_size - offset);

      /* Allocate SHM buffer and copy data */
      auto buffer = CLIO_IPC->AllocateBuffer(this_size);
      if (buffer.IsNull()) return -1;
      std::memcpy(buffer.ptr_, src + offset, this_size);

      ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
      std::string blob_name = dataset->dataset_path + "/chunk_" +
                              std::to_string(i);

      auto future = cte_client->AsyncPutBlob(
          dataset->file->tag_id, blob_name, offset, this_size,
          blob_data, -1.0f, clio::cte::core::Context(), 0);

      dataset->pending_puts.push_back(std::move(future));
      dataset->pending_buffers.push_back(std::move(buffer));
    }

    /* Write to the native VOL -- the authoritative store. Its status is this
       call's status; if it failed, the bytes we just staged into CTE describe
       data the file does not contain, so drop them. */
    herr_t rc = H5VLdataset_write(1, &dataset->obj.under_object,
                       dataset->obj.under_vol_id,
                       &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                       dxpl_id, &buf[d], req);
    if (rc < 0) {
      ret_value = rc;
      drain_dataset_puts(dataset);
      clio_invalidate_dataset(dataset);
    }
    if (tracing)
      clio_trace_access(dataset, clio::trace::Op::kWrite, mem_type_id[d],
                          mem_space_id[d], file_space_id[d], dxpl_id,
                          rc < 0 ? clio::trace::Served::kUncacheable
                                 : clio::trace::Served::kCache,
                          clio_since_us(t0));
  }

  return ret_value;
}

/* True when the dataset's linear chunk cache is populated (a fully-staged cache
   always has a non-empty chunk_0, the hit-test key). */
static bool clio_cache_populated(clio_dataset_t *dataset) {
  auto *cte_client = get_cte_client();
  auto sz = cte_client->AsyncGetBlobSize(dataset->file->tag_id,
                                         dataset->dataset_path + "/chunk_0");
  sz.Wait();
  return sz->size_ > 0;
}

/* Reassemble the full linear dataset image from its CTE chunk blobs into dst
   (which must hold total_size bytes). Returns true on success. Shared by the
   whole-read hit path and selection serving. */
static bool clio_read_cached_image(clio_dataset_t *dataset,
                                     size_t total_size, char *dst) {
  auto *cte_client = get_cte_client();
  size_t chunk_size = dataset->file->chunk_size;
  size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
  std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> futures;
  std::vector<ctp::ipc::FullPtr<char>> buffers;
  futures.reserve(num_chunks);
  buffers.reserve(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    size_t offset = i * chunk_size;
    size_t this_size = std::min(chunk_size, total_size - offset);
    auto buffer = CLIO_IPC->AllocateBuffer(this_size);
    if (buffer.IsNull()) return false;
    ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
    std::string blob_name = dataset->dataset_path + "/chunk_" +
                            std::to_string(i);
    futures.push_back(cte_client->AsyncGetBlob(dataset->file->tag_id, blob_name,
                                               offset, this_size, 0, blob_data));
    buffers.push_back(std::move(buffer));
  }
  for (size_t i = 0; i < futures.size(); ++i) {
    futures[i].Wait();
    size_t offset = i * chunk_size;
    size_t this_size = std::min(chunk_size, total_size - offset);
    std::memcpy(dst + offset, buffers[i].ptr_, this_size);
  }
  return true;
}

/* H5Dscatter source callback: hand the whole gathered selection to HDF5 in one
   shot. dst_space selects exactly as many elements as we provide, so HDF5
   consumes it in a single call. */
struct clio_scatter_ctx {
  const void *buf;
  size_t nbytes;
};
static herr_t clio_scatter_cb(const void **src_buf, size_t *src_bytes,
                                void *op) {
  auto *ctx = static_cast<clio_scatter_ctx *>(op);
  *src_buf = ctx->buf;
  *src_bytes = ctx->nbytes;
  return 0;
}

/* Selection-aware READ serving (serve-only, no prefetch). When a hyperslab/point
   read hits a dataset whose linear chunk cache is already populated, satisfy it
   from the CTE tier instead of native: materialize the full image, then use
   HDF5's own gather/scatter to extract the file-space selection into the user
   buffer per the mem-space selection. Returns true if served from cache, false
   on a cache miss or any failure (caller then falls back to native).
   The whole linear image is materialized per call; range-limited fetch of only
   the touched chunks is a future optimization. */
static bool clio_serve_selection(clio_dataset_t *dataset, hid_t mem_type_id,
                                   hid_t mem_space_id, hid_t file_space_id,
                                   hid_t dxpl_id, void *buf) {
  if (!clio_cache_populated(dataset)) return false;  /* serve-only: no prefetch */

  H5VL_dataset_get_args_t ga;
  ga.op_type = H5VL_DATASET_GET_SPACE;
  ga.args.get_space.space_id = H5I_INVALID_HID;
  if (H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id, &ga,
                      dxpl_id, nullptr) < 0)
    return false;
  hid_t full_space = ga.args.get_space.space_id;
  if (full_space < 0) return false;

  bool ok = false;
  do {
    hssize_t total_nelem = H5Sget_simple_extent_npoints(full_space);
    size_t type_size = H5Tget_size(mem_type_id);
    if (total_nelem <= 0 || type_size == 0) break;
    size_t total_size = static_cast<size_t>(total_nelem) * type_size;

    /* INTERIM GUARD. Serving a selection currently materialises the WHOLE
       dataset and gathers out of it, so a 1 KiB hyperslab against a 100 GiB
       dataset would allocate 100 GiB -- an OOM, and strictly worse than simply
       letting native do the read. Until the fetch is narrowed to the chunks the
       selection actually touches (Q2.3), refuse to serve above a ceiling and
       fall back to native. Tunable via CLIO_VOL_MAX_SERVE_BYTES; default 1 GiB. */
    static const size_t kMaxServeBytes = []() -> size_t {
      const char *v = std::getenv("CLIO_VOL_MAX_SERVE_BYTES");
      if (v && *v) {
        size_t n = std::strtoull(v, nullptr, 10);
        if (n > 0) return n;
      }
      return (size_t)1 << 30;
    }();
    if (total_size > kMaxServeBytes) break;

    std::vector<char> full(total_size);
    if (!clio_read_cached_image(dataset, total_size, full.data())) break;

    /* Gather the file-space selection into a contiguous buffer. H5S_ALL file
       space means the whole image is selected. */
    hid_t fspace = (file_space_id == H5S_ALL) ? full_space : file_space_id;
    hssize_t nsel = H5Sget_select_npoints(fspace);
    if (nsel <= 0) break;
    size_t sel_size = static_cast<size_t>(nsel) * type_size;
    std::vector<char> sel(sel_size);
    if (H5Dgather(fspace, full.data(), mem_type_id, sel_size, sel.data(),
                  nullptr, nullptr) < 0)
      break;

    /* Place the gathered elements into the user buffer. H5S_ALL mem space means
       a contiguous buffer of the selected elements; otherwise scatter per the
       mem-space selection. */
    if (mem_space_id == H5S_ALL) {
      std::memcpy(buf, sel.data(), sel_size);
      ok = true;
    } else {
      clio_scatter_ctx ctx{sel.data(), sel_size};
      ok = (H5Dscatter(clio_scatter_cb, &ctx, mem_type_id, mem_space_id,
                       buf) >= 0);
    }
  } while (false);

  H5Sclose(full_space);
  return ok;
}

/**
 * Dataset read — CTE read-through cache.
 *
 * For whole-dataset, independent reads the data is served from the CTE tier
 * when present (cache hit). On a miss the read is satisfied by the native VOL
 * (the source of truth) and the buffer is then staged into the tier so the
 * next read of the same dataset hits. This is the key correctness fix over the
 * original connector, which served reads ONLY from CTE blobs — returning
 * zero-filled buffers for any pre-existing native file whose data was never
 * written through the connector. All non-whole / collective reads pass through
 * to the native VOL unchanged.
 */
static herr_t clio_dataset_read(size_t count, void *dset[],
                                  hid_t mem_type_id[],
                                  hid_t mem_space_id[],
                                  hid_t file_space_id[],
                                  hid_t dxpl_id, void *buf[],
                                  void **req) {
  const bool tracing = clio::trace::enabled();
  herr_t ret_value = 0;

  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<clio_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;
    auto t0 = std::chrono::steady_clock::now();

    /* Native passthrough when the cache is unusable, there is no file
       reference, the type is not flat, the memory type disagrees in size with
       the stored type, or the read is collective. The native VOL always
       produces correct data. */
    bool cacheable_flat = clio_cache_usable(dataset->file) &&
                          dataset->cacheable &&
                          clio_type_is_read_cacheable(mem_type_id[d]) &&
                          clio_type_matches_file(dataset, mem_type_id[d],
                                                 dxpl_id) &&
                          !clio_is_collective(dxpl_id);
    if (!cacheable_flat) {
      /* Propagate the native status: a failed read previously returned success
         with the user's buffer untouched. */
      herr_t rc = H5VLdataset_read(1, &dataset->obj.under_object,
                        dataset->obj.under_vol_id,
                        &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                        dxpl_id, &buf[d], req);
      if (rc < 0) ret_value = rc;
      if (tracing)
        clio_trace_access(dataset, clio::trace::Op::kRead, mem_type_id[d],
                            mem_space_id[d], file_space_id[d], dxpl_id,
                            clio::trace::Served::kUncacheable,
                            clio_since_us(t0));
      continue;
    }
    auto *cte_client = get_cte_client();

    /* Partial (non-full-extent) selection → serve from the CTE tier when the
       linear cache is already populated (serve-only). On a miss, fall back to
       native (unchanged). A full-extent selection (incl. a full hyperslab, as
       h5dump issues) is treated as whole so it populates/serves via CTE. */
    if (!clio_read_is_whole(mem_space_id[d], file_space_id[d])) {
      bool served = clio_serve_selection(dataset, mem_type_id[d],
                                           mem_space_id[d], file_space_id[d],
                                           dxpl_id, buf[d]);
      if (!served) {
        herr_t rc = H5VLdataset_read(1, &dataset->obj.under_object,
                          dataset->obj.under_vol_id,
                          &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                          dxpl_id, &buf[d], req);
        if (rc < 0) ret_value = rc;
      }
      if (tracing)
        clio_trace_access(dataset, clio::trace::Op::kRead, mem_type_id[d],
                            mem_space_id[d], file_space_id[d], dxpl_id,
                            served ? clio::trace::Served::kCache
                                   : clio::trace::Served::kNative,
                            clio_since_us(t0));
      continue;
    }

    /* Whole, independent read → size the dataset from its native dataspace. */
    H5VL_dataset_get_args_t get_args;
    get_args.op_type = H5VL_DATASET_GET_SPACE;
    get_args.args.get_space.space_id = H5I_INVALID_HID;
    H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id,
                     &get_args, dxpl_id, nullptr);
    hid_t space = get_args.args.get_space.space_id;
    hssize_t nelem = (space >= 0) ? H5Sget_simple_extent_npoints(space) : -1;
    if (space >= 0) H5Sclose(space);
    if (nelem <= 0) {
      /* Can't size it — fall back to native for safety. */
      herr_t rc = H5VLdataset_read(1, &dataset->obj.under_object,
                        dataset->obj.under_vol_id,
                        &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                        dxpl_id, &buf[d], req);
      if (rc < 0) ret_value = rc;
      if (tracing)
        clio_trace_access(dataset, clio::trace::Op::kRead, mem_type_id[d],
                            mem_space_id[d], file_space_id[d], dxpl_id,
                            clio::trace::Served::kNative, clio_since_us(t0));
      continue;
    }

    size_t type_size = H5Tget_size(mem_type_id[d]);
    size_t total_size = static_cast<size_t>(nelem) * type_size;
    size_t chunk_size = dataset->file->chunk_size;
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    char *dst = static_cast<char *>(buf[d]);

    /* Hit test: a fully-populated cache always has a non-empty chunk_0. */
    clio::run::u64 cached = 0;
    {
      auto sz = cte_client->AsyncGetBlobSize(
          dataset->file->tag_id, dataset->dataset_path + "/chunk_0");
      sz.Wait();
      cached = sz->size_;
    }

    if (cached == 0) {
      /* MISS — native read is the source of truth, then stage into the tier. */
      herr_t rc = H5VLdataset_read(1, &dataset->obj.under_object,
                                   dataset->obj.under_vol_id,
                                   &mem_type_id[d], &mem_space_id[d],
                                   &file_space_id[d], dxpl_id, &buf[d], req);
      if (rc < 0) return rc;
      /* Stage into the tier. Best-effort, but a PARTIAL stage must not be left
         behind: chunk_0 present is the hit test, so a run that stages chunk_0
         and then fails would make the next read a "hit" on an incomplete image.
         Track it and invalidate rather than leave that trap. */
      bool staged_ok = true;
      for (size_t i = 0; i < num_chunks && staged_ok; ++i) {
        size_t offset = i * chunk_size;
        size_t this_size = std::min(chunk_size, total_size - offset);
        auto buffer = CLIO_IPC->AllocateBuffer(this_size);
        if (buffer.IsNull()) { staged_ok = false; break; }
        std::memcpy(buffer.ptr_, dst + offset, this_size);
        ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
        std::string blob_name = dataset->dataset_path + "/chunk_" +
                                std::to_string(i);
        auto fut = cte_client->AsyncPutBlob(
            dataset->file->tag_id, blob_name, offset, this_size, blob_data,
            -1.0f, clio::cte::core::Context(), 0);
        fut.Wait();
        if (fut->GetReturnCode() != 0) staged_ok = false;
      }
      if (!staged_ok) {
        clio_invalidate_dataset(dataset);
      }
    } else {
      /* HIT — serve the whole image from the CTE tier. If reassembly fails we
         have no data to return, so fall back to native rather than handing back
         a partially-filled buffer with a success status. */
      if (!clio_read_cached_image(dataset, total_size, dst)) {
        clio_invalidate_dataset(dataset);
        herr_t rc = H5VLdataset_read(1, &dataset->obj.under_object,
                          dataset->obj.under_vol_id,
                          &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                          dxpl_id, &buf[d], req);
        if (rc < 0) ret_value = rc;
      }
    }
    if (tracing)
      clio_trace_access(dataset, clio::trace::Op::kRead, mem_type_id[d],
                          mem_space_id[d], file_space_id[d], dxpl_id,
                          cached == 0 ? clio::trace::Served::kNative
                                      : clio::trace::Served::kCache,
                          clio_since_us(t0));
  }

  return ret_value;
}

static herr_t clio_dataset_get(void *obj, H5VL_dataset_get_args_t *args,
                                 hid_t dxpl_id, void **req) {
  auto *dset = static_cast<clio_dataset_t *>(obj);
  return H5VLdataset_get(dset->obj.under_object, dset->obj.under_vol_id,
                          args, dxpl_id, req);
}

static herr_t clio_dataset_specific(void *obj,
                                      H5VL_dataset_specific_args_t *args,
                                      hid_t dxpl_id, void **req) {
  auto *dset = static_cast<clio_dataset_t *>(obj);
  /* Safe mode: H5Dflush is a durability barrier at DATASET granularity, and the
     spec names it alongside H5Fflush -- so drain this dataset's pending CTE puts
     before delegating, exactly as file_specific does for the whole file. This
     was previously a bare pass-through, so H5Dflush returned with async puts
     still in flight.

     H5Dset_extent changes the dataset's element count, which is what the linear
     blob image is sized by, so the cached image no longer describes the dataset:
     invalidate rather than serve a stale-length image. */
  if (args) {
    if (args->op_type == H5VL_DATASET_FLUSH) {
      if (!drain_dataset_puts(dset)) {
        clio_invalidate_dataset(dset);
      }
    } else if (args->op_type == H5VL_DATASET_SET_EXTENT) {
      drain_dataset_puts(dset);
      clio_invalidate_dataset(dset);
    }
  }
  return H5VLdataset_specific(dset->obj.under_object, dset->obj.under_vol_id,
                               args, dxpl_id, req);
}

static herr_t clio_dataset_close(void *obj, hid_t dxpl_id, void **req) {
  auto *dset = static_cast<clio_dataset_t *>(obj);

  /* Unregister from the file's Safe-mode set before draining, so a concurrent
     flush/close does not touch this dataset while it is being torn down. */
  if (dset->file && dset->cacheable) {
    std::lock_guard<std::mutex> lk(dset->file->ds_mtx);
    dset->file->open_datasets.erase(dset);
  }

  /* Flush all pending async writes; a failed put leaves a partial image, so
     invalidate rather than let the next open treat it as a hit. */
  if (!drain_dataset_puts(dset)) {
    clio_invalidate_dataset(dset);
  }

  if (dset->file_type >= 0) H5Tclose(dset->file_type);
  herr_t ret = H5VLdataset_close(dset->obj.under_object,
                                  dset->obj.under_vol_id, dxpl_id, req);
  delete dset;
  return ret;
}

/* ========================================================================
 * Passthrough: group, attribute, datatype, link, object, introspect
 * ======================================================================== */

/* Group */
static void *clio_group_create(void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 const char *name, hid_t lcpl_id,
                                 hid_t gcpl_id, hid_t gapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  void *under = H5VLgroup_create(o->under_object, loc_params,
                                  o->under_vol_id, name, lcpl_id,
                                  gcpl_id, gapl_id, dxpl_id, req);
  if (!under) return nullptr;
  auto *grp = new clio_obj_t;
  grp->under_object = under;
  grp->under_vol_id = o->under_vol_id;
  grp->parent_file = o->parent_file;     /* inherit from parent */
  return grp;
}

static void *clio_group_open(void *obj,
                               const H5VL_loc_params_t *loc_params,
                               const char *name, hid_t gapl_id,
                               hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  void *under = H5VLgroup_open(o->under_object, loc_params,
                                o->under_vol_id, name, gapl_id,
                                dxpl_id, req);
  if (!under) return nullptr;
  auto *grp = new clio_obj_t;
  grp->under_object = under;
  grp->under_vol_id = o->under_vol_id;
  grp->parent_file = o->parent_file;     /* inherit from parent */
  return grp;
}

static herr_t clio_group_get(void *obj, H5VL_group_get_args_t *args,
                               hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLgroup_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t clio_group_specific(void *obj,
                                    H5VL_group_specific_args_t *args,
                                    hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLgroup_specific(o->under_object, o->under_vol_id, args,
                             dxpl_id, req);
}

static herr_t clio_group_close(void *obj, hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  herr_t ret = H5VLgroup_close(o->under_object, o->under_vol_id,
                                dxpl_id, req);
  delete o;
  return ret;
}

/* Attribute — full passthrough via native VOL, with proper wrapping */
static void *clio_attr_create(void *obj,
                                const H5VL_loc_params_t *loc_params,
                                const char *name, hid_t type_id,
                                hid_t space_id, hid_t acpl_id,
                                hid_t aapl_id, hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  void *under = H5VLattr_create(o->under_object, loc_params, o->under_vol_id,
                                 name, type_id, space_id, acpl_id, aapl_id,
                                 dxpl_id, req);
  if (!under) return nullptr;
  auto *attr = new clio_obj_t;
  attr->under_object = under;
  attr->under_vol_id = o->under_vol_id;
  attr->parent_file = o->parent_file;
  return attr;
}

static void *clio_attr_open(void *obj,
                              const H5VL_loc_params_t *loc_params,
                              const char *name, hid_t aapl_id,
                              hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  void *under = H5VLattr_open(o->under_object, loc_params, o->under_vol_id,
                               name, aapl_id, dxpl_id, req);
  if (!under) return nullptr;
  auto *attr = new clio_obj_t;
  attr->under_object = under;
  attr->under_vol_id = o->under_vol_id;
  attr->parent_file = o->parent_file;
  return attr;
}

static herr_t clio_attr_read(void *attr, hid_t dtype_id, void *buf,
                               hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(attr);
  return H5VLattr_read(o->under_object, o->under_vol_id, dtype_id, buf,
                        dxpl_id, req);
}

static herr_t clio_attr_write(void *attr, hid_t dtype_id, const void *buf,
                                hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(attr);
  return H5VLattr_write(o->under_object, o->under_vol_id, dtype_id, buf,
                         dxpl_id, req);
}

static herr_t clio_attr_get(void *obj, H5VL_attr_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLattr_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t clio_attr_specific(void *obj, const H5VL_loc_params_t *lp,
                                   H5VL_attr_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLattr_specific(o->under_object, lp, o->under_vol_id, args,
                            dxpl_id, req);
}

static herr_t clio_attr_close(void *attr, hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(attr);
  herr_t ret = H5VLattr_close(o->under_object, o->under_vol_id, dxpl_id, req);
  delete o;
  return ret;
}

/* Link — passthrough */
static herr_t clio_link_create(H5VL_link_create_args_t *args,
                                 void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 hid_t lcpl_id, hid_t lapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);

  /* For a HARD link the target object carried in args is an clio-wrapped
     pointer. The native VOL must receive the native under-object, or it links
     against a foreign pointer and the resulting structure diverges from native
     (h5diff dirty). Shallow-copy the args and unwrap the target before
     delegating; mirrors the reference H5VLpassthru. Soft/UD links carry a path
     or user data, not a wrapped object, so they pass through unchanged. */
  H5VL_link_create_args_t under_args;
  if (args && args->op_type == H5VL_LINK_CREATE_HARD &&
      args->args.hard.curr_obj) {
    under_args = *args;
    under_args.args.hard.curr_obj =
        static_cast<clio_obj_t *>(args->args.hard.curr_obj)->under_object;
    args = &under_args;
  }

  return H5VLlink_create(args, o ? o->under_object : nullptr, loc_params,
                          o ? o->under_vol_id : H5VL_NATIVE,
                          lcpl_id, lapl_id, dxpl_id, req);
}

/* H5Lmove / H5Lrename. Left null, these failed outright through the connector —
   a pass-through must forward every op it does not itself implement. */
static herr_t clio_link_move(void *src_obj,
                               const H5VL_loc_params_t *loc_params1,
                               void *dst_obj,
                               const H5VL_loc_params_t *loc_params2,
                               hid_t lcpl_id, hid_t lapl_id,
                               hid_t dxpl_id, void **req) {
  auto *s = static_cast<clio_obj_t *>(src_obj);
  auto *d = static_cast<clio_obj_t *>(dst_obj);
  /* Either endpoint may be null (a move can name one side purely by path). */
  hid_t vol_id = s ? s->under_vol_id : (d ? d->under_vol_id : H5VL_NATIVE);
  return H5VLlink_move(s ? s->under_object : nullptr, loc_params1,
                        d ? d->under_object : nullptr, loc_params2,
                        vol_id, lcpl_id, lapl_id, dxpl_id, req);
}

static herr_t clio_link_copy(void *src_obj,
                               const H5VL_loc_params_t *loc_params1,
                               void *dst_obj,
                               const H5VL_loc_params_t *loc_params2,
                               hid_t lcpl_id, hid_t lapl_id,
                               hid_t dxpl_id, void **req) {
  auto *s = static_cast<clio_obj_t *>(src_obj);
  auto *d = static_cast<clio_obj_t *>(dst_obj);
  return H5VLlink_copy(s->under_object, loc_params1,
                        d->under_object, loc_params2,
                        s->under_vol_id, lcpl_id, lapl_id, dxpl_id, req);
}

static herr_t clio_link_get(void *obj, const H5VL_loc_params_t *loc_params,
                              H5VL_link_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLlink_get(o->under_object, loc_params, o->under_vol_id,
                       args, dxpl_id, req);
}

static herr_t clio_link_specific(void *obj,
                                   const H5VL_loc_params_t *loc_params,
                                   H5VL_link_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLlink_specific(o->under_object, loc_params, o->under_vol_id,
                            args, dxpl_id, req);
}

/* Object — passthrough */
static void *clio_object_open(void *obj,
                                const H5VL_loc_params_t *loc_params,
                                H5I_type_t *opened_type,
                                hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  void *under = H5VLobject_open(o->under_object, loc_params, o->under_vol_id,
                                 opened_type, dxpl_id, req);
  if (!under) return nullptr;

  /* CRITICAL: when the opened object is a dataset, HDF5 will subsequently route
     the dataset_read/write/close callbacks to this wrapper and cast it to
     clio_dataset_t. h5py opens datasets through H5Oopen, so returning a bare
     clio_obj_t here would be mis-cast and corrupt the heap. Build the right
     wrapper type. The dataset path is recovered from a by-name location when
     available; otherwise the dataset is marked non-cacheable. */
  if (opened_type && *opened_type == H5I_DATASET) {
    const char *name = nullptr;
    if (loc_params && loc_params->type == H5VL_OBJECT_BY_NAME) {
      name = loc_params->loc_data.loc_by_name.name;
    }
    return make_dataset_wrapper(under, o->under_vol_id, o->parent_file, name);
  }

  auto *wrapped = new clio_obj_t;
  wrapped->under_object = under;
  wrapped->under_vol_id = o->under_vol_id;
  wrapped->parent_file = o->parent_file;
  return wrapped;
}

static herr_t clio_object_copy(void *src_obj,
                                 const H5VL_loc_params_t *loc_params1,
                                 const char *src_name,
                                 void *dst_obj,
                                 const H5VL_loc_params_t *loc_params2,
                                 const char *dst_name,
                                 hid_t ocpypl_id, hid_t lcpl_id,
                                 hid_t dxpl_id, void **req) {
  auto *s = static_cast<clio_obj_t *>(src_obj);
  auto *d = static_cast<clio_obj_t *>(dst_obj);
  return H5VLobject_copy(s->under_object, loc_params1, src_name,
                          d->under_object, loc_params2, dst_name,
                          s->under_vol_id, ocpypl_id, lcpl_id, dxpl_id, req);
}

static herr_t clio_object_get(void *obj,
                                const H5VL_loc_params_t *loc_params,
                                H5VL_object_get_args_t *args,
                                hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLobject_get(o->under_object, loc_params, o->under_vol_id,
                         args, dxpl_id, req);
}

static herr_t clio_object_specific(void *obj,
                                     const H5VL_loc_params_t *loc_params,
                                     H5VL_object_specific_args_t *args,
                                     hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLobject_specific(o->under_object, loc_params, o->under_vol_id,
                              args, dxpl_id, req);
}

/* Datatype — passthrough. neuroh5/MiV store committed named datatypes under
   /H5Types (population enums, etc.) and open them with H5Topen; without these
   callbacks the connector's datatype_cls is all-null and every H5Topen through
   the VOL fails. Every op delegates to the native VOL. */
static void *clio_datatype_commit(void *obj,
                                    const H5VL_loc_params_t *loc_params,
                                    const char *name, hid_t type_id,
                                    hid_t lcpl_id, hid_t tcpl_id,
                                    hid_t tapl_id, hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  void *under = H5VLdatatype_commit(o->under_object, loc_params, o->under_vol_id,
                                     name, type_id, lcpl_id, tcpl_id, tapl_id,
                                     dxpl_id, req);
  if (!under) return nullptr;
  auto *dt = new clio_obj_t;
  dt->under_object = under;
  dt->under_vol_id = o->under_vol_id;
  dt->parent_file = o->parent_file;
  return dt;
}

static void *clio_datatype_open(void *obj,
                                  const H5VL_loc_params_t *loc_params,
                                  const char *name, hid_t tapl_id,
                                  hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  void *under = H5VLdatatype_open(o->under_object, loc_params, o->under_vol_id,
                                   name, tapl_id, dxpl_id, req);
  if (!under) return nullptr;
  auto *dt = new clio_obj_t;
  dt->under_object = under;
  dt->under_vol_id = o->under_vol_id;
  dt->parent_file = o->parent_file;
  return dt;
}

static herr_t clio_datatype_get(void *obj, H5VL_datatype_get_args_t *args,
                                  hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLdatatype_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t clio_datatype_specific(void *obj,
                                       H5VL_datatype_specific_args_t *args,
                                       hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLdatatype_specific(o->under_object, o->under_vol_id, args, dxpl_id,
                                req);
}

static herr_t clio_datatype_close(void *obj, hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  herr_t ret = H5VLdatatype_close(o->under_object, o->under_vol_id, dxpl_id,
                                   req);
  delete o;
  return ret;
}

/* Introspect */
static herr_t clio_introspect_get_conn_cls(void *obj,
                                             H5VL_get_conn_lvl_t lvl,
                                             const H5VL_class_t **conn_cls) {
  (void)obj; (void)lvl;
  *conn_cls = &H5VL_clio_cls;
  return 0;
}

static herr_t clio_introspect_get_cap_flags(const void *info,
                                              uint64_t *cap_flags) {
  (void)info;
  /* Same constant the class advertises. These previously disagreed -- the class
     literal omitted LINK_BASIC and OBJECT_BASIC that this callback reported --
     so what the connector claimed depended on which one HDF5 asked. */
  *cap_flags = CLIO_VOL_CAP_FLAGS;
  return 0;
}

static herr_t clio_introspect_opt_query(void *obj, H5VL_subclass_t cls,
                                          int opt_type, uint64_t *flags) {
  /* Forward to the under-VOL so native-specific optional ops are reported as
     supported (mirrors H5VLpassthru). Critically, HDF5 only issues the
     H5VL_NATIVE_FILE_POST_OPEN op — which sets the file's VOL object, required by
     variable-length datatypes — when this query reports it supported. Hardcoding
     *flags=0 suppressed post-open, leaving H5F_VOL_OBJ NULL and crashing vlen. */
  auto *o = static_cast<clio_obj_t *>(obj);
  if (!o) { *flags = 0; return 0; }
  return H5VLintrospect_opt_query(o->under_object, o->under_vol_id, cls,
                                  opt_type, flags);
}

/* ------------------------------------------------------------------------
 * Blob callbacks — pass-through. HDF5 stores variable-length data (vlen
 * strings, vlen sequences) and references in the file's global heap via the
 * VOL "blob" interface. A pass-through connector MUST forward these to the
 * under-VOL; leaving them null makes any vlen/reference dataset crash at
 * create time (H5T__vlen_set_loc dereferences the missing handler). The blob
 * `obj` is a file object — clio_obj_t is its first member, so the cast
 * yields the native under-object. Blob data lives in the native file's heap;
 * it is not mirrored into CTE (consistent with the vlen cache bypass).
 * ------------------------------------------------------------------------ */
static herr_t clio_blob_put(void *obj, const void *buf, size_t size,
                              void *blob_id, void *ctx) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLblob_put(o->under_object, o->under_vol_id, buf, size, blob_id, ctx);
}

static herr_t clio_blob_get(void *obj, const void *blob_id, void *buf,
                              size_t size, void *ctx) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLblob_get(o->under_object, o->under_vol_id, blob_id, buf, size, ctx);
}

static herr_t clio_blob_specific(void *obj, void *blob_id,
                                   H5VL_blob_specific_args_t *args) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLblob_specific(o->under_object, o->under_vol_id, blob_id, args);
}

static herr_t clio_blob_optional(void *obj, void *blob_id,
                                   H5VL_optional_args_t *args) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLblob_optional(o->under_object, o->under_vol_id, blob_id, args);
}

/* ------------------------------------------------------------------------
 * Optional-op passthrough. Every subclass's `optional` callback carries the
 * connector-specific ops HDF5 (and other pass-through connectors) dispatch
 * through the VOL — for datasets that includes direct chunk I/O
 * (H5Dread_chunk / H5Dwrite_chunk / H5Dchunk_iter). Left null, those calls fail
 * through this connector even though the native VOL beneath supports them. The
 * file subclass already forwarded (it must, for H5VL_NATIVE_FILE_POST_OPEN);
 * these complete the set. Mirrors H5VLpassthru.
 * ------------------------------------------------------------------------ */
static herr_t clio_attr_optional(void *obj, H5VL_optional_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLattr_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t clio_dataset_optional(void *obj, H5VL_optional_args_t *args,
                                      hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLdataset_optional(o->under_object, o->under_vol_id, args, dxpl_id,
                               req);
}

static herr_t clio_datatype_optional(void *obj, H5VL_optional_args_t *args,
                                       hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLdatatype_optional(o->under_object, o->under_vol_id, args, dxpl_id,
                                req);
}

static herr_t clio_group_optional(void *obj, H5VL_optional_args_t *args,
                                    hid_t dxpl_id, void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLgroup_optional(o->under_object, o->under_vol_id, args, dxpl_id,
                             req);
}

static herr_t clio_link_optional(void *obj,
                                   const H5VL_loc_params_t *loc_params,
                                   H5VL_optional_args_t *args, hid_t dxpl_id,
                                   void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLlink_optional(o->under_object, loc_params, o->under_vol_id, args,
                            dxpl_id, req);
}

static herr_t clio_object_optional(void *obj,
                                     const H5VL_loc_params_t *loc_params,
                                     H5VL_optional_args_t *args, hid_t dxpl_id,
                                     void **req) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLobject_optional(o->under_object, loc_params, o->under_vol_id, args,
                              dxpl_id, req);
}

/* ------------------------------------------------------------------------
 * Object tokens — passthrough. Tokens are the VOL-agnostic identity of an
 * object; H5Oget_info-driven code compares and serialises them via these. Left
 * null, H5Otoken_cmp/to_str/from_str fail through the connector.
 * ------------------------------------------------------------------------ */
static herr_t clio_token_cmp(void *obj, const H5O_token_t *token1,
                               const H5O_token_t *token2, int *cmp_value) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLtoken_cmp(o->under_object, o->under_vol_id, token1, token2,
                        cmp_value);
}

static herr_t clio_token_to_str(void *obj, H5I_type_t obj_type,
                                  const H5O_token_t *token, char **token_str) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLtoken_to_str(o->under_object, obj_type, o->under_vol_id, token,
                           token_str);
}

static herr_t clio_token_from_str(void *obj, H5I_type_t obj_type,
                                    const char *token_str, H5O_token_t *token) {
  auto *o = static_cast<clio_obj_t *>(obj);
  return H5VLtoken_from_str(o->under_object, obj_type, o->under_vol_id,
                             token_str, token);
}

/* Connector-info comparison. HDF5 uses this to decide whether two FAPLs select
   the same connector configuration; without it, infos that differ are treated
   as equal. Compares the fields that actually change behaviour. */
static herr_t clio_info_cmp(int *cmp_value, const void *_info1,
                              const void *_info2) {
  const auto *i1 = static_cast<const clio_vol_info_t *>(_info1);
  const auto *i2 = static_cast<const clio_vol_info_t *>(_info2);
  if (!i1 || !i2) {
    *cmp_value = (i1 == i2) ? 0 : (i1 ? 1 : -1);
    return 0;
  }
  if (i1->under_vol_id != i2->under_vol_id) {
    *cmp_value = (i1->under_vol_id < i2->under_vol_id) ? -1 : 1;
    return 0;
  }
  if (i1->chunk_size != i2->chunk_size) {
    *cmp_value = (i1->chunk_size < i2->chunk_size) ? -1 : 1;
    return 0;
  }
  *cmp_value = 0;
  return 0;
}

/* ========================================================================
 * VOL connector class definition
 * ======================================================================== */

const H5VL_class_t H5VL_clio_cls = {
    /* version      */ H5VL_VERSION,
    /* value        */ CLIO_VOL_CONNECTOR_VALUE,
    /* name         */ CLIO_VOL_CONNECTOR_NAME,
    /* conn_version */ CLIO_VOL_CONNECTOR_VERSION,
    /* cap_flags    */ CLIO_VOL_CAP_FLAGS,
    /* initialize   */ nullptr,
    /* terminate    */ nullptr,

    /* info_cls */ {
        /* size    */ sizeof(clio_vol_info_t),
        /* copy    */ clio_info_copy,
        /* cmp     */ clio_info_cmp,
        /* free    */ clio_info_free,
        /* to_str  */ nullptr,
        /* from_str*/ nullptr,
    },

    /* wrap_cls */ {
        /* get_object  */ clio_wrap_get_object,
        /* get_wrap_ctx*/ clio_get_wrap_ctx,
        /* wrap_object */ clio_wrap_object,
        /* unwrap_object*/ clio_unwrap_object,
        /* free_wrap_ctx*/ clio_free_wrap_ctx,
    },

    /* attr_cls */ {
        /* create   */ clio_attr_create,
        /* open     */ clio_attr_open,
        /* read     */ clio_attr_read,
        /* write    */ clio_attr_write,
        /* get      */ clio_attr_get,
        /* specific */ clio_attr_specific,
        /* optional */ clio_attr_optional,
        /* close    */ clio_attr_close,
    },

    /* dataset_cls */ {
        /* create   */ clio_dataset_create,
        /* open     */ clio_dataset_open,
        /* read     */ clio_dataset_read,
        /* write    */ clio_dataset_write,
        /* get      */ clio_dataset_get,
        /* specific */ clio_dataset_specific,
        /* optional */ clio_dataset_optional,
        /* close    */ clio_dataset_close,
    },

    /* datatype_cls */ {
        /* commit   */ clio_datatype_commit,
        /* open     */ clio_datatype_open,
        /* get      */ clio_datatype_get,
        /* specific */ clio_datatype_specific,
        /* optional */ clio_datatype_optional,
        /* close    */ clio_datatype_close,
    },

    /* file_cls */ {
        /* create   */ clio_file_create,
        /* open     */ clio_file_open,
        /* get      */ clio_file_get,
        /* specific */ clio_file_specific,
        /* optional */ clio_file_optional,
        /* close    */ clio_file_close,
    },

    /* group_cls */ {
        /* create   */ clio_group_create,
        /* open     */ clio_group_open,
        /* get      */ clio_group_get,
        /* specific */ clio_group_specific,
        /* optional */ clio_group_optional,
        /* close    */ clio_group_close,
    },

    /* link_cls */ {
        /* create   */ clio_link_create,
        /* copy     */ clio_link_copy,
        /* move     */ clio_link_move,
        /* get      */ clio_link_get,
        /* specific */ clio_link_specific,
        /* optional */ clio_link_optional,
    },

    /* object_cls */ {
        /* open     */ clio_object_open,
        /* copy     */ clio_object_copy,
        /* get      */ clio_object_get,
        /* specific */ clio_object_specific,
        /* optional */ clio_object_optional,
        /* (no close — objects are closed by their specific type class) */
    },

    /* introspect_cls */ {
        /* get_conn_cls   */ clio_introspect_get_conn_cls,
        /* get_cap_flags  */ clio_introspect_get_cap_flags,
        /* opt_query      */ clio_introspect_opt_query,
    },

    /* request_cls */ {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    },

    /* blob_cls */ {
        clio_blob_put, clio_blob_get, clio_blob_specific, clio_blob_optional,
    },

    /* token_cls */ {
        clio_token_cmp, clio_token_to_str, clio_token_from_str,
    },

    /* optional */ nullptr,
};

hid_t H5VL_clio_register(void) {
  return H5VLregister_connector(&H5VL_clio_cls, H5P_DEFAULT);
}

/* ========================================================================
 * HDF5 plugin entry points
 *
 * These two exports let HDF5 discover and load the connector dynamically via
 * the standard environment-variable mechanism — no application change needed:
 *
 *   export HDF5_PLUGIN_PATH=<dir containing libclio_hdf5_vol.so>
 *   export HDF5_VOL_CONNECTOR="clio"
 *
 * On the first H5Fopen/H5Fcreate, HDF5 dlopen()s plugins on HDF5_PLUGIN_PATH,
 * calls H5PLget_plugin_type() (must be H5PL_TYPE_VOL) and H5PLget_plugin_info()
 * (returns the connector class), and matches by connector name ("clio").
 * Without these the connector could only be installed by an application calling
 * H5VL_clio_register() + H5Pset_vol() itself.
 * ======================================================================== */
extern "C" H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_VOL; }
extern "C" const void *H5PLget_plugin_info(void) { return &H5VL_clio_cls; }
