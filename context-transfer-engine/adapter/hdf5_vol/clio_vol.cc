/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Clio HDF5 VOL Connector
 *
 * Intercepts H5Dwrite/H5Dread and maps them to CTE AsyncPutBlob/AsyncGetBlob.
 * Large writes are split into configurable-size chunks and submitted
 * asynchronously. All other HDF5 operations pass through to the native VOL.
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
struct clio_obj_t {
  void           *under_object;
  hid_t           under_vol_id;
  clio_file_t  *parent_file;
};

struct clio_file_t {
  clio_obj_t obj;
  clio::cte::core::TagId tag_id;
  std::string file_name;
  size_t chunk_size;
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
  dset->file = parent_file;
  dset->obj.parent_file = parent_file;
  dset->dataset_path = path ? path : "";
  dset->cacheable = (parent_file != nullptr) && path && path[0] != '\0';
  /* Register with the file so Safe-mode flush/close can drain this dataset's
     pending puts. Only cacheable datasets accumulate CTE puts. */
  if (dset->cacheable) {
    std::lock_guard<std::mutex> lk(parent_file->ds_mtx);
    parent_file->open_datasets.insert(dset);
  }
  return dset;
}

/* Block until every async CTE put for this dataset has completed, then release
   the shared-memory buffers. Shared by dataset_close and Safe-mode flush/close. */
static void drain_dataset_puts(clio_dataset_t *dset) {
  for (auto &future : dset->pending_puts) {
    future.Wait();
  }
  dset->pending_puts.clear();
  dset->pending_buffers.clear();
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
     Config comes from CLIO_SERVER_CONF, same as the runtime. */
  static std::once_flag once;
  std::call_once(once, []() { clio::cte::core::CLIO_CTE_CLIENT_INIT(); });
  return CLIO_CTE_CLIENT;
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

static void *clio_unwrap_object(void *obj) {
  auto *o = static_cast<clio_obj_t *>(obj);
  /* Symmetric with clio_wrap_object()'s H5VLwrap_object(): peel the native
     wrapper back off before discarding our wrapper. */
  void *under = H5VLunwrap_object(o->under_object, o->under_vol_id);
  if (under) delete o;
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

static void *clio_file_create(const char *name, unsigned flags,
                                hid_t fcpl_id, hid_t fapl_id,
                                hid_t dxpl_id, void **req) {
  /* Get underlying VOL info */
  hid_t under_vol_id = H5VL_NATIVE;
  void *under_vol_info = nullptr;

  /* Check if user provided Clio VOL info.
   * HDF5 2.x split H5Pget_vol() into H5Pget_vol_id() + H5Pget_vol_info();
     we only care about the info blob here. */
  clio_vol_info_t *vol_info = nullptr;
  H5Pget_vol_info(fapl_id, reinterpret_cast<void **>(&vol_info));
  size_t chunk_size = CLIO_VOL_DEFAULT_CHUNK_SIZE;
  if (vol_info) {
    under_vol_id = vol_info->under_vol_id;
    under_vol_info = vol_info->under_vol_info;
    if (vol_info->chunk_size > 0) {
      chunk_size = vol_info->chunk_size;
    }
  }

  /* Check environment variable for chunk size */
  const char *env_chunk = std::getenv("CLIO_VOL_CHUNK_SIZE");
  if (env_chunk) {
    chunk_size = std::strtoul(env_chunk, nullptr, 10);
    if (chunk_size == 0) chunk_size = CLIO_VOL_DEFAULT_CHUNK_SIZE;
  }

  /* Create file via native VOL */
  hid_t native_fapl = H5Pcopy(fapl_id);
  H5Pset_vol(native_fapl, H5VL_NATIVE, nullptr);
  void *under_file = H5VLfile_create(name, flags, fcpl_id, native_fapl,
                                      dxpl_id, req);
  H5Pclose(native_fapl);
  if (!under_file) return nullptr;

  /* Create CTE tag for this file */
  auto *cte_client = get_cte_client();
  auto tag_task = cte_client->AsyncGetOrCreateTag(std::string("hdf5:") + name);
  tag_task.Wait();

  /* Create file state */
  auto *file = new clio_file_t;
  file->obj.under_object = under_file;
  file->obj.under_vol_id = H5VL_NATIVE;
  file->obj.parent_file = file;          /* self-pointer */
  file->tag_id = tag_task->tag_id_;
  file->file_name = name;
  file->chunk_size = chunk_size;
  file->trace = clio::trace::open_file(name);

  return file;
}

static void *clio_file_open(const char *name, unsigned flags,
                              hid_t fapl_id, hid_t dxpl_id, void **req) {
  size_t chunk_size = CLIO_VOL_DEFAULT_CHUNK_SIZE;
  const char *env_chunk = std::getenv("CLIO_VOL_CHUNK_SIZE");
  if (env_chunk) {
    chunk_size = std::strtoul(env_chunk, nullptr, 10);
    if (chunk_size == 0) chunk_size = CLIO_VOL_DEFAULT_CHUNK_SIZE;
  }

  /* Open file via native VOL */
  hid_t native_fapl = H5Pcopy(fapl_id);
  H5Pset_vol(native_fapl, H5VL_NATIVE, nullptr);
  void *under_file = H5VLfile_open(name, flags, native_fapl, dxpl_id, req);
  H5Pclose(native_fapl);
  if (!under_file) return nullptr;

  /* Get or create CTE tag */
  auto *cte_client = get_cte_client();
  auto tag_task = cte_client->AsyncGetOrCreateTag(std::string("hdf5:") + name);
  tag_task.Wait();

  auto *file = new clio_file_t;
  file->obj.under_object = under_file;
  file->obj.under_vol_id = H5VL_NATIVE;
  file->obj.parent_file = file;          /* self-pointer */
  file->tag_id = tag_task->tag_id_;
  file->file_name = name;
  file->chunk_size = chunk_size;
  file->trace = clio::trace::open_file(name);

  return file;
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
  auto *file = static_cast<clio_file_t *>(obj);
  /* Safe mode: on H5Fflush, drain every open cacheable dataset's pending CTE
     puts BEFORE delegating the flush to native. This makes the flush a real
     durability barrier for the cache path (fail-closed: we only return once all
     async puts have landed), with the native file remaining authoritative. */
  if (args && args->op_type == H5VL_FILE_FLUSH) {
    std::lock_guard<std::mutex> lk(file->ds_mtx);
    for (auto *dset : file->open_datasets) {
      drain_dataset_puts(dset);
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
    std::lock_guard<std::mutex> lk(file->ds_mtx);
    for (auto *dset : file->open_datasets) {
      drain_dataset_puts(dset);
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
  auto *cte_client = get_cte_client();

  const bool tracing = clio::trace::enabled();
  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<clio_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;
    auto t0 = std::chrono::steady_clock::now();

    /* Only whole-dataset, independent writes can be represented in the linear
       CTE chunk cache. For no-file-reference, partial (hyperslab), or
       collective writes, persist to the native VOL only — caching a partial
       write under a whole-dataset key would poison a later whole read. */
    if (!dataset->file || !dataset->cacheable ||
        !clio_is_whole_read(mem_space_id[d], file_space_id[d]) ||
        !clio_type_is_cacheable(mem_type_id[d]) ||
        clio_is_collective(dxpl_id)) {
      H5VLdataset_write(1, &dataset->obj.under_object,
                         dataset->obj.under_vol_id,
                         &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                         dxpl_id, &buf[d], req);
      /* This write did NOT refresh the whole linear image, so any cached image
         is now stale. Invalidate it (drop the chunk_0 hit-test key) so later
         whole/selection reads miss and re-stage fresh data from native. Without
         this, a partial write followed by a cached read returns pre-write data. */
      if (dataset->file && dataset->cacheable) {
        auto del = cte_client->AsyncDelBlob(
            dataset->file->tag_id, dataset->dataset_path + "/chunk_0");
        del.Wait();
      }
      if (tracing)
        clio_trace_access(dataset, clio::trace::Op::kWrite, mem_type_id[d],
                            mem_space_id[d], file_space_id[d], dxpl_id,
                            clio::trace::Served::kUncacheable,
                            clio_since_us(t0));
      continue;
    }

    /* Compute total data size from memory dataspace */
    hid_t space = mem_space_id[d];
    if (space == H5S_ALL) {
      /* Get dataspace from the native dataset */
      H5VL_dataset_get_args_t get_args;
      get_args.op_type = H5VL_DATASET_GET_SPACE;
      get_args.args.get_space.space_id = H5I_INVALID_HID;
      H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id,
                       &get_args, dxpl_id, nullptr);
      space = get_args.args.get_space.space_id;
    }
    hssize_t nelem = H5Sget_simple_extent_npoints(space);
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

    /* Also write to native VOL for metadata consistency */
    H5VLdataset_write(1, &dataset->obj.under_object,
                       dataset->obj.under_vol_id,
                       &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                       dxpl_id, &buf[d], req);
    if (tracing)
      clio_trace_access(dataset, clio::trace::Op::kWrite, mem_type_id[d],
                          mem_space_id[d], file_space_id[d], dxpl_id,
                          clio::trace::Served::kCache, clio_since_us(t0));
  }

  return 0;
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
  auto *cte_client = get_cte_client();
  const bool tracing = clio::trace::enabled();

  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<clio_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;
    auto t0 = std::chrono::steady_clock::now();

    /* Native passthrough when there is no file reference, the type is not flat,
       or the read is collective. The native VOL always produces correct data. */
    bool cacheable_flat = dataset->file && dataset->cacheable &&
                          clio_type_is_read_cacheable(mem_type_id[d]) &&
                          !clio_is_collective(dxpl_id);
    if (!cacheable_flat) {
      H5VLdataset_read(1, &dataset->obj.under_object,
                        dataset->obj.under_vol_id,
                        &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                        dxpl_id, &buf[d], req);
      if (tracing)
        clio_trace_access(dataset, clio::trace::Op::kRead, mem_type_id[d],
                            mem_space_id[d], file_space_id[d], dxpl_id,
                            clio::trace::Served::kUncacheable,
                            clio_since_us(t0));
      continue;
    }

    /* Partial (non-full-extent) selection → serve from the CTE tier when the
       linear cache is already populated (serve-only). On a miss, fall back to
       native (unchanged). A full-extent selection (incl. a full hyperslab, as
       h5dump issues) is treated as whole so it populates/serves via CTE. */
    if (!clio_read_is_whole(mem_space_id[d], file_space_id[d])) {
      bool served = clio_serve_selection(dataset, mem_type_id[d],
                                           mem_space_id[d], file_space_id[d],
                                           dxpl_id, buf[d]);
      if (!served) {
        H5VLdataset_read(1, &dataset->obj.under_object,
                          dataset->obj.under_vol_id,
                          &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                          dxpl_id, &buf[d], req);
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
      H5VLdataset_read(1, &dataset->obj.under_object,
                        dataset->obj.under_vol_id,
                        &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                        dxpl_id, &buf[d], req);
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
      for (size_t i = 0; i < num_chunks; ++i) {
        size_t offset = i * chunk_size;
        size_t this_size = std::min(chunk_size, total_size - offset);
        auto buffer = CLIO_IPC->AllocateBuffer(this_size);
        if (buffer.IsNull()) break;            /* caching is best-effort */
        std::memcpy(buffer.ptr_, dst + offset, this_size);
        ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
        std::string blob_name = dataset->dataset_path + "/chunk_" +
                                std::to_string(i);
        auto fut = cte_client->AsyncPutBlob(
            dataset->file->tag_id, blob_name, offset, this_size, blob_data,
            -1.0f, clio::cte::core::Context(), 0);
        fut.Wait();
      }
    } else {
      /* HIT — serve the whole image from the CTE tier. */
      if (!clio_read_cached_image(dataset, total_size, dst)) return -1;
    }
    if (tracing)
      clio_trace_access(dataset, clio::trace::Op::kRead, mem_type_id[d],
                          mem_space_id[d], file_space_id[d], dxpl_id,
                          cached == 0 ? clio::trace::Served::kNative
                                      : clio::trace::Served::kCache,
                          clio_since_us(t0));
  }

  return 0;
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

  /* Flush all pending async writes */
  drain_dataset_puts(dset);

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
  *cap_flags = H5VL_CAP_FLAG_FILE_BASIC | H5VL_CAP_FLAG_DATASET_BASIC |
               H5VL_CAP_FLAG_GROUP_BASIC | H5VL_CAP_FLAG_ATTR_BASIC |
               H5VL_CAP_FLAG_LINK_BASIC | H5VL_CAP_FLAG_OBJECT_BASIC;
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

/* ========================================================================
 * VOL connector class definition
 * ======================================================================== */

const H5VL_class_t H5VL_clio_cls = {
    /* version      */ H5VL_VERSION,
    /* value        */ CLIO_VOL_CONNECTOR_VALUE,
    /* name         */ CLIO_VOL_CONNECTOR_NAME,
    /* conn_version */ CLIO_VOL_CONNECTOR_VERSION,
    /* cap_flags    */ H5VL_CAP_FLAG_FILE_BASIC | H5VL_CAP_FLAG_DATASET_BASIC |
                       H5VL_CAP_FLAG_GROUP_BASIC | H5VL_CAP_FLAG_ATTR_BASIC,
    /* initialize   */ nullptr,
    /* terminate    */ nullptr,

    /* info_cls */ {
        /* size    */ sizeof(clio_vol_info_t),
        /* copy    */ clio_info_copy,
        /* cmp     */ nullptr,
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
        /* optional */ nullptr,
        /* close    */ clio_attr_close,
    },

    /* dataset_cls */ {
        /* create   */ clio_dataset_create,
        /* open     */ clio_dataset_open,
        /* read     */ clio_dataset_read,
        /* write    */ clio_dataset_write,
        /* get      */ clio_dataset_get,
        /* specific */ clio_dataset_specific,
        /* optional */ nullptr,
        /* close    */ clio_dataset_close,
    },

    /* datatype_cls */ {
        /* commit   */ clio_datatype_commit,
        /* open     */ clio_datatype_open,
        /* get      */ clio_datatype_get,
        /* specific */ clio_datatype_specific,
        /* optional */ nullptr,
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
        /* optional */ nullptr,
        /* close    */ clio_group_close,
    },

    /* link_cls */ {
        /* create   */ clio_link_create,
        /* copy     */ clio_link_copy,
        /* move     */ nullptr,
        /* get      */ clio_link_get,
        /* specific */ clio_link_specific,
        /* optional */ nullptr,
    },

    /* object_cls */ {
        /* open     */ clio_object_open,
        /* copy     */ clio_object_copy,
        /* get      */ clio_object_get,
        /* specific */ clio_object_specific,
        /* optional */ nullptr,
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
        nullptr, nullptr,
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
