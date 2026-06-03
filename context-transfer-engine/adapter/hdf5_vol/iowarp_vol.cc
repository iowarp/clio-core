/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * IOWarp HDF5 VOL Connector
 *
 * Intercepts H5Dwrite/H5Dread and maps them to CTE AsyncPutBlob/AsyncGetBlob.
 * Large writes are split into configurable-size chunks and submitted
 * asynchronously. All other HDF5 operations pass through to the native VOL.
 */

#include "iowarp_vol.h"

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

#include <clio_runtime/clio_runtime.h>
/* transport_factory_impl.h provides the inline definitions of
   ctp::lbm::Transport::ClearRecvHandles / Send<...>. They are required
   when AsyncPutBlob template-instantiates here (it doesn't fire from
   chimaera.h alone). Without this include the linker leaves the
   templated symbols undefined in our .so. */
#include <clio_ctp/lightbeam/transport_factory_impl.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>

/* ========================================================================
 * Internal state structures
 * ======================================================================== */

struct iowarp_file_t;  /* fwd */

/* parent_file points to the iowarp_file_t this object belongs to. For
   files it is a self-pointer; for groups, datasets, attributes it is
   inherited from the containing file/group at create-time. The
   dataset_t branch reads it to find the CTE tag for AsyncPutBlob /
   AsyncGetBlob. */
struct iowarp_obj_t {
  void           *under_object;
  hid_t           under_vol_id;
  iowarp_file_t  *parent_file;
};

struct iowarp_file_t {
  iowarp_obj_t obj;
  clio::cte::core::TagId tag_id;
  std::string file_name;
  size_t chunk_size;
};

struct iowarp_dataset_t {
  iowarp_obj_t obj;
  iowarp_file_t *file;
  std::string dataset_path;
  /* When false the CTE cache is bypassed and every transfer goes to the native
     VOL. Set for datasets whose stable path is unknown (opened via the generic
     object-open / wrap paths), so we never key a blob by an empty/ambiguous
     name. */
  bool cacheable;
  /* Pending async writes flushed on close */
  std::vector<chi::Future<clio::cte::core::PutBlobTask>> pending_puts;
  std::vector<ctp::ipc::FullPtr<char>> pending_buffers;
};

/* Build a dataset wrapper. Centralised so every code path that can produce a
   dataset object (dataset_open/create, object_open, wrap_object) yields the
   same fully-formed iowarp_dataset_t — otherwise a dataset returned as a bare
   iowarp_obj_t would be fatally mis-cast when HDF5 later routes
   dataset_read/close to it. */
static iowarp_dataset_t *make_dataset_wrapper(void *under, hid_t under_vol_id,
                                              iowarp_file_t *parent_file,
                                              const char *path) {
  auto *dset = new iowarp_dataset_t;
  dset->obj.under_object = under;
  dset->obj.under_vol_id = under_vol_id;
  dset->file = parent_file;
  dset->obj.parent_file = parent_file;
  dset->dataset_path = path ? path : "";
  dset->cacheable = (parent_file != nullptr) && path && path[0] != '\0';
  return dset;
}

/* VOL object-wrap context. HDF5 uses this during link/object iteration
   (H5Literate2 / H5Lvisit2 / H5Ovisit2): before invoking the user's operator it
   saves a wrap context from this connector, then for every iterated object it
   calls iowarp_wrap_object() to re-wrap the native object back into our VOL so
   the operator's hid_t routes through this connector. Mirrors the reference
   pass-through connector (H5VLpassthru). The previous implementation returned
   the native wrap context directly and had iowarp_wrap_object() skip
   H5VLwrap_object(), which left the iterated object only half-wrapped and made
   H5Literate2 abort with "... is not a VOL connector ID" — the deeper blocker
   that kept neuroh5's group enumeration from completing. */
struct iowarp_wrap_ctx_t {
  hid_t under_vol_id;
  void *under_wrap_ctx;
};

/* ========================================================================
 * Helper: Get CTE client
 * ======================================================================== */

static clio::cte::core::Client *get_cte_client() {
  /* Lazily attach this process to the running chimaera/CTE runtime on first
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

static void *iowarp_info_copy(const void *_info) {
  const auto *info = static_cast<const iowarp_vol_info_t *>(_info);
  auto *new_info = new iowarp_vol_info_t(*info);
  if (info->under_vol_info) {
    H5VLcopy_connector_info(info->under_vol_id, &new_info->under_vol_info,
                            info->under_vol_info);
  }
  return new_info;
}

static herr_t iowarp_info_free(void *_info) {
  auto *info = static_cast<iowarp_vol_info_t *>(_info);
  if (info->under_vol_info) {
    H5VLfree_connector_info(info->under_vol_id, info->under_vol_info);
  }
  delete info;
  return 0;
}

/* ========================================================================
 * Wrap / unwrap callbacks
 * ======================================================================== */

static void *iowarp_wrap_get_object(const void *obj) {
  auto *o = static_cast<const iowarp_obj_t *>(obj);
  return H5VLget_object(o->under_object, o->under_vol_id);
}

static herr_t iowarp_get_wrap_ctx(const void *obj, void **wrap_ctx) {
  auto *o = static_cast<const iowarp_obj_t *>(obj);
  /* Carry the under VOL id alongside its wrap context so iowarp_wrap_object()
     can call H5VLwrap_object() against the right connector. */
  auto *ctx = new iowarp_wrap_ctx_t;
  ctx->under_vol_id = o->under_vol_id;
  H5Iinc_ref(ctx->under_vol_id);
  ctx->under_wrap_ctx = nullptr;
  H5VLget_wrap_ctx(o->under_object, o->under_vol_id, &ctx->under_wrap_ctx);
  *wrap_ctx = ctx;
  return 0;
}

static void *iowarp_wrap_object(void *under_obj, H5I_type_t obj_type,
                                void *_wrap_ctx) {
  auto *ctx = static_cast<iowarp_wrap_ctx_t *>(_wrap_ctx);
  hid_t under_vol_id = ctx ? ctx->under_vol_id : H5VL_NATIVE;
  void *under_wrap_ctx = ctx ? ctx->under_wrap_ctx : nullptr;

  /* Let the underlying (native) VOL wrap the raw iteration object first.
     Skipping this step — as the old code did — left the object unusable by the
     native VOL and made link/object iteration abort. */
  void *under = H5VLwrap_object(under_obj, obj_type, under_vol_id,
                                under_wrap_ctx);
  if (!under) return nullptr;

  /* parent_file is unknown during iteration, so everything wrapped here falls
     back to the native VOL (correct, just uncached). A wrapped *dataset* must
     still be a full iowarp_dataset_t (non-cacheable), or a later
     dataset_read/close would mis-cast it. */
  if (obj_type == H5I_DATASET) {
    return make_dataset_wrapper(under, under_vol_id, nullptr, nullptr);
  }
  auto *o = new iowarp_obj_t;
  o->under_object = under;
  o->under_vol_id = under_vol_id;
  o->parent_file = nullptr;
  return o;
}

static void *iowarp_unwrap_object(void *obj) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  /* Symmetric with iowarp_wrap_object()'s H5VLwrap_object(): peel the native
     wrapper back off before discarding our wrapper. */
  void *under = H5VLunwrap_object(o->under_object, o->under_vol_id);
  if (under) delete o;
  return under;
}

static herr_t iowarp_free_wrap_ctx(void *_wrap_ctx) {
  auto *ctx = static_cast<iowarp_wrap_ctx_t *>(_wrap_ctx);
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

static void *iowarp_file_create(const char *name, unsigned flags,
                                hid_t fcpl_id, hid_t fapl_id,
                                hid_t dxpl_id, void **req) {
  /* Get underlying VOL info */
  hid_t under_vol_id = H5VL_NATIVE;
  void *under_vol_info = nullptr;

  /* Check if user provided IOWarp VOL info.
   * HDF5 2.x split H5Pget_vol() into H5Pget_vol_id() + H5Pget_vol_info();
     we only care about the info blob here. */
  iowarp_vol_info_t *vol_info = nullptr;
  H5Pget_vol_info(fapl_id, reinterpret_cast<void **>(&vol_info));
  size_t chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
  if (vol_info) {
    under_vol_id = vol_info->under_vol_id;
    under_vol_info = vol_info->under_vol_info;
    if (vol_info->chunk_size > 0) {
      chunk_size = vol_info->chunk_size;
    }
  }

  /* Check environment variable for chunk size */
  const char *env_chunk = std::getenv("IOWARP_VOL_CHUNK_SIZE");
  if (env_chunk) {
    chunk_size = std::strtoul(env_chunk, nullptr, 10);
    if (chunk_size == 0) chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
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
  auto *file = new iowarp_file_t;
  file->obj.under_object = under_file;
  file->obj.under_vol_id = H5VL_NATIVE;
  file->obj.parent_file = file;          /* self-pointer */
  file->tag_id = tag_task->tag_id_;
  file->file_name = name;
  file->chunk_size = chunk_size;

  return file;
}

static void *iowarp_file_open(const char *name, unsigned flags,
                              hid_t fapl_id, hid_t dxpl_id, void **req) {
  size_t chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
  const char *env_chunk = std::getenv("IOWARP_VOL_CHUNK_SIZE");
  if (env_chunk) {
    chunk_size = std::strtoul(env_chunk, nullptr, 10);
    if (chunk_size == 0) chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
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

  auto *file = new iowarp_file_t;
  file->obj.under_object = under_file;
  file->obj.under_vol_id = H5VL_NATIVE;
  file->obj.parent_file = file;          /* self-pointer */
  file->tag_id = tag_task->tag_id_;
  file->file_name = name;
  file->chunk_size = chunk_size;

  return file;
}

static herr_t iowarp_file_get(void *obj, H5VL_file_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  return H5VLfile_get(file->obj.under_object, file->obj.under_vol_id,
                      args, dxpl_id, req);
}

static herr_t iowarp_file_specific(void *obj,
                                   H5VL_file_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  return H5VLfile_specific(file->obj.under_object, file->obj.under_vol_id,
                           args, dxpl_id, req);
}

static herr_t iowarp_file_close(void *obj, hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  herr_t ret = H5VLfile_close(file->obj.under_object, file->obj.under_vol_id,
                               dxpl_id, req);
  delete file;
  return ret;
}

/* ========================================================================
 * Dataset callbacks
 * ======================================================================== */

/**
 * Helper: extract the iowarp_file_t from an obj pointer.
 *
 * Every wrapper (file, group, dataset, attr) has an iowarp_obj_t as its
 * first member with a parent_file pointer. Files set it to themselves;
 * groups inherit it from their parent; datasets/attrs inherit it from
 * their containing file or group.
 *
 * If obj came in without parent_file populated (e.g. from
 * iowarp_wrap_object where the wrap context didn't include a file
 * reference), this returns nullptr and the caller falls back to pure
 * native-VOL passthrough.
 */
static iowarp_file_t *find_parent_file(void *obj) {
  if (!obj) return nullptr;
  /* All iowarp wrappers (iowarp_obj_t, iowarp_file_t, iowarp_dataset_t)
     start with iowarp_obj_t as their first member, so reading
     parent_file via this cast is safe. */
  return static_cast<iowarp_obj_t *>(obj)->parent_file;
}

/**
 * Helper: is this a whole-dataset transfer we can represent as linear CTE
 * chunks? The blob cache is keyed by linear byte offset over the full dataset
 * extent, so it can only correctly serve transfers covering the entire dataset
 * contiguously (H5S_ALL on both mem and file spaces). Any hyperslab / point
 * selection must go to the native VOL.
 */
static bool iowarp_is_whole_read(hid_t mem_space_id, hid_t file_space_id) {
  return mem_space_id == H5S_ALL && file_space_id == H5S_ALL;
}

/**
 * Helper: true when the transfer plist requests collective MPI-IO. Collective
 * transfers must stay on the native VOL — serving some ranks from cache while
 * others miss would desynchronise the collective call and deadlock.
 */
static bool iowarp_is_collective(hid_t dxpl_id) {
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

static void *iowarp_dataset_create(void *obj,
                                   const H5VL_loc_params_t *loc_params,
                                   const char *name, hid_t lcpl_id,
                                   hid_t type_id, hid_t space_id,
                                   hid_t dcpl_id, hid_t dapl_id,
                                   hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);

  /* Create dataset via native VOL */
  void *under_dset = H5VLdataset_create(
      o->under_object, loc_params, o->under_vol_id, name,
      lcpl_id, type_id, space_id, dcpl_id, dapl_id, dxpl_id, req);
  if (!under_dset) return nullptr;

  return make_dataset_wrapper(under_dset, o->under_vol_id,
                              find_parent_file(obj), name);
}

static void *iowarp_dataset_open(void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 const char *name, hid_t dapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);

  void *under_dset = H5VLdataset_open(
      o->under_object, loc_params, o->under_vol_id, name,
      dapl_id, dxpl_id, req);
  if (!under_dset) return nullptr;

  return make_dataset_wrapper(under_dset, o->under_vol_id,
                              find_parent_file(obj), name);
}

/**
 * Dataset write: chunk data into async PutBlob calls.
 *
 * Each chunk is copied to shared memory and submitted via AsyncPutBlob.
 * Futures are collected in the dataset state and flushed on close.
 */
static herr_t iowarp_dataset_write(size_t count, void *dset[],
                                   hid_t mem_type_id[],
                                   hid_t mem_space_id[],
                                   hid_t file_space_id[],
                                   hid_t dxpl_id, const void *buf[],
                                   void **req) {
  auto *cte_client = get_cte_client();

  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<iowarp_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;

    /* Only whole-dataset, independent writes can be represented in the linear
       CTE chunk cache. For no-file-reference, partial (hyperslab), or
       collective writes, persist to the native VOL only — caching a partial
       write under a whole-dataset key would poison a later whole read. */
    if (!dataset->file || !dataset->cacheable ||
        !iowarp_is_whole_read(mem_space_id[d], file_space_id[d]) ||
        iowarp_is_collective(dxpl_id)) {
      H5VLdataset_write(1, &dataset->obj.under_object,
                         dataset->obj.under_vol_id,
                         &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                         dxpl_id, &buf[d], req);
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
  }

  return 0;
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
static herr_t iowarp_dataset_read(size_t count, void *dset[],
                                  hid_t mem_type_id[],
                                  hid_t mem_space_id[],
                                  hid_t file_space_id[],
                                  hid_t dxpl_id, void *buf[],
                                  void **req) {
  auto *cte_client = get_cte_client();

  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<iowarp_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;

    /* Native passthrough when there is no file reference, the selection is
       partial, or the read is collective. The native VOL always produces
       correct data; only the whole/independent case is cacheable. */
    if (!dataset->file || !dataset->cacheable ||
        !iowarp_is_whole_read(mem_space_id[d], file_space_id[d]) ||
        iowarp_is_collective(dxpl_id)) {
      H5VLdataset_read(1, &dataset->obj.under_object,
                        dataset->obj.under_vol_id,
                        &mem_type_id[d], &mem_space_id[d], &file_space_id[d],
                        dxpl_id, &buf[d], req);
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
      continue;
    }

    size_t type_size = H5Tget_size(mem_type_id[d]);
    size_t total_size = static_cast<size_t>(nelem) * type_size;
    size_t chunk_size = dataset->file->chunk_size;
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    char *dst = static_cast<char *>(buf[d]);

    /* Hit test: a fully-populated cache always has a non-empty chunk_0. */
    chi::u64 cached = 0;
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
      /* HIT — serve every chunk from the CTE tier. */
      std::vector<chi::Future<clio::cte::core::GetBlobTask>> futures;
      std::vector<ctp::ipc::FullPtr<char>> buffers;
      for (size_t i = 0; i < num_chunks; ++i) {
        size_t offset = i * chunk_size;
        size_t this_size = std::min(chunk_size, total_size - offset);
        auto buffer = CLIO_IPC->AllocateBuffer(this_size);
        if (buffer.IsNull()) return -1;
        ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
        std::string blob_name = dataset->dataset_path + "/chunk_" +
                                std::to_string(i);
        futures.push_back(cte_client->AsyncGetBlob(
            dataset->file->tag_id, blob_name, offset, this_size, 0, blob_data));
        buffers.push_back(std::move(buffer));
      }
      for (size_t i = 0; i < futures.size(); ++i) {
        futures[i].Wait();
        size_t offset = i * chunk_size;
        size_t this_size = std::min(chunk_size, total_size - offset);
        std::memcpy(dst + offset, buffers[i].ptr_, this_size);
      }
    }
  }

  return 0;
}

static herr_t iowarp_dataset_get(void *obj, H5VL_dataset_get_args_t *args,
                                 hid_t dxpl_id, void **req) {
  auto *dset = static_cast<iowarp_dataset_t *>(obj);
  return H5VLdataset_get(dset->obj.under_object, dset->obj.under_vol_id,
                          args, dxpl_id, req);
}

static herr_t iowarp_dataset_specific(void *obj,
                                      H5VL_dataset_specific_args_t *args,
                                      hid_t dxpl_id, void **req) {
  auto *dset = static_cast<iowarp_dataset_t *>(obj);
  return H5VLdataset_specific(dset->obj.under_object, dset->obj.under_vol_id,
                               args, dxpl_id, req);
}

static herr_t iowarp_dataset_close(void *obj, hid_t dxpl_id, void **req) {
  auto *dset = static_cast<iowarp_dataset_t *>(obj);

  /* Flush all pending async writes */
  for (auto &future : dset->pending_puts) {
    future.Wait();
  }
  dset->pending_puts.clear();
  dset->pending_buffers.clear();

  herr_t ret = H5VLdataset_close(dset->obj.under_object,
                                  dset->obj.under_vol_id, dxpl_id, req);
  delete dset;
  return ret;
}

/* ========================================================================
 * Passthrough: group, attribute, datatype, link, object, introspect
 * ======================================================================== */

/* Group */
static void *iowarp_group_create(void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 const char *name, hid_t lcpl_id,
                                 hid_t gcpl_id, hid_t gapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = H5VLgroup_create(o->under_object, loc_params,
                                  o->under_vol_id, name, lcpl_id,
                                  gcpl_id, gapl_id, dxpl_id, req);
  if (!under) return nullptr;
  auto *grp = new iowarp_obj_t;
  grp->under_object = under;
  grp->under_vol_id = o->under_vol_id;
  grp->parent_file = o->parent_file;     /* inherit from parent */
  return grp;
}

static void *iowarp_group_open(void *obj,
                               const H5VL_loc_params_t *loc_params,
                               const char *name, hid_t gapl_id,
                               hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = H5VLgroup_open(o->under_object, loc_params,
                                o->under_vol_id, name, gapl_id,
                                dxpl_id, req);
  if (!under) return nullptr;
  auto *grp = new iowarp_obj_t;
  grp->under_object = under;
  grp->under_vol_id = o->under_vol_id;
  grp->parent_file = o->parent_file;     /* inherit from parent */
  return grp;
}

static herr_t iowarp_group_get(void *obj, H5VL_group_get_args_t *args,
                               hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLgroup_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t iowarp_group_specific(void *obj,
                                    H5VL_group_specific_args_t *args,
                                    hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLgroup_specific(o->under_object, o->under_vol_id, args,
                             dxpl_id, req);
}

static herr_t iowarp_group_close(void *obj, hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  herr_t ret = H5VLgroup_close(o->under_object, o->under_vol_id,
                                dxpl_id, req);
  delete o;
  return ret;
}

/* Attribute — full passthrough via native VOL, with proper wrapping */
static void *iowarp_attr_create(void *obj,
                                const H5VL_loc_params_t *loc_params,
                                const char *name, hid_t type_id,
                                hid_t space_id, hid_t acpl_id,
                                hid_t aapl_id, hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = H5VLattr_create(o->under_object, loc_params, o->under_vol_id,
                                 name, type_id, space_id, acpl_id, aapl_id,
                                 dxpl_id, req);
  if (!under) return nullptr;
  auto *attr = new iowarp_obj_t;
  attr->under_object = under;
  attr->under_vol_id = o->under_vol_id;
  attr->parent_file = o->parent_file;
  return attr;
}

static void *iowarp_attr_open(void *obj,
                              const H5VL_loc_params_t *loc_params,
                              const char *name, hid_t aapl_id,
                              hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = H5VLattr_open(o->under_object, loc_params, o->under_vol_id,
                               name, aapl_id, dxpl_id, req);
  if (!under) return nullptr;
  auto *attr = new iowarp_obj_t;
  attr->under_object = under;
  attr->under_vol_id = o->under_vol_id;
  attr->parent_file = o->parent_file;
  return attr;
}

static herr_t iowarp_attr_read(void *attr, hid_t dtype_id, void *buf,
                               hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(attr);
  return H5VLattr_read(o->under_object, o->under_vol_id, dtype_id, buf,
                        dxpl_id, req);
}

static herr_t iowarp_attr_write(void *attr, hid_t dtype_id, const void *buf,
                                hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(attr);
  return H5VLattr_write(o->under_object, o->under_vol_id, dtype_id, buf,
                         dxpl_id, req);
}

static herr_t iowarp_attr_get(void *obj, H5VL_attr_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLattr_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t iowarp_attr_specific(void *obj, const H5VL_loc_params_t *lp,
                                   H5VL_attr_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLattr_specific(o->under_object, lp, o->under_vol_id, args,
                            dxpl_id, req);
}

static herr_t iowarp_attr_close(void *attr, hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(attr);
  herr_t ret = H5VLattr_close(o->under_object, o->under_vol_id, dxpl_id, req);
  delete o;
  return ret;
}

/* Link — passthrough */
static herr_t iowarp_link_create(H5VL_link_create_args_t *args,
                                 void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 hid_t lcpl_id, hid_t lapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLlink_create(args, o ? o->under_object : nullptr, loc_params,
                          o ? o->under_vol_id : H5VL_NATIVE,
                          lcpl_id, lapl_id, dxpl_id, req);
}

static herr_t iowarp_link_copy(void *src_obj,
                               const H5VL_loc_params_t *loc_params1,
                               void *dst_obj,
                               const H5VL_loc_params_t *loc_params2,
                               hid_t lcpl_id, hid_t lapl_id,
                               hid_t dxpl_id, void **req) {
  auto *s = static_cast<iowarp_obj_t *>(src_obj);
  auto *d = static_cast<iowarp_obj_t *>(dst_obj);
  return H5VLlink_copy(s->under_object, loc_params1,
                        d->under_object, loc_params2,
                        s->under_vol_id, lcpl_id, lapl_id, dxpl_id, req);
}

static herr_t iowarp_link_get(void *obj, const H5VL_loc_params_t *loc_params,
                              H5VL_link_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLlink_get(o->under_object, loc_params, o->under_vol_id,
                       args, dxpl_id, req);
}

static herr_t iowarp_link_specific(void *obj,
                                   const H5VL_loc_params_t *loc_params,
                                   H5VL_link_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLlink_specific(o->under_object, loc_params, o->under_vol_id,
                            args, dxpl_id, req);
}

/* Object — passthrough */
static void *iowarp_object_open(void *obj,
                                const H5VL_loc_params_t *loc_params,
                                H5I_type_t *opened_type,
                                hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = H5VLobject_open(o->under_object, loc_params, o->under_vol_id,
                                 opened_type, dxpl_id, req);
  if (!under) return nullptr;

  /* CRITICAL: when the opened object is a dataset, HDF5 will subsequently route
     the dataset_read/write/close callbacks to this wrapper and cast it to
     iowarp_dataset_t. h5py opens datasets through H5Oopen, so returning a bare
     iowarp_obj_t here would be mis-cast and corrupt the heap. Build the right
     wrapper type. The dataset path is recovered from a by-name location when
     available; otherwise the dataset is marked non-cacheable. */
  if (opened_type && *opened_type == H5I_DATASET) {
    const char *name = nullptr;
    if (loc_params && loc_params->type == H5VL_OBJECT_BY_NAME) {
      name = loc_params->loc_data.loc_by_name.name;
    }
    return make_dataset_wrapper(under, o->under_vol_id, o->parent_file, name);
  }

  auto *wrapped = new iowarp_obj_t;
  wrapped->under_object = under;
  wrapped->under_vol_id = o->under_vol_id;
  wrapped->parent_file = o->parent_file;
  return wrapped;
}

static herr_t iowarp_object_copy(void *src_obj,
                                 const H5VL_loc_params_t *loc_params1,
                                 const char *src_name,
                                 void *dst_obj,
                                 const H5VL_loc_params_t *loc_params2,
                                 const char *dst_name,
                                 hid_t ocpypl_id, hid_t lcpl_id,
                                 hid_t dxpl_id, void **req) {
  auto *s = static_cast<iowarp_obj_t *>(src_obj);
  auto *d = static_cast<iowarp_obj_t *>(dst_obj);
  return H5VLobject_copy(s->under_object, loc_params1, src_name,
                          d->under_object, loc_params2, dst_name,
                          s->under_vol_id, ocpypl_id, lcpl_id, dxpl_id, req);
}

static herr_t iowarp_object_get(void *obj,
                                const H5VL_loc_params_t *loc_params,
                                H5VL_object_get_args_t *args,
                                hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLobject_get(o->under_object, loc_params, o->under_vol_id,
                         args, dxpl_id, req);
}

static herr_t iowarp_object_specific(void *obj,
                                     const H5VL_loc_params_t *loc_params,
                                     H5VL_object_specific_args_t *args,
                                     hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLobject_specific(o->under_object, loc_params, o->under_vol_id,
                              args, dxpl_id, req);
}

/* Datatype — passthrough. neuroh5/MiV store committed named datatypes under
   /H5Types (population enums, etc.) and open them with H5Topen; without these
   callbacks the connector's datatype_cls is all-null and every H5Topen through
   the VOL fails. Every op delegates to the native VOL. */
static void *iowarp_datatype_commit(void *obj,
                                    const H5VL_loc_params_t *loc_params,
                                    const char *name, hid_t type_id,
                                    hid_t lcpl_id, hid_t tcpl_id,
                                    hid_t tapl_id, hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = H5VLdatatype_commit(o->under_object, loc_params, o->under_vol_id,
                                     name, type_id, lcpl_id, tcpl_id, tapl_id,
                                     dxpl_id, req);
  if (!under) return nullptr;
  auto *dt = new iowarp_obj_t;
  dt->under_object = under;
  dt->under_vol_id = o->under_vol_id;
  dt->parent_file = o->parent_file;
  return dt;
}

static void *iowarp_datatype_open(void *obj,
                                  const H5VL_loc_params_t *loc_params,
                                  const char *name, hid_t tapl_id,
                                  hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = H5VLdatatype_open(o->under_object, loc_params, o->under_vol_id,
                                   name, tapl_id, dxpl_id, req);
  if (!under) return nullptr;
  auto *dt = new iowarp_obj_t;
  dt->under_object = under;
  dt->under_vol_id = o->under_vol_id;
  dt->parent_file = o->parent_file;
  return dt;
}

static herr_t iowarp_datatype_get(void *obj, H5VL_datatype_get_args_t *args,
                                  hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLdatatype_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t iowarp_datatype_specific(void *obj,
                                       H5VL_datatype_specific_args_t *args,
                                       hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLdatatype_specific(o->under_object, o->under_vol_id, args, dxpl_id,
                                req);
}

static herr_t iowarp_datatype_close(void *obj, hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  herr_t ret = H5VLdatatype_close(o->under_object, o->under_vol_id, dxpl_id,
                                   req);
  delete o;
  return ret;
}

/* Introspect */
static herr_t iowarp_introspect_get_conn_cls(void *obj,
                                             H5VL_get_conn_lvl_t lvl,
                                             const H5VL_class_t **conn_cls) {
  (void)obj; (void)lvl;
  *conn_cls = &H5VL_iowarp_cls;
  return 0;
}

static herr_t iowarp_introspect_get_cap_flags(const void *info,
                                              uint64_t *cap_flags) {
  (void)info;
  *cap_flags = H5VL_CAP_FLAG_FILE_BASIC | H5VL_CAP_FLAG_DATASET_BASIC |
               H5VL_CAP_FLAG_GROUP_BASIC | H5VL_CAP_FLAG_ATTR_BASIC |
               H5VL_CAP_FLAG_LINK_BASIC | H5VL_CAP_FLAG_OBJECT_BASIC;
  return 0;
}

static herr_t iowarp_introspect_opt_query(void *obj, H5VL_subclass_t cls,
                                          int opt_type, uint64_t *flags) {
  (void)obj; (void)cls; (void)opt_type;
  *flags = 0;
  return 0;
}

/* ========================================================================
 * VOL connector class definition
 * ======================================================================== */

const H5VL_class_t H5VL_iowarp_cls = {
    /* version      */ H5VL_VERSION,
    /* value        */ IOWARP_VOL_CONNECTOR_VALUE,
    /* name         */ IOWARP_VOL_CONNECTOR_NAME,
    /* conn_version */ IOWARP_VOL_CONNECTOR_VERSION,
    /* cap_flags    */ H5VL_CAP_FLAG_FILE_BASIC | H5VL_CAP_FLAG_DATASET_BASIC |
                       H5VL_CAP_FLAG_GROUP_BASIC | H5VL_CAP_FLAG_ATTR_BASIC,
    /* initialize   */ nullptr,
    /* terminate    */ nullptr,

    /* info_cls */ {
        /* size    */ sizeof(iowarp_vol_info_t),
        /* copy    */ iowarp_info_copy,
        /* cmp     */ nullptr,
        /* free    */ iowarp_info_free,
        /* to_str  */ nullptr,
        /* from_str*/ nullptr,
    },

    /* wrap_cls */ {
        /* get_object  */ iowarp_wrap_get_object,
        /* get_wrap_ctx*/ iowarp_get_wrap_ctx,
        /* wrap_object */ iowarp_wrap_object,
        /* unwrap_object*/ iowarp_unwrap_object,
        /* free_wrap_ctx*/ iowarp_free_wrap_ctx,
    },

    /* attr_cls */ {
        /* create   */ iowarp_attr_create,
        /* open     */ iowarp_attr_open,
        /* read     */ iowarp_attr_read,
        /* write    */ iowarp_attr_write,
        /* get      */ iowarp_attr_get,
        /* specific */ iowarp_attr_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_attr_close,
    },

    /* dataset_cls */ {
        /* create   */ iowarp_dataset_create,
        /* open     */ iowarp_dataset_open,
        /* read     */ iowarp_dataset_read,
        /* write    */ iowarp_dataset_write,
        /* get      */ iowarp_dataset_get,
        /* specific */ iowarp_dataset_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_dataset_close,
    },

    /* datatype_cls */ {
        /* commit   */ iowarp_datatype_commit,
        /* open     */ iowarp_datatype_open,
        /* get      */ iowarp_datatype_get,
        /* specific */ iowarp_datatype_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_datatype_close,
    },

    /* file_cls */ {
        /* create   */ iowarp_file_create,
        /* open     */ iowarp_file_open,
        /* get      */ iowarp_file_get,
        /* specific */ iowarp_file_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_file_close,
    },

    /* group_cls */ {
        /* create   */ iowarp_group_create,
        /* open     */ iowarp_group_open,
        /* get      */ iowarp_group_get,
        /* specific */ iowarp_group_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_group_close,
    },

    /* link_cls */ {
        /* create   */ iowarp_link_create,
        /* copy     */ iowarp_link_copy,
        /* move     */ nullptr,
        /* get      */ iowarp_link_get,
        /* specific */ iowarp_link_specific,
        /* optional */ nullptr,
    },

    /* object_cls */ {
        /* open     */ iowarp_object_open,
        /* copy     */ iowarp_object_copy,
        /* get      */ iowarp_object_get,
        /* specific */ iowarp_object_specific,
        /* optional */ nullptr,
        /* (no close — objects are closed by their specific type class) */
    },

    /* introspect_cls */ {
        /* get_conn_cls   */ iowarp_introspect_get_conn_cls,
        /* get_cap_flags  */ iowarp_introspect_get_cap_flags,
        /* opt_query      */ iowarp_introspect_opt_query,
    },

    /* request_cls */ {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    },

    /* blob_cls */ {
        nullptr, nullptr, nullptr, nullptr,
    },

    /* token_cls */ {
        nullptr, nullptr,
    },

    /* optional */ nullptr,
};

hid_t H5VL_iowarp_register(void) {
  return H5VLregister_connector(&H5VL_iowarp_cls, H5P_DEFAULT);
}

/* ========================================================================
 * HDF5 plugin entry points
 *
 * These two exports let HDF5 discover and load the connector dynamically via
 * the standard environment-variable mechanism — no application change needed:
 *
 *   export HDF5_PLUGIN_PATH=<dir containing libiowarp_hdf5_vol.so>
 *   export HDF5_VOL_CONNECTOR="iowarp"
 *
 * On the first H5Fopen/H5Fcreate, HDF5 dlopen()s plugins on HDF5_PLUGIN_PATH,
 * calls H5PLget_plugin_type() (must be H5PL_TYPE_VOL) and H5PLget_plugin_info()
 * (returns the connector class), and matches by connector name ("iowarp").
 * Without these the connector could only be installed by an application calling
 * H5VL_iowarp_register() + H5Pset_vol() itself.
 * ======================================================================== */
extern "C" H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_VOL; }
extern "C" const void *H5PLget_plugin_info(void) { return &H5VL_iowarp_cls; }
