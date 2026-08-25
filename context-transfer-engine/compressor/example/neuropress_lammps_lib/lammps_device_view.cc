/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file lammps_device_view.cc
 * @brief The driver's window onto LAMMPS' Kokkos DEVICE views. No LAMMPS patch.
 *
 * The rest of the driver speaks LAMMPS' C library interface (src/library.h),
 * which exposes only the HOST mirror of Atom::x/v/f -- `lammps_extract_atom`
 * returns `double **` into host memory and `lammps_gather_atoms` copies out of
 * it. There is no device accessor in library.h at all (the only GPU-adjacent
 * entries are lammps_has_gpu_device / lammps_get_gpu_device_info), so a driver
 * built against it alone cannot hand Clio anything but host bytes -- which is
 * what the `--order id|local` paths do, and why their chunks arrive at the
 * compressor with device=0 even when LAMMPS itself is running on the GPU.
 *
 * This translation unit is the exception. It includes LAMMPS' own C++ headers
 * and reads `LAMMPS::atomKK` (lammps.h:44, public) to reach
 * `AtomKokkos::k_x / k_v / k_f` (atom_kokkos.h:34-36, public DualViews). The
 * device side of those DualViews is real CUDA memory owned by LAMMPS, and
 * handing its address to Clio is what makes `--order device` a genuine
 * GPU-resident hand-over rather than a host copy with a different name.
 *
 * Nothing here writes to LAMMPS memory, calls sync(), or runs during a
 * timestep. It is a read of state LAMMPS has already declared final.
 *
 * WHY THIS IS A SEPARATE FILE
 * ---------------------------
 * Kokkos' CUDA headers refuse to compile under a host compiler
 * (Kokkos_Setup_Cuda.hpp: "KOKKOS_ENABLE_CUDA defined but the compiler is not
 * defining the __CUDACC__ macro as expected"), so this one file is built with
 * nvcc through LAMMPS' own `nvcc_wrapper` and LAMMPS' own flags -- see the
 * CMakeLists. neuropress_lammps_lib.cc stays plain C++ against library.h and
 * calls in here through the extern "C" surface below. That split is the whole
 * reason the earlier gpu-direct attempt's LAMMPS patch is unnecessary: the
 * code that needs Kokkos lives in the driver, not in a LAMMPS `fix`.
 *
 * WHEN THE DEVICE VIEWS ARE CURRENT
 * ---------------------------------
 * VerletKokkos::run syncs to HOST before output->write on output steps and
 * unconditionally at the end of run (verlet_kokkos.cpp:516,524), then sets
 * auto_sync = 1. `sync(Host, ...)` makes the host current WITHOUT marking the
 * device stale, so between `run` segments -- the driver's hand-over point --
 * BOTH sides hold the same bytes. Measured: need_sync<Device>() is 0 for x, v,
 * f and tag there, and a D2H of the device view is bit-identical to the host
 * mirror.
 *
 * Do NOT "fix" that by calling atomKK->sync(Device, ...) first.
 * AtomKokkos::sync (atom_kokkos.cpp:156-165) marks the HOST side modified
 * before syncing whenever auto_sync is on -- which VerletKokkos::run has just
 * turned on -- so a blanket sync(Device) schedules a host->device copy of the
 * whole payload. Harmless here only because the two images are equal; pure
 * cost, and actively wrong at any other point in the step. The driver asks
 * clio_lmp_need_sync_device() instead and REFUSES when it is non-zero, rather
 * than papering over a stale image with a copy.
 */

#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos_type.h"

#include <Kokkos_Core.hpp>

#include <cstring>
#include <type_traits>

using namespace LAMMPS_NS;

namespace {

/* LayoutRight is what makes "nlocal*3 contiguous doubles" mean "xyz per atom".
 * kokkos_type.h gives tdual_x_array LayoutLeft when LMP_KOKKOS_NO_LEGACY is
 * defined and LayoutRight otherwise (kokkos_type.h:827-830); v and f are
 * always LayoutRight. Under LayoutLeft the same bytes are all the *x*
 * components of the first 3*nlocal atoms -- data that compresses, stores and
 * decompresses perfectly and is completely wrong, with no runtime symptom to
 * catch it. So it fails the BUILD instead. */
static_assert(std::is_same<DAT::tdual_x_array::array_layout,
                           Kokkos::LayoutRight>::value,
              "tdual_x_array is not LayoutRight (LMP_KOKKOS_NO_LEGACY?): "
              "nlocal*3 doubles from the base pointer would not be xyz");
static_assert(std::is_same<DAT::tdual_v_array::array_layout,
                           Kokkos::LayoutRight>::value,
              "tdual_v_array is not LayoutRight");
static_assert(std::is_same<DAT::tdual_f_array::array_layout,
                           Kokkos::LayoutRight>::value,
              "tdual_f_array is not LayoutRight");

/** The LAMMPS instance, or nullptr when this run is not a Kokkos run.
 *  atomKK is null without `-k on g 1 -sf kk`, which is exactly the case the
 *  caller must refuse rather than quietly store host bytes. */
inline LAMMPS *KokkosLammps(void *handle) {
  auto *lmp = static_cast<LAMMPS *>(handle);
  if (lmp == nullptr || lmp->atomKK == nullptr) return nullptr;
  return lmp;
}

}  // namespace

extern "C" {

/**
 * A device buffer the ID gather can scatter into, grown on demand and reused.
 *
 * The gather needs somewhere the size of a WHOLE field, because scattering by
 * tag writes anywhere in it -- so it cannot write straight into the per-chunk
 * buffers the compressor is handed. Owned here rather than by the driver so
 * neuropress_lammps_lib.cc keeps needing no CUDA or Kokkos header of its own.
 *
 * One allocation for the whole run: every frame gathers the same field size,
 * so after the first call this is a no-op.
 */
static void *g_scratch = nullptr;
static size_t g_scratch_bytes = 0;

void *clio_lmp_device_scratch(size_t bytes) {
  if (bytes == 0) return nullptr;
  if (g_scratch != nullptr && g_scratch_bytes >= bytes) return g_scratch;
  if (g_scratch != nullptr) {
    Kokkos::kokkos_free<LMPDeviceType::memory_space>(g_scratch);
    g_scratch = nullptr;
    g_scratch_bytes = 0;
  }
  g_scratch = Kokkos::kokkos_malloc<LMPDeviceType::memory_space>(
      "clio_lmp_gather_scratch", bytes);
  if (g_scratch != nullptr) g_scratch_bytes = bytes;
  return g_scratch;
}

/** Release it. Must run BEFORE lammps_kokkos_finalize(), or the free lands
 *  after Kokkos has torn its memory spaces down. */
void clio_lmp_device_scratch_free(void) {
  if (g_scratch != nullptr) {
    Kokkos::kokkos_free<LMPDeviceType::memory_space>(g_scratch);
    g_scratch = nullptr;
    g_scratch_bytes = 0;
  }
}

/** Non-zero when this handle is a Kokkos run with device-resident atoms. */
int clio_lmp_device_available(void *handle) {
  return KokkosLammps(handle) != nullptr ? 1 : 0;
}

/** Atoms owned by this rank. Serial runs have nlocal == natoms; the driver
 *  refuses otherwise rather than storing a fraction of the system. */
long clio_lmp_device_nlocal(void *handle) {
  LAMMPS *lmp = KokkosLammps(handle);
  return lmp ? static_cast<long>(lmp->atomKK->nlocal) : -1;
}

/* The three field accessors. Each returns the DEVICE side of the DualView --
 * a real CUDA pointer into LAMMPS' own allocation, in LAMMPS' INTERNAL atom
 * order. The view is allocated with nmax rows, so the caller takes nlocal*3
 * elements from the base pointer (valid under LayoutRight, asserted above).
 * Nothing is copied here; the pointer is only meaningful while LAMMPS holds
 * it, i.e. until the next reallocation, which is why the driver stages out of
 * it immediately. */

double *clio_lmp_device_x(void *handle, long *nlocal_out) {
  LAMMPS *lmp = KokkosLammps(handle);
  if (!lmp) return nullptr;
  if (nlocal_out) *nlocal_out = static_cast<long>(lmp->atomKK->nlocal);
  return lmp->atomKK->k_x.view<LMPDeviceType>().data();
}

double *clio_lmp_device_v(void *handle, long *nlocal_out) {
  LAMMPS *lmp = KokkosLammps(handle);
  if (!lmp) return nullptr;
  if (nlocal_out) *nlocal_out = static_cast<long>(lmp->atomKK->nlocal);
  return lmp->atomKK->k_v.view<LMPDeviceType>().data();
}

double *clio_lmp_device_f(void *handle, long *nlocal_out) {
  LAMMPS *lmp = KokkosLammps(handle);
  if (!lmp) return nullptr;
  if (nlocal_out) *nlocal_out = static_cast<long>(lmp->atomKK->nlocal);
  return lmp->atomKK->k_f.view<LMPDeviceType>().data();
}

/** Which device views would be read STALE right now, as a bitmask:
 *  1 = x, 2 = v, 4 = f, 8 = tag. 0 means every one of them is current and can
 *  be read as-is; -1 means this is not a Kokkos run.
 *
 *  tag is included because the ID gather below indexes by it: a current x with
 *  a stale tag scatters the right coordinates to the wrong atoms, which is
 *  precisely the class of error that survives a round-trip check.
 *
 *  This REPORTS; it does not repair. See the header comment for why calling
 *  sync(Device, ...) to make the answer zero is the wrong move. */
int clio_lmp_need_sync_device(void *handle) {
  LAMMPS *lmp = KokkosLammps(handle);
  if (!lmp) return -1;
  int mask = 0;
  if (lmp->atomKK->k_x.need_sync<LMPDeviceType>()) mask |= 1;
  if (lmp->atomKK->k_v.need_sync<LMPDeviceType>()) mask |= 2;
  if (lmp->atomKK->k_f.need_sync<LMPDeviceType>()) mask |= 4;
  if (lmp->atomKK->k_tag.need_sync<LMPDeviceType>()) mask |= 8;
  return mask;
}

/** Are atom IDs present and consecutive (1..natoms)?
 *  The gather below writes to index tag[i]-1 and cannot be correct otherwise.
 *  lammps_gather_atoms raises the identical precondition as a LAMMPS error
 *  (library.cpp:3597-3601); here it is a question so the driver can refuse
 *  with its own message instead of aborting inside LAMMPS. */
int clio_lmp_tags_consecutive(void *handle) {
  LAMMPS *lmp = KokkosLammps(handle);
  if (!lmp) return 0;
  return (lmp->atomKK->tag_enable != 0 && lmp->atomKK->tag_consecutive() != 0)
             ? 1
             : 0;
}

/**
 * Gather one field into ATOM-ID order, on the device, into `dst_dev`.
 *
 * This is `lammps_gather_atoms`' arithmetic moved onto the GPU:
 * library.cpp:3663-3672 zero-fills a natoms*3 buffer and then, for each owned
 * atom i, writes `copy[3*(tag[i]-1) + j] = array[i][j]`. Same permutation,
 * same zero fill, same result -- but the destination is the Clio-registered
 * device buffer the compressor will read, so the payload never becomes host
 * bytes on the way.
 *
 * Why bother, rather than storing the device view as it stands: LAMMPS'
 * internal order is not stable. Atom::sort() re-bins atoms spatially every
 * `atom_modify sort` interval (default 1000 steps, atom.cpp:103), so two
 * frames stored in internal order are not atom-for-atom comparable and are
 * not what `dump h5md` writes -- which would take crosscheck_h5md.sh, the one
 * check that ties this example's bytes to a stock LAMMPS dump, out of play.
 * ID order costs one kernel and one buffer and keeps it.
 *
 * The zero fill is not defensive padding: it is part of the definition. A
 * multi-rank gather sums each rank's sparse contribution, so every slot this
 * rank does not own must be exactly 0.0. The driver refuses multi-rank runs,
 * but the arithmetic stays the same one either way.
 *
 * dst_dev must be device memory of at least natoms*3 doubles. Returns 0 on
 * success, non-zero on a refused precondition -- never a partial gather.
 */
int clio_lmp_device_gather_id(void *handle, const char *field, double *dst_dev,
                              long natoms) {
  LAMMPS *lmp = KokkosLammps(handle);
  if (!lmp || !field || !dst_dev || natoms <= 0) return 1;
  if (!clio_lmp_tags_consecutive(handle)) return 2;

  const int nlocal = lmp->atomKK->nlocal;
  if (nlocal < 0 || static_cast<long>(nlocal) > natoms) return 3;

  // Unmanaged: Clio owns these bytes (a registered kDeviceMem backend), so the
  // View must never take a reference to them.
  using DstView = Kokkos::View<double *, LMPDeviceType,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  DstView dst(dst_dev, static_cast<size_t>(natoms) * 3);

  auto tag = lmp->atomKK->k_tag.view<LMPDeviceType>();

  // Every slot, then the owned ones. Two passes rather than one because the
  // permutation is a scatter: which slots this rank writes is not known
  // without reading tag, and a slot left untouched must still be 0.0.
  Kokkos::parallel_for(
      "clio_gather_zero",
      Kokkos::RangePolicy<LMPDeviceType>(0, static_cast<size_t>(natoms) * 3),
      KOKKOS_LAMBDA(const size_t k) { dst(k) = 0.0; });

  if (std::strcmp(field, "x") == 0) {
    auto src = lmp->atomKK->k_x.view<LMPDeviceType>();
    Kokkos::parallel_for(
        "clio_gather_x", Kokkos::RangePolicy<LMPDeviceType>(0, nlocal),
        KOKKOS_LAMBDA(const int i) {
          const size_t off = static_cast<size_t>(tag(i) - 1) * 3;
          dst(off + 0) = src(i, 0);
          dst(off + 1) = src(i, 1);
          dst(off + 2) = src(i, 2);
        });
  } else if (std::strcmp(field, "v") == 0) {
    auto src = lmp->atomKK->k_v.view<LMPDeviceType>();
    Kokkos::parallel_for(
        "clio_gather_v", Kokkos::RangePolicy<LMPDeviceType>(0, nlocal),
        KOKKOS_LAMBDA(const int i) {
          const size_t off = static_cast<size_t>(tag(i) - 1) * 3;
          dst(off + 0) = src(i, 0);
          dst(off + 1) = src(i, 1);
          dst(off + 2) = src(i, 2);
        });
  } else if (std::strcmp(field, "f") == 0) {
    auto src = lmp->atomKK->k_f.view<LMPDeviceType>();
    Kokkos::parallel_for(
        "clio_gather_f", Kokkos::RangePolicy<LMPDeviceType>(0, nlocal),
        KOKKOS_LAMBDA(const int i) {
          const size_t off = static_cast<size_t>(tag(i) - 1) * 3;
          dst(off + 0) = src(i, 0);
          dst(off + 1) = src(i, 1);
          dst(off + 2) = src(i, 2);
        });
  } else {
    return 4;
  }

  // The caller stages and submits straight after this returns, and Clio's
  // copy runs on a different stream, so the gather has to be COMPLETE here
  // rather than merely issued.
  Kokkos::fence("clio_lmp_device_gather_id");
  return 0;
}

}  // extern "C"
