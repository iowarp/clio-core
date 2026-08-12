#!/usr/bin/env python3
"""Phase 1 / Part A — clio HDF5 compatibility suite (differential testing).

Does routing HDF5 through a clio connector preserve native HDF5 semantics?
Method: the NATIVE stack is the oracle. For each feature case, exercise four arms
and assert the file content (data + metadata) matches:

  native write    -> native read      (reference)
  connector write -> native read      (write compat: emits a valid native file)
  native write    -> connector read   (read compat)
  connector write -> connector read   (round-trip)

plus the native-compatibility gates of VFD_VOL_TECHNICAL_GOALS.md §1.1:
h5diff(native, connector) is clean (a), the produced file is structurally valid
with no connector loaded (b), and h5dump/h5ls/h5stat produce *identical output*
on the two files, not merely a zero exit status (c).

TWO CONNECTORS, ONE TEMPLATE. `--modes` selects which are exercised:

  vol   HDF5_VOL_CONNECTOR=clio  — sees semantic ops (datasets, groups, attrs)
  vfd   HDF5_DRIVER=clio_vfd     — sees only (addr, size)

They deliberately are NOT one merged suite. A VOL connector and a VFD intercept
different things, so a case means something different to each: to the VOL the
corpus below is a semantic matrix; to the VFD the very same programs are
BYTE-PATTERN GENERATORS (compact layout -> tiny writes, chunked+filters ->
variable-size scattered writes, extendible append -> EOF growth). The VFD is
datatype-agnostic; running a datatype matrix through it tests HDF5, not the
driver. Read the corpus that way when looking at the vfd column. What is shared
is the method, the harness and the corpus; what diverges stays in each
connector's own suite (VOL: iteration, tokens, direct chunk I/O — the C tests
below; VFD: locking, EOA/EOF, vector I/O, get_handle — test_vfd_adapter.cc).

Output: a per-connector pass/fail matrix. CI-shaped (nonzero exit on any
failure). Runs INSIDE the clio-core dev container; the driver keeps clio_run up
and spawns each arm as a subprocess with the connector env toggled.

  python3 hdf5_compat_suite.py --modes vol,vfd --out hdf5_compat_results.json
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BIN = os.environ.get("CLIO_VOL_BIN", "/workspace/build/bin")
CLIO_HOME = os.environ.get("CLIO_HOME_DIR", "/home/iowarp")
RUNTIME_LOG = "/tmp/clio_run_hdf5compat.log"
# Count-agnostic: the compose emits "All <N> pools created successfully"; the
# suite's own config (below) makes 2, a dev box's ~/.clio/clio.yaml may make 3.
RUNTIME_READY = "pools created successfully"
TMP = "/tmp/hdf5compat"

# Connector modes. "native" is the oracle (no clio in the process at all); the
# other two are the connectors under test. Both are selected by an environment
# variable read at HDF5 library init, which is why every arm must be a FRESH
# subprocess — see _env/_run.
CONNECTORS = ("vol", "vfd")

# Entries named by --expect-fail, populated by driver(). These are DOCUMENTED
# open gaps, not broken tests: they are deterministic, they always fail the same
# way, and the suite still exits 0. They print as KNOWN-GAP rather than FAIL so
# an expected gap is not mistaken for a regression at a glance -- the distinction
# the exit code already makes, made visible in the output too.
EXPECT_FAIL = set()

# One line per known gap, printed next to it. A gap with no explanation is
# indistinguishable from an allowlist someone added to make a test go quiet.
GAP_NOTES = {
    # Empty, and that is the point. The only entry this suite ever carried --
    # the VOL cache serving a pre-corruption copy of a file damaged behind
    # HDF5's back -- was closed by the coherence stamp (file identity captured
    # at close, validated at open). An entry here is a gap that is understood
    # and written down, never one that is merely inconvenient.
}


def _mark(key, ok):
    """PASS / KNOWN-GAP / FAIL for one result entry."""
    if ok:
        return "PASS"
    return "KNOWN-GAP" if key in EXPECT_FAIL else "FAIL"

# Each arm writes into its own directory, and those directory names are all the
# SAME LENGTH on purpose. This is not cosmetic.
#
# The two files being compared must have different paths (they coexist), but
# some layouts store a FILENAME INSIDE the file: an external dataset records the
# raw-data file's name, and a VDS records its source files' names in the local
# heap. So the name's LENGTH becomes file content, and a native/connector name
# that differs in length shows up as a genuine byte difference -- h5stat reports
# a different Heap/File-metadata/Unaccounted-space total, and h5ls sizes its
# column rule to the longest name. Tool parity then fails for a reason that has
# nothing to do with the connector.
#
# Equal-length arm directories plus an identical basename ("<case>.h5" in
# nat/vol/vfd) make the two files structurally identical, so parity compares
# what it is supposed to. Any connector added here must keep a 3-character
# directory token; the assertion below is the reminder.
ARM_DIR = {"native": "nat", "vol": "vol", "vfd": "vfd"}
assert len({len(v) for v in ARM_DIR.values()}) == 1, \
    "arm directory names must be equal length -- see the comment above"

# CTest maps this exit code to "Skipped" (wired via SKIP_RETURN_CODE in the
# adapter CMakeLists). This is a DIFFERENTIAL test whose oracle is a native HDF5
# stack: every write/read arm runs as a subprocess of THIS interpreter and needs
# h5py, plus the HDF5 CLI tools (h5diff/h5dump/h5ls) it cross-checks the VOL file
# with. On a host without that toolchain there is nothing to compare against, so
# the suite reports SKIP rather than a wall of false FAILs — a missing h5py is an
# environment gap, not a VOL incompatibility.
SKIP_RC = 125

# Self-contained runtime config. The suite must NOT depend on a pre-existing
# ~/.clio/clio.yaml: it is absent in CI (the deps-cpu image has no such file),
# so `clio_run start` would compose only the built-in admin pool and never emit
# the readiness marker ("clio_run did not become ready"). We ship a minimal
# config via CLIO_SERVER_CONF instead: a small DRAM bdev + the CTE core pool,
# with capacities bounded to fit a constrained CI /dev/shm (docker --shm-size=2g).
# The CTE pool is named to match clio::cte::core::kCtePoolName so the client's
# get-or-create resolves it by name (Local routing) rather than broadcasting.
SUITE_CONF = os.path.join(TMP, "clio_suite.yaml")
SUITE_CONF_YAML = """\
networking:
  port: 9413
  neighborhood_size: 32
memory:
  main_segment_size: 256MB
  client_data_segment_size: 256MB
  runtime_data_segment_size: 256MB
runtime:
  num_threads: 4
  queue_depth: 1024
  local_sched: "default"
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "256MB"
  - mod_name: clio_cte_core
    pool_name: clio_cte_core
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "ram::cte_ram_tier1"
        bdev_type: "ram"
        capacity_limit: "256MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
"""

# ---------------------------------------------------------------- write fixtures
# Each writer builds a deterministic file at `path` via h5py. The connector is
# selected by the driver through the process env (HDF5_VOL_CONNECTOR /
# HDF5_DRIVER); the fixtures themselves are plain HDF5 programs and contain no
# clio-specific anything. That is the point: the same program is a valid VOL
# semantic case AND a valid VFD byte-pattern generator.

def w_int32_1d_contig(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(4096, dtype="i4"))

def w_float64_2d_chunked(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("m", data=np.linspace(0, 1, 64 * 64, dtype="f8").reshape(64, 64),
                         chunks=(16, 64))

def w_float32_3d_contig(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("v", data=np.arange(16 * 16 * 16, dtype="f4").reshape(16, 16, 16))

def w_compound(path):
    import h5py, numpy as np
    dt = np.dtype([("id", "i4"), ("x", "f8"), ("y", "f8")])
    arr = np.zeros(256, dtype=dt)
    arr["id"] = np.arange(256); arr["x"] = np.arange(256) * 1.5; arr["y"] = -np.arange(256)
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=arr)

def w_strings(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("fixed", data=np.array([b"alpha", b"beta", b"gamma"], dtype="S8"))
        f.create_dataset("vlen", data=["one", "two", "three"],
                         dtype=h5py.string_dtype())

def w_attrs(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        d = f.create_dataset("d", data=np.arange(10, dtype="i4"))
        d.attrs["units"] = "meters"
        d.attrs["scale"] = np.float64(2.5)
        d.attrs["shape3"] = np.arange(3, dtype="i8")
        g = f.create_group("grp")
        g.attrs["title"] = "a group"
        f.attrs["root_note"] = "file-level attr"

def w_groups_links(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        g = f.create_group("outer/inner")
        ds = g.create_dataset("leaf", data=np.arange(20, dtype="i4"))
        f["hardlink"] = ds                      # hard link to the dataset
        f["softlink"] = h5py.SoftLink("/outer/inner/leaf")

def w_chunked_shuffle(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("s", data=np.arange(4096, dtype="i4").reshape(64, 64),
                         chunks=(16, 16), shuffle=True)

def w_chunked_ragged(path):
    """Chunked with dims that do NOT divide the shape — partial edge chunks.

    Every other chunked case in this corpus divides evenly, which leaves the
    edge case untested on both sides: HDF5 does not trim chunks at the dataset
    boundary, so the last chunk in each dimension hangs past the end of the
    logical array. That is the path where a reader (including the §1.1(d)
    library-free one) either reads the padding into the tail or walks off its
    buffer, and where a connector caching whole chunks can return more data
    than the dataset has.

    Ragged in BOTH dimensions, since ragged-in-the-fastest and
    ragged-in-the-slowest are different code paths.
    """
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=(np.arange(37 * 19, dtype="i4") * 7 + 3)
                         .reshape(37, 19), chunks=(8, 5))


def w_hyperslab_src(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("g", data=np.arange(100 * 100, dtype="f4").reshape(100, 100),
                         chunks=(25, 100))

def w_uint_types(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("u8", data=np.arange(256, dtype="u1"))
        f.create_dataset("u16", data=np.arange(4096, dtype="u2"))
        f.create_dataset("u32", data=(np.arange(1024, dtype="u4") * 100003))

def w_enum(path):
    import h5py, numpy as np
    dt = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 2}, basetype="i4")
    with h5py.File(path, "w") as f:
        f.create_dataset("e", data=np.array([0, 1, 2, 1, 0, 2, 2], dtype="i4"), dtype=dt)

def w_array_dtype(path):
    """H5T_ARRAY of 3 doubles, dataspace (50,).

    NOTE, because the obvious spelling is silently broken. This case used to be
        create_dataset("arr", data=<(50,3) f8 array>, dtype=np.dtype(("f8",(3,))))
    which h5py resolves to a dataspace of (50,3) whose elements are THEMSELVES
    3-arrays -- logical shape (50,3,3). The supplied data does not match that, so
    the write was dropped: the dataset ended up with storage_size 0, no allocated
    storage (OFFSET HADDR_UNDEF), and read back as all zeros.

    That made this case pass all four differential arms VACUOUSLY -- native wrote
    nothing, the connector wrote nothing, both read zeros, the digests agreed.
    It verified nothing whatsoever about array datatypes. Found by the
    library-free reader (§1.1(d)), which reported unallocated storage where
    h5dump had said CONTIGUOUS.

    Building the typed array first and assigning into an explicitly shaped
    dataset keeps H5T_ARRAY on disk (verified with h5dump -p) AND actually
    allocates and writes the 1200 bytes.
    """
    import h5py, numpy as np
    dt = np.dtype(("f8", (3,)))
    a = np.zeros((50,), dtype=dt)
    a[...] = np.arange(50 * 3, dtype="f8").reshape(50, 3)
    with h5py.File(path, "w") as f:
        d = f.create_dataset("arr", shape=(50,), dtype=dt)
        d[...] = a

def w_scalar(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("sc", data=np.float64(3.14159265358979))

def w_extendible_append(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        d = f.create_dataset("ext", shape=(4,), maxshape=(None,), chunks=(4,), dtype="i4")
        d[:] = np.arange(4, dtype="i4")
    with h5py.File(path, "r+") as f:            # reopen and grow
        d = f["ext"]
        d.resize((8,))
        d[4:8] = np.arange(4, 8, dtype="i4")

def w_fletcher32(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("fl", data=np.arange(4096, dtype="i4").reshape(64, 64),
                         chunks=(16, 16), fletcher32=True)

def w_point_src(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("p", data=np.arange(1000, dtype="f8"))

# ---------------------------------------------------------------- layouts
# The three layouts below break the assumption every case above quietly makes:
# that a dataset's bytes live in one contiguous raw-data block inside the .h5
# file. Compact puts them in the object header; external puts them in a
# different, non-HDF5 file; virtual puts them in OTHER HDF5 files and assembles
# them on read. That is precisely where a cache keyed on a linear file image is
# most likely to be wrong -- for external and virtual datasets the raw data is
# not in the file being cached at all.
#
# Note for the vfd column specifically: these are also the most distinctive
# byte-pattern generators in the corpus. Compact produces a header write and no
# raw-data block; external produces writes the driver never sees (they go to a
# file HDF5 opens itself); virtual produces reads that fan out to other files.
#
# The helper files are named "<h5 file>.raw" / "<h5 file>.src.h5" deliberately.
# Their names are STORED IN the .h5 file and printed by h5dump -pBH, and the
# native and connector copies necessarily have different names. Deriving them
# from the .h5 path means _normalize_tool_output's existing filename scrub
# collapses both to "<FILE>.raw"/"<FILE>.src.h5", so tool parity compares what
# it should instead of failing on a difference that is purely the harness's own
# file naming.

def w_compact_layout(path):
    """Raw data stored INSIDE the dataset object header (no raw-data block).

    h5py's create_dataset has no layout= knob, so this drops to the low-level
    API. Compact is capped at 64KB of raw data by the format, hence the small
    dataset."""
    import h5py, numpy as np
    data = np.arange(256, dtype="i4")
    with h5py.File(path, "w") as f:
        space = h5py.h5s.create_simple(data.shape)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_layout(h5py.h5d.COMPACT)
        tid = h5py.h5t.py_create(data.dtype, logical=True)
        dsid = h5py.h5d.create(f.id, b"c", tid, space, dcpl)
        dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, data)


def w_external_layout(path):
    """Raw data in a separate, NON-HDF5 file; the .h5 holds only the pointer."""
    import h5py, numpy as np
    raw = path + ".raw"
    if os.path.exists(raw):
        os.remove(raw)          # HDF5 does not truncate an existing external file
    n = 512
    with h5py.File(path, "w") as f:
        d = f.create_dataset("e", shape=(n,), dtype="i4",
                             external=[(raw, 0, n * 4)])
        d[:] = np.arange(n, dtype="i4")


def w_virtual_layout(path):
    """VDS: raw data lives in ANOTHER HDF5 file, mapped in on read."""
    import h5py, numpy as np
    src = path + ".src.h5"
    if os.path.exists(src):
        os.remove(src)
    n = 100
    with h5py.File(src, "w") as f:
        f.create_dataset("s", data=np.arange(n, dtype="i4"))
    layout = h5py.VirtualLayout(shape=(n,), dtype="i4")
    layout[:] = h5py.VirtualSource(src, "s", shape=(n,))
    with h5py.File(path, "w") as f:
        f.create_virtual_dataset("v", layout)


# --------------------------------------------------------------- T5 corpus
# Priorities 2-6 of the corpus widening in Q1_COMPAT_SUITE_PLAN.md §T5. These
# are written to MEASURE, not because the connectors are known to handle them:
# the suite is the deliverable here and support for what it finds is separate
# work. A case that fails is a real, named result, not a broken test.
#
# Two rules these cases follow, both learned the hard way elsewhere in this file:
#
#   1. NEVER hash a raw file offset. Object and region references encode an
#      address in the file, and the same logical file written twice may place an
#      object differently. Hashing the reference bytes would compare storage
#      layout and call a difference an incompatibility. Dereference, then hash
#      what the reference POINTS AT.
#   2. Never iterate. h5py's visititems/keys go through the deprecated
#      H5Ovisit_by_name1/H5Literate_by_name1, which HDF5 restricts to the native
#      connector, so they fail through ANY pass-through VOL. Address objects by
#      name and read structure from property lists instead.

# ---- priority 2: datatypes ----
def w_vlen_sequence(path):
    """H5T_VLEN of int32 -- ragged rows, each row separately heap-allocated."""
    import h5py, numpy as np
    dt = h5py.vlen_dtype(np.dtype("i4"))
    with h5py.File(path, "w") as f:
        d = f.create_dataset("vl", (5,), dtype=dt)
        for i in range(5):
            d[i] = np.arange(i + 1, dtype="i4") * (i + 3)


def read_vlen_sequence(path):
    """Hash element-by-element. The generic digest would hash object-array
    contents via repr(); these are ndarrays, so hash their bytes explicitly."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        d = f["vl"]
        h.update(("%s|%s" % (d.dtype, d.shape)).encode())
        for i in range(d.shape[0]):
            row = d[i]
            h.update(("%d|%s|" % (i, row.shape)).encode())
            h.update(row.tobytes())
    return h.hexdigest()


def w_object_reference(path):
    """H5R object references to two datasets and a group."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("target_a", data=np.arange(16, dtype="i4"))
        f.create_dataset("target_b", data=np.arange(8, dtype="f8") * 1.5)
        f.create_group("target_g")
        r = f.create_dataset("refs", (3,), dtype=h5py.ref_dtype)
        r[0] = f["target_a"].ref
        r[1] = f["target_b"].ref
        r[2] = f["target_g"].ref


def read_object_reference(path):
    """DEREFERENCE, then hash. A reference's raw bytes are a file address and
    may legitimately differ between two files holding identical data."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        r = f["refs"]
        for i in range(r.shape[0]):
            obj = f[r[i]]
            h.update(("R|%d|%s|" % (i, obj.name)).encode())
            if isinstance(obj, h5py.Dataset):
                v = obj[()]
                h.update(("%s|%s|" % (obj.dtype, obj.shape)).encode())
                h.update(v.tobytes())
    return h.hexdigest()


def w_region_reference(path):
    """H5R dataset-region references: a reference that carries a selection."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        d = f.create_dataset("grid", data=np.arange(100, dtype="i4").reshape(10, 10))
        r = f.create_dataset("regions", (2,), dtype=h5py.regionref_dtype)
        r[0] = d.regionref[2:5, 3:7]
        r[1] = d.regionref[0, :]


def read_region_reference(path):
    """Dereference each region and hash the SELECTED data, not the ref bytes."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        d, r = f["grid"], f["regions"]
        for i in range(r.shape[0]):
            sel = d[r[i]]
            h.update(("RR|%d|%s|" % (i, sel.shape)).encode())
            h.update(sel.tobytes())
    return h.hexdigest()


def w_opaque_dtype(path):
    """H5T_OPAQUE -- bytes HDF5 stores without interpreting, carrying a tag."""
    import h5py, numpy as np
    dt = h5py.opaque_dtype(np.dtype("V8"))
    raw = np.array([bytes(bytearray(range(i, i + 8))) for i in range(6)], dtype="V8")
    with h5py.File(path, "w") as f:
        f.create_dataset("op", data=raw.astype(dt))


def w_committed_dtype(path):
    """A NAMED (committed) datatype: a datatype stored as its own object in the
    file and shared by reference, rather than copied into each dataset."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f["shared_i64"] = np.dtype("i8")          # commits the type
        t = f["shared_i64"]
        f.create_dataset("uses_a", (10,), dtype=t)
        f.create_dataset("uses_b", (4,), dtype=t)
        f["uses_a"][...] = np.arange(10, dtype="i8") * 7
        f["uses_b"][...] = np.arange(4, dtype="i8") * 11


def read_committed_dtype(path):
    """Hash the data AND the fact that the type is still committed and shared --
    a connector that silently copied the type would still return right data."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        t = f["shared_i64"]
        h.update(("T|committed=%s|%s" % (isinstance(t, h5py.Datatype), t.dtype)).encode())
        for p in ("uses_a", "uses_b"):
            d = f[p]
            h.update(("D|%s|%s|%s|" % (p, d.dtype, d.shape)).encode())
            h.update(d[()].tobytes())
            # Committed means the dataset's type is the named one, not a copy.
            h.update(("committed=%s|" % d.id.get_type().committed()).encode())
    return h.hexdigest()


def w_compound_nested_array(path):
    """An ARRAY field nested inside a compound -- a two-level type."""
    import h5py, numpy as np
    dt = np.dtype([("id", "i4"), ("vec", "f4", (3,)), ("mat", "i2", (2, 2))])
    a = np.zeros(6, dtype=dt)
    for i in range(6):
        a[i]["id"] = i
        a[i]["vec"] = [i * 1.5, i * 2.5, i * 3.5]
        a[i]["mat"] = [[i, i + 1], [i + 2, i + 3]]
    with h5py.File(path, "w") as f:
        f.create_dataset("nested", data=a)


# ---- priority 3: selections ----
def w_selection_src(path):
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("sel", data=np.arange(1000, dtype="i4").reshape(20, 50))


def read_irregular_hyperslab(path):
    """A UNION of two disjoint hyperslabs -- an irregular selection, which the
    library cannot serve as one contiguous run. Built with the low-level
    H5S_SELECT_OR because h5py's slicing cannot express a union."""
    import h5py, numpy as np
    from h5py import h5s, h5t, h5d
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        dset = f["sel"]
        sp = dset.id.get_space()
        sp.select_hyperslab((2, 3), (4, 5), op=h5s.SELECT_SET)
        sp.select_hyperslab((12, 30), (3, 6), op=h5s.SELECT_OR)
        npoints = sp.get_select_npoints()
        mem = h5s.create_simple((npoints,))
        out = np.zeros(npoints, dtype="i4")
        dset.id.read(mem, sp, out, h5t.NATIVE_INT32)
        h.update(("IRR|%d|" % npoints).encode())
        h.update(out.tobytes())
    return h.hexdigest()


def read_zero_element(path):
    """A selection of zero elements. Must succeed and return nothing -- an easy
    place for an off-by-one or an empty-buffer assumption to hide."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        sub = f["sel"][5:5, 0:10]
        h.update(("ZERO|%s|%s|" % (sub.dtype, sub.shape)).encode())
        h.update(sub.tobytes())
    return h.hexdigest()


def read_single_element(path):
    """One element, the smallest possible transfer."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        v = f["sel"][7, 23]
        h.update(("ONE|%s|%r" % (f["sel"].dtype, v)).encode())
    return h.hexdigest()


# ---- priority 4: dataspaces ----
def w_null_dataspace(path):
    """H5S_NULL: a dataset with a dataspace containing NO elements at all --
    distinct from a zero-length dimension and from a scalar."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("nul", data=h5py.Empty("i4"))
        f.create_dataset("companion", data=np.arange(4, dtype="i4"))


def read_null_dataspace(path):
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        d = f["nul"]
        # shape is None for a null dataspace; the class is the assertion.
        h.update(("NULL|%s|%r|%s" % (d.dtype, d.shape,
                                     d.id.get_space().get_simple_extent_type())).encode())
        c = f["companion"]
        h.update(c[()].tobytes())
    return h.hexdigest()


# ---- priority 5: metadata ----
def w_creation_order(path):
    """Link creation order TRACKED and INDEXED on a group -- group metadata that
    a connector must carry through, and that is stored in the group's creation
    property list rather than in any dataset."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        g = f.create_group("ordered", track_order=True)
        for name in ("zebra", "alpha", "mike", "bravo"):
            g.create_dataset(name, data=np.arange(3, dtype="i4"))


def read_creation_order(path):
    """Read the tracking FLAGS from the group creation property list. Does not
    iterate: h5py iteration is unavailable through a pass-through VOL, so the
    property list is the observable that works for both arms."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        g = f["ordered"]
        flags = g.id.get_create_plist().get_link_creation_order()
        h.update(("CO|%d|" % flags).encode())
        for name in ("zebra", "alpha", "mike", "bravo"):   # by name, not iteration
            h.update(("%s|%s|" % (name, name in g)).encode())
            h.update(g[name][()].tobytes())
    return h.hexdigest()


def w_object_copy(path):
    """H5Ocopy: a whole object duplicated inside the file."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        src = f.create_group("src")
        src.create_dataset("payload", data=np.arange(20, dtype="i4") * 3)
        src["payload"].attrs["note"] = "copied"
        f.copy("src", "dst")


def w_object_delete(path):
    """A deleted link. What must survive is the file being correct AFTERWARDS --
    the deleted name gone, the others intact."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        for name in ("keep_a", "doomed", "keep_b"):
            f.create_dataset(name, data=np.arange(6, dtype="i4"))
        del f["doomed"]


def read_object_delete(path):
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        h.update(("DEL|present=%s|" % ("doomed" in f)).encode())
        for name in ("keep_a", "keep_b"):
            h.update(("%s|" % name).encode())
            h.update(f[name][()].tobytes())
    return h.hexdigest()


def w_external_link(path):
    """An EXTERNAL link: a link whose target lives in a different file."""
    import h5py, numpy as np
    target = path + ".target"
    if os.path.exists(target):
        os.remove(target)
    with h5py.File(target, "w") as t:
        t.create_dataset("far", data=np.arange(12, dtype="i4") * 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("near", data=np.arange(5, dtype="i4"))
        # RELATIVE target, deliberately. An external link stores the target
        # filename verbatim, and each arm writes into its own directory, so an
        # absolute path would embed a DIFFERENT string per arm and h5diff would
        # report a difference that is an artifact of where the test ran rather
        # than anything about the connector. A bare filename resolves against
        # the linking file's own directory, so both arms embed the identical
        # link value and each still finds its own target.
        f["elink"] = h5py.ExternalLink(os.path.basename(target), "/far")


def read_external_link(path):
    """Traverse the link and read through it -- the traversal is the test.

    Reads with the CWD set to the file's own directory, and that is load-bearing
    rather than tidiness. The link stores a RELATIVE target (an absolute one
    would differ per arm and make h5diff report a difference that is an artifact
    of where the test ran). HDF5 resolves a relative external link against
    several candidate prefixes -- an explicit elink prefix, HDF5_EXT_PREFIX, the
    parent file's own directory, and the working directory -- and which one wins
    is exactly the kind of platform detail that made this case pass on Linux and
    fail on macOS. Making the CWD equal to the parent file's directory collapses
    those candidates onto the same location, so the result no longer depends on
    which rule fires first.

    The chdir is a hypothesis about WHY macOS differs, not a confirmed
    cause, so the resolution facts it depends on are reported on stderr. stderr
    is free here: the driver parses only the DIGEST line on stdout, and prints
    what a worker wrote to stderr only when the case fails. Without this, a
    traversal that resolves to the wrong file is indistinguishable in the output
    from one that fails to resolve at all -- both are a bare FAIL.
    """
    import h5py
    prev = os.getcwd()
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    h = hashlib.sha256()
    try:
        os.chdir(target_dir)
        print(f"elink: path={path} abspath={os.path.abspath(path)} "
              f"cwd={os.getcwd()} realcwd={os.path.realpath(os.getcwd())}",
              file=sys.stderr)
        with h5py.File(os.path.abspath(path), "r") as f:
            h.update(f["near"][()].tobytes())
            # VOL-SAFE probes only. `f.get(name, getlink=True)` reads well but
            # goes through H5Lget_info1, which HDF5 restricts to the native
            # connector -- so as a diagnostic it does not report on the failure,
            # it CAUSES one, in every arm running under a connector. Same family
            # as the visititems()/keys()/dereference restriction this suite
            # already keeps in C. `in` (H5Lexists) and .filename have no such
            # limit.
            print(f"elink: link_present={'elink' in f} "
                  f"target_exists={os.path.exists(path + '.target')} "
                  f"file.filename={f.filename}", file=sys.stderr)
            far = f["elink"]
            h.update(("EL|%s|%s|" % (far.dtype, far.shape)).encode())
            h.update(far[()].tobytes())
    finally:
        os.chdir(prev)
    return h.hexdigest()


def w_space_status(path):
    """Two datasets differing ONLY in whether their raw storage is allocated."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("written", data=np.arange(64, dtype="i4"),
                         chunks=(16,))
        f.create_dataset("untouched", shape=(64,), dtype="i4", chunks=(16,))


def read_space_status(path):
    """H5Dget_space_status: is the raw storage allocated? A connector that
    reported everything as allocated would look right on every other case."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        for name in ("written", "untouched"):
            d = f[name]
            h.update(("SS|%s|%s|" % (name, d.id.get_space_status())).encode())
        h.update(f["written"][()].tobytes())
    return h.hexdigest()


# ---- priority 6: filters ----
def w_scaleoffset(path):
    """The scale-offset filter: LOSSY for floats, so the corpus uses the
    integer form, where it is lossless and a digest comparison is meaningful."""
    import h5py, numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("so", data=np.arange(1000, dtype="i4") % 251,
                         chunks=(100,), scaleoffset=0)


def w_nbit(path):
    """The N-Bit filter, which stores only the significant bits of a datatype.
    h5py exposes no keyword for it, so this drops to the low-level DCPL."""
    import h5py, numpy as np
    from h5py import h5p, h5s, h5t, h5d, h5z
    with h5py.File(path, "w") as f:
        dcpl = h5p.create(h5p.DATASET_CREATE)
        dcpl.set_chunk((100,))
        # h5py wraps H5Pset_nbit with no dedicated method; the generic
        # set_filter is the same call underneath.
        dcpl.set_filter(h5z.FILTER_NBIT, h5z.FLAG_MANDATORY)
        # N-Bit only does anything for a type with reduced precision; set 16 of
        # the 32 bits so the filter has something to strip.
        tid = h5t.STD_I32LE.copy()
        tid.set_precision(16)
        tid.set_offset(0)
        sp = h5s.create_simple((1000,))
        dsid = h5d.create(f.id, b"nb", tid, sp, dcpl=dcpl)
        data = (np.arange(1000, dtype="i4") % 30000).astype("i4")
        dsid.write(h5s.ALL, h5s.ALL, data, h5t.NATIVE_INT32)


def read_point(path):
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        sub = f["p"][[0, 10, 33, 100, 777, 999]]   # scattered point/fancy selection
        h.update(("%s|%s" % (sub.dtype, sub.shape)).encode())
        h.update(sub.tobytes())
    return h.hexdigest()

# ---------------------------------------------------------------- readers/digests
def digest_by_spec(path, paths, attrs):
    """sha256 over declared dataset paths (dtype/shape/bytes) + declared attrs.
    Accesses objects BY NAME only — no visititems/H5Ovisit (the VOL does not
    support link iteration; that is tested separately by the 'iteration' case)."""
    import h5py, numpy as np
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        for p in paths:
            o = f[p]
            d = o[()]
            h.update(("D|%s|%s|%s" % (p, o.dtype, o.shape)).encode())
            # Variable-length datatypes (vlen strings) come back as object arrays;
            # .tobytes() would hash the element POINTERS (nondeterministic), not the
            # string content. Hash the decoded content element-by-element instead.
            if getattr(d, "dtype", None) is not None and d.dtype == object:
                for el in np.ravel(np.asarray(d, dtype=object)):
                    h.update(el if isinstance(el, bytes)
                             else el.encode() if isinstance(el, str)
                             else repr(el).encode())
            elif isinstance(d, (bytes, str)):
                h.update(d if isinstance(d, bytes) else d.encode())
            else:
                h.update(d.tobytes() if hasattr(d, "tobytes") else repr(d).encode())
        for (op, an) in attrs:
            obj = f if op == "/" else f[op]
            v = obj.attrs[an]
            val = v.tolist() if hasattr(v, "tolist") else v
            h.update(("A|%s|%s|%r" % (op, an, val)).encode())
    return h.hexdigest()


def read_iteration(path):
    """Structure iteration through whatever VOL is active (visititems). Exercises
    H5Ovisit/H5Literate — a known VOL gap; expected to differ/fail through the VOL."""
    import h5py
    h = hashlib.sha256()
    names = []
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: names.append(n))
    h.update("|".join(sorted(names)).encode())
    return h.hexdigest()

def read_hyperslab(path):
    """Selection read: a strided hyperslab through whatever VOL is active."""
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        sub = f["g"][10:90:3, 5:95:2]        # strided 2D hyperslab
        h.update(("%s|%s" % (sub.dtype, sub.shape)).encode())
        h.update(sub.tobytes())
    return h.hexdigest()

CASES = {
    "int32_1d_contig":   {"write": w_int32_1d_contig, "paths": ["a"]},
    "float64_2d_chunked": {"write": w_float64_2d_chunked, "paths": ["m"]},
    "float32_3d_contig": {"write": w_float32_3d_contig, "paths": ["v"]},
    "compound":          {"write": w_compound, "paths": ["t"]},
    "strings":           {"write": w_strings, "paths": ["fixed", "vlen"]},
    "attrs":             {"write": w_attrs, "paths": ["d"],
                          "attrs": [("d", "units"), ("d", "scale"), ("d", "shape3"),
                                    ("grp", "title"), ("/", "root_note")]},
    "groups_links":      {"write": w_groups_links,
                          "paths": ["outer/inner/leaf", "hardlink", "softlink"]},
    "chunked_shuffle":   {"write": w_chunked_shuffle, "paths": ["s"]},
    "chunked_ragged":    {"write": w_chunked_ragged, "paths": ["r"]},
    "hyperslab_read":    {"write": w_hyperslab_src, "read": read_hyperslab},
    "uint_types":        {"write": w_uint_types, "paths": ["u8", "u16", "u32"]},
    "enum":              {"write": w_enum, "paths": ["e"]},
    "array_dtype":       {"write": w_array_dtype, "paths": ["arr"]},
    "scalar":            {"write": w_scalar, "paths": ["sc"]},
    "extendible_append": {"write": w_extendible_append, "paths": ["ext"]},
    "fletcher32":        {"write": w_fletcher32, "paths": ["fl"]},
    "point_selection":   {"write": w_point_src, "read": read_point},
    # Layouts whose raw data is not a contiguous block in the .h5 file.
    "compact_layout":    {"write": w_compact_layout, "paths": ["c"]},
    "external_layout":   {"write": w_external_layout, "paths": ["e"]},
    # no_storage: a VDS has no raw storage of its own by definition -- the data
    # lives in its source files. Exempts it from the fixture storage check only;
    # it must still read back real (non-zero) content, which is what proves the
    # mapping to the source resolved.
    "virtual_layout":    {"write": w_virtual_layout, "paths": ["v"],
                          "no_storage": True},

    # ---- T5 priority 2: datatypes ----
    "vlen_sequence":     {"write": w_vlen_sequence, "read": read_vlen_sequence},
    # VFD-ONLY, and for the same reason the iteration case is disabled below.
    # h5py dereferences with H5Rdereference2, which HDF5 hard-restricts to the
    # native connector -- through the clio VOL it raises outright:
    #   "H5Rdereference2 is only meant to be used with the native VOL connector"
    # and h5py's reference WRITE path silently stores a null reference. Measured,
    # not assumed: the VOL-written file reads back <HDF5 object reference (null)>
    # while the native-written one resolves to /target_a, /target_b, /target_g.
    # That is an h5py/HDF5 API-version restriction, NOT a clio defect -- the
    # reference pass-through connector fails identically -- so allowlisting it
    # as a VOL gap would record a false fact about this connector.
    # A VFD imposes no API restriction, so the VFD arm runs them normally.
    # Covering references through the VOL needs a C probe on the modern
    # H5Rcreate_object/H5Ropen_object API (1.12+), like vol_c_iteration_test.c
    # does for iteration. Do not "fix" this by re-enabling the h5py arm.
    "object_reference":  {"write": w_object_reference, "read": read_object_reference,
                          "modes": ["vfd"],
                          "skip_reason": "h5py uses H5Rdereference2, which HDF5 "
                                         "restricts to the native VOL connector"},
    "region_reference":  {"write": w_region_reference, "read": read_region_reference,
                          "modes": ["vfd"],
                          "skip_reason": "h5py uses H5Rdereference2, which HDF5 "
                                         "restricts to the native VOL connector"},
    "opaque_dtype":      {"write": w_opaque_dtype, "paths": ["op"]},
    "committed_dtype":   {"write": w_committed_dtype, "read": read_committed_dtype},
    "compound_nested":   {"write": w_compound_nested_array, "paths": ["nested"]},

    # ---- T5 priority 3: selections ----
    # One writer, three readers: the selection is what varies, so re-writing the
    # same array three times would measure nothing extra.
    "irregular_hyperslab": {"write": w_selection_src, "read": read_irregular_hyperslab},
    "zero_element_sel":  {"write": w_selection_src, "read": read_zero_element},
    "single_element_sel": {"write": w_selection_src, "read": read_single_element},

    # ---- T5 priority 4: dataspaces (scalar and unlimited are covered above) ----
    # The digest comes from `read` (the null dataspace has no elements to hash,
    # so the assertion is its extent CLASS). `paths` names the companion instead:
    # a null dataset would trip the fixture storage check for a legitimate
    # reason, and pointing the check at the companion keeps the file covered
    # without exempting it.
    "null_dataspace":    {"write": w_null_dataspace, "read": read_null_dataspace,
                          "paths": ["companion"]},

    # ---- T5 priority 5: metadata ----
    "creation_order":    {"write": w_creation_order, "read": read_creation_order},
    "object_copy":       {"write": w_object_copy,
                          "paths": ["src/payload", "dst/payload"],
                          "attrs": [("src/payload", "note"),
                                    ("dst/payload", "note")]},
    "object_delete":     {"write": w_object_delete, "read": read_object_delete},
    "external_link":     {"write": w_external_link, "read": read_external_link},
    "space_status":      {"write": w_space_status, "read": read_space_status},

    # ---- T5 priority 6: filters ----
    "scaleoffset":       {"write": w_scaleoffset, "paths": ["so"]},
    "nbit":              {"write": w_nbit, "paths": ["nb"]},
    # "iteration" is DISABLED here on purpose. h5py has poor VOL support: its
    # visititems()/keys() call the DEPRECATED H5Ovisit_by_name1 / H5Literate_by_name1,
    # which HDF5 hard-restricts to the native VOL connector, so they fail through ANY
    # non-native VOL (the reference H5VLpassthru included) — before our callbacks are
    # even reached. The clio VOL's iteration is actually correct: a C client using
    # the modern H5Ovisit3 / H5Literate2 traverses fine. That is verified by the
    # isolated C test `vol_c_iteration_test.c` (run via _run_c_tests below), which is
    # the accurate way to test VOL iteration. Do not re-add an h5py iteration case.
    # "iteration":       {"write": w_groups_links, "read": read_iteration},
}

# szip is registered only where the ENCODER is actually built -- it is patent-
# encumbered and many HDF5 builds ship decode-only or omit it. Registering it
# unconditionally would fail on those builds for a reason that has nothing to do
# with the connector. Conditional at import so the parent and the worker
# subprocess agree on the corpus; SZIP_STATUS is printed by the driver, because
# a case that quietly is not there is the failure mode this suite exists to
# avoid.
SZIP_STATUS = "not probed (h5py unavailable)"
try:                                            # pragma: no cover - env dependent
    import h5py as _h5py_probe

    if _h5py_probe.h5z.filter_avail(_h5py_probe.h5z.FILTER_SZIP):
        def w_szip(path):
            """The szip filter, where the build provides an encoder."""
            import h5py, numpy as np
            with h5py.File(path, "w") as f:
                f.create_dataset("sz", data=np.arange(1000, dtype="i4") % 977,
                                 chunks=(100,), compression="szip")

        CASES["szip"] = {"write": w_szip, "paths": ["sz"]}
        SZIP_STATUS = "registered (encoder available)"
    else:
        SZIP_STATUS = "OMITTED: no szip filter in this HDF5 build"
    del _h5py_probe
except Exception as _e:                         # pragma: no cover - env dependent
    SZIP_STATUS = f"OMITTED: szip probe failed ({_e})"

# ---------------------------------------------------------------- worker
def worker(case, action, path):
    spec = CASES[case]
    if action == "write":
        spec["write"](path)
        print("WROTE")
    else:  # read
        rd = spec.get("read")
        if rd is None:
            digest = digest_by_spec(path, spec["paths"], spec.get("attrs", []))
        else:
            digest = rd(path)
        print("DIGEST:" + digest)

# ---------------------------------------------------------------- driver
def _env(mode):
    """Environment for one arm. `mode` is "native", "vol" or "vfd".

    Both connectors are chosen by an env var that HDF5 reads once at library
    init, so a mode cannot be switched inside a live process — every arm is a
    fresh subprocess. "native" must actively CLEAR both vars rather than merely
    not set them: this driver may itself have been launched with one of them set,
    and a leaked var would silently make the oracle not-native, which would make
    every comparison vacuous.
    """
    if mode not in ("native",) + CONNECTORS:
        raise ValueError("unknown mode: %r" % (mode,))
    e = dict(os.environ, HOME="/home/iowarp",
             LD_LIBRARY_PATH=BIN + ":/usr/local/lib:/usr/lib/x86_64-linux-gnu",
             PYTHONPATH=BIN)
    e.pop("HDF5_VOL_CONNECTOR", None)
    e.pop("HDF5_DRIVER", None)
    if mode == "vol":
        e["HDF5_PLUGIN_PATH"] = BIN
        e["HDF5_VOL_CONNECTOR"] = "clio"
    elif mode == "vfd":
        e["HDF5_PLUGIN_PATH"] = BIN
        e["HDF5_DRIVER"] = "clio_vfd"
    return e

# Worker stderr from the arms of the case currently being run, keyed by
# "<action>/<mode> on <arm-dir>". A worker that RUNS but reads different bytes
# leaves no trace at all in the columns -- the arm is just a FAIL -- so anything
# it wrote to stderr is kept here and printed by run_connector if that case ends
# up failing. Cleared per case; only the failing case pays the output.
#
# The arm DIRECTORY is part of the key on purpose: two of the four arms are
# read/native, differing only in which file they read (the native oracle vs the
# connector's output), and keying on action/mode alone would silently drop one.
ARM_STDERR = {}


def _run(case, action, path, mode):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--case", case, "--action", action, "--file", path]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_env(mode), timeout=180)
    if p.stderr.strip():
        arm_dir = os.path.basename(os.path.dirname(os.path.abspath(path)))
        ARM_STDERR[f"{action}/{mode} on {arm_dir}"] = p.stderr.strip()
    for line in p.stdout.splitlines():
        if line.startswith("DIGEST:"):
            return line[7:], p.returncode
        if line == "WROTE":
            return "WROTE", p.returncode
    # No output means the worker died, and the column it feeds then prints a
    # bare FAIL with no reason -- which is how `external_link` reached CI as an
    # unexplained platform difference. The traceback and the HDF5 error stack
    # are the whole diagnosis; print them here rather than discarding them and
    # inferring the cause from a boolean two screens later.
    print(f"  {case:<22} ARM-CRASHED  {action}/{mode} rc={p.returncode}; "
          f"worker output follows")
    for stream, text in (("err", p.stderr), ("out", p.stdout)):
        for line in text.strip().splitlines():
            print(f"      {stream}| {line}")
    return None, p.returncode  # crash / no output

def _h5diff(a, b):
    p = subprocess.run(["h5diff", a, b], capture_output=True, text=True,
                       env=_env("native"))
    return p.returncode == 0

def _tool_ok(path):
    for tool in (["h5dump", "-H"], ["h5ls", "-r"], ["h5stat"]):
        p = subprocess.run(tool + [path], capture_output=True, text=True,
                           env=_env("native"))
        if p.returncode != 0:
            return False
    return True


# ------------------------------------------------- §1.1(b) structural validity
def _struct_valid(path):
    """§1.1(b): is the produced file structurally conformant, CLIO not loaded?

    MECHANISM, stated plainly because it is weaker than the criterion's first
    choice: `h5check` is NOT present in the deps-cpu image (verified 2026-07-29),
    so this is the criterion's own documented substitute — `h5clear -s` followed
    by a clean reopen. What that actually proves is narrower than h5check:
    h5check validates the file against the format specification object by object,
    whereas this only shows the superblock is well-formed enough to rewrite its
    status_flags and that the metadata graph is still fully traversable
    afterwards. It would not catch a spec violation that HDF5's own reader is
    tolerant of. Do not report this as "h5check clean"; the results table names
    it `h5clear+reopen` for that reason.

    Runs on a COPY. h5clear -s writes to the file, and the h5diff/tool-parity
    gates must see the connector's output exactly as it was produced.
    """
    if not os.path.exists(path):
        return False
    probe = path + ".structcheck.h5"
    try:
        shutil.copy2(path, probe)
        clear = subprocess.run(["h5clear", "-s", probe], capture_output=True,
                               text=True, env=_env("native"))
        if clear.returncode != 0:
            return False
        # Clean reopen. h5stat is used rather than `h5dump -H` because it walks
        # the whole object header / B-tree / heap structure and so exercises far
        # more of the file than a header dump does.
        reopen = subprocess.run(["h5stat", probe], capture_output=True,
                                text=True, env=_env("native"))
        return reopen.returncode == 0
    finally:
        if os.path.exists(probe):
            os.remove(probe)


# ------------------------------------------------------- §1.1(c) tool parity
# The criterion is OUTPUT equality, not exit status: "h5dump -pBH, h5ls -vlr and
# h5stat produce output identical to the native-produced file modulo whitespace,
# timestamps, and file-size fields."
_PARITY_TOOLS = (["h5dump", "-pBH"], ["h5ls", "-vlr"], ["h5stat"])


def _normalize_tool_output(text, path):
    """Erase the differences that are legitimately allowed to differ.

    Measured 2026-07-29 across the whole corpus at HDF5 2.1.1: the ONLY
    difference between the native-produced and connector-produced files' tool
    output is the file NAME (h5dump's `HDF5 "<name>" {` banner, h5ls's `Opened
    "<name>" with sec2 driver.`, h5stat's `Filename:` line). No timestamp and no
    file-size field differed anywhere.

    Only the DIRECTORY component actually differs, because the two arms share a
    basename by construction (see ARM_DIR). That matters for the layouts that
    embed a filename in the file: it keeps the embedded names the same length,
    so no size field diverges and there is nothing here that needs scrubbing.

    So only the name and trailing whitespace are normalized here. The criterion
    also permits scrubbing timestamps and file-size fields, and deliberately none
    is scrubbed: a scrubber for a field that does not currently differ is
    untested code that can only ever hide a future regression. If such a field
    starts differing, this gate goes RED and that is the correct outcome — the
    difference gets looked at and then either fixed or explicitly excused here.
    """
    base = os.path.basename(path)
    stem = base[:-3] if base.endswith(".h5") else base
    # Longest first: after `path` and `base` are gone, `stem` catches bare stems.
    for needle in (path, base, stem):
        text = text.replace(needle, "<FILE>")
    return "\n".join(l.rstrip() for l in text.splitlines() if l.strip())


def _tool_parity(native_path, conn_path):
    """Return (ok, first_mismatching_tool_or_None)."""
    for tool in _PARITY_TOOLS:
        outs = []
        for p in (native_path, conn_path):
            r = subprocess.run(tool + [p], capture_output=True, text=True,
                               env=_env("native"))
            if r.returncode != 0:
                return False, tool[0] + "(rc=%d)" % r.returncode
            outs.append(_normalize_tool_output(r.stdout, p))
        if outs[0] != outs[1]:
            return False, tool[0]
    return True, None

def _wipe_clio_shm():
    """Remove leftover clio/chimaera POSIX shm segments from a prior run.
    /dev/shm is Linux-only; on macOS (no /dev/shm) there is nothing to sweep
    here and clio_run's own shm_unlink handles cleanup."""
    try:
        names = os.listdir("/dev/shm")
    except OSError:
        return
    for f in names:
        if f.startswith("chimaera") or f.startswith("clio"):
            try:
                os.remove(os.path.join("/dev/shm", f))
            except OSError:
                pass


def restart_runtime():
    """Kill any clio_run, wipe shm, start fresh from BIN, wait until ready."""
    # -x (exact process NAME), not -f (full command line). -f matches any
    # process whose cmdline merely CONTAINS "clio_run" -- which includes the
    # shell, CI wrapper, or editor that happens to have the word in its command,
    # and, most reliably, the harness that invoked this suite. It killed three
    # consecutive admission runs from a wrapper whose only sin was containing
    # the string in a cleanup command, each time as a bare SIGTERM with an empty
    # log, which looks like an infrastructure failure rather than a self-inflicted
    # one. The runtime's process name is exactly "clio_run", so -x is both
    # narrower and correct.
    subprocess.run(["pkill", "-x", "clio_run"], check=False)
    time.sleep(2)
    _wipe_clio_shm()
    # Provide clio_run a self-contained compose config (see SUITE_CONF_YAML) so
    # readiness does not depend on a ~/.clio/clio.yaml that CI lacks.
    os.makedirs(TMP, exist_ok=True)
    with open(SUITE_CONF, "w") as cf:
        cf.write(SUITE_CONF_YAML)
    env = dict(os.environ, HOME=CLIO_HOME, CLIO_SERVER_CONF=SUITE_CONF)
    with open(RUNTIME_LOG, "w") as log:
        proc = subprocess.Popen([os.path.join(BIN, "clio_run"), "start"],
                                stdout=log, stderr=log,
                                cwd=os.path.dirname(BIN) or "/", env=env)
    for _ in range(60):
        try:
            with open(RUNTIME_LOG) as fh:
                if RUNTIME_READY in fh.read():
                    time.sleep(1)
                    return True
        except FileNotFoundError:
            pass
        time.sleep(1)
    # Never became ready. This has been opaque in CI (the runtime log is not
    # surfaced), so dump why before the caller's assert fails.
    _dump_restart_diagnostics(proc, env)
    return False


def _dump_restart_diagnostics(proc, env):
    """Print why clio_run failed to become ready — visible in CI output."""
    print("=" * 72)
    print("restart_runtime: clio_run did NOT become ready — diagnostics")
    print("=" * 72)
    rc = proc.poll()
    print(f"clio_run pid={proc.pid} alive={rc is None} returncode={rc}")
    home = env.get("HOME", "")
    cfg = os.path.join(home, ".clio", "clio.yaml")
    print(f"HOME(clio_run)={home}  BIN={BIN}")
    print(f"config {cfg} exists={os.path.exists(cfg)}")
    for cmd in (["df", "-h", "/dev/shm"], ["free", "-m"],
                ["ls", "-la", os.path.join(BIN, "clio_run")]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            print(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}", end="")
        except Exception as e:  # noqa: BLE001 - diagnostics must never raise
            print(f"$ {' '.join(cmd)} -> {e}")
    print(f"--- {RUNTIME_LOG} (last 80 lines) ---")
    try:
        with open(RUNTIME_LOG) as fh:
            lines = fh.read().splitlines()
        print("\n".join(lines[-80:]) if lines else "(runtime log is empty)")
    except OSError as e:
        print(f"(could not read runtime log: {e})")
    print("=" * 72)
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def _hdf5_prefix():
    """Return the install prefix of the HDF5 that this suite links against, or
    None. The suite is launched by ctest with an explicit interpreter path
    (e.g. <conda>/bin/python3), so the HDF5 headers/libs it exercises live under
    that interpreter's prefix (<conda>/include, <conda>/lib). Deriving the prefix
    from sys.executable keeps the on-the-fly C compiler pointed at the same HDF5
    as the h5py arms, instead of a hardcoded /usr/local that may not have it."""
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    if os.path.isfile(os.path.join(prefix, "include", "hdf5.h")):
        return prefix
    return None


def _find_h5cc():
    """Locate the h5cc wrapper, preferring one next to the running interpreter
    (the HDF5 this suite actually uses) over anything on PATH."""
    prefix = _hdf5_prefix()
    if prefix:
        cand = os.path.join(prefix, "bin", "h5cc")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which("h5cc")


def _compile_c(binp, srcp):
    """Compile a single C test, preferring the HDF5 wrapper h5cc and falling back
    to gcc + -lhdf5. Returns the CompletedProcess of whichever compiler ran, or
    None if neither h5cc nor gcc is installed. Selecting the compiler with
    shutil.which first is deliberate: invoking a non-existent h5cc directly raises
    FileNotFoundError, which previously aborted the whole suite instead of letting
    the gcc fallback run."""
    comp = None
    h5cc = _find_h5cc()
    if h5cc:
        comp = subprocess.run([h5cc, "-o", binp, srcp], capture_output=True,
                              text=True, env=_env("native"))
    if (comp is None or comp.returncode != 0) and shutil.which("gcc"):
        # Point gcc at the HDF5 under the running interpreter's prefix (the same
        # one the h5py arms use); fall back to /usr/local for standalone runs.
        prefix = _hdf5_prefix()
        inc = os.path.join(prefix, "include") if prefix else "/usr/local/include"
        lib = os.path.join(prefix, "lib") if prefix else "/usr/local/lib"
        comp = subprocess.run(["gcc", "-o", binp, srcp, "-I" + inc,
                               "-L" + lib, "-Wl,-rpath," + lib, "-lhdf5"],
                              capture_output=True, text=True, env=_env("native"))
    return comp


# Probes that did not COMPILE, kept apart from probes that compiled and failed.
# A build error means the connector's behaviour was never measured, which is a
# different fact with a different owner and a different fix than "the connector
# is incompatible". Both are fatal; only one of them is a statement about the
# VOL. Populated by _build_probe, reported separately by driver().
BUILD_ERRORS = {}

# Every C probe the suite compiles, name -> source. Built up front by
# _build_probe_set so a build error is reported before any case runs, rather
# than surfacing mid-run as a single mysterious case failure.
C_PROBES = {
    "c_iteration":         "vol_c_iteration_test.c",
    "c_safeflush":         "vol_c_safeflush_test.c",
    "c_selection":         "vol_c_selection_test.c",
    "c_cache_identity":    "vol_c_cache_identity_test.c",
    "c_error_propagation": "vol_c_error_propagation_test.c",
    "c_passthrough_ops":   "vol_c_passthrough_ops_test.c",
    "c_isaccessible":      "vol_c_isaccessible_test.c",
    # Shared by the error-parity axis (both connectors) and the
    # unselected-connector inertness pin, which run the same binary.
    "c_error_parity":      "hdf5_c_error_parity_test.c",
}

# The subset _run_c_tests EXECUTES as standalone pass/fail VOL cases. The rest
# of C_PROBES is built here but driven by its own section, with its own argv and
# its own verdict parsing (c_error_parity), so it must not be run as one of
# these. Build set and run set are deliberately not the same list.
VOL_C_TESTS = ("c_iteration", "c_safeflush", "c_selection",
               "c_cache_identity", "c_error_propagation", "c_passthrough_ops",
               # From dev (99183b33), carried across the rename of this file.
               # It has to be in BOTH lists: dev's version ran every C_PROBES
               # entry, and this branch split build-set from run-set, so adding
               # it only to C_PROBES would have compiled it and never run it --
               # a silently dropped test rather than a visible conflict.
               "c_isaccessible")


def _record_build_error(name, comp):
    """Record a failed compile in BUILD_ERRORS and print the compiler's FULL
    diagnostics.

    Full, not truncated: a tail slice is the wrong end of a compiler error.
    Compilers put the message first and the caret art last, so keeping the last
    N characters reliably discards the sentence naming the problem and keeps
    `~~~~~~~` -- which is exactly how an HDF5 API-arity mismatch reached CI
    looking like an unexplained incompatibility."""
    err = comp.stderr.strip() or comp.stdout.strip() or "(no compiler output)"
    # One-line summary for the end-of-run block: the first line that actually
    # says "error", not merely the first line -- compilers open with context
    # ("In function 'main':") and with warnings that precede the real cause.
    lines = err.splitlines()
    BUILD_ERRORS[name] = next((l.strip() for l in lines if "error:" in l),
                              lines[0].strip() if lines else "compile failed")
    print(f"  {name:<20} BUILD-ERROR  did not compile; full output follows")
    for line in lines:
        print(f"      | {line}")


def _build_probe(name, binp, srcp):
    """Compile one C probe. True on success; on failure records the reason in
    BUILD_ERRORS and prints the compiler's full diagnostics."""
    comp = _compile_c(binp, srcp)
    if comp is not None and comp.returncode == 0:
        return True
    if comp is None:
        BUILD_ERRORS[name] = "no C compiler (h5cc/gcc) found"
        print(f"  {name:<20} BUILD-ERROR  no C compiler (h5cc/gcc) found")
        return False
    _record_build_error(name, comp)
    return False


def _build_probe_set():
    """Build every C probe before any case runs. Fails loudly and completely:
    all build errors are reported in one pass, not one per run."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(TMP, exist_ok=True)
    for name, src in C_PROBES.items():
        _build_probe(name, os.path.join(TMP, name),
                     os.path.join(src_dir, src))
    if BUILD_ERRORS:
        print(f"  {len(BUILD_ERRORS)} of {len(C_PROBES)} C probes did not build; "
              "their coverage is UNMEASURED (not 'passing', not 'incompatible')")
    else:
        print(f"  all {len(C_PROBES)} C probes built")


def _run_c_tests():
    """Compile + run the isolated C tests through the VOL. Returns
    {name: {"pass": bool}}; each C program exits 0 on pass. These cover ops h5py
    cannot exercise well via a non-native VOL: modern-API iteration
    (c_iteration), Safe-mode H5Fflush durability (c_safeflush),
    selection-aware read caching + partial-write invalidation (c_selection), and
    the cache-identity regressions (c_cache_identity): mem/file datatype
    mismatch, a cache surviving H5F_ACC_TRUNC of its file, and H5Dflush as a
    barrier. Each of those three returned wrong data with a success status.
    c_isaccessible covers the filename-scoped file_specific ops HDF5 invokes
    with a NULL object; h5py never calls H5Fis_accessible, so no h5py case
    reaches that callback."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for name in VOL_C_TESTS:
        binp = os.path.join(TMP, name)
        srcp = os.path.join(src_dir, C_PROBES[name])
        # Already built (and already reported) by _build_probe_set; retry here
        # only for a standalone caller that skipped that step.
        if name in BUILD_ERRORS:
            print(f"  {name:<20} BUILD-ERROR  not run: {BUILD_ERRORS[name]}")
            out[name] = {"pass": False}
            continue
        if not os.path.exists(binp) and not _build_probe(name, binp, srcp):
            out[name] = {"pass": False}
            continue
        r = subprocess.run([binp], capture_output=True, text=True,
                           env=_env("vol"), timeout=120)
        ok = (r.returncode == 0)
        out[name] = {"pass": ok}
        # The binary's own verdict line ends in PASS/FAIL; isolate it from any
        # interleaved clio runtime INFO logging on stdout (e.g. PoolId(major:...)).
        verdict = [l for l in r.stdout.strip().splitlines()
                   if l.rstrip().endswith(("PASS", "FAIL"))]
        detail = verdict[-1] if verdict else r.stdout.strip()[-70:]
        print(f"  {name:<20} {'PASS' if ok else 'FAIL'}  ({detail})")
    return out


def _run_cache_reuse_check():
    """Does the cache still work on the SECOND file a process touches?

    This exists because a defect that disabled caching for every file after the
    first passed every other check in this suite. Nothing here asserted that the
    tier is USED -- only that whatever it returns is correct -- so a connector
    that silently fell back to native reads was indistinguishable from a working
    one. Correct, fast, and correct-but-never-cached all look the same to a
    differential test.

    Method: create N distinct files in ONE process, each written, closed, aged
    briefly, then read TWICE. The second read of each file must be a cache hit.
    The failing mode this pins is "hits == 1 regardless of N".

    WHY THE PAUSE AND THE SECOND READ, since neither is what the check is about.
    The connector will not record a coherence stamp for a file whose mtime is
    younger than the filesystem's timestamp granularity, because within that
    window mtime cannot rule out a later same-granule modification (see
    clio_stamp_ambiguous). A file written and immediately closed is always
    inside that window, so the closing session records no stamp and the next
    open has nothing to validate against -- it MUST miss, by design.

    That makes "write, close, reopen, read" the one access shape guaranteed
    never to hit, which is what this check used to do. It was passing on
    timing: the workload happened to be slow enough that the close landed
    outside the window. The pause removes that dependency, and the second read
    is what actually exercises the tier -- the first read's close is the one
    that gets to record a stamp, because by then the file has aged.

    This is not the timing dependency being papered over. The behaviour it
    would otherwise measure -- whether an instant reopen skips the cache -- is
    asserted deliberately in _run_instant_reopen_check() below, so the trade is
    pinned somewhere rather than silently baked into this check's timing.
    """
    n_files = 4
    tdir = os.path.join(TMP, "reuse_trace")
    shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(tdir, exist_ok=True)

    prog = (
        "import h5py, numpy as np, sys, time\n"
        "for i in range(%d):\n"
        "    p = '%s/reuse_%%d.h5' %% i\n"
        "    a = np.arange(4096, dtype='i4')\n"
        "    with h5py.File(p, 'w') as f: f.create_dataset('d', data=a)\n"
        "    time.sleep(0.05)   # age past the stamp window; see the docstring\n"
        "    for _ in range(2):\n"
        "        with h5py.File(p, 'r') as f:\n"
        "            assert (f['d'][()] == a).all(), 'data differs on reread'\n"
    ) % (n_files, TMP)

    env = dict(_env("vol"), CLIO_VOL_TRACE=tdir)
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, env=env, timeout=240)
    if r.returncode != 0:
        print(f"  {'cache_reuse':<22} FAIL  workload failed: "
              f"{r.stderr.strip()[:160]}")
        return {"vol/cache_reuse": {"workload_ok": False}}

    hits = 0
    for fn in glob.glob(os.path.join(tdir, "*.access.json")):
        try:
            with open(fn) as fh:
                doc = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        for _name, ds in doc.get("datasets", {}).items():
            hits += ds.get("read_served", {}).get("cache", 0)

    # Every file after the first must also hit. Asserting ">1" rather than
    # "== n_files" keeps this from being brittle about a single cold start,
    # while still failing hard on the "only ever one" signature.
    ok = hits > 1
    print(f"  {'cache_reuse':<22} {'PASS' if ok else 'FAIL'}  "
          f"({hits}/{n_files} reads served from the tier across distinct files"
          + ("" if ok else " -- caching stops after the first file") + ")")
    return {"vol/cache_reuse": {"reused_across_files": ok}}


def _run_instant_reopen_check():
    """A file reopened the instant it is closed must NOT be served from cache.

    This is the deliberate half of the coherence stamp's fail-closed rule, and
    it needs its own check because it is a behaviour nobody would infer from a
    correctness test: the data is right either way. Without this the rule lives
    only in a comment and in the timing of _run_cache_reuse_check(), and the
    first person to "optimise away" the withheld stamp would find every test
    still green while quietly restoring the bug it exists to prevent -- a
    corrupt file read as fine, whenever the corruption lands in the same
    timestamp granule as the write it followed.

    What it asserts, in order of what actually matters:
      1. the data read back is correct (fail-closed must never mean wrong);
      2. the reopen served ZERO reads from the tier;
      3. the telemetry NAMES the reason -- the closing session reports
         `ambiguous`, the reopening session reports `absent` -- so a miss here
         is attributable rather than mysterious.

    (3) is the part that makes a future performance measurement legible: these
    are self-inflicted misses, and a benchmark that cannot see them will blame
    the workload.
    """
    tdir = os.path.join(TMP, "instant_trace")
    shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(tdir, exist_ok=True)
    p = os.path.join(TMP, "instant_reopen.h5")
    if os.path.exists(p):
        os.remove(p)

    prog = (
        "import h5py, numpy as np\n"
        "a = np.arange(4096, dtype='i4')\n"
        "with h5py.File('%s', 'w') as f: f.create_dataset('d', data=a)\n"
        "with h5py.File('%s', 'r') as f:\n"
        "    assert (f['d'][()] == a).all(), 'data differs on instant reread'\n"
    ) % (p, p)

    env = dict(_env("vol"), CLIO_VOL_TRACE=tdir)
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, env=env, timeout=240)
    checks = {"workload_ok": r.returncode == 0}
    if r.returncode != 0:
        print(f"  {'instant_reopen':<22} FAIL  workload failed: "
              f"{r.stderr.strip()[:160]}")
        return {"vol/instant_reopen": checks}

    cache_reads, ambiguous, absent = 0, 0, 0
    for fn in sorted(glob.glob(os.path.join(tdir, "*.access.json"))):
        try:
            with open(fn) as fh:
                doc = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        coh = doc.get("coherence", {})
        ambiguous += coh.get("ambiguous", 0)
        absent += coh.get("absent", 0)
        for _name, ds in doc.get("datasets", {}).items():
            cache_reads += ds.get("read_served", {}).get("cache", 0)

    checks["not_served_from_tier"] = cache_reads == 0
    checks["reason_recorded"] = ambiguous >= 1 and absent >= 1
    ok = all(checks.values())
    print(f"  {'instant_reopen':<22} {'PASS' if ok else 'FAIL'}  "
          f"(tier served {cache_reads} reads; stamp withheld x{ambiguous}, "
          f"absent-at-open x{absent})")
    return {"vol/instant_reopen": checks}


def _run_bbox_fetch_check():
    """§4(A): does serving a selection fetch only the chunks its bounding box
    touches, or the whole cached image?

    Needs a small cache chunk to be observable at all. The default is 1 MiB and
    the c_selection corpus is 192 bytes, so the entire image is ONE chunk and a
    bounding box cannot narrow anything -- the check would pass on broken code.
    CLIO_VOL_CHUNK_SIZE=32 makes the image span six chunks, which is what gives
    the assertion teeth.

    The measure is `bytes_fetched_from_tier` (bytes pulled OUT of the tier), not
    read_bytes_from_cache (bytes handed to the application). Before §4(A) every
    served selection fetched the whole image, so fetched == served_reads x
    image_bytes exactly; narrowing makes it strictly less. That equality is what
    this fails on.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    binp = os.path.join(TMP, "c_selection")
    srcp = os.path.join(src_dir, "vol_c_selection_test.c")
    if not os.path.exists(binp) and not _build_probe("bbox_fetch", binp, srcp):
        return {"bbox_fetch": {"pass": False}}
    tdir = os.path.join(TMP, "trace_bbox")
    os.makedirs(tdir, exist_ok=True)
    for f in glob.glob(tdir + "/*"):
        os.remove(f)
    env = dict(_env("vol"), CLIO_VOL_TRACE=tdir, CLIO_VOL_CHUNK_SIZE="32")
    r = subprocess.run([binp], capture_output=True, text=True, env=env, timeout=120)
    checks = {"workload_ok": r.returncode == 0}
    image_bytes = 8 * 6 * 4          # R x C x sizeof(int), see vol_c_selection_test.c
    narrowed = False
    summaries = glob.glob(tdir + "/*.access.json")
    detail = ""
    if summaries:
        try:
            d = json.load(open(summaries[0]))["datasets"]["m"]
            served = d["read_served"]["cache"]
            fetched = d["bytes_fetched_from_tier"]
            whole_image_cost = served * image_bytes
            narrowed = 0 < fetched < whole_image_cost
            detail = (f"fetched {fetched}B for {served} cache-served reads; "
                      f"whole-image would be {whole_image_cost}B")
        except Exception as e:
            detail = f"summary unreadable: {e}"
    checks["fetch_narrowed"] = narrowed
    ok = all(checks.values())
    print(f"  {'bbox_fetch':<20} {'PASS' if ok else 'FAIL'}  ({detail})")
    return {"bbox_fetch": checks}


def _run_trace_check():
    """Verify access telemetry (Part B). Runs the c_selection workload with
    CLIO_VOL_TRACE set and asserts the summary JSON + per-access JSONL are
    produced with sane, self-consistent fields — including read cache-hit rate in
    [0,1] and repeated-selection detection (the workload reads one hyperslab
    twice). Observe-only: does not change data-path behavior."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    binp = os.path.join(TMP, "c_selection")  # reuse the c_selection binary
    srcp = os.path.join(src_dir, "vol_c_selection_test.c")
    if not os.path.exists(binp) and not _build_probe("telemetry", binp, srcp):
        return {"telemetry": {"pass": False}}
    tdir = os.path.join(TMP, "trace")
    os.makedirs(tdir, exist_ok=True)
    for f in glob.glob(tdir + "/*"):
        os.remove(f)
    env = dict(_env("vol"), CLIO_VOL_TRACE=tdir)
    r = subprocess.run([binp], capture_output=True, text=True, env=env, timeout=120)
    checks = {"workload_ok": r.returncode == 0}
    summaries = glob.glob(tdir + "/*.access.json")
    jsonls = glob.glob(tdir + "/*.access.jsonl")
    checks["summary_written"] = len(summaries) == 1
    checks["jsonl_written"] = len(jsonls) == 1 and os.path.getsize(jsonls[0]) > 0 if jsonls else False
    fields_ok = repeat_ok = False
    if summaries:
        try:
            s = json.load(open(summaries[0]))
            d = s["datasets"]["m"]
            hr = d["cache_hit_rate"]
            lay = d["layout"]
            fields_ok = (d["reads"] > 0 and d["writes"] > 0 and 0.0 <= hr <= 1.0
                         and d["read_served"]["cache"] > 0 and d["ndims"] == 2
                         and d["dtype"] == "integer"
                         # chunk-alignment probe: dataset is chunked {4,3}; case D
                         # is aligned, cases A/B are misaligned.
                         and lay["chunked"] is True and lay["chunk_dims"] == "[4,3]"
                         and lay["read_aligned"] >= 1 and lay["read_misaligned"] >= 1
                         # latency split is present and non-negative
                         and d["read_latency_us"]["cache_mean"] >= 0.0)
            repeat_ok = d["max_repeated_selection"] >= 2  # A and B read the same hyperslab
        except Exception:
            pass
    checks["fields_sane"] = fields_ok
    checks["repeat_detected"] = repeat_ok

    # WRITE-SIDE ACCOUNTING. Nothing here asserted anything about writes, which
    # is why `write_served.mirrored` could mean "the native write succeeded"
    # through a release: every read-side field was checked and every write-side
    # field was taken on faith.
    #
    # The pin is an invariant, not a restatement of the implementation: a WRITE
    # can never be `served` from the cache. Serving is what reads do; a write
    # puts bytes in, it does not get bytes out. The old code emitted
    # "served":"cache" on every successfully cached write, so this fails on it.
    write_ok = False
    if summaries and jsonls:
        try:
            s = json.load(open(summaries[0]))
            recs = [json.loads(l) for l in open(jsonls[0]) if l.strip()]
            writes = [r for r in recs if r.get("op") == "write"]
            ws = s["datasets"]["m"]["write_staged"]
            write_ok = (
                bool(writes)
                and all(r["served"] != "cache" for r in writes)
                # a staged write reports what it submitted, and only then
                and all(("staged_bytes" in r) == (r["served"] == "staged")
                        for r in writes)
                and ws["staged"] >= 1
                and ws["bytes_staged"] > 0
                # the surviving-bytes identity the admission denominator rests on
                and ws["bytes_resident"] ==
                    max(0, ws["bytes_staged"] - ws["bytes_discarded"])
                and s["v"] >= 2)
        except Exception:
            pass
    checks["write_accounting"] = write_ok

    ok = all(checks.values())
    print(f"  {'telemetry':<20} {'PASS' if ok else 'FAIL'}  "
          f"(summary+jsonl, hit_rate/repeat sane, write accounting)")
    return {"telemetry": checks}


# ------------------------------------------------- §1.1(e) error-code parity
# The cache opt-out for each connector. Both are env vars so the negative-path
# program can stay connector-agnostic (it sets nothing and includes no clio
# header); the driver alone decides the configuration.
CACHE_OFF_ENV = {"vol": {"CLIO_VOL_CACHE": "0"},
                 "vfd": {"CLIO_VFD_CACHE": "0"}}


def _parse_eparity(stdout):
    """{case: 'rc=..,maj=..,min=..'} from the C program's EPARITY lines."""
    out = {}
    for line in stdout.splitlines():
        if not line.startswith("EPARITY "):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        out[parts[1]] = " ".join(parts[2:])
    return out


def _run_error_parity(mode):
    """§1.1(e): do failures fail the SAME WAY through the connector as natively?

    Runs hdf5_c_error_parity_test.c once under native (the oracle) and once per
    cache setting under `mode`, then diffs the per-case (status, major, minor)
    triples. Returns {"<mode>/eparity_cache<on|off>/<case>": {"parity": bool}} —
    one entry PER CASE, deliberately. A single entry for the whole program would
    force --expect-fail to allowlist every negative case at once to excuse the
    one known gap, which is precisely the blanket allowlist the plan rules out.

    Both cache settings are run because they ask different questions. Cache-OFF
    is where error parity is simply expected to hold. Cache-ON is where a cache
    can mask a file's own error by answering from a staged copy — see
    case_checksum in the C program.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    binp = os.path.join(TMP, "c_error_parity")
    srcp = os.path.join(src_dir, "hdf5_c_error_parity_test.c")
    if not os.path.exists(binp) and not _build_probe("c_error_parity", binp, srcp):
        return {f"{mode}/c_error_parity": {"compiled": False}}

    def emit(env):
        r = subprocess.run([binp, TMP], capture_output=True, text=True,
                           env=env, timeout=240)
        return _parse_eparity(r.stdout), r.returncode

    ref, rc_ref = emit(_env("native"))
    if not ref:
        print(f"  {'error_parity':<22} FAIL  native oracle emitted nothing "
              f"(rc={rc_ref}) — cannot compare")
        return {f"{mode}/eparity_oracle": {"oracle_ran": False}}

    results = {}
    for setting in ("on", "off"):
        env = dict(_env(mode))
        if setting == "off":
            env.update(CACHE_OFF_ENV[mode])
        got, _rc = emit(env)
        for case, want in ref.items():
            have = got.get(case)
            ok = (have == want)
            key = f"{mode}/eparity_cache{setting}/{case}"
            results[key] = {"parity": ok}
            line = f"  cache{setting}: {case:<30} {_mark(key, ok)}"
            if not ok:
                line += f"\n      native   : {want}\n      {mode:<9}: {have}"
                note = GAP_NOTES.get(key)
                if key in EXPECT_FAIL:
                    line += ("\n      why      : "
                             + (note or "documented open gap (no note recorded)"))
            print(line)
    return results


# ----------------------------------------------------- fixture sanity guard
def _run_fixture_sanity():
    """Do the fixtures actually WRITE anything?

    Regression guard for a defect this suite shipped with. `array_dtype` passed
    a (50,3) array with dtype ("f8",(3,)), which h5py resolves to a (50,3)
    dataspace OF 3-arrays -- logical shape (50,3,3). The data did not match, so
    the write was silently dropped and the dataset ended up with no allocated
    storage, reading back as all zeros.

    Nothing caught it, and nothing could have: a DIFFERENTIAL test compares the
    two arms against each other, and both arms ran the same broken fixture. They
    agreed perfectly -- on nothing. The case reported PASS on all four arms for
    as long as it existed.

    So the differential method has a blind spot exactly here, and it needs a
    non-differential check to cover it: assert against the file itself that the
    bytes are there. Cheap, and it makes the whole corpus's PASS mean something.

    Runs on the native files the corpus loop already wrote -- no extra writes.
    """
    checks = {}
    code = (
        "import sys, json, h5py, numpy as np\n"
        "path, names = sys.argv[1], sys.argv[2:]\n"
        "out = {}\n"
        "with h5py.File(path, 'r') as f:\n"
        "    for n in names:\n"
        "        d = f[n]\n"
        "        a = np.asarray(d[()])\n"
        # np.any() raises on structured (compound) dtypes, so test the raw bytes
        # instead -- that works for every dtype the corpus contains. Object
        # dtypes (vlen) hold pointers, whose bytes say nothing about content, so
        # they are exempt from the all-zero test rather than wrongly flagged.
        "        if a.dtype == object:\n"
        "            allzero = False\n"
        "        else:\n"
        "            raw = a.tobytes()\n"
        "            allzero = len(raw) > 0 and not any(raw)\n"
        "        out[n] = [int(d.id.get_storage_size()), bool(allzero)]\n"
        "print(json.dumps(out))\n"
    )
    bad = []
    for case, spec in CASES.items():
        dsets = spec.get("paths")
        if not dsets:
            continue
        fn = os.path.join(TMP, ARM_DIR["native"], f"{case}.h5")
        if not os.path.exists(fn):
            continue
        r = subprocess.run([sys.executable, "-c", code, fn] + list(dsets),
                           capture_output=True, text=True, env=_env("native"),
                           timeout=180)
        try:
            got = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            checks[f"fixture_sanity/{case}"] = {"inspected": False}
            bad.append(f"{case}: could not inspect")
            continue
        for d, (storage, allzero) in got.items():
            # A virtual dataset legitimately has NO storage of its own -- the raw
            # data is in its source files. That is the definition of a VDS, not a
            # dropped write, so it is exempt from the storage check but NOT from
            # the all-zero check (which is what would catch a VDS that failed to
            # map through to its source).
            exempt = spec.get("no_storage", False)
            ok = (exempt or storage > 0) and not allzero
            checks[f"fixture_sanity/{case}:{d}"] = {"wrote_data": ok}
            if not ok:
                bad.append(f"{case}:{d} storage={storage} all_zero={allzero}")
    n_ok = sum(1 for v in checks.values() if all(v.values()))
    print(f"  {'fixture_wrote_data':<22} {n_ok}/{len(checks)} declared datasets "
          f"have real content")
    for b in bad:
        print(f"      VACUOUS: {b}")
    return checks


# ------------------------------------------- §1.1(d) library-free readability
def _fnv1a(data, h=0xCBF29CE484222325):
    """Must match fnv1a() in hdf5_c_libfree_reader.c exactly."""
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _libfree_expected(path, dsets):
    """What a NATIVE h5py read says the raw dataset bytes are.

    Runs in a subprocess under the native env so no connector is in the loop.
    Returns {dset: (nbytes, fnv)} for datasets whose in-memory bytes are
    expected to equal their on-disk bytes; anything where they would not be
    (vlen, filtered, unallocated) is left out and the C reader skips it too.
    """
    code = (
        "import sys, json, numpy as np, h5py\n"
        "path, names = sys.argv[1], sys.argv[2:]\n"
        "out = {}\n"
        "with h5py.File(path, 'r') as f:\n"
        "    for n in names:\n"
        "        d = f[n]\n"
        "        if d.id.get_storage_size() == 0: continue\n"
        "        a = np.ascontiguousarray(d[()])\n"
        "        if a.dtype == object: continue\n"
        "        out[n] = [int(a.nbytes), a.tobytes().hex()]\n"
        "print(json.dumps(out))\n"
    )
    r = subprocess.run([sys.executable, "-c", code, path] + list(dsets),
                       capture_output=True, text=True, env=_env("native"),
                       timeout=180)
    try:
        raw = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {}
    return {n: (nb, _fnv1a(bytes.fromhex(hx))) for n, (nb, hx) in raw.items()}


def _parse_libfree(stdout):
    """{dset: ('ok', nbytes, fnv) | ('skip', reason) | ('fail', reason)}"""
    out = {}
    for line in stdout.splitlines():
        if not line.startswith("LIBFREE "):
            continue
        p = line.split()
        if len(p) < 3:
            continue
        name, verdict = p[1], p[2]
        fields = dict(kv.split("=", 1) for kv in p[3:] if "=" in kv)
        if verdict == "ok":
            out[name] = ("ok", int(fields.get("nbytes", -1)),
                         int(fields.get("fnv", "0"), 16))
        else:
            out[name] = (verdict, fields.get("reason", "?"))
    return out


def _run_libfree_check(mode, cases_with_paths):
    """§1.1(d): is the data readable with NO HDF5 library in the process?

    The strongest of the five clauses and the one that most cleanly falsifies a
    design keeping data outside the native file -- (a)-(c)/(e) all ask HDF5
    whether HDF5 likes the file, so the library is in the loop and cannot
    distinguish "the bytes are in the file" from "CLIO can find the bytes".
    Before this the clause was an 8-byte superblock signature probe.

    Method stays differential: hdf5_c_libfree_reader.c parses the superblock,
    walks the group graph, and preads the raw data with no HDF5 code anywhere in
    the process; the driver compares its hash to a native h5py read of the same
    dataset. Reports one entry per dataset, and counts skips separately so the
    coverage of this gate is visible rather than implied -- a skip is NOT a pass.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    binp = os.path.join(TMP, "c_libfree_reader")
    srcp = os.path.join(src_dir, "hdf5_c_libfree_reader.c")
    # Compiled WITHOUT h5cc on purpose: this program must not link libhdf5, and
    # using the HDF5 compiler wrapper would quietly link it.
    if shutil.which("gcc") is None:
        print(f"  {'libfree':<22} SKIP  (no gcc)")
        return {}
    comp = subprocess.run(["gcc", "-O2", "-o", binp, srcp],
                          capture_output=True, text=True, env=_env("native"))
    if comp.returncode != 0:
        # Same treatment as every other probe: BUILD-ERROR, full diagnostics,
        # and NOT MEASURED in the summary rather than a compatibility verdict.
        # Recorded under the key the summary matches on (trailing segment).
        _record_build_error("libfree_compile", comp)
        return {f"{mode}/libfree_compile": {"compiled": False}}
    # Guard the central claim of this gate rather than trusting it: if this
    # binary links libhdf5, it is not reading the file "without the library"
    # and the whole §1.1(d) result would be worthless.
    #
    # The tool differs by platform -- ldd is Linux, macOS uses otool -L -- and
    # calling the wrong one raises FileNotFoundError, which took the ENTIRE
    # suite down on macOS rather than this one gate. A check that cannot run is
    # reported as a check that did not run; it never crashes the run and never
    # silently passes.
    linkage_tool = ["otool", "-L"] if sys.platform == "darwin" else ["ldd"]
    if shutil.which(linkage_tool[0]) is None:
        print(f"  {'libfree':<22} SKIP  (no {linkage_tool[0]}: cannot verify the "
              f"reader is libhdf5-free, so §1.1(d) is NOT measured here)")
        return {}   # same shape as the "no gcc" skip above: loud, not fatal
    link = subprocess.run(linkage_tool + [binp], capture_output=True, text=True)
    # Match on the DEPENDENCY lines only, never on the whole output. `otool -L`
    # echoes the binary's own path as an unindented first line, and this binary
    # lives under TMP = /tmp/hdf5compat -- so a whole-output substring search
    # matched the ARGUMENT rather than any library, and this gate reported
    # "reader links libhdf5" on macOS for a path the suite had itself chosen.
    # `ldd` does not echo the path, which is the only reason Linux never saw it.
    # Both tools indent one dependency per line, so the indent is the shared
    # discriminator between "what I asked about" and "what it links".
    deps = [ln for ln in (link.stdout + link.stderr).splitlines() if ln[:1].isspace()]
    if any("hdf5" in ln.lower() for ln in deps):
        print(f"  {'libfree':<22} FAIL  reader links libhdf5; it cannot test §1.1(d)")
        for ln in deps:
            if "hdf5" in ln.lower():
                print(f"  {'':<22}       {ln.strip()}")
        return {f"{mode}/libfree_no_hdf5_linkage": {"unlinked": False}}

    results, n_ok, skips = {}, 0, {}
    for case, dsets in cases_with_paths:
        fc = os.path.join(TMP, ARM_DIR[mode], f"{case}.h5")
        fn = os.path.join(TMP, ARM_DIR["native"], f"{case}.h5")
        if not os.path.exists(fc) or not os.path.exists(fn):
            continue
        # The reference comes from the NATIVE file, not from re-reading fc.
        # Comparing libfree(fc) against h5py(fc) would be self-referential and
        # would MISS the exact failure this gate exists to catch: a connector
        # that writes metadata pointing at a hole and keeps the real data
        # elsewhere. Both readers would then see the same hole and agree on
        # garbage. Against the native file's contents, that connector fails here.
        want = _libfree_expected(fn, dsets)
        r = subprocess.run([binp, fc] + list(dsets), capture_output=True,
                           text=True, env=_env("native"), timeout=180)
        got = _parse_libfree(r.stdout)
        for d in dsets:
            g = got.get(d)
            if g is None:
                results[f"{mode}/libfree/{case}:{d}"] = {"read": False}
                print(f"  libfree {case}:{d:<22} FAIL  (reader emitted nothing)")
                continue
            if g[0] == "skip":
                # Declared out of scope by the reader, with a named reason.
                skips.setdefault(g[1], []).append(f"{case}:{d}")
                continue
            if g[0] == "fail":
                # NOT a skip. The reader tried and could not parse it, which is a
                # real result and must be reported as one -- bucketing a parse
                # failure under "out of scope" would let a broken reader look
                # like a narrow one.
                results[f"{mode}/libfree/{case}:{d}"] = {"read": False}
                print(f"  libfree {case}:{d:<22} FAIL  (reader: {g[1]})")
                continue
            _, nbytes, fnv = g
            exp = want.get(d)
            ok = exp is not None and exp[0] == nbytes and exp[1] == fnv
            results[f"{mode}/libfree/{case}:{d}"] = {"read": ok}
            if ok:
                n_ok += 1
            else:
                print(f"  libfree {case}:{d:<22} FAIL  "
                      f"(native={exp}, libfree=({nbytes},{fnv:#018x}))")
    print(f"  {'libfree_readable':<22} {n_ok}/{len(results)} datasets read with "
          f"no HDF5 library in the process")
    for reason, who in sorted(skips.items()):
        print(f"      out of scope ({reason}): {', '.join(who)}")
    return results


def _run_plugin_presence_check():
    """A connector that is PRESENT but NOT SELECTED must be completely inert.

    Regression pin for a segfault this suite found. HDF5 resolves which
    connector can open a file by iterating every plugin on HDF5_PLUGIN_PATH and
    asking each one H5VL_FILE_IS_ACCESSIBLE — an operation on a FILENAME, so it
    passes obj == NULL. clio_file_specific cast that to a file and dereferenced
    it, which meant that merely having libclio_hdf5_vol.so sitting in the plugin
    directory turned a plain H5Fopen of a missing file into a CRASH: connector
    not selected, not requested, nothing in the application referring to it.

    Why it needs its own check rather than riding along on the other arms:
    every one of those selects a connector, and selecting it is exactly what
    HID this bug — the explicit path opens directly and never probes. The
    configuration under test here is the one no existing arm covers.

    Method is the same as everywhere else: run the negative-path program with
    HDF5_PLUGIN_PATH set but NO connector selected, and require it to behave
    identically to a run with no plugin path at all.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    binp = os.path.join(TMP, "c_error_parity")
    srcp = os.path.join(src_dir, "hdf5_c_error_parity_test.c")
    if not os.path.exists(binp) and not _build_probe("c_error_parity", binp, srcp):
        return {"plugin_presence_inert": {"compiled": False}}

    def emit(env):
        r = subprocess.run([binp, TMP], capture_output=True, text=True,
                           env=env, timeout=240)
        return _parse_eparity(r.stdout), r.returncode

    clean_env = _env("native")
    ref, rc_ref = emit(clean_env)
    present_env = dict(clean_env, HDF5_PLUGIN_PATH=BIN)
    got, rc_got = emit(present_env)

    props = {
        # A crash shows up here first: the old code exited 139 with no output.
        "no_crash": rc_got == rc_ref,
        "same_cases": bool(ref) and set(got) == set(ref),
        "same_errors": bool(ref) and got == ref,
    }
    mark = "PASS" if all(props.values()) else "FAIL"
    print(f"  {'plugin_presence_inert':<22} {mark}  "
          f"(plugin dir on HDF5_PLUGIN_PATH, no connector selected; "
          f"rc {rc_ref} vs {rc_got}, {len(ref)} vs {len(got)} cases)")
    return {"plugin_presence_inert": props}


def stop_runtime():
    """Tear down the clio_run that restart_runtime started, so it does not leak
    into later steps of the same CI job. The Linux adapters job runs a FUSE
    mount smoke test after ctest; a leftover runtime holds port 9413 and the
    smoke's clio_run then dies with 'Address already in use'."""
    # -x, not -f: see restart_runtime() above.
    subprocess.run(["pkill", "-x", "clio_run"], check=False)
    time.sleep(1)
    _wipe_clio_shm()


# Column order for the per-connector results table. Each maps to a §1.1
# criterion; `mech` is what the column is actually measured WITH, printed in the
# table legend so nobody has to read this file to know what a PASS means.
COLUMNS = (
    ("write_compat",   "1.1(a)", "digest vs native"),
    ("read_compat",    "1.1(a)", "digest vs native"),
    ("roundtrip",      "1.1(a)", "digest vs native"),
    ("h5diff_clean",   "1.1(a)", "h5diff"),
    ("tools_ok",       "1.1(c)", "h5dump/h5ls/h5stat exit status"),
    ("struct_valid",   "1.1(b)", "h5clear+reopen (h5check unavailable)"),
    ("tool_parity",    "1.1(c)", "h5dump -pBH / h5ls -vlr / h5stat OUTPUT equality"),
)


def run_connector(mode):
    """Run the whole corpus against one connector. Returns {"<mode>/<case>": props}.

    The native oracle is regenerated inside this loop rather than shared across
    connectors, so a vol run and a vfd run are independent and either can be run
    alone with identical meaning.
    """
    print(f"\n== connector: {mode} ==")
    results = {}
    for case, spec in CASES.items():
        # A case may declare which connectors it is meaningful for. Most do not:
        # an ordinary HDF5 program is meaningful to both. Skipping is LOUD.
        applies = spec.get("modes", CONNECTORS)
        if mode not in applies:
            print(f"  {case:<22} SKIP  (not applicable to {mode}: "
                  f"{spec.get('skip_reason', 'declared modes=' + ','.join(applies))})")
            continue
        ndir = os.path.join(TMP, ARM_DIR["native"])
        cdir = os.path.join(TMP, ARM_DIR[mode])
        os.makedirs(ndir, exist_ok=True)
        os.makedirs(cdir, exist_ok=True)
        fn = os.path.join(ndir, f"{case}.h5")
        fc = os.path.join(cdir, f"{case}.h5")
        for f in (fn, fc):
            if os.path.exists(f):
                os.remove(f)
        # four arms
        ARM_STDERR.clear()
        _run(case, "write", fn, "native")
        ref, _ = _run(case, "read", fn, "native")     # reference (native/native)
        _run(case, "write", fc, mode)
        d_wc, _ = _run(case, "read", fc, "native")    # write compat
        d_rc, _ = _run(case, "read", fn, mode)        # read compat
        d_rt, _ = _run(case, "read", fc, mode)        # round-trip
        both = os.path.exists(fn) and os.path.exists(fc)
        parity_ok, parity_who = _tool_parity(fn, fc) if both else (False, "missing file")
        props = {
            "write_compat": (ref is not None and d_wc == ref),
            "read_compat":  (ref is not None and d_rc == ref),
            "roundtrip":    (ref is not None and d_rt == ref),
            "h5diff_clean": (both and _h5diff(fn, fc)),
            "tools_ok":     (os.path.exists(fc) and _tool_ok(fc)),
            "struct_valid": _struct_valid(fc),
            "tool_parity":  parity_ok,
        }
        results[f"{mode}/{case}"] = props
        # Per-COLUMN marks share the case's key: if the case is an expected gap,
        # whichever column it fails on is that gap, not a regression.
        ckey = f"{mode}/{case}"
        mark = lambda b: _mark(ckey, b)
        detail = "" if parity_ok else f"  [parity differs: {parity_who}]"
        print(f"  {case:<22} "
              + " ".join(f"{k}={mark(props[k])}" for k, _c, _m in COLUMNS)
              + detail)
        # A failing digest column is otherwise a bare FAIL, which cannot
        # distinguish "the arm crashed" (digest None -- see ARM-CRASHED above)
        # from "the arm ran and read something else". Those need different
        # investigations, so name which one it was at the point of failure.
        if not all(props[k] for k in ("write_compat", "read_compat", "roundtrip")):
            short = lambda d: "CRASHED" if d is None else str(d)[:16]
            print(f"  {'':<22}   digests: native={short(ref)} "
                  f"wc={short(d_wc)} rc={short(d_rc)} rt={short(d_rt)}")
            for arm, err in sorted(ARM_STDERR.items()):
                print(f"  {'':<22}   stderr[{arm}]:")
                for line in err.splitlines():
                    print(f"  {'':<22}     | {line}")
    return results


def driver(args):
    # Set BEFORE any check runs: the marks are printed as the checks execute,
    # long before the summary computes the same set.
    global EXPECT_FAIL
    EXPECT_FAIL = {x for x in (args.expect_fail or "").split(",") if x}
    unknown = EXPECT_FAIL - set(GAP_NOTES)
    if unknown:
        # An allowlist entry with no recorded reason is how a permanent
        # exemption gets in unnoticed. Loud, but not fatal -- a caller
        # investigating a new gap should still be able to run.
        print("WARNING: --expect-fail entries with no note in GAP_NOTES "
              f"(add one saying WHY): {sorted(unknown)}")
    assert restart_runtime(), "clio_run did not become ready"
    os.makedirs(TMP, exist_ok=True)
    # Build every C probe first. A probe that does not compile is a broken
    # harness, and finding that out now -- with all of them reported together --
    # beats discovering it as one opaque case failure minutes into the run.
    print("-- building C probes --")
    _build_probe_set()
    print(f"-- corpus: {len(CASES)} cases; szip: {SZIP_STATUS} --")
    results = {}

    print("columns: " + " | ".join(f"{k} [{crit} via {mech}]"
                                   for k, crit, mech in COLUMNS))
    for mode in args.modes:
        results.update(run_connector(mode))
        # §1.1(d). Runs on the CONNECTOR-produced files, which is the whole
        # point: it asks whether what the connector wrote is readable with the
        # HDF5 library removed from the process entirely.
        print(f"\n-- library-free readability [{mode}] (§1.1(d)) --")
        with_paths = [(c, s["paths"]) for c, s in CASES.items()
                      if s.get("paths") and mode in s.get("modes", CONNECTORS)]
        results.update(_run_libfree_check(mode, with_paths))
        # §1.1(e). Applies to BOTH connectors — the negative path is the one
        # axis where the VFD and the VOL are asked the identical question.
        print(f"\n-- error-code parity vs native [{mode}] (§1.1(e)) --")
        results.update(_run_error_parity(mode))

    # VOL-only sections. The C tests cover ops h5py cannot exercise through a
    # non-native VOL, and the telemetry check is Part B observability that lives
    # in the VOL (OD-2). Neither has a VFD analogue — the VFD's own C suite is
    # test_vfd_adapter.cc — so both are gated on the vol arm actually running.
    # Connector-independent checks. Run once, after the corpus loop has produced
    # the native files these inspect.
    print("\n-- fixture sanity (does the corpus actually write data?) --")
    results.update(_run_fixture_sanity())
    print("\n-- unselected-connector inertness (regression pin) --")
    results.update(_run_plugin_presence_check())

    if "vol" in args.modes:
        print("\n-- C tests (VOL-aware APIs h5py can't exercise) --")
        results.update({"vol/" + k: v for k, v in _run_c_tests().items()})
        print("\n-- cache reuse across files (regression pin) --")
        results.update(_run_cache_reuse_check())
        results.update(_run_instant_reopen_check())
        print("\n-- telemetry (access observability) --")
        results.update({"vol/" + k: v for k, v in _run_trace_check().items()})
        results.update({"vol/" + k: v for k, v in _run_bbox_fetch_check().items()})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    total = len(results)
    expect_fail = EXPECT_FAIL
    failed = {c for c, p in results.items() if not all(p.values())}
    unexpected = sorted(failed - expect_fail)      # honest failures (or regressions)
    fixed = sorted(expect_fail - failed)           # known gap now passes
    # A probe that never compiled says nothing about the connector. Report it as
    # its own category so the summary does not assert an incompatibility it did
    # not measure. Still fatal -- unmeasured is not the same as passing either.
    # Result keys are connector-scoped ("vol/c_passthrough_ops") while a probe
    # builds once for all of them, so match on the trailing segment.
    def _probe_of(key):
        return BUILD_ERRORS.get(key.rsplit("/", 1)[-1])
    unbuilt = sorted(k for k in failed if _probe_of(k))
    measured_fail = sorted(k for k in failed if not _probe_of(k))

    # Per-connector tally, so a red VFD arm is never read as a red VOL arm, and
    # so a documented gap is never read as a regression.
    print()
    for mode in args.modes:
        keys = [k for k in results if k.startswith(mode + "/")]
        bad = [k for k in keys if k in failed]
        gaps = [k for k in bad if k in expect_fail]
        real = [k for k in bad if k not in expect_fail]
        summary = f"{len(keys) - len(bad)}/{len(keys)} pass for {mode}"
        if gaps:
            summary += f", {len(gaps)} known gap{'s' if len(gaps) > 1 else ''}"
        if real:
            summary += f" — FAILING: {sorted(real)}"
        print(summary)
    print(f"{total - len(failed)}/{total} cases fully pass overall. wrote {args.out}")

    if unbuilt:
        print(f"BUILD-ERROR — NOT MEASURED, the probe did not compile: {unbuilt}")
        for name in unbuilt:
            print(f"    {name}: {_probe_of(name)}")
        print("    This is a harness/toolchain failure, not a compatibility "
              "result. Fix the build, then re-run to learn what these cases say.")

    if expect_fail:
        # regression-gate mode: only unexpected failures are fatal
        for k in sorted(failed & expect_fail):
            print(f"KNOWN-GAP {k}\n    {GAP_NOTES.get(k, 'no note recorded')}")
        if fixed:
            print(f"NOTE: previously-failing cases now PASS (drop from --expect-fail): {fixed}")
        # A probe that did not build is already reported above as UNMEASURED;
        # calling it a regression would name the wrong problem. It still counts
        # toward the exit status.
        regressions = [c for c in unexpected if not _probe_of(c)]
        if regressions:
            print(f"REGRESSION — these should pass but FAILED: {regressions}")
    elif measured_fail:
        # honest mode (default): every failure is reported and fatal
        print(f"FAILING — not yet native-compatible for: {measured_fail}")
    return 1 if unexpected else 0


def _missing_deps():
    """External dependencies the suite hard-requires but that may be absent.
    Returns a human-readable list (empty when all present). h5py must import in
    THIS interpreter — the write/read arms are subprocesses of sys.executable —
    and the HDF5 CLI tools are the differential oracle. Missing any of them means
    the suite cannot produce a meaningful comparison, so main() reports SKIP."""
    missing = []
    try:
        import h5py  # noqa: F401
    except Exception:
        missing.append("python module 'h5py'")
    # h5stat and h5clear joined the hard requirements with the §1.1(b)/(c) gates:
    # h5stat is part of the tool-parity triple and the structural reopen, h5clear
    # is the structural check itself. Without them those columns cannot be
    # evaluated at all, so their absence is an environment gap like the rest.
    for tool in ("h5diff", "h5dump", "h5ls", "h5stat", "h5clear"):
        if shutil.which(tool) is None:
            missing.append(tool)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--action", choices=["write", "read"])
    ap.add_argument("--file")
    ap.add_argument("--out", default="hdf5_compat_results.json")
    ap.add_argument("--bin", help="dir with libclio_hdf5_vol.so / libclio_vfd.so "
                    "+ clio_run (default $CLIO_VOL_BIN or /workspace/build/bin)")
    ap.add_argument("--modes", default="vol",
                    help="comma-separated connectors to exercise: "
                    + ",".join(CONNECTORS) + " (default: vol)")
    ap.add_argument("--expect-fail", default="",
                    help="comma-separated '<mode>/<case>' entries allowed to fail "
                    "(known gaps); exit is nonzero only on a REGRESSION outside "
                    "this set")
    a = ap.parse_args()
    if a.bin:
        global BIN
        BIN = a.bin
    if a.worker:
        worker(a.case, a.action, a.file)
        return 0
    a.modes = [m for m in (a.modes or "").split(",") if m]
    bad = [m for m in a.modes if m not in CONNECTORS]
    if bad or not a.modes:
        print(f"ERROR: --modes must be a non-empty subset of {CONNECTORS}; got {bad or 'nothing'}")
        return 2
    missing = _missing_deps()
    if missing:
        print("SKIP: the HDF5 compat suite needs a native HDF5 toolchain that is "
              "not installed on this host (" + ", ".join(missing) + "). This suite "
              "differentially compares a clio connector against native HDF5, so it "
              "is skipped rather than reported as failing.")
        return SKIP_RC
    try:
        return driver(a)
    finally:
        # Never leak the clio_run started by restart_runtime into later CI steps
        # (the Linux adapters FUSE mount smoke binds the same port). Tear it down
        # even if the suite raised.
        stop_runtime()

if __name__ == "__main__":
    sys.exit(main())
