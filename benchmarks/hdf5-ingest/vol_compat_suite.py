#!/usr/bin/env python3
"""Phase 1 / Part A — clio VOL compatibility suite (differential testing).

Does routing HDF5 through the clio VOL preserve native HDF5 semantics? Method:
the NATIVE VOL is the oracle. For each feature case, exercise four arms and assert
the file content (data + metadata) matches:

  native write -> native read   (reference)
  VOL write    -> native read    (write compat: VOL emits a valid native file)
  native write -> VOL read       (read compat)
  VOL write    -> VOL read        (round-trip)

plus h5diff(native_file, vol_file) is clean and h5dump/h5ls succeed on the
VOL-written file. Output: a per-case pass/fail matrix. CI-shaped (nonzero exit on
any failure). Runs INSIDE the clio-core dev container; the driver keeps clio_run
up and spawns each arm as a subprocess with the VOL env toggled.

  python3 vol_compat_suite.py --out vol_compat_results.json
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
RUNTIME_LOG = "/tmp/clio_run_volcompat.log"
# Count-agnostic: the compose emits "All <N> pools created successfully"; the
# suite's own config (below) makes 2, a dev box's ~/.clio/clio.yaml may make 3.
RUNTIME_READY = "pools created successfully"
TMP = "/tmp/volcompat"

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
# Each writer builds a deterministic file at `path` via h5py. VOL on/off is set
# by the driver through the process env (HDF5_VOL_CONNECTOR).

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
    import h5py, numpy as np
    dt = np.dtype(("f8", (3,)))
    with h5py.File(path, "w") as f:
        f.create_dataset("arr", data=np.arange(50 * 3, dtype="f8").reshape(50, 3), dtype=dt)

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
    "hyperslab_read":    {"write": w_hyperslab_src, "read": read_hyperslab},
    "uint_types":        {"write": w_uint_types, "paths": ["u8", "u16", "u32"]},
    "enum":              {"write": w_enum, "paths": ["e"]},
    "array_dtype":       {"write": w_array_dtype, "paths": ["arr"]},
    "scalar":            {"write": w_scalar, "paths": ["sc"]},
    "extendible_append": {"write": w_extendible_append, "paths": ["ext"]},
    "fletcher32":        {"write": w_fletcher32, "paths": ["fl"]},
    "point_selection":   {"write": w_point_src, "read": read_point},
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
def _env(vol):
    e = dict(os.environ, HOME="/home/iowarp",
             LD_LIBRARY_PATH=BIN + ":/usr/local/lib:/usr/lib/x86_64-linux-gnu",
             PYTHONPATH=BIN)
    if vol:
        e["HDF5_PLUGIN_PATH"] = BIN
        e["HDF5_VOL_CONNECTOR"] = "clio"
    else:
        e.pop("HDF5_VOL_CONNECTOR", None)
    return e

def _run(case, action, path, vol):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--case", case, "--action", action, "--file", path]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_env(vol), timeout=180)
    for line in p.stdout.splitlines():
        if line.startswith("DIGEST:"):
            return line[7:], p.returncode
        if line == "WROTE":
            return "WROTE", p.returncode
    return None, p.returncode  # crash / no output

def _h5diff(a, b):
    p = subprocess.run(["h5diff", a, b], capture_output=True, text=True,
                       env=_env(False))
    return p.returncode == 0

def _tool_ok(path):
    for tool in (["h5dump", "-H"], ["h5ls", "-r"]):
        p = subprocess.run(tool + [path], capture_output=True, text=True, env=_env(False))
        if p.returncode != 0:
            return False
    return True

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
    subprocess.run(["pkill", "-f", "clio_run"], check=False)
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
                              text=True, env=_env(False))
    if (comp is None or comp.returncode != 0) and shutil.which("gcc"):
        # Point gcc at the HDF5 under the running interpreter's prefix (the same
        # one the h5py arms use); fall back to /usr/local for standalone runs.
        prefix = _hdf5_prefix()
        inc = os.path.join(prefix, "include") if prefix else "/usr/local/include"
        lib = os.path.join(prefix, "lib") if prefix else "/usr/local/lib"
        comp = subprocess.run(["gcc", "-o", binp, srcp, "-I" + inc,
                               "-L" + lib, "-Wl,-rpath," + lib, "-lhdf5"],
                              capture_output=True, text=True, env=_env(False))
    return comp


def _run_c_tests():
    """Compile + run the isolated C tests through the VOL. Returns
    {name: {"pass": bool}}; each C program exits 0 on pass. These cover ops h5py
    cannot exercise well via a non-native VOL: modern-API iteration
    (c_iteration), Safe-mode H5Fflush durability (c_safeflush), and
    selection-aware read caching + partial-write invalidation (c_selection)."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    tests = {"c_iteration": "vol_c_iteration_test.c",
             "c_safeflush": "vol_c_safeflush_test.c",
             "c_selection": "vol_c_selection_test.c"}
    out = {}
    for name, src in tests.items():
        binp = os.path.join(TMP, name)
        srcp = os.path.join(src_dir, src)
        comp = _compile_c(binp, srcp)
        if comp is None or comp.returncode != 0:
            detail = (comp.stderr.strip()[-120:] if comp is not None
                      else "no C compiler (h5cc/gcc) found")
            print(f"  {name:<20} COMPILE-FAIL  {detail}")
            out[name] = {"pass": False}
            continue
        r = subprocess.run([binp], capture_output=True, text=True,
                           env=_env(True), timeout=120)
        ok = (r.returncode == 0)
        out[name] = {"pass": ok}
        # The binary's own verdict line ends in PASS/FAIL; isolate it from any
        # interleaved clio runtime INFO logging on stdout (e.g. PoolId(major:...)).
        verdict = [l for l in r.stdout.strip().splitlines()
                   if l.rstrip().endswith(("PASS", "FAIL"))]
        detail = verdict[-1] if verdict else r.stdout.strip()[-70:]
        print(f"  {name:<20} {'PASS' if ok else 'FAIL'}  ({detail})")
    return out


def _run_trace_check():
    """Verify access telemetry (Part B). Runs the c_selection workload with
    CLIO_VOL_TRACE set and asserts the summary JSON + per-access JSONL are
    produced with sane, self-consistent fields — including read cache-hit rate in
    [0,1] and repeated-selection detection (the workload reads one hyperslab
    twice). Observe-only: does not change data-path behavior."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    binp = os.path.join(TMP, "c_selection")  # reuse the c_selection binary
    srcp = os.path.join(src_dir, "vol_c_selection_test.c")
    if not os.path.exists(binp):
        comp = _compile_c(binp, srcp)
        if comp is None or comp.returncode != 0:
            print(f"  {'telemetry':<20} COMPILE-FAIL")
            return {"telemetry": {"pass": False}}
    tdir = os.path.join(TMP, "trace")
    os.makedirs(tdir, exist_ok=True)
    for f in glob.glob(tdir + "/*"):
        os.remove(f)
    env = dict(_env(True), CLIO_VOL_TRACE=tdir)
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
    ok = all(checks.values())
    print(f"  {'telemetry':<20} {'PASS' if ok else 'FAIL'}  "
          f"(summary+jsonl, hit_rate/repeat sane)")
    return {"telemetry": checks}


def stop_runtime():
    """Tear down the clio_run that restart_runtime started, so it does not leak
    into later steps of the same CI job. The Linux adapters job runs a FUSE
    mount smoke test after ctest; a leftover runtime holds port 9413 and the
    smoke's clio_run then dies with 'Address already in use'."""
    subprocess.run(["pkill", "-f", "clio_run"], check=False)
    time.sleep(1)
    _wipe_clio_shm()


def driver(args):
    assert restart_runtime(), "clio_run did not become ready"
    os.makedirs(TMP, exist_ok=True)
    results, n_fail = {}, 0
    for case in CASES:
        fn = os.path.join(TMP, case + "_native.h5")
        fv = os.path.join(TMP, case + "_vol.h5")
        for f in (fn, fv):
            if os.path.exists(f):
                os.remove(f)
        # arm setup
        _run(case, "write", fn, vol=False)
        ref, _ = _run(case, "read", fn, vol=False)          # reference (native/native)
        _run(case, "write", fv, vol=True)
        d_wc, rc_wc = _run(case, "read", fv, vol=False)      # write compat
        d_rc, rc_rc = _run(case, "read", fn, vol=True)       # read compat
        d_rt, rc_rt = _run(case, "read", fv, vol=True)       # round-trip
        props = {
            "write_compat": (ref is not None and d_wc == ref),
            "read_compat":  (ref is not None and d_rc == ref),
            "roundtrip":    (ref is not None and d_rt == ref),
            "h5diff_clean": (os.path.exists(fn) and os.path.exists(fv) and _h5diff(fn, fv)),
            "tools_ok":     (os.path.exists(fv) and _tool_ok(fv)),
        }
        results[case] = props
        if not all(props.values()):
            n_fail += 1
        mark = lambda b: "PASS" if b else "FAIL"
        print(f"  {case:<20} " + " ".join(f"{k}={mark(v)}" for k, v in props.items()))

    # C-API tests: features h5py cannot exercise through a non-native VOL
    # (modern-API iteration). Accurate way to test the VOL as C/C++/NetCDF apps use it.
    print("\n-- C tests (VOL-aware APIs h5py can't exercise) --")
    results.update(_run_c_tests())

    # Access telemetry (Part B observability): summary + JSONL, fields sane.
    print("\n-- telemetry (access observability) --")
    results.update(_run_trace_check())

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    total = len(results)
    expect_fail = {x for x in (args.expect_fail or "").split(",") if x}
    failed = {c for c, p in results.items() if not all(p.values())}
    unexpected = sorted(failed - expect_fail)      # honest failures (or regressions)
    fixed = sorted(expect_fail - failed)           # known gap now passes
    print(f"\n{total - len(failed)}/{total} cases fully pass. wrote {args.out}")
    if expect_fail:
        # regression-gate mode: only unexpected failures are fatal
        print(f"expected gaps still failing: {sorted(failed & expect_fail)}")
        if fixed:
            print(f"NOTE: previously-failing cases now PASS (drop from --expect-fail): {fixed}")
        if unexpected:
            print(f"REGRESSION — these should pass but FAILED: {unexpected}")
    elif failed:
        # honest mode (default): every failure is reported and fatal
        print(f"FAILING — VOL not yet compatible for: {sorted(failed)}")
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
    for tool in ("h5diff", "h5dump", "h5ls"):
        if shutil.which(tool) is None:
            missing.append(tool)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--action", choices=["write", "read"])
    ap.add_argument("--file")
    ap.add_argument("--out", default="vol_compat_results.json")
    ap.add_argument("--bin", help="dir with libclio_hdf5_vol.so + clio_run "
                    "(default $CLIO_VOL_BIN or /workspace/build/bin)")
    ap.add_argument("--expect-fail", default="",
                    help="comma-separated cases allowed to fail (known gaps); "
                    "exit is nonzero only on a REGRESSION outside this set")
    a = ap.parse_args()
    if a.bin:
        global BIN
        BIN = a.bin
    if a.worker:
        worker(a.case, a.action, a.file)
        return 0
    missing = _missing_deps()
    if missing:
        print("SKIP: VOL compat suite needs a native HDF5 toolchain that is not "
              "installed on this host (" + ", ".join(missing) + "). This suite "
              "differentially compares the clio VOL against native HDF5, so it is "
              "skipped rather than reported as failing.")
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
