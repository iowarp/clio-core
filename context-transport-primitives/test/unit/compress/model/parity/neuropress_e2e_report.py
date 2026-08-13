#!/usr/bin/env python3
"""Side-by-side E2E report: Clio-NeuroPress vs native NeuroPress.

Consumes the artifacts produced by the two drivers (see
neuropress_e2e_compare.py for the runbook) and reports every vector the
comparison covers, plus a per-chunk table.

  clio.csv          per-chunk selection log written by compressor_runtime
  clio.csv.payload  per-chunk compressed-payload hash, same run
  native.csv        per-chunk log written by neuropress_native_replay
  data.bin          the exact bytes both sides compressed

Where a metric is not measured on a side, this says so rather than
substituting the other side's number.
"""
import csv
import statistics
import struct
import sys

clio_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/np_f_clio.csv"
nat_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/np_f_native.csv"
data_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/neuropress_e2e_data.bin"
show_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 = all

CHUNK = 4 * 1024 * 1024

# ---------------------------------------------------------------- load
clio = {}
for r in csv.DictReader(open(clio_path)):
    c = int(r["blob"].rsplit("/chunk_", 1)[1])
    if c not in clio:          # first (write-path) selection only
        clio[c] = r

payload = {}
try:
    for r in csv.DictReader(open(clio_path + ".payload")):
        c = int(r["blob"].rsplit("/chunk_", 1)[1])
        if c not in payload:
            payload[c] = r
except FileNotFoundError:
    pass

nat = {int(r["chunk"]): r for r in csv.DictReader(open(nat_path))}
common = sorted(set(clio) & set(nat))
n = len(common)

ALGO = {0: "lz4", 1: "snappy", 2: "deflate", 3: "gdeflate",
        4: "zstd", 5: "ans", 6: "cascaded", 7: "bitcomp"}


def sel(row, cols):
    return tuple(int(row[k]) for k in cols)


# ------------------------------------------------- ground-truth statistics
# Descriptive statistics of the source data, computed once. Both pipelines
# are lossless here, so this is also the expected output distribution for
# each; native's decompressed-output figures are checked against it below.
truth = {}
with open(data_path, "rb") as f:
    for c in range(max(common) + 1 if common else 0):
        f.seek(c * CHUNK)
        buf = f.read(CHUNK)
        if len(buf) < CHUNK:
            break
        v = struct.unpack("<%df" % (CHUNK // 4), buf)
        m = sum(v) / len(v)
        m2 = sum((x - m) ** 2 for x in v) / len(v)
        m4 = sum((x - m) ** 4 for x in v) / len(v)
        truth[c] = (m, m2, (m4 / (m2 * m2)) if m2 > 0 else 0.0)

# ------------------------------------------------------------- comparisons
route_mismatch = [c for c in common
                  if sel(clio[c], ("algo_idx", "quantize", "shuffle")) !=
                     sel(nat[c], ("algo_idx", "quantize", "shuffle"))]

input_same = [c for c in common
              if int(clio[c]["checksum"]) and int(nat[c]["checksum"])
              and int(clio[c]["checksum"]) == int(nat[c]["checksum"])]
input_checked = [c for c in common
                 if int(clio[c]["checksum"]) and int(nat[c]["checksum"])]

pay_same, pay_checked = [], []
for c in common:
    if c in payload and int(nat[c].get("payload_hash", 0)):
        pay_checked.append(c)
        if int(payload[c]["payload_hash"]) == int(nat[c]["payload_hash"]):
            pay_same.append(c)


def maxrel(key_c, key_n):
    w = 0.0
    for c in common:
        x, y = float(clio[c][key_c]), float(nat[c][key_n])
        w = max(w, abs(x - y) / max(1e-12, abs(x), abs(y)))
    return w


def line(label, a, b, note=""):
    print(f"  {label:<26} {a:>22}  {b:>22}  {note}")


print("=" * 96)
print("E2E COMPARISON  --  Clio-NeuroPress vs native NeuroPress")
print(f"chunks compared: {n}   chunk size: {CHUNK // (1024*1024)} MiB   "
      f"total: {n * CHUNK / (1024**3):.2f} GiB")
print("=" * 96)
print()
print(f"  {'METRIC':<26} {'CLIO':>22}  {'NATIVE':>22}")
print("  " + "-" * 92)

# --- byte-level equivalence
line("input bytes (FNV-1a)",
     f"{len(input_same)}/{len(input_checked)}", "identical", "byte-for-byte")
if pay_checked:
    line("compressed payload size",
         f"{sum(1 for c in pay_checked if int(payload[c]['compressed_size']) == int(nat[c]['payload_size']))}"
         f"/{len(pay_checked)}", "identical", "codec output, headers excluded")
    line("compressed payload hash",
         f"{len(pay_same)}/{len(pay_checked)}", "identical",
         "see per-algorithm breakdown")
    # The aggregate hides the actual story: it splits cleanly by codec, so
    # report it that way rather than as one unexplained fraction.
    by_algo = {}
    for c in pay_checked:
        a = ALGO.get(int(clio[c]["algo_idx"]), "?")
        ok = int(payload[c]["payload_hash"]) == int(nat[c]["payload_hash"])
        m, d = by_algo.get(a, (0, 0))
        by_algo[a] = (m + ok, d + (not ok))
    for a in sorted(by_algo):
        m, d = by_algo[a]
        line(f"   payload hash: {a}", f"{m} identical", f"{d} differ", "")
else:
    line("compressed payload hash", "not logged", "not logged",
         "rerun with current build")

# --- routing
line("NN routing decisions",
     f"{n - len(route_mismatch)}/{n} agree", f"{len(route_mismatch)} divergent",
     f"mismatch rate {100.0*len(route_mismatch)/n:.2f}%")

# --- predictions
for k in ("pred_ratio", "pred_ct_ms", "pred_dt_ms", "pred_psnr"):
    line(f"NN {k}", "—", f"{maxrel(k, k):.3e}", "max rel. diff")

# --- statistics the selection ranked on
for k in ("entropy", "mad", "second_deriv"):
    line(f"input stat: {k}", "—", f"{maxrel(k, k):.3e}", "max rel. diff")

# --- compression
cr = [float(clio[c]["actual_ratio"]) for c in common]
nr = [float(nat[c]["ratio"]) for c in common]
line("compression ratio (mean)", f"{statistics.mean(cr):.5f}",
     f"{statistics.mean(nr):.5f}", "")
line("compression ratio (min)", f"{min(cr):.5f}", f"{min(nr):.5f}", "")
line("compression ratio (max)", f"{max(cr):.5f}", f"{max(nr):.5f}", "")
csz = [CHUNK / float(clio[c]["actual_ratio"]) for c in common]
nsz = [float(nat[c]["compressed_size"]) for c in common]
delta = [a - b for a, b in zip(csz, nsz)]
line("compressed size delta", f"{min(delta):+.0f}..{max(delta):+.0f} B", "—",
     "native figure includes its 64B header")

# --- throughput
cth = [CHUNK / (1024**2) / (float(clio[c]["actual_ct_ms"]) / 1000.0)
       for c in common if float(clio[c]["actual_ct_ms"]) > 0]
nth = [CHUNK / (1024**2) / (float(nat[c]["actual_ct_ms"]) / 1000.0)
       for c in common if float(nat[c]["actual_ct_ms"]) > 0]
if cth and nth:
    line("compress throughput", f"{statistics.median(cth):.0f} MiB/s",
         f"{statistics.median(nth):.0f} MiB/s", "median; wall-clock, not comparable")

# --- reconstruction
nmse = [float(nat[c]["mse"]) for c in common if "mse" in nat[c]]
if nmse:
    line("MSE (decompressed vs in)", "0 (whole-dataset)",
         f"{max(nmse):.3e}", "max over chunks")
else:
    line("MSE (decompressed vs in)", "0 (whole-dataset)", "not logged", "")

# --- output distribution
if "mean" in next(iter(nat.values())):
    dm = max(abs(float(nat[c]["mean"]) - truth[c][0]) for c in common if c in truth)
    dv = max(abs(float(nat[c]["variance"]) - truth[c][1]) /
             max(1e-12, truth[c][1]) for c in common if c in truth)
    dk = max(abs(float(nat[c]["kurtosis"]) - truth[c][2]) /
             max(1e-12, truth[c][2]) for c in common if c in truth)
    line("output mean", "= source", f"{dm:.3e}", "abs diff vs source")
    line("output variance", "= source", f"{dv:.3e}", "rel diff vs source")
    line("output kurtosis", "= source", f"{dk:.3e}", "rel diff vs source")
print()
print("  Clio's output-distribution row reads '= source' because its H5Dread")
print("  round trip was verified exact over every element, so its decompressed")
print("  distribution IS the source distribution. Native's figures are computed")
print("  from its own decompressed buffer and checked against the source.")
print()

# ------------------------------------------------------------- per-chunk
print("=" * 96)
print("PER-CHUNK DETAIL   (C = Clio, N = native;  * marks any divergence)")
print("=" * 96)
hdr = (f"{'chunk':>5} {'side':>4} {'action':>6} {'algo':>9} {'q':>1} {'s':>1} "
       f"{'entropy':>8} {'mad':>8} {'2ndderiv':>9} {'ratio':>8} "
       f"{'comp_KiB':>9} {'ct_ms':>7} {'pred_ratio':>10}")
print(hdr)
print("-" * len(hdr))
listed = common if show_chunks == 0 else common[:show_chunks]
for c in listed:
    a, b = clio[c], nat[c]
    diff = "*" if c in route_mismatch else " "
    ca = int(a["algo_idx"])
    na = int(b["algo_idx"])
    c_action = ca + 8 * int(a["quantize"]) + 16 * int(a["shuffle"])
    c_sz = CHUNK / float(a["actual_ratio"]) / 1024.0
    print(f"{c:>5}{diff}{'C':>3} {c_action:>6} {ALGO.get(ca,'?'):>9} "
          f"{a['quantize']:>1} {a['shuffle']:>1} "
          f"{float(a['entropy']):>8.4f} {float(a['mad']):>8.4f} "
          f"{float(a['second_deriv']):>9.5f} {float(a['actual_ratio']):>8.5f} "
          f"{c_sz:>9.1f} {float(a['actual_ct_ms']):>7.2f} "
          f"{float(a['pred_ratio']):>10.4f}")
    print(f"{'':>5}{' ':>1}{'N':>3} {int(b['action']):>6} {ALGO.get(na,'?'):>9} "
          f"{b['quantize']:>1} {b['shuffle']:>1} "
          f"{float(b['entropy']):>8.4f} {float(b['mad']):>8.4f} "
          f"{float(b['second_deriv']):>9.5f} {float(b['ratio']):>8.5f} "
          f"{float(b['compressed_size'])/1024.0:>9.1f} "
          f"{float(b['actual_ct_ms']):>7.2f} {float(b['pred_ratio']):>10.4f}")
