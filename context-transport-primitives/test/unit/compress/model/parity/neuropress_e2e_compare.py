#!/usr/bin/env python3
"""Compare Clio's per-chunk NeuroPress selections against native NeuroPress.

Third stage of the GPU -> HDF5 end-to-end check (issue #693). Full runbook:

  # 1. Clio: 1 GiB generated on the GPU, written through the HDF5 VOL in
  #    4 MiB chunks, NeuroPress selecting per chunk. Dumps the exact bytes
  #    it compressed and logs every selection.
  #    Drop CLIO_NEUROPRESS_E2E_LEARNING=0 for inference + online learning.
  CLIO_NEUROPRESS_E2E_LEARNING=0 \\
  CLIO_NEUROPRESS_SELECTION_LOG=/tmp/clio.csv bin/neuropress_e2e

  # 2. Native NeuroPress over the SAME bytes in the SAME order.
  #    Needs a NeuroPress build; it is not a Clio build dependency:
  #      cmake -S ~/NeuroPress -B <bld> -DNVCOMP_PREFIX=<prefix having
  #        include/nvcomp.hpp and lib/libnvcomp.so>
  #      cmake --build <bld> --target gpucompress -j8
  #      g++ -std=c++17 -O2 neuropress_native_replay.cc \\
  #        -I ~/NeuroPress/include -I /usr/local/cuda/include \\
  #        -L <bld> -lgpucompress -Wl,-rpath,<bld> -lcudart -o replay
  ./replay --weights .../model.nnwt --clio-csv /tmp/clio.csv \\
      --out /tmp/native.csv --inference-only

  # 3. Diff.
  ./neuropress_e2e_compare.py /tmp/clio.csv /tmp/native.csv

With learning off, selection is a pure function of each chunk's own
statistics, so the two sides must agree chunk for chunk. With learning on
they need not: see the ordering discussion in neuropress_native_replay.cc.
"""
import csv, sys, collections

clio_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/neuropress_e2e_clio.csv"
nat_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/neuropress_e2e_native.csv"

# Clio logs each chunk twice (write pass, then read-miss re-stage). Keep the
# first occurrence per chunk: that is the write-path selection.
clio = {}
with open(clio_path) as f:
    for r in csv.DictReader(f):
        c = int(r["blob"].rsplit("/chunk_", 1)[1])
        if c in clio:
            continue
        clio[c] = r

nat = {}
with open(nat_path) as f:
    for r in csv.DictReader(f):
        nat[int(r["chunk"])] = r

common = sorted(set(clio) & set(nat))
print(f"chunks: clio={len(clio)} native={len(nat)} compared={len(common)}\n")

sel_match = 0
algo_match = 0
mismatches = collections.Counter()
stat_err = {"entropy": 0.0, "mad": 0.0, "second_deriv": 0.0}
ratio_rel = []
examples = []

for c in common:
    a, b = clio[c], nat[c]
    ca = (int(a["algo_idx"]), int(a["quantize"]), int(a["shuffle"]))
    na = (int(b["algo_idx"]), int(b["quantize"]), int(b["shuffle"]))
    if ca == na:
        sel_match += 1
    else:
        mismatches[(ca, na)] += 1
        if len(examples) < 8:
            examples.append((c, ca, na, a["lib_name"]))
    if ca[0] == na[0]:
        algo_match += 1

    for k_c, k_n in (("entropy", "entropy"), ("mad", "mad"),
                     ("second_deriv", "second_deriv")):
        x, y = float(a[k_c]), float(b[k_n])
        d = abs(x - y) / max(1e-12, abs(x), abs(y))
        stat_err[k_c] = max(stat_err[k_c], d)

    cr, nr = float(a["actual_ratio"]), float(b["ratio"])
    if cr > 0 and nr > 0:
        ratio_rel.append(abs(cr - nr) / max(cr, nr))

n = len(common)
print(f"SELECTION (algo, quantize, shuffle)")
print(f"  exact match : {sel_match}/{n} ({100.0*sel_match/n:.1f}%)")
print(f"  algo only   : {algo_match}/{n} ({100.0*algo_match/n:.1f}%)\n")

# Input identity, measured rather than assumed. Both sides hash the chunk
# they actually handed to the compressor; a zero means the side could not
# hash it (Clio skips device-resident chunks rather than adding a staging
# copy to the storage path), which is reported as "not checked" rather than
# counted as agreement.
print("INPUT BYTES (FNV-1a over each chunk as compressed)")
if "checksum" in next(iter(clio.values())) and "checksum" in next(iter(nat.values())):
    both = [c for c in common
            if int(clio[c]["checksum"]) != 0 and int(nat[c]["checksum"]) != 0]
    same = sum(1 for c in both
               if int(clio[c]["checksum"]) == int(nat[c]["checksum"]))
    skipped = len(common) - len(both)
    print(f"  identical : {same}/{len(both)} chunks hashed on both sides")
    if skipped:
        print(f"  not checked: {skipped} (a side could not hash the chunk)")
else:
    print("  no checksum column -- rerun both sides with the current build")
print()

# Both sides now report the statistics the selection was actually made from:
# Clio via EstCompressionStats' out-params, native via
# gpucompress_compute_stats_gpu (gpucompress_compress_gpu leaves the fields
# in gpucompress_stats_t zeroed, so they are asked for explicitly).
native_stats_present = any(float(nat[c]["entropy"]) != 0.0 for c in common)
clio_stats_present = any(float(clio[c]["entropy"]) != 0.0 for c in common)
print("DATA STATISTICS (max relative difference)")
if not native_stats_present or not clio_stats_present:
    missing = []
    if not clio_stats_present:
        missing.append("clio")
    if not native_stats_present:
        missing.append("native")
    print(f"  NOT COMPARED: {' and '.join(missing)} logged zeros for every")
    print("  chunk, so there is nothing to diff here.")
else:
    for k, v in stat_err.items():
        print(f"  {k:13s}: {v:.3e}")
print()

if ratio_rel:
    ratio_rel.sort()
    print("ACHIEVED COMPRESSION (relative difference in ratio)")
    print(f"  median : {ratio_rel[len(ratio_rel)//2]:.4e}")
    print(f"  max    : {ratio_rel[-1]:.4e}\n")

# The NN's four outputs, which is where a porting error would show up before
# it ever reached a selection.
pred_worst = {}
for k in ("pred_ratio", "pred_ct_ms", "pred_dt_ms", "pred_psnr"):
    w = 0.0
    for c in common:
        x, y = float(clio[c][k]), float(nat[c][k])
        w = max(w, abs(x - y) / max(1e-12, abs(x), abs(y)))
    pred_worst[k] = w
print("NN PREDICTIONS (max relative difference)")
for k, v in pred_worst.items():
    print(f"  {k:11s}: {v:.3e}")
print()

if mismatches:
    print("SELECTION MISMATCHES (clio -> native), most common first")
    for (ca, na), k in mismatches.most_common(10):
        print(f"  {k:4d}x  algo/q/s {ca} -> {na}")
    print("\n  examples:")
    for c, ca, na, name in examples:
        print(f"    chunk {c:3d}: clio {ca} ({name})  native {na}")
