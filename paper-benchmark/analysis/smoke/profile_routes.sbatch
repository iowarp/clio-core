#!/usr/bin/env bash
#SBATCH --job-name=np-profile-routes
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=/projects/bekn/imuradli/np-profile/slurm-%j.out
#
# MEASURE the data path of every workload route instead of reading its
# comments. Each route runs under `nsys profile -t cuda`, and
# `nsys stats --report cuda_gpu_mem_size_sum` then reports how many bytes
# actually crossed PCIe in each direction.
#
# The discriminator is simple and does not depend on any claim in the source:
#
#   HtoD ~= payload   the payload was UPLOADED -> it was not device-resident
#   HtoD ~= 0         the payload was produced on the GPU and never left
#   DtoH ~= payload   the payload was pulled back -> an extra copy exists
#   DtoH ~= stored    only the compressed bytes came back -> minimal
#
# EXPLORATION AND QUALITY ARE TURNED OFF for these runs (--explore-k 0,
# MEASURE_DT=0, MEASURE_QUALITY=0). Both do their own device work -- a
# decompress per candidate, and a decompress-plus-inverse for quality -- and
# their copies would swamp the staging traffic this is trying to isolate. The
# question here is where the PAYLOAD travels between the simulation and the
# compressor, not what the sweep costs.
set -uo pipefail

REPO=/u/imuradli/clio-core
PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export WARPX_BIN=/projects/bekn/imuradli/np-src/warpx/build-clio/bin/warpx.3d
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MEASURE_DT=0 MEASURE_QUALITY=0

ROOT=/projects/bekn/imuradli/np-profile
mkdir -p "$ROOT"; cd "$ROOT"
SUM=$ROOT/summary.csv
echo "route,workload,coupling,payload_bytes,htod_bytes,dtoh_bytes,htod_ops,dtoh_ops" > "$SUM"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# prof <name> <payload_bytes> -- everything after is the command.
prof() {
  local name=$1 payload=$2; shift 2
  say "== profiling $name"
  rm -f "$ROOT/$name.nsys-rep" "$ROOT/$name.sqlite"
  nsys profile -t cuda --force-overwrite true -o "$ROOT/$name" \
       --sample none --cpuctxsw none \
       "$@" > "$ROOT/$name.stdout" 2> "$ROOT/$name.stderr"
  local rc=$?
  [ $rc -ne 0 ] && { say "   run rc=$rc (profiling may still have data)"; tail -3 "$ROOT/$name.stderr" | sed 's/^/      /'; }
  nsys stats --report cuda_gpu_mem_size_sum --format csv \
       -o "$ROOT/${name}_mem" "$ROOT/$name.nsys-rep" > /dev/null 2>&1
  local f="$ROOT/${name}_mem_cuda_gpu_mem_size_sum.csv"
  [ -f "$f" ] || { say "   no memcpy report produced"; return; }
  python3 - "$f" "$name" "$payload" "$SUM" <<'PY'
import csv, sys
f, name, payload, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
h = d = 0.0; ho = do = 0
for r in csv.DictReader(open(f)):
    op = (r.get('Operation') or '').lower()
    tot = float(r.get('Total (MB)') or 0) * 1e6
    cnt = int(float(r.get('Count') or 0))
    if 'htod' in op or 'host-to-device' in op: h += tot; ho += cnt
    elif 'dtoh' in op or 'device-to-host' in op: d += tot; do += cnt
print(f"      payload {payload/1048576:9.1f} MiB")
print(f"      HtoD    {h/1048576:9.1f} MiB in {ho:6d} ops   ({h/payload if payload else 0:5.2f}x payload)")
print(f"      DtoH    {d/1048576:9.1f} MiB in {do:6d} ops   ({d/payload if payload else 0:5.2f}x payload)")
open(out,'a').write(f"{name},,,{payload},{int(h)},{int(d)},{ho},{do}\n")
PY
  echo
}

# ---------------------------------------------------------------------------
# 1. NYX -- dump & replay. Existing dumps; --max-files keeps it short.
# ---------------------------------------------------------------------------
NF=/scratch/bekn/imuradli/np-smoke/fields/nyx
if [ -d "$NF" ]; then
  P=$(( 12 * 8388608 ))
  prof nyx_replay "$P" env CLIO_NEUROPRESS_STAGE_H2D=1 \
      "$PB/nyx/run_config.sh" explore-balance --fields "$NF" --max-files 12 \
      --chunk 8388608 --bw 5e6 --eb 1e-3 --explore-k 0 --no-verify \
      --results "$ROOT/r" --tag nyx
fi

# ---------------------------------------------------------------------------
# 2/3. VPIC -- in situ, device pointers. Lossy has no --verify; lossless does.
#      The pair isolates what the digest copy costs.
# ---------------------------------------------------------------------------
VOX=$(( 64 * 64 * 64 )); P=$(( VOX * 4 * 16 * 2 ))
prof vpic_insitu_lossy "$P" \
    "$PB/vpic/run_config_insitu.sh" explore-balance --ncell 62 --steps 50 --int 25 \
    --chunk 1048576 --bw 5e6 --eb 1e-3 --explore-k 0 --results "$ROOT/r" --tag vpl
prof vpic_insitu_lossless_verify "$P" \
    "$PB/vpic/run_config_insitu.sh" explore-balance --ncell 62 --steps 50 --int 25 \
    --chunk 1048576 --bw 5e6 --explore-k 0 --verify --results "$ROOT/r" --tag vpv

# ---------------------------------------------------------------------------
# 4. LAMMPS -- in situ, device gather. run_config.sh ALWAYS passes --report,
#    which is the clause under test.
# ---------------------------------------------------------------------------
NAT=$(( 4 * 20 * 20 * 20 )); P=$(( NAT * 3 * 8 * 3 * 3 ))
prof lammps_insitu "$P" \
    "$PB/lammps/run_config.sh" explore-balance --box 20 --steps 50 --gap 25 \
    --require-device --chunk 1048576 --bw 5e6 --eb 1e-3 --explore-k 0 \
    --results "$ROOT/r" --tag lmp

# ---------------------------------------------------------------------------
# 5. WARPX -- VOL, in situ but host-assembled. Short run; the grid cannot be
#    shrunk (README), so steps are what gets cut.
# ---------------------------------------------------------------------------
P=$(( 64 * 64 * 512 * 4 * 10 * 2 ))
prof warpx_vol "$P" \
    "$PB/warpx/run_config.sh" explore-balance --ncell "64 64 512" --steps 8 --interval 4 \
    --stage-h2d --chunk 1048576 --bw 5e6 --eb 1e-3 --explore-k 0 \
    --deck /projects/bekn/imuradli/np-src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d \
    --results "$ROOT/r" --tag wxv
rm -rf "$ROOT/r/wxv/run/diags"

say "== summary"; column -s, -t "$SUM"
