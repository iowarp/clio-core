#!/usr/bin/env bash
# Ares reliability + tiering performance suite: one command, results you can
# compare across major updates.
#
#   CI/ares/run.sh --build <build dir> [--suite reliability|perf|all|smoke]
#                  [--nodes 6] [--out DIR] [--baseline DIR] [--partition P]
#                  [--time HH:MM:SS] [--threshold 0.20]
#
# Outside a SLURM allocation it submits itself (sbatch --wait, exclusive
# nodes) and waits; inside one (salloc / sbatch) it runs on the allocation.
# Every leg is a clio_fs_suite run with real daemons on every node; see
# CI/ares/README.md for what each leg covers. Results land in --out:
#   <out>/<leg>/{results.jsonl,summary.md,suite.log,logs/}
#   <out>/report.md   (compare.py: failures, regressions vs --baseline)
#   <out>/meta.json   (git sha, nodes, legs, start/end)
# Exit status: 0 = every test passed and nothing regressed.
set -uo pipefail

# sbatch runs a COPY of this script from its spool directory, so the job
# learns where the checkout is from CLIO_ARES_HERE (set before submitting).
HERE="${CLIO_ARES_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "${HERE}/../.." && pwd)"
SUITE_DIR="${REPO}/context-transfer-engine/test/integration/clio_fs_suite"

BUILD="" SUITE="all" NODES=6 OUT="" BASELINE="" PARTITION="compute"
TIME="24:00:00" THRESHOLD="0.20" WORKING_SET_MB=4096 EXCLUDE="" NODELIST="" REL_DISK_GB=13
# Node screening before submit: idle nodes whose /mnt/nvme has less than
# MIN_NVME_GB free, or whose /mnt/ssd or /mnt/hdd cannot be read (a dead
# drive), are excluded automatically. 0 disables.
MIN_NVME_GB=30
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --working-set-mb) WORKING_SET_MB="$2"; shift 2 ;;
    --exclude) EXCLUDE="$2"; shift 2 ;;
    --nodelist) NODELIST="$2"; shift 2 ;;
    --rel-disk-gb) REL_DISK_GB="$2"; shift 2 ;;
    --min-nvme-gb) MIN_NVME_GB="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "${BUILD}" ] || { echo "--build <build dir> is required" >&2; exit 2; }
BUILD="$(cd "${BUILD}" && pwd)" || exit 2
[ -x "${BUILD}/bin/clio_run" ] || { echo "${BUILD}/bin/clio_run missing: build first" >&2; exit 2; }
SHA="$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
[ -n "${OUT}" ] || OUT="${HOME}/clio_ares_runs/$(date +%Y%m%d-%H%M%S)-${SHA}-${SUITE}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"

# ---- not inside an allocation: submit ourselves and wait -------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "submitting ${NODES}-node job on '${PARTITION}' (output: ${OUT})"
  args=(--build "${BUILD}" --suite "${SUITE}" --nodes "${NODES}" --out "${OUT}"
        --threshold "${THRESHOLD}" --working-set-mb "${WORKING_SET_MB}"
        --rel-disk-gb "${REL_DISK_GB}")
  [ -n "${BASELINE}" ] && args+=(--baseline "${BASELINE}")
  # --exclude: nodes to keep out of the allocation, e.g. ones whose shared
  # NVMe is too full for the tiers (the suite's deploy preflight names them).
  # Screen idle nodes (shared cluster: other users fill /mnt/nvme, and dead
  # drives return EIO) so the allocation never lands on one the deploy
  # preflight would reject -- that fails every leg of the run.
  if [ "${MIN_NVME_GB}" -gt 0 ] && [ -z "${NODELIST}" ]; then
    idle="$(sinfo -p "${PARTITION}" -h -t idle -o '%N')"
    if [ -n "${idle}" ]; then
      nidle="$(scontrol show hostnames "${idle}" | wc -l)"
      unfit="$(timeout 240 srun -p "${PARTITION}" -w "${idle}" -N "${nidle}" -t 4 bash -c '
          h=$(hostname -s)
          nv=$(df -BG --output=avail /mnt/nvme 2>/dev/null | tail -1 | tr -dc 0-9)
          for d in /mnt/ssd /mnt/hdd; do ls "$d" >/dev/null 2>&1 || { echo "$h $d unreadable"; exit; }; done
          [ "${nv:-0}" -lt '"${MIN_NVME_GB}"' ] && echo "$h /mnt/nvme ${nv:-0} GB free"
          true' 2>/dev/null)"
      if [ -n "${unfit}" ]; then
        echo "excluding unfit nodes:"; echo "${unfit}" | sed 's/^/  /'
        bad="$(echo "${unfit}" | awk '{print $1}' | paste -sd,)"
        EXCLUDE="${EXCLUDE:+${EXCLUDE},}${bad}"
      fi
    fi
  fi
  xargs_=()
  [ -n "${EXCLUDE}" ] && xargs_=(--exclude "${EXCLUDE}")
  [ -n "${NODELIST}" ] && xargs_+=(-w "${NODELIST}")
  CLIO_ARES_HERE="${HERE}" sbatch "${xargs_[@]}" --export=ALL --wait --exclusive -N "${NODES}" -p "${PARTITION}" -t "${TIME}" \
         -J clio-ares-"${SUITE}" -o "${OUT}/slurm-%j.out" \
         "${HERE}/run.sh" "${args[@]}"
  rc=$?
  [ -f "${OUT}/report.md" ] && sed -n '1,40p' "${OUT}/report.md"
  exit $rc
fi

# ---- inside an allocation --------------------------------------------------
HOSTS="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n "${NODES}" | paste -sd,)"
NHOSTS="$(echo "${HOSTS}" | tr ',' '\n' | wc -l)"
echo "nodes (${NHOSTS}): ${HOSTS}"

# Tier capacities for one ares_tiers leg from DRAM:NVMe:SSD:HDD percentages
# of where each node's working set (WORKING_SET_MB, which the overflow,
# spill and hot-set tests write) ends up, e.g. tier_mix 25:25:25:25.
#   DRAM and NVMe hold their share of ONE copy of the working set, so
#   together they are smaller than it and the data has to move down.
#   SSD and HDD (in their ratio) take the rest of the durable footprint:
#   every fsynced byte is kept 3 times on the file tiers (flushed primary,
#   local replica, a peer's remote copy), and the overflow tests also
#   overwrite and add files, so 6 x the working set across NVMe + SSD + HDD
#   (4.5 x filled the HDD tier and ended in a short write, tiermix3).
# A plain fraction of one fixed budget cannot work: with DRAM-heavy mixes the
# slow tiers are too small to hold three copies of more than RAM, so data
# would never reach SSD/HDD (or the writes end in ENOSPC). Splitting the
# durable footprint over NVMe too makes NVMe hold everything (tiermix2).
tier_mix() {
  local ram nvme ssd hdd
  IFS=: read -r ram nvme ssd hdd <<< "$1"
  local w="${WORKING_SET_MB}"
  local fast=$(( w * nvme / 100 ))
  local rest=$(( w * 6 - fast ))
  local slow=$(( ssd + hdd ))
  local hdd_gb=$(( (rest * hdd / slow + 1023) / 1024 ))
  [ "${hdd_gb}" -ge 1 ] || hdd_gb=1
  echo "--working-set-mb ${w} --ram-mb $(( w * ram / 100 ))" \
       "--fast-mb ${fast} --ssd-mb $(( rest * ssd / slow ))" \
       "--disk-gb ${hdd_gb}"
}

# One leg = "name|profile|groups|extra suite args". Fault groups break the
# deployment on purpose, so each profile gets its own leg (fresh deploy).
# REL_DISK_GB sizes every reliability leg's disk tier (or safe array). The
# tests size their data from the tiers, so this is not what they measure;
# keeping it small lets nodes whose shared NVMe is mostly full take part
# (13 GB: ~16 GB free per node with the 1 GB fast tier and logs). The
# overflow tests write 1.5x RAM + fast and keep 3 copies of every fsynced
# byte plus rewrite garbage: with a 2 GB fast tier that filled a 13 GB disk
# tier (ENOSPC, reliability6), with 1 GB it is ~7 GB of 14.
LEGS_RELIABILITY=(
  "posix_dist|persistent|posix,dist,dirs,apps,cache|--disk-gb ${REL_DISK_GB}"
  "node_faults|persistent|fault,repl|--disk-gb ${REL_DISK_GB}"
  "stress_tiers|tiered|stress,cache|--ram-mb 512 --fast-mb 1024 --disk-gb ${REL_DISK_GB} --probe-silence-s 30"
  "safe_bdev|safe|safe|--ram-mb 512 --fast-mb 1024 --disk-gb ${REL_DISK_GB}"
)
# Tiering under different DRAM:NVMe:SSD:HDD fractions. Each mix runs the
# performance studies and the stress tests (writes 1.5x the fast tiers, so
# data is forced down to SSD/HDD and migrated back); the balanced and
# HDD-heavy mixes also take the aimed node faults while data lives on the
# slow tiers.
# Load-only legs run with the runtime's 30 s failure window: under this
# load a node's workers saturate and it can send nothing for >10 s, and the
# fault legs' 10 s window declared it dead and failed fsyncs EIO. The
# aimed node faults (repl) keep the fast 10 s window, in their own legs.
LEGS_TIERMIX=(
  "tiermix_dram|ares_tiers|tier,stress|$(tier_mix 40:30:20:10) --probe-silence-s 30"
  "tiermix_balanced|ares_tiers|tier,stress|$(tier_mix 25:25:25:25) --probe-silence-s 30"
  "tiermix_balanced_faults|ares_tiers|repl|$(tier_mix 25:25:25:25)"
  "tiermix_hdd|ares_tiers|tier,stress|$(tier_mix 5:10:15:70) --probe-silence-s 30"
  "tiermix_hdd_faults|ares_tiers|repl|$(tier_mix 5:10:15:70)"
)
LEGS_PERF=(
  "tier4_ares|ares_tiers|tier,perf|--ram-mb 2048 --fast-mb 4096 --ssd-mb 8192 --disk-gb 32 --probe-silence-s 30"
  "tier_default|persistent|tier|--probe-silence-s 30"
)
LEGS_SMOKE=(
  "smoke|ares_tiers|tier|--ram-mb 256 --fast-mb 512 --ssd-mb 1024 --disk-gb 4 --only tier_small_io,tier_producer_consumer"
)
case "${SUITE}" in
  reliability) LEGS=("${LEGS_RELIABILITY[@]}") ;;
  tiermix) LEGS=("${LEGS_TIERMIX[@]}") ;;
  perf) LEGS=("${LEGS_PERF[@]}") ;;
  all) LEGS=("${LEGS_RELIABILITY[@]}" "${LEGS_TIERMIX[@]}" "${LEGS_PERF[@]}") ;;
  smoke) LEGS=("${LEGS_SMOKE[@]}") ;;
  *) echo "unknown --suite ${SUITE}" >&2; exit 2 ;;
esac

python3 - "${OUT}/meta.json" "${SHA}" "${HOSTS}" "${SUITE}" <<'EOF'
import json, sys, time
json.dump({'sha': sys.argv[2], 'hosts': sys.argv[3].split(','),
           'suite': sys.argv[4], 'start': time.strftime('%F %T')},
          open(sys.argv[1], 'w'), indent=1)
EOF

# Unit/integration ctest for CTE core, cache, replication and safe_bdev on
# the first node (single-node component checks before the cluster legs).
if [ "${SUITE}" = reliability ] || [ "${SUITE}" = all ]; then
  mkdir -p "${OUT}/ctest_components"
  first="${HOSTS%%,*}"
  echo "== ctest components on ${first}"
  # Tests that start their own runtime (restart, persist, stdio interceptor)
  # collide on its port and shared memory when run side by side -- they
  # failed in ~1 s under -j 4 and pass alone -- so they get a serial pass.
  SERIAL_RE='restart|persist'
  # The interceptor unit tests (stdio/mpiio/posix/vfd adapters) write fixed
  # /tmp/clio_cte_* paths, so on a shared node another user's leftover file
  # fails them (EPERM on remove). They are single-node API tests, not part
  # of the distributed evaluation: run them in the regular CI instead.
  ADAPTER_RE='clio_cte_(stdio|mpiio|posix|vfd)'
  srun -N1 -w "${first}" bash -c "cd '${BUILD}' && ctest -R 'safe_bdev|cte|replica|cache|cfs' -E '${SERIAL_RE}|${ADAPTER_RE}' -LE 'gpu|docker|manual|bench|mpi' \
      --timeout 900 -j 4 --output-junit '${OUT}/ctest_components/junit.xml' \
      > '${OUT}/ctest_components/ctest.log' 2>&1; \
      ctest -R '${SERIAL_RE}' -LE 'gpu|docker|manual|bench|mpi' --timeout 900 -j 1 \
      --output-junit '${OUT}/ctest_components/junit_serial.xml' \
      >> '${OUT}/ctest_components/ctest.log' 2>&1"
  python3 - "${OUT}/ctest_components" <<'EOF'
import json, os, sys, xml.etree.ElementTree as ET
d = sys.argv[1]
out = open(os.path.join(d, 'results.jsonl'), 'w')
cases = []
for jf in ('junit.xml', 'junit_serial.xml'):
  try:
    cases += list(ET.parse(os.path.join(d, jf)).getroot().iter('testcase'))
  except Exception as e:
    out.write(json.dumps({'name': f'ctest ({jf})', 'group': 'ctest',
                          'status': 'ERROR', 'error': f'no junit: {e}'}) + '\n')
for tc in cases:
  st = 'PASS'
  if tc.find('failure') is not None or tc.find('error') is not None:
    st = 'FAIL'
  elif tc.find('skipped') is not None:
    st = 'UNSUP'
  out.write(json.dumps({'name': tc.get('name'), 'group': 'ctest',
                        'status': st, 'secs': float(tc.get('time') or 0),
                        'metrics': {}}) + '\n')
EOF
fi

overall=0
for leg in "${LEGS[@]}"; do
  IFS='|' read -r name profile groups extra <<< "${leg}"
  echo "== leg ${name}: profile=${profile} groups=${groups} ${extra}"
  # shellcheck disable=SC2086
  python3 -B "${SUITE_DIR}/suite.py" --bin "${BUILD}/bin" --out "${OUT}/${name}" \
      --hosts "${HOSTS}" --profile "${profile}" --groups "${groups}" ${extra} \
      || overall=1
done

python3 - "${OUT}/meta.json" <<'EOF'
import json, sys, time
m = json.load(open(sys.argv[1])); m['end'] = time.strftime('%F %T')
json.dump(m, open(sys.argv[1], 'w'), indent=1)
EOF

cmp_args=("${OUT}" --threshold "${THRESHOLD}")
[ -n "${BASELINE}" ] && cmp_args+=(--baseline "${BASELINE}")
python3 -B "${HERE}/compare.py" "${cmp_args[@]}" || overall=1
echo "results: ${OUT}  (report: ${OUT}/report.md)"
exit ${overall}
