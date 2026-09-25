#!/usr/bin/env bash
# Produce the sanitized clio-core source tree every trial starts from.
#
#   ./sanitize.sh [out_dir]        default: $RUNS/base/clio-core
#
# "From scratch" means the agent cannot read, run, or recover any existing
# implementation of the workloads it is asked to build. So the copy drops:
#   - git history (a fresh single-commit repo is created instead)
#   - every workload under gpu_vector/benchmark/ (only the generic build
#     tooling survives), and the benchmark/pipeline/design docs that describe
#     them (eternia.md, PREFETCH.md, PAGING_DEFECT.md, SYCL_PORT.md)
#   - workload-specific tests and the distributed test harnesses
#   - jarvis packages and scratch dirs that wrap or record the workloads
#   - model weights, virtualenvs, docs site, result dumps and cargo targets
#     (size, and result dumps include workload outputs)
#   - project skills (.claude/) -- the arm under test adds its own
#   - this harness itself, and all build trees / in-source build junk
# Residual, accepted leakage (identical in every arm): comments in the
# gpu_vector and runtime headers mention the workloads by name, and the CTE
# core ships data organizers named after two of them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "${HERE}/../../../.." && pwd)"          # clio-core root
RUNS="${RUNS:-${SRC}/agent_eval_runs}"
OUT="${1:-${RUNS}/base/clio-core}"
GV=context-transfer-engine/adapter/gpu_vector

rm -rf "${OUT}"
mkdir -p "${OUT}"

rsync -a \
  --exclude='/.git' --exclude='/build*' --exclude='/wt' --exclude='/external' \
  --exclude='/agent_eval_runs' --exclude='/.claude' --exclude='/.claude-plugin' \
  --exclude='/jarvis_clio_core' --exclude='/.eternia_wl_work' --exclude='/.vscode' \
  --exclude='/models' --exclude='/venv' --exclude='/.venv' --exclude='/results' --exclude='/docs' \
  --exclude='target' --exclude='node_modules' --exclude='/context-transfer-engine/compressor/results' \
  --exclude='CMakeFiles' --exclude='CMakeCache.txt' --exclude='cmake_install.cmake' \
  --exclude='Testing' --exclude='CTestTestfile.cmake' --exclude='__pycache__' \
  --exclude="/${GV}/benchmark" --exclude="/${GV}/pipelines" --exclude="/${GV}/agent_eval" \
  --exclude="/${GV}/eternia.md" --exclude="/${GV}/PREFETCH.md" \
  --exclude="/${GV}/PAGING_DEFECT.md" --exclude="/${GV}/SYCL_PORT.md" \
  --exclude="/${GV}/gpu_vector_md.yaml" \
  --exclude="/${GV}/test/distributed*" --exclude="/${GV}/test/workloads" \
  --exclude="/${GV}/test/gnn_dataset.h" \
  --exclude="/${GV}/test/test_gpu_vector_workloads_gpu.cc" \
  --exclude="/${GV}/test/test_gpu_vector_gnn*_gpu.cc" \
  --exclude="/${GV}/test/test_gpu_vector_md_distributed_gpu.cc" \
  --exclude="/${GV}/test/test_gpu_vector_distributed_gpu.cc" \
  "${SRC}/" "${OUT}/"

# Generic benchmark tooling only: the build script and the register cap.
mkdir -p "${OUT}/${GV}/benchmark"
cp "${SRC}/${GV}/benchmark/build_newcoro.sh" "${SRC}/${GV}/benchmark/gv_launch_bounds.h" \
   "${OUT}/${GV}/benchmark/"
cat > "${OUT}/${GV}/benchmark/CMakeLists.txt" <<'EOF'
# Benchmarks are built outside CMake by build_newcoro.sh.
EOF

# Anything left that still names a removed file is reported, not silently kept.
if grep -rIl --exclude-dir=.git -E 'clio_(kmeans|grayscott|lbann|lammps_md)_paged' "${OUT}" \
     2>/dev/null | grep -v '\.md$' ; then
  echo "sanitize: WARNING -- files above still reference removed benchmarks" >&2
fi

# Fresh history: one commit, no way back to the originals.
git -C "${OUT}" init -q
git -C "${OUT}" add -A
git -C "${OUT}" -c user.name=sandbox -c user.email=sandbox@localhost \
  commit -q -m "sandbox snapshot"
echo "sanitized tree: ${OUT} ($(du -sh "${OUT}" | cut -f1))"
