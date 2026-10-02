#!/usr/bin/env bash
#===============================================================================
# build_lossy.sh -- build one lossy_sweep_<codec> binary per GPU lossy codec,
# out of tree in $BUILD_DIR. Each binary links exactly one codec (several
# export clashing symbols). Libraries are found through RPATH.
#
#   build_lossy.sh              # every codec
#   build_lossy.sh fsz cuszp3   # just these
#
# Environment: NPENV ($HOME/np-env), SRC ($HOME/np-ext-src), BUILD_DIR
# ($HOME/np-build/lossy-sweep), ARCH (80).
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NPENV=${NPENV:-$HOME/np-env}
SRC=${SRC:-$HOME/np-ext-src}
BUILD_DIR=${BUILD_DIR:-$HOME/np-build/lossy-sweep}
ARCH=${ARCH:-80}
mkdir -p "$BUILD_DIR"
ALL=(fsz cuszp3 cuszp2 cuszi cuszhi pfpl fzgpu cuszx cuzfp)

# nv NAME SOURCES... -- FLAGS... : compile the harness plus SOURCES into
# $BUILD_DIR/lossy_sweep_NAME (written to a private name, then renamed).
nv() {
  local name=$1; shift
  local srcs=() flags=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do srcs+=("$1"); shift; done
  [ "${1:-}" = "--" ] && shift
  flags=("$@")
  local out=$BUILD_DIR/lossy_sweep_$name tmp=$BUILD_DIR/.lossy_sweep_$name.$$
  echo "== $name"
  if nvcc -O3 -std=c++17 -arch=sm_$ARCH --expt-relaxed-constexpr \
       -I"$HERE" "$HERE/lossy_sweep.cu" "${srcs[@]}" -o "$tmp" "${flags[@]}"; then
    mv -f "$tmp" "$out" && echo "   ok: $out"
  else
    rm -f "$tmp"; echo "   FAILED: $name"; return 1
  fi
}

# rp DIR... -- RPATH entries. --disable-new-dtags writes DT_RPATH, which (unlike
# DT_RUNPATH) also resolves the codec libraries' own dependencies on siblings.
rp() { local a="-Xlinker --disable-new-dtags "; for d in "$@"; do a+="-Xlinker -rpath=$d "; done; echo "$a"; }

build_one() {
  case "$1" in
    fsz)    nv fsz "$HERE/codec_fsz.cu" -- -I"$NPENV/fsz/include" \
              -L"$NPENV/fsz/lib64" -L"$NPENV/fsz/lib" -lfsz $(rp "$NPENV/fsz/lib64" "$NPENV/fsz/lib") ;;
    cuszp3) nv cuszp3 "$HERE/codec_cuszp.cu" -- -DCUSZP_V3 -I"$NPENV/cuszp/include" \
              -L"$NPENV/cuszp/lib64" -lcuSZp $(rp "$NPENV/cuszp/lib64") ;;
    cuszp2) nv cuszp2 "$HERE/codec_cuszp.cu" -- -I"$NPENV/cuszp2/include" \
              -L"$NPENV/cuszp2/lib64" -L"$NPENV/cuszp2/lib" -lcuSZp \
              $(rp "$NPENV/cuszp2/lib64" "$NPENV/cuszp2/lib") ;;
    pfpl)   nv pfpl "$HERE/codec_pfpl.cu" "$HERE/pfpl_comp.cu" "$HERE/pfpl_decomp.cu" -- \
              -fmad=false -I"$SRC/PFPL/src" ;;
    cuszi)  nv cuszi "$HERE/codec_cuszi.cu" -- -I"$NPENV/cusz019/include" \
              -I"$NPENV/cusz019/include/cusz/include" -L"$NPENV/cusz019/lib64" -lcusz \
              $(rp "$NPENV/cusz019/lib64") ;;
    cuszhi) nv cuszhi "$HERE/codec_cuszhi.cu" -- -DPSZ_USE_CUDA -I"$NPENV/cuszhi/include/cusz" \
              -include thrust/extrema.h -include thrust/transform_reduce.h \
              -L"$NPENV/cuszhi/lib64" -lcusz -lpszcomp_cu -lpszkernel_cu -lpszmem -lpszstat_cu \
              -lpszhf_cu -lpszspv_cu -lpsztime -lpszutils_seq -llc $(rp "$NPENV/cuszhi/lib64") ;;
    fzgpu)  nv fzgpu "$HERE/codec_fzgpu.cu" -- --extended-lambda -I"$SRC/FZ-GPU/src" \
              -I"$SRC/FZ-GPU/include" ;;
    cuszx)  nv cuszx "$HERE/codec_cuszx.cu" -- -I"$NPENV/szx-cuda-fix/include" \
              -L"$NPENV/szx-cuda-fix/lib64" -L"$NPENV/szx-cuda-fix/lib" -lSZx \
              $(rp "$NPENV/szx-cuda-fix/lib64" "$NPENV/szx-cuda-fix/lib") ;;
    cuzfp)  nv cuzfp "$HERE/codec_cuzfp.cu" -- -I"$NPENV/zfp-cuda/include" \
              -L"$NPENV/zfp-cuda/lib64" -lzfp $(rp "$NPENV/zfp-cuda/lib64") ;;
    *) echo "unknown codec $1"; return 1 ;;
  esac
}

rc=0
for c in "${@:-${ALL[@]}}"; do build_one "$c" || rc=1; done
exit $rc
