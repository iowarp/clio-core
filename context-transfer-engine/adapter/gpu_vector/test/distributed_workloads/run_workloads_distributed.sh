#!/bin/bash
# Distributed (2-node docker) run of a paged workload benchmark.
#
#   ./run_workloads_distributed.sh kmeans
#   ./run_workloads_distributed.sh weights
#   ./run_workloads_distributed.sh gnn
#   ./run_workloads_distributed.sh all
#
# Requires: nvidia container toolkit; the repo built into build/ (or set
# BUILD_DIR); the iowarp/deps-cpu image.
#
# WHAT THIS GATES. Not just "it ran": each workload's distributed result is
# compared against its own SINGLE-NODE reference, because these benches all
# report a checksum whose whole purpose is to be configuration-independent.
# A 2-node run that merely exits 0 proves nothing -- the failure mode these
# workloads actually have is a plausible, wrong number.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"

# TWO PATHS TO THE SAME CHECKOUT, and they are not always the same string.
#
# REPO_ROOT is where this script reads and writes: the binaries it runs for
# the single-node reference, the build scripts it invokes. HOST_WORKSPACE is
# the bind SOURCE handed to the docker daemon, which resolves paths in the
# DAEMON's filesystem, not in ours.
#
# On a developer host those are identical and this is all a no-op. Inside the
# repo's own devcontainer they are not: /workspace here is
# /home/<user>/.../core on the host, and binding "/workspace" hands the daemon
# a path that does not exist. Docker then creates it as an empty directory and
# every node dies with
#
#   sh: 1: /workspace/<build>/bin/clio_<wl>_paged_newcoro: not found
#   == gvw-node1 exited 127
#
# -- which reads as a missing binary rather than a missing mount, and the
# binary is right there. So translate: ask the daemon what THIS container's
# mounts are and rewrite the path through the one that contains us.
gvw_daemon_path() {   # gvw_daemon_path <path here> -> the same path, as the daemon sees it
  local p="$1" src dst best_src="" best_dst=""
  [ -f /.dockerenv ] || { echo "$p"; return; }
  while IFS='|' read -r src dst; do
    [ -n "$dst" ] || continue
    case "$p" in
      "$dst"|"$dst"/*)
        # Longest destination wins: /workspace/sub beats /workspace.
        if [ ${#dst} -gt ${#best_dst} ]; then best_dst="$dst"; best_src="$src"; fi ;;
    esac
  done < <(docker inspect "$(hostname)" \
             --format '{{range .Mounts}}{{.Source}}|{{.Destination}}{{"\n"}}{{end}}' 2>/dev/null)
  if [ -n "$best_dst" ]; then echo "${best_src}${p#"$best_dst"}"; else echo "$p"; fi
}
export HOST_WORKSPACE="${HOST_WORKSPACE:-$(gvw_daemon_path "$REPO_ROOT")}"
if [ "$HOST_WORKSPACE" != "$REPO_ROOT" ]; then
  echo "== docker-outside-of-docker: binding $HOST_WORKSPACE (this is $REPO_ROOT)"
elif [ -f /.dockerenv ]; then
  # In a container, but the translation found nothing to translate through --
  # `docker inspect $(hostname)` only works while the hostname is still the
  # container id, and a compose-set hostname breaks that. Say so now: the
  # alternative is four nodes dying with "not found" on a binary that exists,
  # which is a much longer walk to the same conclusion.
  echo "== WARNING: running inside a container, but could not work out how the"
  echo "   docker daemon sees $REPO_ROOT, so it is being passed through as-is."
  echo "   If the nodes exit 127 with 'not found', set HOST_WORKSPACE to this"
  echo "   checkout's path on the DOCKER HOST and re-run."
fi
export HOST_UID=$(id -u) HOST_GID=$(id -g)
export GVW_NODES="${GVW_NODES:-2}"
SVCS="gvw-node1 gvw-node2"
if [ "$GVW_NODES" -ge 3 ]; then
  # Nodes 3 and 4 live behind a profile so a 2-node run does not start them.
  export COMPOSE_PROFILES=four
  # EVERY node must read the 4-entry hostfile, not just the new ones: a node
  # on the 2-entry file never learns the others exist.
  export GVW_CONF=gvw_conf4.yaml
  SVCS="$SVCS gvw-node3"
fi
[ "$GVW_NODES" -ge 4 ] && SVCS="$SVCS gvw-node4"
BIN_DIR="$REPO_ROOT/${BUILD_DIR:-build}/bin"

# Per-workload: the binary, the deck, and how to pull the gated number out of
# its summary line. The decks are small on purpose -- this harness is testing
# the cross-node path, not throughput.
deck() {
  WITNESS=""
  CONTROL_ENV=""
  case "$1" in
    kmeans)
      BENCH=clio_kmeans_paged_bench
      ARGS="--data-mb 64 --blocks 8 --iters 3 --repeat 1"
      # Float sums under atomicAdd are not bit-reproducible; this bench
      # documents ~1e-8 relative spread across page/block layouts, so the
      # comparison below is relative and NOT an equality test.
      KEY='centroid_checksum=([0-9.]+)'; TOL=1e-4 ;;
    weights)
      BENCH=clio_weights_paged_bench
      ARGS="--blocks 8 --pages 4 --repeat 1"
      # Integer accumulation commutes, so this one IS exact at any node count.
      KEY='checksum=([A-Z]+)'; TOL=exact
      # checksum=OK ALONE IS A VACUOUS GATE HERE. Each node compares its own
      # partial `got` against its own partial `want`, so a reduction that
      # silently did nothing still passes. checksum_total is the REDUCED,
      # whole-model number: it must differ from the single-node total, or the
      # nodes are not covering different shards / not combining at all.
      # Measured 2-node/1-node ratio for this deck: 2.00002 -- not 1.0 (no
      # reduction) and not exactly 2.0 (both nodes on the same shard).
      WITNESS='checksum_total=([0-9]+)' ;;
    gmx)
      BENCH=clio_gmx_paged_bench
      ARGS="--page-kb 32 --atoms 20000 --repeat 1"
      # Fixed-point science: CONSERVATION, MESH and GATHER are all bit-exact,
      # so this gate needs no tolerance at any node count. That also makes it
      # the loudest of the three -- a stale mesh page or a double-counted bin
      # fails outright rather than drifting.
      KEY='(ALL GATES PASS)'; TOL=exact ;;
    grayscott)
      BENCH=clio_grayscott_paged_bench
      ARGS="--blocks 8 --steps 4 --data-mb 64 --repeat 1"
      # A 3D stencil: the only workload here whose slabs must exchange a halo
      # every step. Its checksum has a tolerance (float, atomicAdd), so unlike
      # gmx it cannot fail loudly on a stale plane -- hence the negative
      # control below.
      # Now bit-exact at 2 nodes, so the tolerance is only here to absorb the
      # float summation order the bench documents -- it should never be needed.
      KEY='v_checksum=([0-9.]+)'; TOL=1e-9
      CONTROL_ENV='GS_NO_HALO=1' ;;
    lbann)
      BENCH=clio_lbann_paged_bench
      # --blocks 4, not 8: a node's W2 band must be a whole number of pages,
      # and at 4 nodes 8 blocks gives half a page per band. 4 blocks holds
      # for 1, 2 and 4 nodes alike.
      ARGS="--blocks 4"
      # Fixed reference: the bench trains against its own dense in-VRAM copy
      # and asserts bit-equality, so this gate has no tolerance at all.
      KEY='(ALL GATES PASS)'; TOL=exact ;;
    lammps_md)
      BENCH=clio_lammps_md_paged_bench
      # --lattice 28, not 20: a slab must END ON A PAGE BOUNDARY because the
      # cache transfers whole pages, and only nb a multiple of 16 gives a
      # z-plane that is a whole number of them. The bench refuses anything else
      # rather than corrupt quietly. (distributed_md_bench/ documents the same
      # constraint; this entry exists to cover the SYCL edition, which that
      # harness does not build.)
      ARGS="--md --lattice 28 --steps 20"
      KEY='E0=(-?[0-9.]+)'; TOL=1e-6 ;;
    gnn)
      BENCH=clio_gnn_paged_bench
      # 2-layer GraphSAGE: 128 pages per region split by ownership, every
      # vertex's neighbours drawn from the whole graph, so at N nodes about
      # (N-1)/N of the edges read a peer's page -- features in layer 1, the
      # peer's freshly written embeddings in layer 2.
      ARGS="--vertices 8192 --blocks 8 --slots 64"
      # No atomics, fixed summation order: bit-identical at any node count,
      # and the digest is an integer sum, so the reduction is exact too.
      KEY='logit_digest=([0-9]+)'; TOL=exact
      # 0 on one node. Nonzero proves the decomposition crosses nodes, which
      # is what makes the control below mean anything.
      WITNESS='remote_edges=([0-9]+)'
      # Skip peer-owned neighbours (still counted in the mean's divisor). The
      # digest must move, or the cross-node reads were not load-bearing.
      CONTROL_ENV='GNN_NO_REMOTE=1' ;;
    reput_stale)
      # Self-checking CTE probe: no single-node reference, no checksum math.
      # Runs both containers, each exits nonzero on a stale serve.
      BENCH=test_cte_reput_stale
      ARGS="--rounds 200 --burners 6"
      KEY=''; TOL=selfcheck ;;
    *) echo "unknown workload: $1" >&2; return 2 ;;
  esac
}

# GVW_VARIANT selects WHICH EDITION of each benchmark to run. All four are the
# same driver, the same decks and the same gates -- only how the device
# coroutines were lowered, and which compiler lowered them, differs -- so a
# divergence between editions is a real backend or lowering difference rather
# than a differently-configured run.
#
#   (unset)       clio_<wl>_paged_bench            nvcc/clang, co_await
#   sycl          clio_<wl>_paged_bench_sycl       DPC++,      co_await
#   newcoro       clio_<wl>_paged_newcoro          nvcc,       clio-coroc
#   newcoro_sycl  clio_<wl>_paged_newcoro_sycl     DPC++,      clio-coroc
#
# The newcoro editions are the reason this block grew: they are NOT CMake
# targets. Nothing in `cmake --build` produces them, because the transpile is
# a source-to-source pass that runs before the compiler, so without the
# on-demand build below a `ctest -L gv_dist_newcoro` on a fresh tree would
# find no binary at all -- and, worse, on a stale tree would silently gate a
# binary built from an older benchmark or an older transpiler.
case "${GVW_VARIANT:-}" in
  ""|sycl|newcoro|newcoro_sycl) ;;
  *) echo "unknown GVW_VARIANT: ${GVW_VARIANT}" >&2; exit 2 ;;
esac

# Both SYCL editions link libsycl from the DPC++ prefix, which is not in the
# deps image; the compose file mounts it.
case "${GVW_VARIANT:-}" in
  sycl|newcoro_sycl)
    export DPCPP_HOME="${DPCPP_HOME:-$HOME/opt/dpcpp}"
    if [ ! -d "$DPCPP_HOME/lib" ]; then
      echo "GVW_VARIANT=${GVW_VARIANT} but no DPC++ at $DPCPP_HOME; set DPCPP_HOME" >&2
      exit 2
    fi ;;
esac

bench_name() {   # deck() sets BENCH for the CUDA edition; adjust for the rest
  # Only a *_paged_bench has a newcoro edition. Anything else -- today just
  # test_cte_reput_stale -- passes through unchanged rather than being
  # rewritten into the name of a binary nobody ever built. run_one already
  # skips that case, so this is belt and braces, but a silent
  # "test_cte_reput_stale_paged_newcoro" would be a confusing thing to debug.
  local base="${1%_paged_bench}"
  case "${GVW_VARIANT:-}" in
    sycl)         echo "${1}_sycl" ;;
    newcoro)      [ "$base" = "$1" ] && echo "$1" || echo "${base}_paged_newcoro" ;;
    newcoro_sycl) [ "$base" = "$1" ] && echo "$1" || echo "${base}_paged_newcoro_sycl" ;;
    *)            echo "$1" ;;
  esac
}

# THE NEWCORO EDITIONS ARE BUILT ON DEMAND, by the same scripts a developer
# runs by hand, into the same bin/ the containers mount. Rebuilt on every
# invocation rather than only when missing: the pipeline has two inputs the
# binary's timestamp cannot see past -- the benchmark source and the
# transpiler itself -- and a gate that silently ran last week's lowering is
# worse than no gate. The build is a single TU and takes well under a minute.
#
#   GVW_NEWCORO_BUILD=0   skip it and use whatever is already in bin/
#   GVW_BUILD_CUDA_HOME   the toolkit the BUILD uses (nvcc + the transpiler's
#                         parse), when it must differ from the CUDA_HOME the
#                         compose file binds into the containers. Unset by
#                         default, which lets build_newcoro.sh take nvcc from
#                         PATH.
#   GVW_CUDA_BUILD_DIR    the CUDA tree whose compile_commands.json supplies
#                         the transpile's -I/-D set. Only needed for
#                         newcoro_sycl, where BUILD_DIR is the DPC++ tree and
#                         cannot answer that question.
BENCH_SRC_DIR="$REPO_ROOT/context-transfer-engine/adapter/gpu_vector/benchmark"
ensure_newcoro() {   # ensure_newcoro <workload>
  local wl="$1" script rc
  case "${GVW_VARIANT:-}" in
    newcoro)      script=build_newcoro.sh ;;
    newcoro_sycl) script=run_newcoro_sycl.sh ;;
    *) return 0 ;;
  esac
  if [ "${GVW_NEWCORO_BUILD:-1}" = 0 ]; then
    echo "  (GVW_NEWCORO_BUILD=0 -- using the binary already in bin/)"
    return 0
  fi
  echo "=== $wl [$GVW_VARIANT]: transpile + build"
  local log="/tmp/gvw_${wl}_${GVW_VARIANT}_build.log"
  rc=0
  if ! (
    cd "$BENCH_SRC_DIR" || exit 1
    # THE MOUNT'S CUDA_HOME IS NOT THE BUILD'S CUDA_HOME. Three different
    # requirements have been sharing one variable:
    #
    #   the compose bind   a toolkit the DAEMON can see (on a devcontainer
    #                      host that may be the only CUDA 13 it has)
    #   nvcc               must match the libraries the benches link
    #   the coroc parse    must be a CUDA this clang understands -- clang 18
    #                      cannot read the CUDA 13 headers at all
    #
    # Leaking the mount's value into the build makes EVERY transpile die on
    #   crt/math_functions.hpp: error: expected function body after function
    #   declarator   /  fatal error: 'texture_fetch_functions.h' file not found
    # which reads as a broken transpiler rather than a mis-pointed toolkit.
    # Let build_newcoro.sh discover its own (it takes nvcc from PATH and
    # parses against the same one) unless told otherwise.
    if [ -n "${GVW_BUILD_CUDA_HOME:-}" ]; then
      export CUDA_HOME="$GVW_BUILD_CUDA_HOME"
    else
      unset CUDA_HOME
    fi
    if [ "$GVW_VARIANT" = newcoro_sycl ]; then
      CLIO_SYCL_BUILD_DIR="$REPO_ROOT/${BUILD_DIR:-build}" \
      CLIO_BUILD_DIR="${GVW_CUDA_BUILD_DIR:+$REPO_ROOT/$GVW_CUDA_BUILD_DIR}" \
      NEWCORO_RUN=0 ./"$script" "$wl"
    else
      CLIO_BUILD_DIR="$REPO_ROOT/${BUILD_DIR:-build}" \
      NEWCORO_RUN=0 ./"$script" "$wl"
    fi
  ) > "$log" 2>&1; then rc=1; fi
  sed 's/^/  | /' "$log" | tail -6
  [ $rc -eq 0 ] || { echo "  BUILD FAILED (see $log)"; return 1; }
  return 0
}

# GVW_OOC=1 swaps every deck for an OUT-OF-CORE one: the working set exceeds
# the paged cache, so pages evict to the CTE RAM tier and refault -- while the
# nodes are ALSO exchanging across the wire. The resident decks above never
# touch this path, and it is where this codebase's historical bugs live
# (eviction undoing writes, stale refetch, hold-set livelock).
#
# Every deck was calibrated single-node and its evict count recorded; run_one
# REQUIRES evicts > 0 in both the reference and the distributed log, so a deck
# that silently stops evicting FAILS rather than quietly proving nothing.
ooc_deck() {
  REQUIRE_EVICTS=1
  case "$1" in
    kmeans)    ARGS="--data-mb 256 --blocks 8 --iters 3 --repeat 1 --slots 4" ;;   # 12288 evicts
    weights)   ARGS="--blocks 8 --pages 32 --slots 4 --repeat 1" ;;                # 449 evicts
    gmx)       ARGS="--page-kb 32 --atoms 20000 --repeat 1 --blocks 4 --cap 20" ;; # 248 evicts; cap floor = 4 holds/block x blocks
    # 512, not 256: --data-mb is the GLOBAL field, so at 2 nodes each slab is
    # half of it -- 256 gave a per-node slab whose sliding window fit the
    # cache and the distributed run evicted ZERO (the witness caught it).
    # 512 makes the per-node shape match the single-node one that evicts.
    grayscott) ARGS="--data-mb 512 --blocks 8 --steps 4 --repeat 1" ;;
    lbann)     ARGS="--blocks 4 --cap 24" ;;                                       # 870 evicts (~82 weight pages through 24)
    # lattice 28 (nb=16), NOT 32: the distributed split needs a z-plane that
    # is a whole number of pages, and nb=19's plane (nb^2*512 B) has no sane
    # power-of-two divisor -- the bench refuses it at --nodes 2. Pressure comes
    # from the PAGE SIZE instead: 32KB pages double the page count to 64
    # against 28 slots. --rowchunk 1 because a 6-row held span needs 48KB.
    lammps_md) ARGS="--md --lattice 28 --steps 10 --blocks 8 --slots 28 --page-kb 32 --rowchunk 1" ;; # 3313 evicts, E0 = documented -592121.595111
    # 24 slots x 8 blocks is fewer frames than the 256 pages each node
    # touches (its own X and H, plus nearly every peer page as a neighbour).
    gnn)       ARGS="--vertices 8192 --blocks 8 --slots 24" ;;                     # 185 evicts, same digest as resident
  esac
}

max_evicts() { grep -ohE 'evicts=[0-9]+' "$1" 2>/dev/null | sed 's/evicts=//' | sort -n | tail -1; }

extract() { sed -nE "s/.*${1}.*/\\1/p" "$2" | head -1; }

run_one() {
  local wl="$1"; REQUIRE_EVICTS=""; deck "$wl"
  # reput_stale is a self-checking CTE probe, not a paged benchmark: it has no
  # newcoro edition and is not waiting for one. Skipped rather than failed, so
  # `all` stays green under a newcoro variant.
  if [ "${TOL:-}" = selfcheck ] && [ "${GVW_VARIANT:-}" != "${GVW_VARIANT#newcoro}" ]; then
    echo "=== $wl: no newcoro edition (a CTE probe, not a paged bench) -- skipped"
    return 0
  fi
  ensure_newcoro "$wl" || return 1
  if [ "${TOL:-}" = selfcheck ]; then
    echo "=== $wl: 2-node self-checking probe"
    cd "$SCRIPT_DIR"; rm -f "$SCRIPT_DIR"/.done_* 2>/dev/null || true
    export GVW_BENCH="$BENCH" GVW_ARGS="$ARGS"
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    docker compose up -d gvw-node1 gvw-node2 >/dev/null 2>&1
    docker compose logs -f --no-color > "/tmp/gvw_${wl}_dist.log" 2>&1 &
    local lp=$!; local rc2=0 code
    for n in gvw-node1 gvw-node2; do code=$(docker wait "$n"); echo "  == $n exited $code"; [ "$code" -eq 0 ] || rc2=1; done
    kill "$lp" 2>/dev/null || true
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    grep -E "STALE SERVE|PASS:|FAIL:" "/tmp/gvw_${wl}_dist.log" | sed "s/^/  /" | head -4
    [ "$rc2" -eq 0 ] && echo "  $wl OK" || echo "  $wl FAILED"
    return $rc2
  fi
  # The OOC override replaces only the deck; keys, tolerances and the
  # reference-comparison logic are unchanged -- the reference run recomputes
  # its value from the same deck, so nothing here hardcodes a resident answer.
  [ -n "${GVW_OOC:-}" ] && ooc_deck "$wl"
  echo "=== $wl${GVW_VARIANT:+ [$GVW_VARIANT]}: single-node reference"
  local ref_log="/tmp/gvw_${wl}_ref.log"
  ( cd "$BIN_DIR" && LD_LIBRARY_PATH="${DPCPP_HOME:-/nonexistent}/lib:$BIN_DIR:$LD_LIBRARY_PATH" \
      ./"$(bench_name "$BENCH")" $ARGS ) > "$ref_log" 2>&1 || {
    echo "  reference run FAILED"; tail -5 "$ref_log"; return 1; }
  local ref; ref="$(extract "$KEY" "$ref_log")"
  echo "  reference: $ref"

  echo "=== $wl${GVW_VARIANT:+ [$GVW_VARIANT]}: ${GVW_NODES}-node distributed"
  cd "$SCRIPT_DIR"
  # A leftover barrier file would let a node skip the wait entirely.
  rm -f "$SCRIPT_DIR"/.done_* 2>/dev/null || true
  export GVW_BENCH="$(bench_name "$BENCH")" GVW_ARGS="$ARGS"
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  docker compose up -d $SVCS
  docker compose logs -f --no-color > "/tmp/gvw_${wl}_dist.log" 2>&1 &
  local logs_pid=$!
  local rc=0 code
  # Wait for EVERY node rather than the first exit: --abort-on-container-exit
  # turns a passing run into rc=143.
  for n in $SVCS; do
    code=$(docker wait "$n"); echo "  == $n exited $code"
    [ "$code" -eq 0 ] || rc=1
  done
  kill "$logs_pid" 2>/dev/null || true
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  # A workload whose multi-node path is deliberately refused is NOT a pass and
  # NOT a failure: it is an uncovered gap, and it has to read as one. Counting
  # it green would hide it; counting it red would make the suite permanently
  # noisy and train people to ignore it.
  if grep -q 'Refusing' "/tmp/gvw_${wl}_dist.log" 2>/dev/null; then
    echo "  $wl NOT YET SUPPORTED -- the bench refuses --nodes > 1:"
    grep -h 'ERROR:' "/tmp/gvw_${wl}_dist.log" | head -1 | sed 's/^/    /'
    UNSUPPORTED="$UNSUPPORTED $wl"
    return 0
  fi
  [ "$rc" -eq 0 ] || { echo "  DISTRIBUTED RUN FAILED"; tail -25 "/tmp/gvw_${wl}_dist.log"; return 1; }

  local got; got="$(extract "$KEY" "/tmp/gvw_${wl}_dist.log")"
  echo "  distributed: $got"
  if [ -z "$got" ]; then echo "  GATE FAIL: no checksum in the distributed log"; return 1; fi
  if [ "$TOL" = exact ]; then
    [ "$got" = "$ref" ] || { echo "  GATE FAIL: $got != $ref"; return 1; }
  else
    awk -v a="$ref" -v b="$got" -v t="$TOL" 'BEGIN{
      d = (a>b?a-b:b-a); r = (a!=0 ? d/(a<0?-a:a) : d);
      if (r > t) { printf "  GATE FAIL: rel %.3g > %s\n", r, t; exit 1 }
      printf "  GATE PASS: rel %.3g <= %s\n", r, t }' || return 1
  fi
  # PRESSURE WITNESS. An OOC gate that stopped evicting proves nothing: the
  # deck has quietly become resident and every eviction-path bug is invisible.
  if [ -n "${REQUIRE_EVICTS:-}" ]; then
    local rev dev
    rev="$(max_evicts "$ref_log")"; dev="$(max_evicts "/tmp/gvw_${wl}_dist.log")"
    echo "  evictions: 1-node=${rev:-0} distributed=${dev:-0}"
    if [ "${rev:-0}" -eq 0 ] || [ "${dev:-0}" -eq 0 ]; then
      echo "  GATE FAIL: the OOC deck did not evict -- the pressure this gate"
      echo "             exists to apply is not being applied"
      return 1
    fi
  fi
  # The load-bearing check: with a witness declared, the distributed value
  # must differ from the single-node one. Equality means the extra nodes
  # changed nothing, which is the failure this whole harness exists to catch.
  if [ -n "$WITNESS" ]; then
    local wref wgot
    wref="$(extract "$WITNESS" "$ref_log")"
    wgot="$(extract "$WITNESS" "/tmp/gvw_${wl}_dist.log")"
    echo "  witness: 1-node=$wref ${GVW_NODES}-node=$wgot"
    if [ -z "$wgot" ] || [ -z "$wref" ]; then
      echo "  GATE FAIL: witness missing"; return 1
    fi
    if [ "$wref" = "$wgot" ]; then
      echo "  GATE FAIL: witness identical -- the second node contributed"
      echo "             nothing, so the reduction is not load-bearing"
      return 1
    fi
  fi
  # NEGATIVE CONTROL. If disabling the exchange leaves the answer unchanged,
  # the exchange is not load-bearing and the pass above proves nothing. Only
  # run it after a pass, and only where a control is defined.
  # The control is meaningful only on the RESIDENT deck. Under OOC, eviction
  # invalidates every frame between steps anyway, so the refault fetches the
  # peer's published plane even with the generational demand disabled -- each
  # boundary blob has a single writer, so there is no wrong replica to be
  # served. Measured: at --data-mb 512 the control matches the real run
  # exactly, while the resident control still moves (36410.579344 vs
  # 36104.119147). Running it here would fail every correct OOC run.
  if [ -n "$CONTROL_ENV" ] && [ -z "${GVW_OOC:-}" ]; then
    rm -f "$SCRIPT_DIR"/.done_* 2>/dev/null || true
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    env $CONTROL_ENV docker compose up -d $SVCS >/dev/null 2>&1
    for n in $SVCS; do docker wait "$n" >/dev/null; done
    docker compose logs --no-color > "/tmp/gvw_${wl}_ctrl.log" 2>&1
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    local cgot; cgot="$(extract "$KEY" "/tmp/gvw_${wl}_ctrl.log")"
    echo "  control ($CONTROL_ENV): $cgot"
    if [ "$cgot" = "$got" ]; then
      echo "  GATE FAIL: control matches the real run -- the exchange is a no-op"
      return 1
    fi
  fi
  [ -n "$CONTROL_ENV" ] && [ -n "${GVW_OOC:-}" ] && \
    echo "  (control skipped: eviction invalidates frames regardless, see note)"
  echo "  $wl OK"
}

UNSUPPORTED=""
FAILED=""
TARGET="${1:-all}"
if [ "$TARGET" = all ]; then
  rc=0
  for wl in kmeans weights gmx grayscott lbann lammps_md gnn; do
    run_one "$wl" || { rc=1; FAILED="$FAILED $wl"; }
  done
  echo
  echo "=== summary"
  echo "  validated distributed: kmeans weights gmx grayscott lbann lammps_md gnn"
  [ -n "$UNSUPPORTED" ] && echo "  NOT YET SUPPORTED:    $UNSUPPORTED"
  [ -n "$FAILED" ] && echo "  FAILING:              $FAILED"
  exit $rc
fi
run_one "$TARGET"
