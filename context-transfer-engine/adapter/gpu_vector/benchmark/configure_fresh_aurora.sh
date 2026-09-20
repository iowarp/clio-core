#!/usr/bin/env bash
# Configure a FRESH build of this worktree with the same options as an
# existing SYCL build, so a runtime library can be rebuilt from this branch.
#
#   configure_fresh_aurora.sh <existing-build-dir> <new-build-dir>
#
# The existing build (the main checkout's build/sycl) is on another branch;
# per the CRITICAL BUILD RULE it is never touched. Its user-facing cache
# entries are lifted into -D flags and everything else (find_package results)
# is recomputed in the same environment, which yields the same answers.
set -eu
OLD=${1:?existing build dir}
NEW=${2:?new build dir}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CMAKE=${CMAKE:-$(ls /opt/aurora/*/spack/unified/*/install/linux-x86_64/cmake-*/bin/cmake 2>/dev/null | head -1)}
test -x "$CMAKE" || { echo "no cmake" >&2; exit 1; }

args=()
while IFS= read -r line; do
  args+=("-D$line")
# The <Package>_DIR entries are kept: they are where the old build found each
# dependency's config file, and without them find_package(yaml-cpp) has nothing
# to go on -- the spack prefixes are not on any default search path.
done < <(grep -E "^(CLIO_[A-Z_]+|CTP_ENABLE_[A-Z_]+|SYCL_TARGET|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_C_COMPILER|CMAKE_CXX_FLAGS|CMAKE_PREFIX_PATH|BUILD_TESTING|[A-Za-z0-9_-]+_DIR):[A-Z]+=" "$OLD/CMakeCache.txt" \
        | grep -vE "_INCLUDE_DIR|_LIBRARY|ADVANCED|-STRINGS|_BINARY_DIR|_SOURCE_DIR|CMAKE_HOME_DIRECTORY")
# Tests off: only the runtime library is wanted from this tree.
args+=("-DBUILD_TESTING:BOOL=OFF" "-DCLIO_CORE_ENABLE_TESTS:BOOL=OFF")

echo "configuring $NEW from $SRC with ${#args[@]} options"
"$CMAKE" -S "$SRC" -B "$NEW" "${args[@]}"
