#!/usr/bin/env bash
# The ONE place paths live for the external-baseline campaign. Every script
# here sources it; nothing else hardcodes a path. Override any of these in the
# environment (or in a site file, see SITE below) -- the defaults describe a
# checkout with sibling build/env directories, not one particular user.
#
#   REPO        clio-core checkout            (derived from this file)
#   BUILD       a build with the codecs on    ($REPO/build)
#   NPENV       prefix holding cusz/cuszp/ndzip/np  ($HOME/np-env)
#   WORK        where runs and results land   ($PWD/np-baselines)
#
# Field dumps are NOT here: every consumer of *_FIELDS sources
# ../site.sh instead. This file exists for install_codecs.sh.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
NPENV=${NPENV:-$HOME/np-env}
WORK=${WORK:-$PWD/np-baselines}

# Optional site file for machine-specific overrides, kept out of git.
SITE=${SITE:-$(dirname "${BASH_SOURCE[0]}")/site.sh}
# shellcheck source=/dev/null
[ -f "$SITE" ] && . "$SITE"

# Runtime library path: the build, the dependency prefix, and each external
# codec. A codec whose .so is missing here fails as a load error at run time,
# not as a clean message.
export LD_LIBRARY_PATH="$BUILD/bin:$NPENV/np/lib:$NPENV/cusz/lib64:$NPENV/cuszp/lib64:$NPENV/ndzip/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export REPO BUILD NPENV WORK
