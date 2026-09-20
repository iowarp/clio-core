#!/usr/bin/env bash
# Try a handful of IGC options on one SPIR-V image and report which compile.
#   igc_try_opts.sh <image.spv>
# For the one lammps_md kernel (Submit<LaunchBuildList>, 401 KB of SPIR-V)
# that still crashes IGC with inlining disabled.
set -u
SPV=${1:?image}
export IGC_FunctionControl=3
try() {  # try <label> <ocloc -options string>
  if /usr/bin/ocloc compile -file "$SPV" -spirv_input -device pvc \
       -options "$2" -output /tmp/igc_try_$$ -output_no_suffix > /tmp/igc_try_$$.log 2>&1
  then r=PASS; else r=CRASH; fi
  printf '%-5s %s\n' "$r" "$1"
}
try "opt-disable"   "-cl-opt-disable"
try "simd16"        "-igc_opts 'ForceOCLSIMDWidth=16'"
try "simd8"         "-igc_opts 'ForceOCLSIMDWidth=8'"
try "no-ra-opt"     "-igc_opts 'DisableRecompilation=1'"
try "simd16+optoff" "-cl-opt-disable -igc_opts 'ForceOCLSIMDWidth=16'"
rm -f /tmp/igc_try_$$ /tmp/igc_try_$$.log
echo "TRY-DONE"
