#!/usr/bin/env bash
# Two-node runs of every benchmark that passes single-node, in sequence,
# at the same sizes the single-node runs used (each benchmark splits its
# problem across --nodes, so per-node work only shrinks).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHES=(
  "gmx:--page-kb 128"
  "lbann:"
  "weights:--repeat 1"
  "grayscott:--data-mb 512 --hbm-mb 128 --repeat 1 --steps 2"
  "kmeans:--data-mb 256 --hbm-mb 128"
)
for b in "${BENCHES[@]}"; do
  name=${b%%:*}
  args=${b#*:}
  "${HERE}/submit_2n_aurora.sh" "${name}" "${name}_2n" ${args}
done
echo "ALL DONE"
