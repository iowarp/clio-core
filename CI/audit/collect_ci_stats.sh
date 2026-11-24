#!/usr/bin/env bash
# Reproduce the dev-CI audit behind issue #1035.
#
# Pulls the last N workflow runs on a branch, then the failed jobs inside them,
# and prints the two numbers that mattered: per-workflow failure rate, and the
# push-vs-pull_request split. The split is the point -- the expensive matrix
# legs are gated `event_name != pull_request`, so a PR can be green on a commit
# whose push run is red. Measuring them together hides that; measuring them
# apart is what showed CI Adapters at 0% on PRs and 33% on pushes.
set -euo pipefail
REPO=${REPO:-iowarp/clio-core}
BRANCH=${BRANCH:-dev}
OUT=${OUT:-ci-audit}
mkdir -p "$OUT"

gh api -X GET "repos/$REPO/actions/runs" -f branch="$BRANCH" -f per_page=100 --paginate \
  --jq '.workflow_runs[] | [.id,.name,.head_sha[0:8],.status,.conclusion,.created_at,.run_attempt,.event] | @tsv' \
  > "$OUT/runs.tsv"

echo "=== failure rate by workflow ==="
awk -F'\t' '$4=="completed"{t[$2]++; if($5=="failure")f[$2]++} END{for(w in t) printf "  %-40s total=%-4d fail=%-4d %.0f%%\n", w, t[w], f[w]+0, (f[w]*100.0/t[w])}' "$OUT/runs.tsv" | sort

echo; echo "=== push vs pull_request (the gate gap) ==="
awk -F'\t' '$4=="completed"&&($5=="success"||$5=="failure"){k=$2"|"$8; t[k]++; if($5=="failure")f[k]++} END{for(k in t){split(k,a,"|"); printf "  %-38s %-13s runs=%-4d fail=%-3d %.0f%%\n",a[1],a[2],t[k],f[k]+0,(f[k]*100.0/t[k])}}' "$OUT/runs.tsv" | sort

echo; echo "=== merge_group runs (should be >0 once the merge queue is on; see #1036) ==="
awk -F'\t' '$8=="merge_group"' "$OUT/runs.tsv" | wc -l

echo; echo "=== failing jobs, ranked ==="
awk -F'\t' '$4=="completed" && $5=="failure"{print $1"\t"$2}' "$OUT/runs.tsv" > "$OUT/failed-runs.tsv"
: > "$OUT/failed-jobs.tsv"
while IFS=$'\t' read -r id wf; do
  gh api "repos/$REPO/actions/runs/$id/jobs" --paginate \
    --jq ".jobs[] | select(.conclusion==\"failure\") | [\"$wf\",.name,(.id|tostring)] | @tsv" \
    >> "$OUT/failed-jobs.tsv" 2>/dev/null || true
done < "$OUT/failed-runs.tsv"
awk -F'\t' '{print $1" :: "$2}' "$OUT/failed-jobs.tsv" | sort | uniq -c | sort -rn
