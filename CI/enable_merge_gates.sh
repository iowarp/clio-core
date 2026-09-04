#!/usr/bin/env bash
# Turn on the merge gates for a shared branch: required status checks + a merge
# queue, expressed as a repository ruleset.
#
# WHY
#   dev and main have no protection at all today, so CI is advisory: broken code
#   merges freely and every subsequent push shows red until someone notices. The
#   expensive jobs (macFUSE, WinFsp, the boost backend, the wide macOS sweep)
#   are gated on `event_name != pull_request`, so they only ever spoke up AFTER
#   the merge. A merge queue runs that same full matrix on the PROSPECTIVE merge
#   commit, so a defect is caught while it is still the author's problem and dev
#   stays green by construction.
#
# ORDER MATTERS
#   Run this only AFTER the workflows carrying `merge_group:` triggers are on
#   the target branch. A merge queue whose required checks never fire on
#   merge_group events waits forever, and nothing merges. Verify first:
#     git show origin/dev:.github/workflows/ci-adapters.yml | grep merge_group
#
# USAGE
#   CI/enable_merge_gates.sh dev                       # dry run: print ruleset
#   CI/enable_merge_gates.sh dev --apply --no-queue    # checks only, safe today
#   CI/enable_merge_gates.sh dev --apply               # checks + merge queue
#   CI/enable_merge_gates.sh main --apply
#
# The required set is deliberately SMALL and fast. A required check that is
# slow or flaky teaches people to bypass the gate; one that is quick and
# trustworthy gets respected. Widen it only as jobs earn a clean record --
# `gh run list --branch dev` is the evidence.

set -euo pipefail

BRANCH="${1:?usage: enable_merge_gates.sh <branch> [--apply] [--no-queue]}"
REPO="${REPO:-iowarp/clio-core}"

APPLY=""
NO_QUEUE=""
shift
for arg in "$@"; do
  case "$arg" in
    --apply)    APPLY=1 ;;
    --no-queue) NO_QUEUE=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# --no-queue exists for the ordering problem: required checks are safe to turn
# on the moment they report on pull_request events, but the merge_queue rule is
# only safe once the workflows carrying `merge_group:` triggers are ON the
# target branch. Protect first, queue second, same script both times.

# Check names must match the JOB NAME exactly, and each must run on BOTH
# pull_request and merge_group events -- otherwise the queue stalls on a check
# that never reports.
#   cpplint                          cpplint.yml
#   build-test (unit amd64)          ci-linux.yml   (the only phase that runs
#                                    the core suite; PR + merge_group)
#   adapters (linux, all 5)          ci-adapters.yml (not event-gated)
#   vfd (linux, native write-through) ci-vfd.yml
CHECKS=(
  "cpplint"
  "build-test (unit amd64)"
  "adapters (linux, all 5)"
  "vfd (linux, native write-through)"
)

checks_json=$(printf '%s\n' "${CHECKS[@]}" \
  | python3 -c 'import json,sys; print(json.dumps([{"context": l.rstrip("\n")} for l in sys.stdin if l.strip()]))')

ruleset=$(python3 - "$BRANCH" "$checks_json" "${NO_QUEUE:-}" <<'PY'
import json, sys
branch, checks = sys.argv[1], json.loads(sys.argv[2])
no_queue = bool(sys.argv[3]) if len(sys.argv) > 3 else False
ruleset = {
  "name": f"{branch}: gated merges",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": [f"refs/heads/{branch}"], "exclude": []}},
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    # Require a PR, but do NOT require approvals: the goal here is to gate on
    # CI, and demanding reviews in the same change would stall a small team.
    {"type": "pull_request", "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": False,
        "require_code_owner_review": False,
        "require_last_push_approval": False,
        "required_review_thread_resolution": False}},
    # strict_required_status_checks_policy=False: do not force every PR to
    # rebase onto the tip before merging. The merge queue already builds the
    # prospective merge commit, which is the property that actually matters.
    {"type": "required_status_checks", "parameters": {
        "strict_required_status_checks_policy": False,
        "do_not_enforce_on_create": False,
        "required_status_checks": checks}},
  ],
}
if not no_queue:
    ruleset["rules"].append(
      {"type": "merge_queue", "parameters": {
          "merge_method": "SQUASH",
          "grouping_strategy": "ALLGREEN",
          "max_entries_to_build": 5,
          "min_entries_to_merge": 1,
          "max_entries_to_merge": 5,
          "min_entries_to_merge_wait_minutes": 5,
          # The full matrix runs about an hour at the tail (macOS sweep,
          # boost). A timeout shorter than the slowest gate silently drops
          # entries from the queue.
          "check_response_timeout_minutes": 90}})
print(json.dumps(ruleset, indent=2))
PY
)

if [ -z "$APPLY" ]; then
  echo "DRY RUN for $REPO branch '$BRANCH' -- re-run with --apply to create it."
  echo "$ruleset"
  exit 0
fi

existing=$(gh api "/repos/$REPO/rulesets" --jq \
  ".[] | select(.name==\"$BRANCH: gated merges\") | .id" 2>/dev/null || true)

if [ -n "$existing" ]; then
  echo "Updating existing ruleset $existing for '$BRANCH'..."
  printf '%s' "$ruleset" | gh api -X PUT "/repos/$REPO/rulesets/$existing" --input - >/dev/null
else
  echo "Creating ruleset for '$BRANCH'..."
  printf '%s' "$ruleset" | gh api -X POST "/repos/$REPO/rulesets" --input - >/dev/null
fi

echo "Done. Required checks on '$BRANCH':"
printf '  - %s\n' "${CHECKS[@]}"
if [ -n "$NO_QUEUE" ]; then
  echo "Merge queue: NOT enabled (--no-queue). Re-run without it once the"
  echo "  merge_group: triggers are on '$BRANCH'."
else
  echo "Merge queue: enabled (squash, all-green grouping)."
fi
