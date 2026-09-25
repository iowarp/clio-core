#!/usr/bin/env bash
# Run one agent trial: one model, one task, one skill arm, one repetition.
#
#   ./run_trial.sh <model> <task> <arm> <rep>
#     model  claude-opus-5-5 | claude-sonnet-5 | claude-haiku-4-5-20251001
#     task   kmeans | grayscott | lbann | md       (tasks/<task>/TASK.md)
#     arm    none     no skill
#            skill    the gpu-vector skill (general lessons only; it names no
#                     workload and contains no worked implementation)
#     rep    integer
#
# The CLI is pinned: $RUNS/tools/claude (copied from the harness host), never
# whatever `claude` the image happens to have on PATH -- an old one rejects
# newer models outright.
#
# Background tasks are disabled: headless `claude -p` exits when the main
# turn ends and KILLS any subagent or shell still running in the background,
# silently truncating the trial (observed on the first pilot).
#
# Budget: TRIAL_TIMEOUT (default 4h) wall clock. The container is the trial's
# whole world: the sanitized tree (copied), a private copy of the prebuilt
# build, the transpiler, the GPU, and a fresh Claude config holding only the
# credentials -- no user settings, memory, plugins, or skills leak in.
#
# Output (trials/<task>/<arm>/<model>/r<rep>/):
#   transcript.jsonl  the agent's full stream-json transcript
#   meta.json         model/task/arm/rep, start/end, exit code, timeout
#   clio-core/        the tree as the agent left it (agent_task/ = deliverable)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

MODEL="${1:?model}"; TASK="${2:?task}"; ARM="${3:?arm}"; REP="${4:?rep}"
TIMEOUT="${TRIAL_TIMEOUT:-4h}"
TASKDIR="${HERE}/tasks/${TASK}"
test -f "${TASKDIR}/TASK.md" || { echo "no task ${TASK}" >&2; exit 2; }

REL="trials/${TASK}/${ARM}/${MODEL}/r${REP}"
T="${RUNS_HOST_VIEW}/${REL}"
TD="${RUNS_DOCKER_VIEW}/${REL}"
# Atomic claim: whoever creates the directory owns the trial. A directory
# that exists (finished, running, or abandoned) is never touched here.
mkdir -p "$(dirname "${T}")"
mkdir "${T}" 2>/dev/null || { echo "exists: ${T}" >&2; exit 3; }

# The tree and a private build copy (build_newcoro.sh writes into the build).
cp -a "${RUNS_HOST_VIEW}/base/clio-core" "${T}/clio-core"
cp -a "${RUNS_HOST_VIEW}/base/build" "${T}/build"

# The skill arm.
case "${ARM}" in
  none) ;;
  skill)
    D="${T}/clio-core/.claude/skills/gpu-vector"
    mkdir -p "${D}"
    cp "${SKILL_SRC}"/*.md "${D}/" ;;
  *) echo "bad arm ${ARM}" >&2; exit 2 ;;
esac

# A fresh, empty Claude config. Credentials come from a dedicated token, never
# a copy of the user's own login: a trial that refreshed a copied OAuth token
# could rotate it and sign the user out. Create the token once with
# `claude setup-token` and put CLAUDE_CODE_OAUTH_TOKEN=... (or ANTHROPIC_API_KEY=...)
# in $RUNS/secrets.env (mode 600).
SECRETS="${RUNS_HOST_VIEW}/secrets.env"
test -r "${SECRETS}" || { echo "missing ${SECRETS} (see header)" >&2; exit 2; }
mkdir -p "${T}/claude-config"
echo '{}' > "${T}/claude-config/settings.json"

PROMPT="$(cat "${TASKDIR}/TASK.md" "${HERE}/tasks/common.md")"
printf '%s\n' "${PROMPT}" > "${T}/prompt.md"

"${RUNS_HOST_VIEW}/tools/claude" --version > "${T}/cli_version"
sha256sum "${T}/prompt.md" | cut -c1-16 > "${T}/prompt_sha"
if [ -d "${T}/clio-core/.claude/skills/gpu-vector" ]; then
  (cd "${T}/clio-core/.claude/skills/gpu-vector" && cat *.md | sha256sum | cut -c1-16) > "${T}/skill_sha"
else echo none > "${T}/skill_sha"; fi
# Global concurrency: at most MAX_CONCURRENT trial containers on this host,
# whoever launched them. Checked and started under one lock so two launchers
# cannot both see a free slot.
MAXC="${MAX_CONCURRENT:-3}"
CNAME="agent_eval_${TASK}_${ARM}_${MODEL//[^a-zA-Z0-9]/_}_r${REP}"
exec 9>"${RUNS_HOST_VIEW}/.slots.lock"
flock 9
while [ "$(docker ps --format '{{.Names}}' | grep -c '^agent_eval_')" -ge "${MAXC}" ]; do
  sleep 20
done
START="$(date +%s)"
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  --shm-size=16g --ipc=private \
  --name "${CNAME}" \
  -v "${TD}/clio-core:/work/clio-core" \
  -v "${TD}/build:/work/build" \
  -v "${RUNS_DOCKER_VIEW}/base/build-coroc:/work/build-coroc:ro" \
  -v "${TD}/claude-config:/work/claude-config" \
  -v "${RUNS_DOCKER_VIEW}/tools/claude:/work/tools/claude:ro" \
  -e CLAUDE_CONFIG_DIR=/work/claude-config --env-file "${SECRETS}" \
  -e CLIO_BUILD_DIR=/work/build -e COROC=/work/build-coroc/clio-coroc \
  -e CUDA_HOME=/usr/local/cuda-12.9 \
  -e DISABLE_AUTOUPDATER=1 \
  -e CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 \
  -w /work/clio-core \
  "${IMAGE}" \
  timeout "${TIMEOUT}" /work/tools/claude -p "${PROMPT}" --model "${MODEL}" \
    --output-format stream-json --verbose --dangerously-skip-permissions \
  > "${T}/transcript.jsonl" 2> "${T}/stderr.log" &
DPID=$!
until docker ps --format '{{.Names}}' | grep -qx "${CNAME}" || ! kill -0 "${DPID}" 2>/dev/null; do
  sleep 1
done
flock -u 9; exec 9>&-
wait "${DPID}"
RC=$?
END="$(date +%s)"

python3 - "$T" "$MODEL" "$TASK" "$ARM" "$REP" "$START" "$END" "$RC" <<'EOF'
import json, sys
t, model, task, arm, rep, start, end, rc = sys.argv[1:]
json.dump({"model": model, "task": task, "arm": arm, "rep": int(rep),
           "start": int(start), "end": int(end), "wall_s": int(end) - int(start),
           "exit_code": int(rc), "timed_out": int(rc) == 124,
           "claude_cli": open(f"{t}/cli_version").read().strip(),
           "background_tasks_disabled": True,
           "prompt_sha": open(f"{t}/prompt_sha").read().strip(),
           "skill_sha": open(f"{t}/skill_sha").read().strip()},
          open(f"{t}/meta.json", "w"), indent=1)
EOF
rm -rf "${T}/claude-config"     # session state is in transcript.jsonl
echo "trial done: ${REL} rc=${RC} wall=$((END-START))s"
