# Shared settings for the agent evaluation harness. Sourced, not executed.
#
# This harness runs inside a devcontainer that drives the HOST's docker
# daemon, so every bind mount must name the host path. WS is the clio-core
# checkout as this shell sees it; WS_HOST is the same directory on the host.
HERE_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "${HERE_ENV}/../../../.." && pwd)"
WS_HOST="${WS_HOST:-$(docker inspect "$(hostname)" \
  --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' \
  2>/dev/null)}"
WS_HOST="${WS_HOST:-${WS}}"
RUNS_HOST_VIEW="${WS}/agent_eval_runs"              # path in this shell
RUNS_DOCKER_VIEW="${WS_HOST}/agent_eval_runs"       # path for docker -v
IMAGE="${AGENT_EVAL_IMAGE:-vsc-core-54515e2da5a79650df0b7d02b17873015a1e71cf9446786d51e1373bbab7aa6b-uid}"
SKILL_SRC="${WS}/.claude/skills/gpu-vector"
