#!/bin/bash
# Ensure a python venv with wfcommons==1.2 exists at the interpreter path
# given as $1 (e.g. ${HOME}/venv-dtschedule/bin/python). Idempotent: if the
# interpreter already imports wfcommons and numpy nothing is installed.
# zstandard/lz4 are optional (only used by dt_datagen --selftest ratios).
set -u
py="$1"
if [ -z "$py" ]; then
  echo "ensure_venv.sh: interpreter path required" >&2
  exit 2
fi
venv_dir="$(dirname "$(dirname "$py")")"
if [ ! -x "$py" ]; then
  echo "ensure_venv.sh: creating venv at $venv_dir"
  python3 -m venv "$venv_dir" || exit 1
fi
if "$py" -c 'import wfcommons, numpy' 2>/dev/null; then
  exit 0
fi
echo "ensure_venv.sh: installing wfcommons==1.2 numpy into $venv_dir"
"$py" -m pip install --quiet --upgrade pip || exit 1
"$py" -m pip install --quiet 'wfcommons==1.2' numpy || exit 1
"$py" -m pip install --quiet zstandard lz4 || echo "ensure_venv.sh: zstandard/lz4 optional install failed (ignored)"
"$py" -c 'import wfcommons, numpy'
