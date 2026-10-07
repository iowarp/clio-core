#!/bin/bash
# Kill leftover WfBench task processes of the current user on this node.
# Patterns are bracketed so pgrep/pkill never match this script itself.
pkill -u "$USER" -f '[w]fbench_dt' 2>/dev/null
pkill -u "$USER" -f '[c]pu-benchmark' 2>/dev/null
exit 0
