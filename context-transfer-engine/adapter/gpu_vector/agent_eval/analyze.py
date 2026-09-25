#!/usr/bin/env python3
"""Summarize trials: effort (tokens, cost, time, turns), skill use, and grades.

    analyze.py [--csv out.csv]        all trials under $RUNS/trials

Per trial, from transcript.jsonl (Claude Code stream-json):
  - the final `result` event: usage (input / output / cache-write / cache-read
    tokens), total_cost_usd, duration_ms, num_turns, per-model usage (so
    subagent tokens are counted), is_error
  - tool calls: total, Bash commands that build or run the bench, and whether
    the gpu-vector skill was loaded (Skill tool) or its files were read
From meta.json: wall clock and timeout. From grade.json: built, correctness
per deck, and performance ratios against the paged reference and the in-core
baseline (median over the graded runs).
Then per (task, arm, model) cell: means over repetitions.
"""
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = subprocess.run(["bash", "-c", f"source {HERE}/env.sh && echo $RUNS_HOST_VIEW"],
                      capture_output=True, text=True, check=True).stdout.strip()


def transcript_stats(path):
    """Effort and skill-use numbers from one stream-json transcript."""
    s = {"tool_calls": 0, "build_cmds": 0, "run_cmds": 0, "skill_loaded": False,
         "skill_files_read": set(), "result_event": False,
         "background_calls": 0, "killed_tasks": 0}
    if not os.path.exists(path):
        return s
    for line in open(path, errors="replace"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for c in ev.get("message", {}).get("content", []):
                if c.get("type") != "tool_use":
                    continue
                s["tool_calls"] += 1
                name, inp = c.get("name"), c.get("input", {})
                if inp.get("run_in_background"):
                    s["background_calls"] += 1
                if name == "Skill" and "gpu-vector" in json.dumps(inp):
                    s["skill_loaded"] = True
                blob = json.dumps(inp)
                if ".claude/skills/gpu-vector/" in blob:
                    for f in ("SKILL", "API", "DESIGN", "COSTS", "PITFALLS", "CASES", "ORGANIZER"):
                        if f"gpu-vector/{f}" in blob:
                            s["skill_files_read"].add(f)
                if name == "Bash":
                    cmd = inp.get("command", "")
                    if "build.sh" in cmd or "build_newcoro" in cmd or "nvcc" in cmd:
                        s["build_cmds"] += 1
                    if "bin/bench" in cmd:
                        s["run_cmds"] += 1
        elif ev.get("subtype") == "task_updated" and \
                ev.get("patch", {}).get("status") == "killed":
            s["killed_tasks"] += 1   # work cut off at session end: trial invalid
        elif ev.get("type") == "result":
            s["result_event"] = True
            u = ev.get("usage", {})
            s.update({
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": u.get("output_tokens", 0),
                "cache_write_tokens": u.get("cache_creation_input_tokens", 0),
                "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                "cost_usd": ev.get("total_cost_usd"),
                "duration_ms": ev.get("duration_ms"),
                "api_ms": ev.get("duration_api_ms"),
                "num_turns": ev.get("num_turns"),
                "is_error": ev.get("is_error"),
                "model_usage": ev.get("modelUsage", {}),
            })
            # Subagents bill under their own model; count every model's tokens.
            mu = ev.get("modelUsage") or {}
            if mu:
                s["all_models_tokens"] = sum(
                    m.get("inputTokens", 0) + m.get("outputTokens", 0) +
                    m.get("cacheReadInputTokens", 0) + m.get("cacheCreationInputTokens", 0)
                    for m in mu.values())
    s["skill_files_read"] = ",".join(sorted(s["skill_files_read"]))
    return s


def trial_row(tdir):
    """One flat row per trial."""
    meta = json.load(open(f"{tdir}/meta.json"))
    row = dict(meta)
    ts = transcript_stats(f"{tdir}/transcript.jsonl")
    ts.pop("model_usage", None)
    row.update(ts)
    gp = f"{tdir}/grade.json"
    if os.path.exists(gp):
        g = json.load(open(gp))
        row["built"] = g.get("built")
        row["correct_all"] = g.get("correct_all")
        for name, d in g.get("decks", {}).items():
            row[f"{name}.correct"] = d.get("correct")
            row[f"{name}.ms"] = d.get("ms_median")
            row[f"{name}.vs_paged_ref"] = d.get("vs_paged_ref")
            row[f"{name}.vs_incore"] = d.get("vs_incore")
    return row


def main():
    rows = [trial_row(os.path.dirname(m))
            for m in sorted(glob.glob(f"{RUNS}/trials/*/*/*/r*/meta.json"))]
    if "--csv" in sys.argv:
        out = sys.argv[sys.argv.index("--csv") + 1]
        keys = sorted({k for r in rows for k in r})
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out} ({len(rows)} trials)")
    cells = {}
    for r in rows:
        cells.setdefault((r["task"], r["arm"], r["model"]), []).append(r)

    def mean(rs, k):
        v = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        return statistics.mean(v) if v else None

    hdr = f"{'task':8} {'arm':8} {'model':26} {'n':>2} {'correct':>7} {'wall_min':>8} {'Mtok':>7} {'out_ktok':>8} {'cost$':>7} {'turns':>5} {'skill':>5}"
    print(hdr)
    for (task, arm, model), rs in sorted(cells.items()):
        ok = sum(1 for r in rs if r.get("correct_all"))
        tok = mean(rs, "all_models_tokens")
        print(f"{task:8} {arm:8} {model:26} {len(rs):>2} {ok:>3}/{len(rs):<3} "
              f"{(mean(rs, 'wall_s') or 0) / 60:8.1f} {(tok or 0) / 1e6:7.2f} "
              f"{(mean(rs, 'output_tokens') or 0) / 1e3:8.1f} {mean(rs, 'cost_usd') or 0:7.2f} "
              f"{mean(rs, 'num_turns') or 0:5.0f} {sum(1 for r in rs if r.get('skill_loaded')):>2}/{len(rs)}")


if __name__ == "__main__":
    main()
