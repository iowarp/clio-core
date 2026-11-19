#!/usr/bin/env python3
"""Run a WfBench workflow on the nodes of a PBS job with Parsl.

  wfc_run.py --json W.json --workdir DIR --out OUTDIR [--nodes N] [--wpn K]

Executor: Parsl HighThroughputExecutor, one worker pool per node launched
with mpiexec (LocalProvider + MpiExecLauncher over $PBS_NODEFILE), K
workers per node; the DataFlowKernel and the interchange run here. Every
WfBench task is one Parsl python_app that runs the stock `wfbench` program
with cwd = DIR (wfbench resolves input sizes relative to the cwd) and the
task's arguments from the JSON, exactly as WfCommons' Parsl translator
does with a bash_app, plus:
  * per-task timing on the worker (wall time of the wfbench process);
  * a size check of every input before the task (on the reading node) and
    of every output after it (on the writing node) against the sizes in the
    JSON, so a file that is missing or short when a consumer opens it is
    counted rather than silently read as fewer bytes;
  * task logs stay in memory (stderr tail kept on failure), so the file
    system under test only sees the workflow's own files.
A task whose wfbench exits non-zero raises, and Parsl fails its dependents
(DependencyError) instead of running them on missing inputs.

Before the workflow, two rounds of K null tasks per node (wfbench with no
files, same CPU work): the first warms each node (Python byte-compiles,
stress-ng loads), the second measures the warm per-task overhead
(null_task_median_s).

Writes OUTDIR/tasks.csv and OUTDIR/summary.json and prints one line
  WFRESULT <summary json>
"""
import argparse
import csv
import json
import os
import socket
import statistics
import sys
import time


def parse_tasks(path):
    """Tasks of a WfBench JSON: argv, parents, expected input/output sizes."""
    import ast
    w = json.load(open(path))["workflow"]
    size = {f["id"]: f["sizeInBytes"] for f in w["specification"]["files"]}
    spec = {t["id"]: t for t in w["specification"]["tasks"]}
    tasks = {}
    for e in w["execution"]["tasks"]:
        argv = []
        for a in e["command"]["arguments"]:
            flag, _, val = a.partition(" ")
            if flag in ("--input-files", "--output-files"):
                val = json.dumps(ast.literal_eval(val))
            argv += [flag, val] if val else [flag]
        s = spec[e["id"]]
        tasks[e["id"]] = {
            "argv": argv, "parents": s.get("parents", []),
            "category": s.get("category") or s["name"],
            "inputs": {f: size[f] for f in s.get("inputFiles", [])},
            "outputs": {f: size[f] for f in s.get("outputFiles", [])}}
    return tasks


def topo_order(tasks):
    """Task ids with every parent before its children."""
    order, state = [], {}
    for root in tasks:
        stack = [(root, False)]
        while stack:
            i, done = stack.pop()
            if done:
                state[i] = 2
                order.append(i)
                continue
            if state.get(i):
                continue
            state[i] = 1
            stack.append((i, True))
            stack += [(p, False) for p in tasks[i]["parents"] if not state.get(p)]
    return order


def run_wfbench(tid, argv, workdir, in_sizes, out_sizes, python, wfbench, inputs=()):
    """Parsl app body: check inputs, run wfbench in workdir, check outputs.

    inputs holds the parents' futures (Parsl waits for them); in_sizes and
    out_sizes map file names to the sizes the JSON expects.
    """
    import json, os, socket, subprocess, time

    def bad(files):
        n = 0
        for f, sz in files.items():
            try:
                n += os.stat(os.path.join(workdir, f)).st_size != sz
            except OSError:
                n += 1
        return n
    t_chk = time.time()
    in_bad = bad(in_sizes)
    t0 = time.time()
    p = subprocess.run([python, wfbench] + argv, cwd=workdir, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    t1 = time.time()
    out_bad = bad(out_sizes)
    rec = {"id": tid, "host": socket.gethostname(), "t_check": t_chk,
           "t_start": t0, "t_end": t1, "rc": p.returncode,
           "in_bad": in_bad, "out_bad": out_bad}
    if p.returncode != 0:
        raise RuntimeError(json.dumps(dict(rec, err=p.stderr.decode(errors="replace")[-1500:])))
    return rec


def make_config(a, run_dir):
    """Parsl config: HTEX, one worker pool per node via mpiexec."""
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import MpiExecLauncher, SimpleLauncher
    from parsl.providers import LocalProvider
    venv_bin = os.path.dirname(a.python)
    htex = HighThroughputExecutor(
        label="wf", address=a.address or socket.gethostbyname(socket.gethostname()),
        max_workers_per_node=a.wpn, cores_per_worker=1, cpu_affinity="none",
        encrypted=False, heartbeat_period=30, heartbeat_threshold=300,
        prefetch_capacity=0,
        provider=LocalProvider(
            nodes_per_block=a.nodes, init_blocks=1, min_blocks=1, max_blocks=1,
            launcher=(SimpleLauncher() if a.local else
                      MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--envall --ppn 1")),
            worker_init=f"export PATH={venv_bin}:$PATH; ulimit -c 0",
            cmd_timeout=300))
    return Config(executors=[htex], run_dir=run_dir, strategy="none",
                  retries=0, initialize_logging=True)


def wait_managers(dfk, n, timeout):
    """Block until n worker pools registered with the interchange."""
    ex = dfk.executors["wf"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            got = len(ex.connected_managers())
        except Exception:
            got = 0
        if got >= n:
            return got, time.time() - t0
        time.sleep(2)
    return got, time.time() - t0


def pct(v, q):
    """q-th percentile (0-100) of a list, or None."""
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def summarize(tasks, recs, fails, t_sub, t_done, null_d, a):
    """Makespan, bytes, per-category timing, failure counts."""
    ok = [r for r in recs if r]
    rd = sum(sum(tasks[r["id"]]["inputs"].values()) for r in ok)
    wr = sum(sum(tasks[r["id"]]["outputs"].values()) for r in ok)
    mk = t_done - t_sub
    cats = {}
    for r in ok:
        cats.setdefault(tasks[r["id"]]["category"], []).append(r["t_end"] - r["t_start"])
    ovh = statistics.median(null_d) if null_d else 0.0
    return {
        "workflow": os.path.basename(a.json), "nodes": a.nodes, "wpn": a.wpn,
        "tasks": len(tasks), "ok": len(ok),
        "failed": sum(1 for f in fails.values() if f["kind"] == "task"),
        "dep_failed": sum(1 for f in fails.values() if f["kind"] == "dependency"),
        "other_failed": sum(1 for f in fails.values() if f["kind"] == "other"),
        "makespan_s": round(mk, 3), "bytes_read": rd, "bytes_written": wr,
        "agg_GBps": round((rd + wr) / mk / 1e9, 3) if mk > 0 else None,
        "inputs_bad": sum(r["in_bad"] for r in ok),
        "outputs_bad": sum(r["out_bad"] for r in ok),
        "null_task_median_s": round(ovh, 3),
        "task_s": {k: {"n": len(v), "p50": round(pct(v, 50), 3), "p90": round(pct(v, 90), 3),
                       "p99": round(pct(v, 99), 3), "max": round(max(v), 3),
                       "sum": round(sum(v), 1)} for k, v in sorted(cats.items())},
        "task_s_all": {"p50": pct([r["t_end"] - r["t_start"] for r in ok], 50),
                       "p90": pct([r["t_end"] - r["t_start"] for r in ok], 90),
                       "p99": pct([r["t_end"] - r["t_start"] for r in ok], 99)},
    }


def classify(exc):
    """'dependency', 'task' (wfbench rc != 0) or 'other', plus detail."""
    name = type(exc).__name__
    if "Dependency" in name:
        return "dependency", str(exc)[:300]
    msg = str(exc)
    if msg.startswith("{"):
        return "task", msg
    return "other", f"{name}: {msg[:1500]}"


def main():
    """Run warm-up null tasks, then the workflow, and write the results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--wpn", type=int, default=16)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--address", default=None)
    ap.add_argument("--connect-timeout", type=int, default=600)
    ap.add_argument("--local", action="store_true", help="one local pool (testing)")
    a = ap.parse_args()
    import parsl
    from parsl.app.app import python_app
    os.makedirs(a.out, exist_ok=True)
    wfbench = os.path.join(os.path.dirname(a.python), "wfbench")
    tasks = parse_tasks(a.json)
    order = topo_order(tasks)
    dfk = parsl.load(make_config(a, os.path.join(a.out, "runinfo")))
    got, wt = wait_managers(dfk, a.nodes, a.connect_timeout)
    print(f"=== parsl: {got}/{a.nodes} worker pools after {wt:.1f} s", flush=True)
    app = python_app(run_wfbench, executors=["wf"])
    # Warm-up / overhead: null tasks (no files), K per node.
    null_argv = ["--name", "null", "--percent-cpu", "1.0", "--cpu-work",
                 tasks[order[0]]["argv"][tasks[order[0]]["argv"].index("--cpu-work") + 1],
                 "--num-chunks", "1", "--output-files", "{}", "--input-files", "[]"]
    # Two rounds: the first is cold (byte-compiles the local venv, loads
    # stress-ng); the second's median is the warm per-task overhead.
    for rnd in ("cold", "warm"):
        nf = [app(f"null{i}", null_argv, a.workdir, {}, {}, a.python, wfbench)
              for i in range(a.nodes * a.wpn)]
        null_d = []
        for f in nf:
            try:
                r = f.result()
                null_d.append(r["t_end"] - r["t_start"])
            except Exception as e:  # report, do not abort: the workflow shows it too
                print(f"=== null task failed: {str(e)[:500]}", flush=True)
        print(f"=== null tasks ({rnd}): {len(null_d)}/{len(nf)} ok, median "
              f"{statistics.median(null_d) if null_d else -1:.3f} s", flush=True)
    futs, recs, fails = {}, {}, {}
    t_sub = time.time()
    for i in order:
        t = tasks[i]
        futs[i] = app(i, t["argv"], a.workdir, t["inputs"], t["outputs"], a.python, wfbench,
                      inputs=[futs[p] for p in t["parents"]])
    for i, f in futs.items():
        try:
            recs[i] = f.result()
        except Exception as e:
            kind, det = classify(e)
            fails[i] = {"kind": kind, "detail": det}
    t_done = time.time()
    s = summarize(tasks, list(recs.values()), fails, t_sub, t_done, null_d, a)
    with open(os.path.join(a.out, "tasks.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["id", "category", "host", "t_start", "t_end", "dur_s", "rc",
                    "bytes_in", "bytes_out", "in_bad", "out_bad"])
        for i in order:
            r = recs.get(i)
            t = tasks[i]
            if r:
                w.writerow([i, t["category"], r["host"], f"{r['t_start'] - t_sub:.3f}",
                            f"{r['t_end'] - t_sub:.3f}", f"{r['t_end'] - r['t_start']:.3f}", 0,
                            sum(t["inputs"].values()), sum(t["outputs"].values()),
                            r["in_bad"], r["out_bad"]])
    with open(os.path.join(a.out, "failures.json"), "w") as fp:
        json.dump(fails, fp, indent=1)
    with open(os.path.join(a.out, "summary.json"), "w") as fp:
        json.dump(s, fp, indent=1)
    for i, f in list(fails.items())[:5]:
        print(f"=== FAIL {i} {f['kind']}: {f['detail'][:800]}", flush=True)
    print("WFRESULT " + json.dumps(s), flush=True)
    dfk.cleanup()
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
