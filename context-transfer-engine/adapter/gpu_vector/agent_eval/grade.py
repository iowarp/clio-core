#!/usr/bin/env python3
"""Grade agent trials, and measure the references they are compared with.

    grade.py refs  <task>              measure CPU reference + perf references
    grade.py trial <trial_dir> [...]   grade one or more trials

Everything that touches the GPU runs one job at a time: run this only when no
trial is active, or the performance numbers are meaningless.

A trial is graded in a fresh container with the trial's own tree and build
copy (the same mounts the agent had, and no credentials):
  1. `bash agent_task/build.sh`                    (30 min limit)
  2. every deck in tasks/<task>/spec.py, REPS times (20 min limit each)
  3. correctness from the first run, performance = median of the runs
Writes grade.json into the trial directory.
"""
import importlib.util
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = int(os.environ.get("GRADE_REPS", "3"))
# A preliminary grade (e.g. while other trials still run) must not be named
# grade.json, or the final, idle-GPU grading would skip the trial.
GRADE_OUT = os.environ.get("GRADE_OUT", "grade.json")
# Grading variant (e.g. "_iters50"): suffixes the reference file too, so a
# variant is always compared with references measured the same way.
GRADE_TAG = os.environ.get("GRADE_TAG", "")


def env():
    """Paths and image from env.sh, read through bash so there is one source."""
    out = subprocess.run(
        ["bash", "-c", f"source {HERE}/env.sh && echo $RUNS_HOST_VIEW && "
         f"echo $RUNS_DOCKER_VIEW && echo $IMAGE && echo $WS_HOST"],
        capture_output=True, text=True, check=True).stdout.split()
    return dict(runs=out[0], runs_docker=out[1], image=out[2], ws_host=out[3])


E = env()
REFBIN = f"{E['runs']}/refs/bin"


def load_spec(task):
    """Import tasks/<task>/spec.py."""
    path = os.path.join(HERE, "tasks", task, "spec.py")
    s = importlib.util.spec_from_file_location(f"spec_{task}", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def docker(mounts, cmd, timeout, workdir="/tmp"):
    """Run `cmd` (a shell string) in the trial image with the GPU.
    Returns (rc, combined output); rc 124 on timeout."""
    args = ["docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--shm-size=16g", "--ipc=private", "-w", workdir]
    for src, dst, mode in mounts:
        args += ["-v", f"{src}:{dst}{':' + mode if mode else ''}"]
    args += ["-e", "CLIO_BUILD_DIR=/work/build",
             "-e", "COROC=/work/build-coroc/clio-coroc",
             "-e", "CUDA_HOME=/usr/local/cuda-12.9",
             E["image"], "timeout", str(timeout), "bash", "-lc", cmd]
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def parse_result(out):
    """The single RESULT line as a dict (counts kept as a list of ints)."""
    lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
    if len(lines) != 1:
        return None, len(lines)
    d = dict(t.split("=", 1) for t in lines[0].split()[1:] if "=" in t)
    if "counts" in d:
        try:
            d["counts"] = [int(x) for x in d["counts"].split(",")]
        except ValueError:
            pass
    return d, 1


def refs(task):
    """Measure and cache the CPU reference and the performance references."""
    spec = load_spec(task)
    ref_mounts = [(E["ws_host"], "/workspace", None),
                  (f"{E['runs_docker']}/refs", f"{E['runs']}/refs", None)]
    res = {}
    for deck in spec.DECKS:
        name, data_mb, cache_mb = deck
        cpu = subprocess.run(spec.cpu_ref_cmd(REFBIN, data_mb),
                             capture_output=True, text=True, check=True).stdout
        d = dict(t.split("=", 1) for t in cpu.split()[1:])
        entry = {"counts": [int(x) for x in d["counts"].split(",")],
                 "csum": float(d["csum"]), "paged_ms": [], "incore_ms": []}
        for _ in range(REPS):
            _, out = docker(ref_mounts, " ".join(spec.paged_ref_cmd(data_mb, cache_mb)), 3600)
            entry["paged_ms"].append(spec.parse_paged_ref_ms(out))
            icmd = spec.incore_ref_cmd(REFBIN, data_mb)
            if icmd is not None:          # deck too big for the device
                _, out = docker(ref_mounts, " ".join(icmd), 3600)
                entry["incore_ms"].append(spec.parse_incore_ms(out))
        entry["paged_ms_median"] = _median(entry["paged_ms"])
        entry["incore_ms_median"] = _median(entry["incore_ms"])
        res[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "counts"}))
    json.dump(res, open(f"{E['runs']}/refs/{task}{GRADE_TAG}.json", "w"), indent=1)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def grade_trial(tdir):
    """Build and run one trial's deliverable; write grade.json."""
    meta = json.load(open(f"{tdir}/meta.json"))
    spec = load_spec(meta["task"])
    ref = json.load(open(f"{E['runs']}/refs/{meta['task']}{GRADE_TAG}.json"))
    rel = os.path.relpath(tdir, E["runs"])
    td = f"{E['runs_docker']}/{rel}"
    mounts = [(f"{td}/clio-core", "/work/clio-core", None),
              (f"{td}/build", "/work/build", None),
              (f"{E['runs_docker']}/base/build-coroc", "/work/build-coroc", "ro")]
    g = {"trial": rel, "decks": {}}

    rc, out = docker(mounts, "cd /work/clio-core/agent_task && bash build.sh", 1800,
                     workdir="/work/clio-core")
    open(f"{tdir}/grade_build.log", "w").write(out)
    g["built"] = rc == 0 and os.path.exists(f"{tdir}/clio-core/agent_task/bin/bench")
    if not g["built"]:
        g["correct_all"] = False
        json.dump(g, open(f"{tdir}/{GRADE_OUT}", "w"), indent=1)
        return g

    for deck in spec.DECKS:
        name = deck[0]
        runs = []
        for r in range(REPS):
            cmd = "/work/clio-core/agent_task/bin/bench " + " ".join(spec.bench_args(*deck[1:]))
            rc, out = docker(mounts, cmd, int(os.environ.get("GRADE_RUN_TIMEOUT", "1200")))
            open(f"{tdir}/grade{GRADE_TAG}_{name}_r{r}.log", "w").write(out)
            res, nres = parse_result(out)
            runs.append({"rc": rc, "nresult": nres, "result": res})
        first = runs[0]
        checks = {"exit0": first["rc"] == 0, "one_result": first["nresult"] == 1}
        if first["result"]:
            checks.update(spec.check(first["result"], ref[name], deck))
        ms = []
        for r in runs:
            try:
                if r["rc"] == 0 and r["result"]:
                    ms.append(spec.metric(r["result"]))
            except (KeyError, ValueError):
                pass
        med = _median(ms)
        g["decks"][name] = {
            "checks": checks, "correct": all(checks.values()),
            "ms_per_iter": ms, "ms_median": med,
            "vs_paged_ref": med / ref[name]["paged_ms_median"] if med and ref[name]["paged_ms_median"] else None,
            "vs_incore": med / ref[name]["incore_ms_median"] if med and ref[name]["incore_ms_median"] else None,
            "first": first["result"],
        }
    g["correct_all"] = all(d["correct"] for d in g["decks"].values())
    json.dump(g, open(f"{tdir}/{GRADE_OUT}", "w"), indent=1)
    return g


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "refs":
        refs(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "trial":
        for t in sys.argv[2:]:
            g = grade_trial(os.path.abspath(t))
            print(g["trial"], "built" if g["built"] else "BUILD FAILED",
                  "correct" if g.get("correct_all") else "INCORRECT",
                  {k: (v["ms_median"], v["correct"]) for k, v in g["decks"].items()})
    else:
        print(__doc__)
        sys.exit(2)
