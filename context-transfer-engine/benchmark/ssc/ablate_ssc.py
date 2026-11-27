"""Offline SSC ablation on the benchmark schedules: how much of AgentJev's choice quality comes
from the context it is given? No CLIO needed.

    python3 ablate_ssc.py out.jsonl [--seeds 101-110] [--variants none,raw,v2,v3]

The window features the SSC is built from are the client trace of the window (op counts and
sizes) and the sampled compressibility of its first blob; both are fixed by the workload and the
window index, not by the composition, so they are rebuilt here exactly as the live run sees them
(tiers are drained after every window, so the free space is the full tier size). The loop is
closed: each answer changes the deployed composition for the next window, as in the live run
(blocking; the async lag is not modelled).

Variants (same model, same questions, same gates and derived action):
    none   no context: only the list of options
    raw    the raw measurements as a record (trace, ratio, free space, declared intent, current chain)
    v2     the SSC: five sections + controller summary, question-local evidence
    v3     v2 + spill-aware capacity wording + transition trigger (the benchmark's SSC)

Guard ablation: ng_<variant> (e.g. ng_v10@qwen3:4b) runs the variant with both guards off (the
declared-requirement indexer gate and the capacity gate); "guards" is the default chain with only the
guards (indexer when declared, codec from the capacity rule), no model. A window whose composition
breaks a declared requirement (searched later, no indexer) scores 0, like lost writes.

Each window's recommendation, after the declared-requirement and capacity gates, is scored on the
phase it describes with the measured grid: score 1 = the phase's best case, 0 = its worst case
(a composition that loses writes scores 0), plus per-axis agreement with the best composition.
"""

import argparse
import json
import os
import statistics
import subprocess
import time
import zlib
from collections import defaultdict

import e2e
import jev_bridge
import ssc_live
import ssc_v7
import ssc_v8
import workload
from controller import DECLARED, QUEUE
from e2e import to_agentjev

AXES = ("placement", "cache", "indexer", "compression")
OPTIONS = ("options this system can run: placement: dram, ssd | compression: none, lz4, zstd | "
           "cache: enable, disable | indexer: enable, disable")


def key(c):
    return (c["placement"], c["cache"], c["indexer"], c["compression"])


def grid_table(path):
    acc = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        acc[(r["phase"], key(r["composition"]))].append((r["wall_s"], r.get("failed_puts", 0)))
    cell = {k: (statistics.median(v[0] for v in vs), max(v[1] for v in vs)) for k, vs in acc.items()}
    best, worst = {}, {}
    for (ph, k), (t, lost) in cell.items():
        if lost or k[2] != (DECLARED[ph]["searched_later"] == "yes"):
            continue
        best[ph] = min(best.get(ph, (None, 1e9)), (k, t), key=lambda x: x[1])
        worst[ph] = max(worst.get(ph, (None, -1)), (k, t), key=lambda x: x[1])
    return cell, best, worst


def score_of(cell, best, worst, phase, rec):
    """1 = the phase's best composition, 0 = its worst; lost writes or a broken declared
    requirement (searched later without the indexer) score 0. Returns (score, violation)."""
    t, lost = cell[(phase, key(rec))]
    violation = rec["indexer"] != (DECLARED[phase]["searched_later"] == "yes") and not rec["indexer"]
    b, wst = best[phase], worst[phase]
    if lost or violation:
        return 0.0, violation
    return (wst[1] - t) / (wst[1] - b[1]), False


def window(phase, i):
    """Client trace summary and sampled ratio of window i of `phase` (as step.py reports them)."""
    blob, n, reads, kind = workload.PHASES[phase]
    w = {"n_put": n, "n_get": reads, "bytes_put": blob * n, "bytes_get": blob * reads,
         "mean_put": blob, "mean_get": blob if reads else 0, "reuse": reads / n, "seconds": 0.0}
    data = workload.payload(kind, blob, i, 0)[:workload.MB]
    return w, len(data) / max(1, len(zlib.compress(bytes(data), 1)))


def model_busy():
    """True while a benchmark run is using a model (keep the GPU to it)."""
    return subprocess.run(["pgrep", "-f", "e2e.py (agentjev|llm:)"], capture_output=True).returncode == 0


CELLS = {}
MEMORY_TAG = {}     # --memory-phases: recorded on every row
QUEUE_S1 = "/ssc-bench/live_queue_s1"      # s1_bridge.py: qwen3:8b, thinking off


def s1_queue(variant):
    """v9@<ollama model>: the same SSC answered by that model through s1_bridge.py (thinking off)."""
    model = variant.split("@", 1)[1]
    return os.environ.get("SSC_QUEUE_ROOT", "/ssc-bench") + "/live_queue_s1_" + model.replace(":", "_").replace(".", "_")


def make_request(variant, builder, obs, phase):
    variant = variant.split("@", 1)[0]
    variant = variant[:-1] if variant.endswith("h") else variant
    variant = variant[:-1] if variant.endswith("q") else variant     # q: same SSC, fast LLM (s1_bridge.py)
    # v12 switches default to the frozen v10 behaviour for every other variant
    ssc_v8.SHOW_RATIO, ssc_v8.CAPACITY, e2e.CAPACITY_FRACTION = False, 0.8, 0.8
    if variant in ("v12", "v12r", "v12f", "v12d"):
        # v12 (revised after error analysis, chosen on practice schedules): v12r = measured ratios
        # next to the cost words; v12f = where the window lands + physical capacity (no 80% margin);
        # v12 = both; v12d = v12 + decisive evidence overrides the model (applied in main)
        ssc_v8.SHOW_RATIO = variant in ("v12", "v12r", "v12d")
        cap = 0.8 if variant == "v12r" else 1.0
        ssc_v8.CAPACITY, e2e.CAPACITY_FRACTION = cap, cap
        ssc_v8.SPEED_ONLY = True
        ssc_v8.FIT_AWARE = variant != "v12r"
        st = builder.state(obs["tiers"])
        text, qs = ssc_v8.build_v9(obs["window"], obs["ratio"], DECLARED[phase], obs["tiers"], phase, CELLS)
        moved = len(builder.history) > 1 and builder.history[-2][0] != builder.history[-1][0]
        return to_agentjev({"state_text": text, "questions": qs}), {"st": st, "qs": qs, "moved": moved, "obs": obs}
    if variant in ("v9", "v10", "v11", "v10x", "v11x"):
        # v10x / v11x: cold start, the same SSC without node experience (fit facts only)
        ssc_v8.SPEED_ONLY = variant in ("v10", "v11", "v10x", "v11x")
        ssc_v8.FIT_AWARE = variant in ("v11", "v11x")
        st = builder.state(obs["tiers"])
        text, qs = ssc_v8.build_v9(obs["window"], obs["ratio"], DECLARED[phase], obs["tiers"], phase, CELLS,
                                   experience=not variant.endswith("x"))
        moved = len(builder.history) > 1 and builder.history[-2][0] != builder.history[-1][0]
        return to_agentjev({"state_text": text, "questions": qs}), {"st": st, "qs": qs, "moved": moved, "obs": obs}
    if variant in ("v8", "v8n"):
        # v8 (ssc_v8.py): experience from other workloads on this node + digested window facts;
        # v8n: the same without the experience block (ablation)
        st = builder.state(obs["tiers"])
        text, qs = ssc_v8.build(obs["window"], obs["ratio"], DECLARED[phase], obs["tiers"], phase, CELLS,
                                experience=variant == "v8")
        moved = len(builder.history) > 1 and builder.history[-2][0] != builder.history[-1][0]
        return to_agentjev({"state_text": text, "questions": qs}), {"st": st, "qs": qs, "moved": moved, "obs": obs}
    if variant in ("v7c", "v7"):
        orig = ssc_live.SSCBuilder.composability
        ssc_live.SSCBuilder.composability = staticmethod(ssc_v7.composability)
        try:
            req, ctx = e2e.request(builder, obs)
        finally:
            ssc_live.SSCBuilder.composability = orig
        ssc_v7.add_evidence(ctx["qs"], builder, obs, compression=variant == "v7")
        return to_agentjev({"state_text": ctx["st"]["text"], "questions": ctx["qs"]}), ctx
    req, ctx = e2e.request(builder, obs)
    if variant in ("v2", "v3"):
        return req, ctx
    qs = builder.questions()
    qs.pop("action")
    if variant == "none":
        text = "## CONTROLLER SUMMARY\n" + OPTIONS
    else:
        w = obs["window"]
        rec = {"current_chain": builder.composition_text(), "window_trace": {k: w[k] for k in
               ("n_put", "n_get", "bytes_put", "bytes_get", "mean_put", "mean_get")},
               "sampled_compression_ratio": round(obs["ratio"], 2),
               "free_gb": {t: round(v["remaining_gb"], 2) for t, v in obs["tiers"].items()},
               "declared": DECLARED[phase]}
        text = "## MEASUREMENTS\n" + json.dumps(rec, indent=1) + "\n\n## CONTROLLER SUMMARY\n" + OPTIONS
    ctx["qs"] = qs
    return to_agentjev({"state_text": text, "questions": qs}), ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", default="101-110")
    ap.add_argument("--variants", default="none,raw,v2,v3")
    ap.add_argument("--grid", default="/ssc-bench/live_results/grid_gs.jsonl")
    ap.add_argument("--experience", choices=["gs", "wf"], default="gs",
                    help="node experience: gs = the Gray-Scott phases, leave-one-phase-out (frozen runs); "
                         "wf = the WfCommons workflows, leave-one-workflow-out")
    ap.add_argument("--app", choices=["gs", "gs7", "app2", "app3", "mix", "wf"], default="gs",
                    help="schedule phases: gs = the five base phases (default), gs7 = + generalization "
                         "phases, app2 = the second application (DLIO / h5bench-style), app3 = WfCommons workflows")
    ap.add_argument("--memory-phases", default=None,
                    help="learning curve: comma list of the Gray-Scott phases this node has measured "
                         "(its experience); default all of workload.MEMORY_PHASES. Scoring always uses "
                         "the full grid.")
    a = ap.parse_args()
    if a.experience == "wf":
        workload.MEMORY_PHASES, workload.LEAVE_OUT = workload.WF_PHASES, "workflow"
    if a.memory_phases:
        mem = tuple(a.memory_phases.split(","))
        assert set(mem) <= set(workload.MEMORY_PHASES), mem
        workload.MEMORY_PHASES = mem
        global MEMORY_TAG
        MEMORY_TAG = {"memory": list(mem)}
    phases = {"gs": workload.BASE_PHASES, "gs7": workload.GS_PHASES, "app2": workload.APP2_PHASES,
              "app3": workload.APP3_PHASES, "mix": workload.MIX_PHASES, "wf": workload.WF_PHASES}[a.app]
    lo, hi = (map(int, a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds)))
    cell, best, worst = grid_table(a.grid)
    CELLS.update(cell)
    tiers = {"dram": {"remaining_gb": float(e2e.os.environ.get("SSC_DRAM_GB", 1)), "score": 1.0},
             "disk": {"remaining_gb": float(e2e.os.environ.get("SSC_DISK_GB", 2)), "score": 0.5}}
    out = open(a.out, "a")
    for variant in a.variants.split(","):
        guarded = not variant.startswith("ng_")
        e2e.CAPACITY_GATE = guarded
        e2e.SSC_VERSION = "v2" if variant == "v2" else "v3"
        ssc_live.SSCBuilder.SPILL_AWARE = variant != "v2"
        if "@" in variant:
            # untimed warm-up: the model is loaded before the first timed decision
            base = variant[3:] if variant.startswith("ng_") else variant
            jev_bridge.ask(s1_queue(base), {"state": "warm-up", "questions": []})
        for seed in range(lo, hi + 1):
            schedule = [ph for item in e2e.random_schedule(seed, phases=phases).split(",")
                        for ph in [item.split(":")[0]] * int(item.split(":")[1])]
            c = dict(e2e.compose.DEFAULT)
            builder = ssc_live.SSCBuilder(c)
            for i, phase in enumerate(schedule):
                if guarded and DECLARED[phase]["searched_later"] == "yes":
                    c["indexer"] = True                     # declared-requirement gate
                w, ratio = window(phase, i)
                w["seconds"] = cell.get((phase, key(c)), (0.0, 0))[0]
                obs = {"window": w, "ratio": ratio, "tiers": tiers}
                builder.current = dict(c)
                builder.observe(w, ratio, DECLARED[phase])
                if variant in ("rules", "rulesh", "knn", "wknn", "wknn10", "wknn11", "const", "guards"):   # no model: heuristic2, or v8's nearest workload
                    if variant == "guards":
                        new = dict(e2e.compose.DEFAULT)
                    elif variant == "const":
                        # one fixed guess for every window (what qwen3-4B with thinking and no SSC
                        # answered every time) plus the controller's capacity gate from e2e.finish
                        new = {"placement": "dram", "cache": True, "indexer": False, "compression": "none"}
                        if w["bytes_put"] / (1 << 30) > 0.8 * sum(t["remaining_gb"] for t in tiers.values()):
                            new["compression"] = "lz4"
                    elif variant in ("wknn", "wknn10", "wknn11"):
                        ssc_v8.SPEED_ONLY = variant in ("wknn10", "wknn11")
                        ssc_v8.FIT_AWARE = variant == "wknn11"
                        new = ssc_v8.wknn_decide(w, ratio, DECLARED[phase], tiers, phase, cell)
                    elif variant == "knn":
                        new = ssc_v8.knn_decide(w, ratio, DECLARED[phase], tiers, phase, cell)
                    else:
                        new, _ = e2e.decide_heuristic2(dict(c), obs, DECLARED[phase])
                    info = {"answers": {}, "decision_latency_s": 0.0}
                    rec = dict(new or c)
                    if variant in ("rulesh", "guards"):
                        need = w["bytes_put"] / (1 << 30)
                        free = sum(t["remaining_gb"] for t in tiers.values())
                        rec["compression"] = ("none" if need <= 0.8 * free else
                                              "lz4" if need / max(ratio, 1.0) <= 0.8 * free else "zstd")
                    if DECLARED[phase]["searched_later"] == "yes":
                        rec["indexer"] = True
                    score, _ = score_of(cell, best, worst, phase, rec)
                    b = best[phase]
                    out.write(json.dumps({"variant": variant, "seed": seed, "step": i, "phase": phase,
                                          "recommended": rec, "best": dict(zip(AXES, b[0])),
                                          "score": round(score, 3),
                                          "axis_ok": {ax: rec[ax] == dict(zip(AXES, b[0]))[ax] for ax in AXES},
                                          "answers": {}, "latency_s": 0.0, **MEMORY_TAG}) + "\n")
                    c = rec
                    continue
                base = variant[3:] if variant.startswith("ng_") else variant
                req, ctx = make_request(base, builder, obs, phase)
                while model_busy():
                    time.sleep(5)
                t0 = time.time()
                for attempt in range(5):
                    try:
                        raw = jev_bridge.ask(s1_queue(base) if "@" in base else
                                             QUEUE_S1 if base.endswith("q") else QUEUE, req)
                        break
                    except RuntimeError:              # host bridge can race on the shared folder
                        time.sleep(2)
                else:
                    raise RuntimeError("bridge failed 5 times")
                new, info = e2e.finish(builder, ctx, raw, time.time() - t0)
                rec = dict(new or c)
                if base.endswith("h"):
                    # hybrid: the codec follows from capacity (none if the raw window fits, else the
                    # faster codec if it fits at the sampled ratio, else the stronger one)
                    need = w["bytes_put"] / (1 << 30)
                    free = sum(t["remaining_gb"] for t in tiers.values())
                    rec["compression"] = ("none" if need <= 0.8 * free else
                                          "lz4" if need / max(ratio, 1.0) <= 0.8 * free else "zstd")
                if base.startswith("v12d"):
                    # decisive evidence overrides the model; near-ties stay with the model
                    rec.update(ssc_v8.decisive(w, ratio, DECLARED[phase], tiers, phase, cell))
                if guarded and DECLARED[phase]["searched_later"] == "yes":
                    rec["indexer"] = True
                score, violation = score_of(cell, best, worst, phase, rec)
                b = best[phase]
                row = {"variant": variant, "seed": seed, "step": i, "phase": phase,
                       "recommended": rec, "best": dict(zip(AXES, b[0])), "score": round(score, 3),
                       "axis_ok": {ax: rec[ax] == dict(zip(AXES, b[0]))[ax] for ax in AXES},
                       "answers": info["answers"], "latency_s": round(info["decision_latency_s"], 3),
                       "violation": violation, **MEMORY_TAG}
                out.write(json.dumps(row) + "\n")
                out.flush()
                c = rec
            print(f"{time.strftime('%T')} {variant} seed {seed} done", flush=True)
    summarize(a.out)


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    print(f"\n{'variant':8s} {'score':>6s} " + " ".join(f"{ax[:5]:>6s}" for ax in AXES) + "  windows")
    for v, rs in by.items():
        print(f"{v:8s} {statistics.mean(r['score'] for r in rs):6.2f} "
              + " ".join(f"{100 * statistics.mean(r['axis_ok'][ax] for r in rs):5.0f}%" for ax in AXES)
              + f"  {len(rs)}")
    print("\nscore by phase:")
    phases = sorted({r["phase"] for r in rows})
    print(f"{'variant':8s} " + " ".join(f"{p[3:]:>9s}" for p in phases))
    for v, rs in by.items():
        print(f"{v:8s} " + " ".join(
            f"{statistics.mean([r['score'] for r in rs if r['phase'] == p] or [float('nan')]):9.2f}" for p in phases))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--summary":
        summarize(sys.argv[2])
    else:
        main()
