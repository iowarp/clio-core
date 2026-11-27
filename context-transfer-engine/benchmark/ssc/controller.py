"""Closed loop on live CLIO: workload window -> SSC -> System 1 model -> gate -> recompose.

    python3 controller.py <policy> out.jsonl [--schedule a:3,b:2,...] [--grid grid.jsonl]

policy
    agentjev  SSC (with controller summary) -> AgentJev via jev_bridge -> gate -> actuator
    static    never recompose (the default composition throughout)
    oracle    before each window, compose the best measured composition for that phase
              (needs --grid; an upper bound that knows the future)

Every window's throughput is measured under the composition in effect, so the policies
are compared on the same schedule. Recomposition restarts the runtime; its cost is logged.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import compose
import lifecycle
import ssc_live
import jev_bridge

from bench_shared import to_agentjev, norm_agentjev  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.environ.get("JEV_QUEUE", "/ssc-bench/live_queue")

# What the application declares about each stream (the "declared" context source)
DECLARED = {
    "gs_output": {"durability": "important (simulation output)", "searched_later": "no", "latency_critical": "no"},
    "gs_analysis": {"durability": "regenerable (derived from snapshots)", "searched_later": "no",
                    "latency_critical": "yes"},
    "gs_products": {"durability": "important (published data products)", "searched_later": "no",
                    "latency_critical": "no"},
    "gs_log": {"durability": "scratch", "searched_later": "yes", "latency_critical": "no"},
    "gs_archive": {"durability": "important (campaign logs kept for audit)", "searched_later": "no",
                   "latency_critical": "no"},
    # generalization phases (frozen-design test only)
    "gs_field_reread": {"durability": "regenerable (analysis of stored fields)", "searched_later": "no",
                        "latency_critical": "no"},
    "gs_events": {"durability": "important (detected features)", "searched_later": "yes",
                  "latency_critical": "no"},
    # second application (DLIO / h5bench-style)
    "dl_samples": {"durability": "regenerable (copy of the training set)", "searched_later": "no",
                   "latency_critical": "yes"},
    "dl_ckpt": {"durability": "important (model checkpoint)", "searched_later": "no", "latency_critical": "no"},
    "h5_particles": {"durability": "important (particle dump)", "searched_later": "yes",
                     "latency_critical": "no"},
    # third application (WfCommons workflows: Montage, Epigenomics, 1000Genome)
    "mt_project": {"durability": "regenerable (reprojected images)", "searched_later": "no", "latency_critical": "no"},
    "mt_fit": {"durability": "regenerable (fit parameters)", "searched_later": "no", "latency_critical": "no"},
    "ep_fastq": {"durability": "regenerable (split reads)", "searched_later": "no", "latency_critical": "no"},
    "ep_map": {"durability": "important (alignments)", "searched_later": "yes", "latency_critical": "no"},
    "kg_vcf": {"durability": "important (input genotypes)", "searched_later": "no", "latency_critical": "no"},
    "kg_indiv": {"durability": "regenerable (per-individual results)", "searched_later": "no",
                 "latency_critical": "no"},
    # four more WfCommons workflows (the SSC's experience is the other workflows, workload.WF_FAMILY)
    "cy_season": {"durability": "regenerable (crop season output)", "searched_later": "no", "latency_critical": "no"},
    "sk_realign": {"durability": "important (realigned reads)", "searched_later": "no", "latency_critical": "no"},
    "sk_gvcf": {"durability": "important (per-sample variant calls)", "searched_later": "yes", "latency_critical": "no"},
    "sg_decon": {"durability": "regenerable (receiver functions)", "searched_later": "no", "latency_critical": "no"},
    "sra_fastq": {"durability": "regenerable (downloaded reads)", "searched_later": "no", "latency_critical": "no"},
}
DEFAULT_SCHEDULE = "gs_output:3,gs_analysis:3,gs_log:3,gs_products:3,gs_archive:3,gs_output:2"
PLACE = {"dram": "dram", "ssd": "disk"}


def best_by_phase(grid_path):
    """Best measured composition per phase among those that meet the requirements: every put
    stored, and the indexer on iff the stream is searched later."""
    rows = [json.loads(l) for l in open(grid_path)]
    agg = {}
    for r in rows:
        k = (r["phase"], compose.name(r["composition"]))
        a = agg.setdefault(k, {"c": r["composition"], "v": [], "lost": 0})
        a["v"].append(r["mb_per_s"])
        a["lost"] += r.get("failed_puts", 0)
    best = {}
    for (phase, _), a in agg.items():
        if phase not in DECLARED:          # a grid measured with other phases
            continue
        need_idx = DECLARED[phase]["searched_later"] == "yes"
        if a["c"]["indexer"] != need_idx or a["lost"]:
            continue
        mean = sum(a["v"]) / len(a["v"])
        if phase not in best or mean > best[phase][1]:
            best[phase] = (a["c"], mean)
    return best


def run_window(c, phase, tag, seed):
    r = subprocess.run([sys.executable, os.path.join(HERE, "step.py"), phase, tag, str(seed)],
                       env=lifecycle.env_for(c), capture_output=True, text=True, timeout=900)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"step failed rc={r.returncode}: {r.stderr[-1500:]}")
    return json.loads(lines[-1])


def decide(builder, obs):
    """AgentJev on the live SSC, then the gate. Returns (new composition or None, record)."""
    st = builder.state(obs["tiers"])
    qs = builder.questions()
    req = to_agentjev({"state_text": st["text"], "questions": qs})
    t0 = time.time()
    ans = norm_agentjev(jev_bridge.ask(QUEUE, req), qs)
    lat = time.time() - t0
    # gate: options are already restricted to what this node runs. Capacity (80% headroom):
    # a placement tier that cannot take another window is swapped, and if the stream does not
    # fit both tiers uncompressed, "none" is replaced by the highest-ratio codec (zstd).
    gated = []
    place = PLACE.get(ans.get("placement", {}).get("pred"), builder.current["placement"])
    need_gb = obs["window"]["bytes_put"] / (1 << 30)
    if need_gb > 0.8 * obs["tiers"][place]["remaining_gb"]:
        place = "disk" if place == "dram" else "dram"
        gated.append("placement")
    codec = ans.get("compression", {}).get("pred") or "none"
    if codec == "none" and need_gb > 0.8 * sum(t["remaining_gb"] for t in obs["tiers"].values()):
        codec = "zstd"
        gated.append("compression")
    proposal = {"placement": place,
                "cache": ans.get("cache", {}).get("pred") == "enable",
                "indexer": ans.get("indexer", {}).get("pred") == "enable",
                "compression": codec}
    action = ans.get("action", {}).get("pred")
    if gated and action == "keep_current":
        action = "recompose"    # the current composition would lose data
        gated.append("action")
    rec = {"ssc_drift": st["drift"], "answers": {k: v.get("pred") for k, v in ans.items()},
           "action_conf": ans.get("action", {}).get("conf"), "decision_latency_s": lat,
           "proposal": proposal, "gated": gated, "state_text": st["text"]}
    if action == "recompose" and proposal != builder.current:
        return proposal, rec
    return None, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", choices=["agentjev", "static", "oracle"])
    ap.add_argument("out")
    ap.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    ap.add_argument("--grid", default="/ssc-bench/live_results/grid.jsonl")
    a = ap.parse_args()
    schedule = [p for item in a.schedule.split(",") for p in [item.split(":")[0]] * int(item.split(":")[1])]
    best = best_by_phase(a.grid) if os.path.exists(a.grid) else {}

    c = dict(compose.DEFAULT)
    builder = ssc_live.SSCBuilder(c)
    subprocess.run(["rm", "-rf", lifecycle.ROOT])
    proc, _ = lifecycle.start(c)
    out = open(a.out, "w")
    try:
        for i, phase in enumerate(schedule):
            recompose_s = 0.0
            if a.policy == "oracle" and phase in best and best[phase][0] != c:
                t0 = time.time()
                lifecycle.stop(proc, c)
                c = dict(best[phase][0])
                proc, _ = lifecycle.start(c)
                recompose_s = time.time() - t0
            obs = run_window(c, phase, f"w{i}_{phase}", seed=i)
            row = {"step": i, "phase": phase, "policy": a.policy, "composition": dict(c),
                   "mb_per_s": obs["result"]["mb_per_s"], "failed_puts": obs["result"]["failed_puts"],
                   "window": obs["window"], "core_telemetry": obs["core_telemetry"], "ratio": obs["ratio"],
                   "best_mb_per_s": best.get(phase, (None, None))[1],
                   "best_composition": best.get(phase, (None, None))[0], "recompose_s": recompose_s}
            if a.policy == "agentjev":
                builder.observe(obs["window"], obs["ratio"], DECLARED[phase])
                new, rec = decide(builder, obs)
                row.update(rec)
                if new is not None:
                    t0 = time.time()
                    lifecycle.stop(proc, c)
                    c = new
                    builder.current = dict(c)
                    proc, _ = lifecycle.start(c)
                    row["recomposed_to"] = dict(c)
                    row["recompose_s"] = time.time() - t0
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{i:2d} {phase:14s} {compose.name(row['composition']):30s} {row['mb_per_s']:8.1f} MB/s"
                  + (f"  -> {compose.name(row['recomposed_to'])}" if row.get("recomposed_to") else ""),
                  flush=True)
    finally:
        lifecycle.stop(proc, c)


if __name__ == "__main__":
    main()
