"""End-to-end run on live CLIO: total time to run a workload schedule under a policy.

    python3 e2e.py <policy> out.jsonl [--schedule a:6,b:6,...] [--grid grid.jsonl]

policy
    fixed:<name>     one composition for the whole run, e.g. fixed:dram-cache-idx-lz4
    oracle           before each window, the best measured composition for its phase (knows the future)
    oracle_reactive  after each window, the best composition for the phase just seen (one window late)
    agentjev         SSC v2 -> AgentJev -> derived action -> gate, after each window (one window late)
    heuristic        expert rules over the same window features, after each window, with the safety rules
    heuristic2       the same rules plus a capacity rule (zstd when the raw window would not fit):
                     the fair expert baseline, written before any run with it
    llm:<model>      System 2 baseline: a local reasoning LLM (Ollama) reads the same SSC and answers
                     the same questions, with the same derived action, gates and safety rules

--schedule seed:N draws a random schedule (see random_schedule) instead of a hand-written one;
gseed:N also draws the two generalization phases. SSC_DRAM_GB / SSC_DISK_GB set the tier sizes.

Recomposition is live: the interceptor pools are swapped with `clio_run compose rm/start` on the
running runtime; only a placement change (a core setting) restarts the runtime. Each window's
blobs are deleted after it (drained), so no compressed blobs outlive the compressor that wrote them.

SSC v2 (as selected on SSC-Bench dev): question-local evidence on shift, placement, cache and
indexer; compression and action get none. The action is derived, not asked: recompose when a
per-axis answer differs from the deployed composition (any confidence right after the drift
detector reports a shift, confidence >= 0.6 otherwise), and only those axes change. Gate: if
the stream cannot fit uncompressed, "none" is replaced by the model's most probable codec.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import compose
import lifecycle
import workload
import ssc_live
import jev_bridge
from controller import DECLARED, best_by_phase, run_window, QUEUE

from bench_shared import to_agentjev, norm_agentjev  # noqa: E402

PLACE = {"dram": "dram", "ssd": "disk"}
LOCAL_AXES = ("shift", "placement", "cache", "indexer")
MIN_CONF = 0.6
CAPACITY_GATE = True    # off only in the offline guard ablation (ablate_ssc.py ng_<variant>)
DEFAULT_SCHEDULE = "gs_output:6,gs_analysis:6,gs_log:6,gs_products:6,gs_archive:6,gs_output:6"
SSC_VERSION = "v2"


def parse_comp(name):
    p, cache, idx, codec = name.split("-")
    return {"placement": p, "cache": cache == "cache", "indexer": idx == "idx", "compression": codec}


# ---------------------------------------------------------------- actuator
def clio(args, c):
    r = subprocess.run([os.path.join(lifecycle.BIN, "clio_run")] + args, env=lifecycle.env_for(c),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clio_run {' '.join(args)} failed: {r.stderr[-500:]}")


def recompose(proc, old, new, force_restart=False):
    """Returns (proc, seconds, mode). Live pool swap unless placement (a core setting) changes.
    force_restart: after a window that overflowed the tiers, CLIO keeps refusing large writes even
    though the space reads as free again (seen with a live-added compressor); a fresh runtime clears
    it, so the next recomposition restarts instead of swapping (same rule for every policy)."""
    t0 = time.time()
    if force_restart or old["placement"] != new["placement"]:
        lifecycle.stop(proc, old)
        proc, _ = lifecycle.start(new, fresh=True)
        return proc, time.time() - t0, "restart" + ("-after-overflow" if force_restart else "")
    path = os.path.join(lifecycle.ROOT, "interceptors.yaml")
    old_y = compose.interceptors_file(old, lifecycle.ROOT)
    if old_y:
        open(path, "w").write(old_y)
        clio(["compose", "rm", path], old)
    with open(os.path.join(lifecycle.ROOT, "clio.yaml"), "w") as f:   # clients read the deployment config
        f.write(compose.render(new, lifecycle.ROOT))
    new_y = compose.interceptors_file(new, lifecycle.ROOT)
    if new_y:
        open(path, "w").write(new_y)
        clio(["compose", "start", path], new)
    if not canary(new):
        # a live swap once left the chain refusing nearly every put with free space available;
        # nothing reports it, so the actuator verifies the new chain and restarts if it is broken
        lifecycle.stop(proc, new)
        proc, _ = lifecycle.start(new, fresh=True)
        return proc, time.time() - t0, "live-canary-restart"
    return proc, time.time() - t0, "live"


CANARY = r"""
import os, clio_cte_core_ext as cte
assert cte.clio_init(cte.RuntimeMode.kClient, False)
assert cte.initialize_cte(os.environ["CLIO_SERVER_CONF"], cte.PoolQuery.Dynamic())
data = (b"canary 0123456789 " * 4000)[:64 << 10]
tag = cte.Tag("__canary")
tag.PutBlob("c", data)
ok = bytes(tag.GetBlob("c", len(data))) == data
cte.get_cte_client().DelBlob(tag.GetTagId(), "c")
print("CANARY", ok)
"""


def canary(c):
    """One small put/get/delete through the chain top of c."""
    try:
        r = subprocess.run([sys.executable, "-c", CANARY], env=lifecycle.env_for(c), capture_output=True,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    return "CANARY True" in r.stdout


# ---------------------------------------------------------------- SSC v2 decision
def local_questions(builder, tiers):
    cur, dom, shifted, _ = builder.drift()
    w, d, r = builder.history[-1][1], builder.history[-1][2], builder.history[-1][3]
    cap = builder.capacity(tiers, w, r)
    ev = {"shift": [f"active class: {cur[0]}", f"dominant class: {dom[0] if dom else 'none yet'}",
                    f"drift detector verdict, different in kind: {'yes' if shifted else 'no'}"],
          "placement": [f"latency critical: {d['latency_critical']}",
                        f"bytes read per byte written: {w['reuse']:.1f}", cap],
          "cache": [f"reads: {w['n_get']} ops; bytes read per byte written: {w['reuse']:.1f}",
                    f"writes: {w['n_put']} ops, mean {w['mean_put'] / 1024:.0f} KB"],
          "indexer": [f"searched later: {d['searched_later']}",
                      f"writes: {w['n_put']} ops, mean {w['mean_put'] / 1024:.0f} KB"]}
    axes = LOCAL_AXES
    if SSC_VERSION == "v5":
        # v5: the compression question gets its own evidence (the pilot showed the model answering
        # the same codec for every workload without it). Plain facts; no codec is named.
        need = w["bytes_put"] / (1 << 30)
        free = sum(t["remaining_gb"] for t in tiers.values())
        ev["compression"] = [
            f"sampled compressibility of this stream: {r:.1f}x (size before / after a fast test compression)",
            f"uncompressed, one window of this stream ({need:.2f} GB) "
            f"{'fits' if need <= 0.8 * free else 'does not fit'} in the free space ({free:.1f} GB)",
            f"latency critical: {d['latency_critical']}",
            "every compressed write and read pays CPU time to compress and decompress"]
        axes = LOCAL_AXES + ("compression",)
    elif SSC_VERSION == "v6":
        # v6: the same facts as v5, but the capacity fact is stated as its consequence in words
        # (v5's numbers pushed the model to the stronger codec everywhere). No codec is named.
        need = w["bytes_put"] / (1 << 30)
        free = sum(t["remaining_gb"] for t in tiers.values())
        if need <= 0.8 * free:
            fit = (f"this stream fits uncompressed in the free space ({need:.2f} GB per window, {free:.1f} GB "
                   f"free), so compressing it would only add CPU time to every write and read")
        else:
            fit = (f"this stream does not fit uncompressed in the free space ({need:.2f} GB per window, "
                   f"{free:.1f} GB free); at the sampled ratio it "
                   f"{'fits' if need / max(r, 1.0) <= 0.8 * free else 'still does not fit'} compressed")
        ev["compression"] = [f"sampled compressibility of this stream: {r:.1f}x", fit,
                             f"latency critical: {d['latency_critical']}"]
        axes = LOCAL_AXES + ("compression",)
    qs = builder.questions()
    qs.pop("action")                       # derived, not asked
    for qid in axes:
        qs[qid]["instructions"] += "\nEvidence for this question:\n" + "\n".join(f"- {e}" for e in ev[qid])
    return qs


CELLS = {}          # (phase, composition key) -> (median wall s, lost); the node's measured grid
NONE_TEXT = ("## CONTROLLER SUMMARY\noptions this system can run: placement: dram, ssd | compression: none, lz4, "
             "zstd | cache: enable, disable | indexer: enable, disable")


def request(builder, obs):
    """The model request for the window just observed, and what finish() needs to use the answer."""
    if SSC_VERSION in ("v10", "v11", "none"):
        import ssc_v8
        st = builder.state(obs["tiers"])
        moved = len(builder.history) > 1 and builder.history[-2][0] != builder.history[-1][0]
        if SSC_VERSION == "none":
            # no SSC: only the options (the "without SSC" arm)
            qs = builder.questions()
            qs.pop("action")
            text = NONE_TEXT
        else:
            ssc_v8.SPEED_ONLY = True
            ssc_v8.FIT_AWARE = SSC_VERSION == "v11"
            text, qs = ssc_v8.build_v9(obs["window"], obs["ratio"], DECLARED[obs["phase"]], obs["tiers"],
                                       obs["phase"], CELLS)
        return (to_agentjev({"state_text": text, "questions": qs}),
                {"st": st, "qs": qs, "moved": moved, "obs": obs})
    st = builder.state(obs["tiers"])
    qs = local_questions(builder, obs["tiers"])
    moved = len(builder.history) > 1 and builder.history[-2][0] != builder.history[-1][0]
    return to_agentjev({"state_text": st["text"], "questions": qs}), {"st": st, "qs": qs, "moved": moved, "obs": obs}


def decide(builder, obs, queue=QUEUE):
    """Blocking decision: ask and wait (the decision latency stalls the workload)."""
    req, ctx = request(builder, obs)
    t0 = time.time()
    raw = jev_bridge.ask(queue, req)
    return finish(builder, ctx, raw, time.time() - t0)


def finish(builder, ctx, raw, lat):
    """Derived action + gate from the model's answers. builder.current is the composition
    deployed now, which in async mode can be later than the window the answers are about."""
    st, qs, moved, obs = ctx["st"], ctx["qs"], ctx["moved"], ctx["obs"]
    tiers = obs["tiers"]
    ans = norm_agentjev(raw, qs)
    cur = dict(builder.current)
    rec = {"placement": PLACE.get(ans["placement"]["pred"], cur["placement"]),
           "cache": ans["cache"]["pred"] == "enable", "indexer": ans["indexer"]["pred"] == "enable",
           "compression": ans["compression"]["pred"] or cur["compression"]}
    # v2: low-confidence changes only after a shift from the dominant class. v3: also when the
    # class differs from the previous window (a return to the dominant class is a transition too)
    need = 0.0 if st["drift"]["shift"] or (SSC_VERSION != "v2" and moved) else MIN_CONF
    changed = [ax for ax in rec if rec[ax] != cur[ax] and ans[ax].get("pred") is not None
               and (ans[ax].get("conf") or 0) >= need]
    prop = dict(cur)
    for ax in changed:
        prop[ax] = rec[ax]
    gated = False
    need_gb = obs["window"]["bytes_put"] / (1 << 30)
    if CAPACITY_GATE and prop["compression"] == "none" and need_gb > 0.8 * sum(t["remaining_gb"] for t in tiers.values()):
        dist = ans["compression"].get("dist") or {}
        codecs = {k: float(v) for k, v in dist.items() if k != "none"} or {"lz4": 1.0}
        prop["compression"] = max(codecs, key=codecs.get)
        gated = True
    info = {"answers": {k: v.get("pred") for k, v in ans.items()},
            "conf": {k: v.get("conf") for k, v in ans.items()}, "drift": st["drift"],
            "changed_axes": changed, "gated": gated, "moved": moved, "decision_latency_s": lat}
    return (prop if prop != cur else None), info


# ---------------------------------------------------------------- rule-based baseline
def decide_heuristic(c, obs, declared):
    """A storage expert's static rules over the same window features the SSC carries (written
    once, not tuned): DRAM is the fast tier and CTE spills on its own; cache only for re-read
    data; indexer iff the stream is declared searchable; lz4 on compressible data unless the
    stream is latency critical. Runs with the same safety rules as SSC + AgentJev."""
    w = obs["window"]
    rec = {"placement": "dram",
           "cache": bool(w["n_get"]) and w["reuse"] >= 2,
           "indexer": declared["searched_later"] == "yes",
           "compression": "lz4" if obs["ratio"] >= 1.3 and declared["latency_critical"] != "yes" else "none"}
    return (rec if rec != c else None), {"heuristic": rec}


def decide_heuristic2(c, obs, declared):
    """heuristic + one capacity rule an expert would add: when another raw window of this stream
    would not fit in 80% of the free space, use the strongest codec (zstd). Written once, before
    any run with it, and frozen."""
    _, info = decide_heuristic(c, obs, declared)
    rec = dict(info["heuristic"])
    free = sum(t["remaining_gb"] for t in obs["tiers"].values())
    if obs["window"]["bytes_put"] / (1 << 30) > 0.8 * free and obs["ratio"] >= 1.3:
        rec["compression"] = "zstd"
    return (rec if rec != c else None), {"heuristic": rec}


def random_schedule(seed, blocks=9, phases=workload.BASE_PHASES):
    """A test schedule nobody wrote: `blocks` blocks of 3-5 windows, a random phase per block,
    never the same phase twice in a row."""
    rng = random.Random(seed)
    out, prev = [], None
    for _ in range(blocks):
        ph = rng.choice([p for p in phases if p != prev])
        out.append(f"{ph}:{rng.randint(3, 5)}")
        prev = ph
    return ",".join(out)


# ---------------------------------------------------------------- safety rules (--rules)
WATERMARK = 0.4      # pause a window once free space drops below 40% of what it started with
                     # (20% left too little room to finish with the indexer in the chain)


def step(c, phase, tag, seed, start=0, watermark=None):
    env = lifecycle.env_for(c)
    if watermark:
        env["SSC_WATERMARK"] = str(watermark)
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "step.py"),
                        phase, tag, str(seed), str(start)], env=env, capture_output=True, text=True, timeout=900)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"step failed rc={r.returncode}: {r.stderr[-1500:]}")
    return json.loads(lines[-1])


def merge(a, b):
    """One window observation from a paused first part a and its resumed remainder b."""
    ra, rb = a["result"], b["result"]
    res = {k: ra[k] + rb[k] for k in ("wall_s", "write_s", "read_s", "logical_mb", "failed_puts", "n_puts")}
    res.update(phase=rb["phase"], mb_per_s=res["logical_mb"] / res["wall_s"] if res["wall_s"] else 0.0)
    wa, wb = a["window"], b["window"]
    w = {k: wa[k] + wb[k] for k in ("n_put", "n_get", "bytes_put", "bytes_get", "seconds")}
    w.update(mean_put=w["bytes_put"] / w["n_put"] if w["n_put"] else 0,
             mean_get=w["bytes_get"] / w["n_get"] if w["n_get"] else 0,
             reuse=w["bytes_get"] / w["bytes_put"] if w["bytes_put"] else 0.0)
    return dict(b, result=res, window=w)


ESCALATE = {"none": "zstd", "lz4": "zstd"}
ENVELOPE = 2         # 1: one reflex step (none -> lz4), as in the e2e/heldout/test3/test4 runs
                     # 2: the reflex adds the strongest codec (zstd), also on top of lz4; a two-step
                     #    none -> lz4 -> zstd escalation fired too late (at 16% free) and lost writes


def run_window_rules(proc, c, phase, tag, seed):
    """Overflow reflex: the window pauses at the capacity watermark, the controller live-adds a
    stronger codec (see ENVELOPE; the decider still picks the codec after the window), and a new
    client resumes the window. Returns (proc, c, obs, reflex or None)."""
    obs, reflex, start = None, None, 0
    while True:
        nxt = (ESCALATE.get(c["compression"]) if ENVELOPE == 2
               else "lz4" if c["compression"] == "none" else None)
        part = step(c, phase, tag, seed, start=start, watermark=WATERMARK if nxt else None)
        obs = part if obs is None else merge(obs, part)
        if "paused_at" not in part["result"]:
            return proc, c, obs, reflex
        new = dict(c, compression=nxt)
        proc, s, mode = recompose(proc, c, new)
        start = part["result"]["paused_at"]
        reflex = {"reflex_at": (reflex or {}).get("reflex_at", start),
                  "reflex_s": (reflex or {}).get("reflex_s", 0) + s, "reflex_mode": mode, "reflex_to": nxt}
        c = new


# ---------------------------------------------------------------- loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("out")
    ap.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    ap.add_argument("--grid", default="/ssc-bench/live_results/grid_gs.jsonl")
    ap.add_argument("--ssc", choices=["v2", "v3", "v4", "v5", "v6", "v10", "v11", "none"], default="v2")
    ap.add_argument("--rules", action="store_true",
                    help="agentjev/llm: declared-requirement gate + capacity-watermark overflow reflex")
    ap.add_argument("--async-decide", action="store_true",
                    help="models: the workload keeps running while the model answers; the answer is applied "
                         "at the first window boundary after it arrives (a slow model acts on stale context)")
    ap.add_argument("--envelope", type=int, choices=[1, 2], default=2, help="safety-rule version (see ENVELOPE)")
    a = ap.parse_args()
    global SSC_VERSION, ENVELOPE
    SSC_VERSION, ENVELOPE = a.ssc, a.envelope
    if a.schedule.startswith(("seed:", "gseed:", "dseed:", "wseed:")):
        kind, n = a.schedule.split(":")
        # gseed: the generalization phases join the draw (they were added after the design froze)
        # dseed: the second application's phases only
        a.schedule = random_schedule(int(n), phases=workload.GS_PHASES if kind == "gseed" else
                                     workload.APP2_PHASES if kind == "dseed" else
                                     workload.APP3_PHASES if kind == "wseed" else workload.BASE_PHASES)
        print("schedule", a.schedule, flush=True)
    model = a.policy == "agentjev" or a.policy.startswith(("llm:", "s1:"))
    queue = QUEUE if a.policy == "agentjev" else QUEUE + "_" + a.policy.split(":", 1)[-1].replace(":", "_")
    if a.policy.startswith("s1:"):
        # fast local model through s1_bridge.py (thinking off unless the model ends in +think)
        queue = os.environ.get("SSC_QUEUE_ROOT", "/ssc-bench") + "/live_queue_s1_" + a.policy[3:].replace(":", "_").replace(".", "_")
    if a.ssc in ("v10", "v11") and os.path.exists(a.grid):
        import statistics
        acc = {}
        for l in open(a.grid):
            r = json.loads(l)
            k = r["composition"]
            acc.setdefault((r["phase"], (k["placement"], k["cache"], k["indexer"], k["compression"])), []).append(
                (r["wall_s"], r.get("failed_puts", 0)))
        CELLS.update({k: (statistics.median(x[0] for x in v), max(x[1] for x in v)) for k, v in acc.items()})
    ssc_live.SSCBuilder.SPILL_AWARE = a.ssc in ("v3", "v4", "v5", "v6")
    ssc_live.SSCBuilder.HIDE_CURRENT = a.ssc == "v4"
    schedule = [p for item in a.schedule.split(",") for p in [item.split(":")[0]] * int(item.split(":")[1])]
    best = {ph: dict(v[0]) for ph, v in best_by_phase(a.grid).items()} if os.path.exists(a.grid) else {}

    c = parse_comp(a.policy.split(":", 1)[1]) if a.policy.startswith("fixed:") else dict(compose.DEFAULT)
    if a.policy == "oracle":
        c = dict(best[schedule[0]])
    builder = ssc_live.SSCBuilder(c)
    subprocess.run(["rm", "-rf", lifecycle.ROOT])
    proc, _ = lifecycle.start(c)
    out = open(a.out, "w")
    overflowed = False
    pending = None          # async: (request id, ctx, submit time)
    try:
        for i, phase in enumerate(schedule):
            row = {"step": i, "phase": phase, "policy": a.policy}
            if a.async_decide:
                row["async"] = True
            if a.policy == "oracle" and best[phase] != c:
                proc, s, mode = recompose(proc, c, best[phase], overflowed)
                overflowed = False
                c = dict(best[phase])
                row.update(recompose_s=s, recompose_mode=mode)
            rules = (model and a.rules) or a.policy.startswith("heuristic")
            if rules and DECLARED[phase]["searched_later"] == "yes" and not c["indexer"]:
                # declared requirement: the stream says it will be searched, so the indexer goes
                # in before its first write (like the feasibility gate, never left to the model)
                new = dict(c, indexer=True)
                proc, s, mode = recompose(proc, c, new, overflowed)
                overflowed = False
                c = new
                row.update(recompose_s=s, recompose_mode=mode, declared_gate=True)
            if rules:
                proc, c, obs, reflex = run_window_rules(proc, c, phase, f"w{i}_{phase}", seed=i)
                if reflex:
                    row.update(reflex)
                    row.update(recompose_s=row.get("recompose_s", 0) + reflex["reflex_s"],
                               recompose_mode=row.get("recompose_mode", reflex["reflex_mode"]))
            else:
                obs = run_window(c, phase, f"w{i}_{phase}", seed=i)
            obs["phase"] = phase           # v10/v11 leave the window's own workload out of the experience
            res = obs["result"]
            row.update(composition=dict(c), wall_s=res["wall_s"], mb_per_s=res["mb_per_s"],
                       failed_puts=res["failed_puts"],
                       search_miss=DECLARED[phase]["searched_later"] == "yes" and not c["indexer"],
                       free_gb_after_drain={t: round(v["remaining_gb"], 2) for t, v in obs["tiers"].items()})
            overflowed = overflowed or res["failed_puts"] > 0
            new = None
            if a.policy == "oracle_reactive":
                new = dict(best[phase]) if best[phase] != c else None
            elif model and a.async_decide:
                builder.current = dict(c)
                builder.observe(obs["window"], obs["ratio"], DECLARED[phase])
                if pending:
                    raw = jev_bridge.poll(queue, pending[0])
                    if raw is not None:
                        new, info = finish(builder, pending[1], raw, time.time() - pending[2])
                        row.update(info, answer_for_step=pending[1]["step"])
                        pending = None
                    else:
                        row["decision_pending"] = True
                if pending is None:
                    req, ctx = request(builder, obs)
                    ctx["step"] = i
                    pending = (jev_bridge.submit(queue, req), ctx, time.time())
            elif model:
                builder.current = dict(c)
                builder.observe(obs["window"], obs["ratio"], DECLARED[phase])
                new, info = decide(builder, obs, queue)
                row.update(info)
            elif a.policy == "heuristic":
                new, info = decide_heuristic(c, obs, DECLARED[phase])
                row.update(info)
            elif a.policy == "heuristic2":
                new, info = decide_heuristic2(c, obs, DECLARED[phase])
                row.update(info)
            if new is not None:
                proc, s, mode = recompose(proc, c, new, overflowed)
                overflowed = False
                c = dict(new)
                row.update(recompose_s=row.get("recompose_s", 0) + s, recompose_mode=mode, recomposed_to=dict(c))
            elif overflowed:
                # CLIO keeps refusing writes after an overflow until the runtime restarts (see
                # recompose); restart even when the decider keeps the composition, for every policy
                proc, s, mode = recompose(proc, c, c, True)
                overflowed = False
                row.update(recompose_s=row.get("recompose_s", 0) + s, recompose_mode=mode)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{i:2d} {phase:14s} {compose.name(row['composition']):26s} {row['wall_s']:6.2f} s "
                  f"lost {row['failed_puts']:3d}" + (f"  -> {compose.name(c)} ({row['recompose_mode']} "
                                                     f"{row['recompose_s']:.2f} s)" if 'recompose_mode' in row else ""),
                  flush=True)
    finally:
        lifecycle.stop(proc, c)


if __name__ == "__main__":
    main()
