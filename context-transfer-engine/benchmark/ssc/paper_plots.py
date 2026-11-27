"""Paper figures and tables from the Delta results (host Python, matplotlib; no CLIO needed).

    python3 paper_plots.py <results dir> <figure dir>

Reads <results>/offline/<node>_<app>_<lane>.jsonl (ablate_ssc.py rows) and, when present,
<results>/live/<node>/<arm>_s<seed>_r<rep>.jsonl (e2e.py rows). Writes PDF + PNG figures and
summary.md (every number in the figures, with 95% bootstrap CIs over schedules).

Quality = mean window score (1 = the phase's best measured composition, 0 = its worst; lost
writes or a broken declared requirement = 0). CIs resample whole schedules (seeds), because
windows of one schedule are not independent (the loop is closed).
"""

import glob
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ------------------------------------------------------------------ arms and colors
# fixed categorical order (validated default palette, light mode); color follows the arm
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY, INK, MUTED = "#8a8985", "#1f1f1e", "#6b6a66"
ARMS = {
    # variant: (label, color, family)
    "v10@qwen3:4b": ("S1 + SSC (qwen3-4B)", BLUE, "s1ssc"),
    "v10@llama3.1:8b": ("S1 + SSC (llama3.1-8B)", VIOLET, "s1ssc"),
    "none@qwen3:4b": ("S1, no SSC (qwen3-4B)", AQUA, "s1"),
    "v10x@qwen3:4b": ("S1 + SSC without node experience (qwen3-4B)", "#86b6ef", "s1ssc"),
    "none@llama3.1:8b": ("S1, no SSC (llama3.1-8B)", MAGENTA, "s1"),
    "none@qwen3:4b+think": ("S2, no SSC (qwen3-4B think)", ORANGE, "s2"),
    "none@qwen3:14b+think": ("S2, no SSC (qwen3-14B think)", RED, "s2"),
    "v10@qwen3:4b+think": ("S2 + SSC (qwen3-4B think)", YELLOW, "s2ssc"),
    "rules": ("Expert rules", GRAY, "nomodel"),
    "rulesh": ("Expert rules + capacity codec", "#a8a7a2", "nomodel"),
    "wknn10": ("SSC lookup, no model", GREEN, "nomodel"),
    "const": ("Fixed guess", "#b5b4ae", "nomodel"),
    "guards": ("Guards only", "#5f5e5a", "nomodel"),
    "v12d@qwen3:4b": ("S1 + SSC v12d (qwen3-4B)", BLUE, "s1ssc"),
    "v12d@llama3.1:8b": ("S1 + SSC v12d (llama3.1-8B)", VIOLET, "s1ssc"),
    # fixed I/O stacks (what CLIO runs today), synthesized on the same windows (add_fixed)
    "fixed-default": ("Default stack (as shipped)", "#b5b4ae", "fixed"),
    "fixed-default-idx": ("Default stack + declared indexer", GRAY, "fixed"),
    "fixed-best": ("Best fixed stack (hand-tuned, hindsight)", "#5f5e5a", "fixed"),
}
MAIN = ["const", "guards", "rules", "wknn10", "none@qwen3:4b", "none@llama3.1:8b",
        "none@qwen3:4b+think", "none@qwen3:14b+think", "v10@qwen3:4b+think",
        "v10@llama3.1:8b", "v10@qwen3:4b"]
NODES = {"05g": "0.5 GB DRAM", "1g": "1 GB DRAM", "4g": "4 GB DRAM",
         "05gp": "0.5 GB DRAM + parallel FS",
         # live result directories (node_app)
         "1g_app3": "WfCommons, 1 GB DRAM node", "1g_app2": "DL/HDF5, 1 GB DRAM node",
         "05g_app3": "WfCommons, 0.5 GB DRAM node"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False,
})


def boot_ci(per_seed, n=4000, seed=7):
    """Mean over seeds and a 95% bootstrap CI that resamples seeds."""
    vals = list(per_seed)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    m = statistics.mean(vals)
    if len(vals) == 1:
        return m, m, m
    rng = random.Random(seed)
    bs = sorted(statistics.mean(rng.choice(vals) for _ in vals) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


# ------------------------------------------------------------------ offline
SEARCHED = {"gs_log", "gs_events", "h5_particles", "ep_map"}     # declared "searched later" (controller.DECLARED)


def load_grid(path, rep=None):
    """(phase, key) -> (median wall s, lost) and per phase the best / worst valid median time.
    rep: use only that repeat (noise sensitivity) instead of the median of all repeats."""
    acc = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if rep is not None and r.get("rep") != rep:
            continue
        c = r["composition"]
        acc[(r["phase"], (c["placement"], c["cache"], c["indexer"], c["compression"]))].append(
            (r["wall_s"], r.get("failed_puts", 0)))
    cell = {k: (statistics.median(x[0] for x in v), max(x[1] for x in v)) for k, v in acc.items()}
    best, worst = {}, {}
    for (ph, k), (t, lost) in cell.items():
        if lost or k[2] != (ph in SEARCHED):
            continue
        best[ph] = min(best.get(ph, t), t)
        worst[ph] = max(worst.get(ph, t), t)
    return cell, best, worst


def slowdown(row, grid):
    """Window time of the chosen composition over the phase's best valid composition (measured
    medians); lost writes or a broken declared requirement cost the worst valid time."""
    cell, best, worst = grid
    c = row["recommended"]
    t, lost = cell[(row["phase"], (c["placement"], c["cache"], c["indexer"], c["compression"]))]
    if lost or row.get("violation"):
        t = worst[row["phase"]]
    return t / best[row["phase"]]


def load_offline(res):
    """{(node, app): {variant: [rows]}}; a variant measured twice keeps its first full run.
    Every row gets `slowdown` (from the node's grid) and a score clipped at 0."""
    out = defaultdict(lambda: defaultdict(list))
    grids = {}
    for p in sorted(glob.glob(os.path.join(res, "offline", "*.jsonl"))):
        b = os.path.basename(p)
        if b.startswith("lc_") or "_lc_" in b:
            continue                    # learning curve (load_lc): partial node experience
        node, app = os.path.basename(p).split("_")[:2]
        if node not in grids:
            gp = os.path.join(res, "grid", f"grid_{node}.jsonl")
            grids[node] = load_grid(gp) if os.path.exists(gp) else None
        for l in open(p):
            r = json.loads(l)
            if "memory" in r:
                continue
            r["raw_score"] = r["score"]
            r["score"] = max(0.0, r["score"])
            r["slowdown"] = slowdown(r, grids[node]) if grids[node] else float("nan")
            out[(node, app)][r["variant"]].append(r)
    for d in out.values():
        for v, rows in d.items():
            seen, keep = set(), []
            for r in rows:
                k = (r["seed"], r["step"])
                if k not in seen:
                    seen.add(k)
                    keep.append(r)
            d[v] = keep
    for (node, app), d in out.items():
        if grids.get(node):
            add_fixed(d, grids[node])
    return out


DEFAULT_STACK = ("dram", False, False, "none")     # compose.DEFAULT: core only, DRAM first
ALL_STACKS = [(p, c, i, z) for p in ("dram", "disk") for c in (False, True) for i in (False, True)
              for z in ("none", "lz4", "zstd")]


def add_fixed(d, grid):
    """Fixed I/O stacks on the same windows as the decision arms (the windows of the 'guards' arm,
    which every scenario has): the default stack as shipped (no indexer, so a stream declared
    searched later breaks its requirement), the default with the user's declared indexer, and the
    best single stack for the whole scenario, chosen with the measured grid on these very windows
    (an expert who tuned the YAML with hindsight). One composition per run; latency 0."""
    tmpl = d.get("guards")
    if not tmpl:
        return

    def rows(stack, declared_idx):
        out = []
        for t in tmpl:
            k = list(stack)
            if declared_idx and t["phase"] in SEARCHED:
                k[2] = True
            rec = dict(zip(("placement", "cache", "indexer", "compression"), k))
            r = {"seed": t["seed"], "step": t["step"], "phase": t["phase"], "recommended": rec,
                 "violation": t["phase"] in SEARCHED and not rec["indexer"], "latency_s": 0.0}
            try:
                r["slowdown"] = slowdown(r, grid)
            except KeyError:
                return None
            r["score"] = 0.0 if r["violation"] else 1.0 / r["slowdown"]
            out.append(r)
        return out

    d["fixed-default"] = rows(DEFAULT_STACK, False)
    d["fixed-default-idx"] = rows(DEFAULT_STACK, True)
    cands = [x for x in (rows(s_, True) for s_ in ALL_STACKS) if x]
    d["fixed-best"] = min(cands, key=lambda x: statistics.mean(math.log(r["slowdown"]) for r in x))
    for v in [v for v in ("fixed-default", "fixed-default-idx") if d[v] is None]:
        del d[v]


def load_lc(res):
    """Learning curve: {(node, app, variant): {N measured phases: {subset: [rows]}}}."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in sorted(glob.glob(os.path.join(res, "offline", "*.jsonl"))):
        b = os.path.basename(p)
        if not b.startswith("lc_") and "_lc_" not in b:
            continue
        node = b.split("_")[1] if b.startswith("lc_") else b.split("_")[0]
        app = b.split("_")[2] if b.startswith("lc_") else b.split("_")[1]
        grid = load_grid(os.path.join(res, "grid", f"grid_{node}.jsonl"))
        for l in open(p):
            r = json.loads(l)
            if "memory" not in r:
                continue
            r["slowdown"] = slowdown(r, grid)
            out[(node, app, r["variant"])][len(r["memory"])][tuple(r["memory"])].append(r)
    return out


def summarize(rows):
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r["score"])
    m, lo, hi = boot_ci([statistics.mean(v) for v in by_seed.values()])
    slow_seed = defaultdict(list)
    for r in rows:
        slow_seed[r["seed"]].append(math.log(r["slowdown"]))
    sm, slo, shi = boot_ci([statistics.mean(v) for v in slow_seed.values()])
    lat = [r.get("latency_s", 0.0) for r in rows]
    return {"score": m, "lo": lo, "hi": hi, "slow": math.exp(sm), "slow_lo": math.exp(slo),
            "slow_hi": math.exp(shi), "seeds": len(by_seed), "windows": len(rows),
            "lat_med": statistics.median(lat), "lat_p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
            "lat_mean": statistics.mean(lat), "lat_sum_per_sched": sum(lat) / max(1, len(by_seed)),
            "lost_or_violation": sum(1 for r in rows if r["score"] == 0.0),
            "violations": sum(1 for r in rows if r.get("violation"))}


def common_seeds(d, variants):
    """Seeds every listed variant has (S2 arms ran on fewer schedules): compare on the same ones."""
    sets = [{r["seed"] for r in d[v]} for v in variants if v in d]
    return set.intersection(*sets) if sets else set()


def fig_pareto(off, figdir, node="1g", app="gs", metric="score"):
    """The headline: decision latency vs decision quality (score, or slowdown vs the best setup)."""
    yk, lk, hk = ("score", "lo", "hi") if metric == "score" else ("slow", "slow_lo", "slow_hi")
    d = off.get((node, app))
    if not d:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    lat_floor = 0.05
    for v in MAIN:
        if v not in d:
            continue
        s = summarize(d[v])
        label, color, fam = ARMS[v]
        nm = ["const", "guards", "rulesh", "wknn10"]
        x = max(s["lat_med"], lat_floor) * (1.12 ** (nm.index(v) - 1.5) if v in nm else 1.0)
        hollow = fam in ("s1", "s2")
        ax.errorbar(x, s[yk], yerr=[[s[yk] - s[lk]], [s[hk] - s[yk]]], fmt="none",
                    ecolor=color, elinewidth=1.2, capsize=2, zorder=2)
        ax.scatter(x, s[yk], s=44 if fam != "nomodel" else 30, marker="o" if fam != "nomodel" else "s",
                   facecolor="white" if hollow else color, edgecolor=color, linewidth=1.6, zorder=3,
                   label=label)
    ax.set_xscale("log")
    if metric != "score":
        from matplotlib.ticker import FixedLocator, NullLocator
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([1, 1.25, 1.5, 2, 3, 5, 8]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}x"))
    ax.set_xlabel("decision latency per window, median (s, log)")
    ax.set_ylabel("decision quality (mean score)" if metric == "score" else
                  "window time vs. best setup (x, geomean)\nlower is better")
    ax.set_xlim(lat_floor * 0.7, None)
    ax.axvspan(lat_floor * 0.7, lat_floor * 1.4, color="#f3f2ee", zorder=0)
    ax.text(lat_floor, 0.985, "no\nmodel", fontsize=6, color=MUTED, va="top", ha="center",
            transform=ax.get_xaxis_transform())
    ax.set_title(f"Quality vs. decision latency ({NODES[node]}, test schedules)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), handletextpad=0.3, borderaxespad=0)
    save(fig, figdir, f"fig_pareto_{node}_{app}" + ("" if metric == "score" else "_slowdown"))


def fig_quality_by_node(off, figdir, app="gs", arms=None):
    arms = arms or ["rulesh", "wknn10", "none@qwen3:4b", "none@qwen3:4b+think", "v10@qwen3:4b",
                    "v10@llama3.1:8b"]
    nodes = [n for n in NODES if (n, app) in off]
    if not nodes:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    w = 0.8 / len(arms)
    for j, v in enumerate(arms):
        for i, n in enumerate(nodes):
            d = off[(n, app)]
            if v not in d:
                continue
            s = summarize(d[v])
            label, color, fam = ARMS[v]
            x = i + (j - (len(arms) - 1) / 2) * w
            ax.bar(x, s["score"], w * 0.9, color=color if fam not in ("s1", "s2") else "white",
                   edgecolor=color, linewidth=1.2, label=label if i == 0 else None)
            ax.errorbar(x, s["score"], yerr=[[s["score"] - s["lo"]], [s["hi"] - s["score"]]], fmt="none",
                        ecolor=INK, elinewidth=0.7, capsize=1.5)
    ax.set_xticks(range(len(nodes)), [NODES[n] for n in nodes])
    ax.set_ylabel("decision quality (mean score)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_title("Quality across node sizes" + (" (second application)" if app == "app2" else ""), loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
    save(fig, figdir, f"fig_quality_nodes_{app}")


def fig_model_sweep(off, figdir, node="1g", app="gs"):
    d = off.get((node, app))
    if not d:
        return
    sizes = [("qwen3:1.7b", 1.7), ("qwen3:4b", 4), ("qwen3:8b", 8), ("qwen3:14b", 14)]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for pre, color, label, hollow in (("v10", BLUE, "S1 + SSC", False), ("none", AQUA, "S1, no SSC", True)):
        xs, ys, los, his = [], [], [], []
        for m, b in sizes:
            v = f"{pre}@{m}"
            if v in d:
                s = summarize(d[v])
                xs.append(b), ys.append(s["score"]), los.append(s["lo"]), his.append(s["hi"])
        if xs:
            ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=2, label=label)
            ax.scatter(xs, ys, s=40, facecolor="white" if hollow else color, edgecolor=color, linewidth=1.6,
                       zorder=3)
    for v, ls in (("none@qwen3:4b+think", "--"), ("none@qwen3:14b+think", ":")):
        if v in d:
            s = summarize(d[v])
            ax.axhline(s["score"], color=ARMS[v][1], linestyle=ls, linewidth=1.4,
                       label=ARMS[v][0] + f", {s['lat_med']:.0f} s/decision")
    if "rulesh" in d:
        ax.axhline(summarize(d["rulesh"])["score"], color=GRAY, linewidth=1, label="Expert rules")
    ax.set_xscale("log")
    ax.set_xticks([b for _, b in sizes], [f"{b:g}B" for _, b in sizes])
    ax.minorticks_off()
    ax.set_xlabel("model size (parameters, thinking off)")
    ax.set_ylabel("decision quality (mean score)")
    ax.set_title(f"The SSC works at every model size ({NODES[node]})", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
    save(fig, figdir, f"fig_model_sweep_{node}")


def fig_guards(off, figdir, node="1g", app="gs"):
    d = off.get((node, app))
    if not d:
        return
    rows = [("guards", "Guards only (no model)"), ("ng_v10@qwen3:4b", "Model + SSC, no guards"),
            ("v10@qwen3:4b", "Model + SSC + guards"), ("ng_none@qwen3:4b", "Model, no SSC, no guards"),
            ("none@qwen3:4b", "Model, no SSC, + guards")]
    rows = [(v, l) for v, l in rows if v in d]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    for i, (v, l) in enumerate(rows):
        s = summarize(d[v])
        color = BLUE if "v10" in v else AQUA if "none" in v else GRAY
        ax.barh(i, s["score"], 0.62, color=color)
        ax.errorbar(s["score"], i, xerr=[[s["score"] - s["lo"]], [s["hi"] - s["score"]]], fmt="none",
                    ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.text(s["hi"] + 0.02, i, f"{s['score']:.2f}" + (f"  ({s['violations']} violations)"
                                                          if s["violations"] else ""),
                va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)), [l for _, l in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("decision quality (mean score, qwen3-4B, thinking off)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Are the guards doing the work?", loc="left")
    save(fig, figdir, f"fig_guards_{node}")


def fig_phase_heatmap(off, figdir, node="1g", app="gs"):
    d = off.get((node, app))
    if not d:
        return
    arms = [v for v in MAIN if v in d]
    phases = sorted({r["phase"] for v in arms for r in d[v]})
    m = [[statistics.mean([r["score"] for r in d[v] if r["phase"] == p] or [float("nan")]) for p in phases]
         for v in arms]
    fig, ax = plt.subplots(figsize=(3.4, 0.25 * len(arms) + 0.7))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"])
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(phases)):
            if not math.isnan(m[i][j]):
                ax.text(j, i, f"{m[i][j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if m[i][j] > 0.6 else INK)
    ax.set_xticks(range(len(phases)), [p.split("_", 1)[1] for p in phases], rotation=30, ha="right")
    ax.set_yticks(range(len(arms)), [ARMS[v][0] for v in arms])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
    ax.set_title(f"Score by workload phase ({NODES[node]})", loc="left")
    save(fig, figdir, f"fig_phase_heatmap_{node}_{app}")



ACC_WITHIN = 1.25    # a decision is "accurate" if its window time is within 25% of the best setup
                     # (measurement noise between repeats is ~10% median, so 10% would mostly score noise)
SCENARIOS = [(("1g", "gs"), "Gray-Scott\n1 GB"), (("05g", "gs"), "Gray-Scott\n0.5 GB"),
             (("4g", "gs"), "Gray-Scott\n4 GB"), (("05gp", "gs"), "Gray-Scott\n0.5 GB+PFS"),
             (("1g", "app2"), "DL/HDF5\n1 GB"), (("4g", "app2"), "DL/HDF5\n4 GB"),
             (("1g", "app3"), "WfCommons\n1 GB"), (("05g", "app3"), "WfCommons\n0.5 GB"),
             (("05gp", "app3"), "WfCommons\n0.5 GB+PFS")]
# main figure: automatic composition against the fixed stacks CLIO runs today; the reasoning model
# without the SSC stays as the "why not just ask a thinking model" reference
MAIN_ARMS = [("fixed-default", "Default stack, as shipped", "#cfcec9", True),
             ("fixed-default-idx", "Default stack + declared indexer", "#a8a7a2", True),
             ("fixed-best", "Best fixed stack (hand-tuned, hindsight)", "#5f5e5a", True),
             ("v12d@qwen3:4b", "S1 + SSC (qwen3-4B, thinking off), this work", BLUE, False),
             ("none@qwen3:4b+think", "S2, no SSC (qwen3-4B, thinking on)", ORANGE, False)]


def fig_main(off, figdir):
    """The paper's main figure. Left: slowdown vs. the best measured composition (geomean over the
    windows of 10 unseen schedules, 95% CI over schedules) for the fixed stacks and S1 + SSC. Right:
    decision time per window of the model-based arms (median, IQR)."""
    scen = [(k, lab) for k, lab in SCENARIOS if k in off and "v12d@qwen3:4b" in off[k]]
    if not scen:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.22, "width_ratios": [2.3, 1]})
    n = len(MAIN_ARMS)
    w = 0.84 / n
    for j, (v, label, color, fixed) in enumerate(MAIN_ARMS):
        for i, (k, _) in enumerate(scen):
            d = off[k]
            if v not in d:
                continue
            s = summarize(d[v])
            x = i + (j - (n - 1) / 2) * w
            a1.bar(x, s["slow"], w * 0.92, bottom=0, color=color, edgecolor="none",
                   label=label if i == 0 else None)
            a1.errorbar(x, s["slow"], yerr=[[s["slow"] - s["slow_lo"]], [s["slow_hi"] - s["slow"]]],
                        fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2)
            if s["slow"] > 2.0:
                a1.text(x, 2.02, f"{s['slow']:.1f}", ha="center", va="bottom", fontsize=5.5, color=INK)
    a1.axhline(1.0, color=INK, linewidth=0.6)
    a1.set_ylim(0.9, 2.1)
    a1.set_xticks(range(len(scen)), [lab for _, lab in scen], fontsize=6.3)
    a1.grid(axis="x", visible=False)
    a1.set_ylabel("slowdown vs. best composition\n(geomean, lower is better)")
    a1.set_title("Composition quality", loc="left")
    # decision time: model arms on the scenarios where both ran
    lat_arms = [(v, lab, c) for v, lab, c, fixed in MAIN_ARMS if not fixed]
    lscen = [(k, lab) for k, lab in scen if all(v in off[k] for v, *_ in lat_arms)]
    m = len(lat_arms)
    for j, (v, label, color) in enumerate(lat_arms):
        for i, (k, _) in enumerate(lscen):
            lat = sorted(max(r.get("latency_s", 0.0), 1e-3) for r in off[k][v])
            med, q1, q3 = lat[len(lat) // 2], lat[len(lat) // 4], lat[3 * len(lat) // 4]
            x = i + (j - (m - 1) / 2) * 0.8 / m
            a2.bar(x, med, 0.8 / m * 0.92, color=color, edgecolor="none")
            a2.errorbar(x, med, yerr=[[med - q1], [q3 - med]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2)
            a2.text(x, q3 * 1.3, f"{med:.0f}" if med >= 10 else f"{med:.1f}", ha="center", fontsize=5.5, color=INK)
    a2.set_yscale("log")
    a2.set_ylim(0.1, 300)
    short = lambda lab: lab.replace("Gray-Scott", "GS").replace("WfCommons", "WfC").replace("DL/HDF5", "DL")
    a2.set_xticks(range(len(lscen)), [short(lab) for _, lab in lscen])
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("decision time per window (s, log)")
    a2.set_title("Decision time", loc="left")
    a1.legend(loc="upper center", bbox_to_anchor=(0.72, -0.2), ncol=3, borderaxespad=0)
    save(fig, figdir, "fig_main_accuracy_speed")


def fig_learning_curve(res, figdir, arms=(("wknn10", "SSC lookup, no model", GREEN),
                                          ("v12d@qwen3:4b", "S1 + SSC (qwen3-4B)", BLUE))):
    """Accuracy vs. how many Gray-Scott phases the node has measured (its experience); mean over
    random subsets with min-max band; top axis: one-time measurement minutes for that many phases."""
    lc = load_lc(res)
    scen = [((n, a), lab) for (n, a), lab in SCENARIOS if any((n, a, v) in lc for v, *_ in arms)]
    if not scen:
        return
    fig, axes = plt.subplots(1, len(scen), figsize=(1.55 * len(scen) + 0.4, 1.9), sharey=True)
    axes = [axes] if len(scen) == 1 else list(axes)
    for ax, ((node, app), lab) in zip(axes, scen):
        for v, label, color in arms:
            curve = lc.get((node, app, v))
            if not curve:
                continue
            ns = sorted(curve)
            accs = [[100 * statistics.mean(r["slowdown"] <= ACC_WITHIN for r in rows)
                     for rows in curve[n].values()] for n in ns]
            ax.plot(ns, [statistics.mean(a) for a in accs], "-o", color=color, ms=2.5, lw=1.1, label=label)
            ax.fill_between(ns, [min(a) for a in accs], [max(a) for a in accs], color=color, alpha=0.15, lw=0)
        ax.set_title(lab.replace("\n", ", "), loc="left")
        ax.set_xticks([1, 2, 3, 5, 7])
        ax.set_ylim(0, 105)
        ax.set_xlabel("measured phases")
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=6)
    save(fig, figdir, "fig_learning_curve")

# ------------------------------------------------------------------ live
LIVE_ARMS = {
    "oracle": ("Oracle (knows the future)", "#b5b4ae"),
    "fixed-default": ("Default stack, as shipped", "#cfcec9"),
    "fixed-best": ("Best fixed stack (hindsight)", GRAY),
    "rules": ("Expert rules", "#5f5e5a"),
    "s1-nossc": ("S1, no SSC (qwen3-4B)", AQUA),
    "s2-nossc": ("S2, no SSC (qwen3-4B think)", ORANGE),
    "s2-nossc-async": ("S2, no SSC, async", "#f4a27f"),
    "s1-ssc-llama": ("S1 + SSC (llama3.1-8B)", VIOLET),
    "s1-ssc": ("S1 + SSC v10 (qwen3-4B)", "#5a95dd"),
    "s1-ssc-async": ("S1 + SSC v10, async", "#86b6ef"),
    "s1-ssc12": ("S1 + SSC (qwen3-4B), this work", BLUE),
    "s1-ssc12-async": ("S1 + SSC, async, this work", "#1f5fae"),
}


def live_total(rows):
    io = sum(r["wall_s"] for r in rows)
    rc = sum(r.get("recompose_s", 0) for r in rows)
    stall = sum(r.get("decision_latency_s", 0) for r in rows if not r.get("async"))
    return io, rc, stall


def load_live(res):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # node -> arm -> seed -> [totals]
    meta = defaultdict(lambda: defaultdict(lambda: {"lost": 0, "miss": 0}))
    for p in glob.glob(os.path.join(res, "live", "*", "*.jsonl")):
        node = os.path.basename(os.path.dirname(p))
        name = os.path.basename(p)[:-6]
        arm, seed, _ = name.rsplit("_", 2)
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            continue
        out[node][arm][seed].append(live_total(rows))
        meta[node][arm]["lost"] += sum(r["failed_puts"] for r in rows)
        meta[node][arm]["miss"] += sum(1 for r in rows if r.get("search_miss"))
    return out, meta


def fig_live(live, meta, figdir, node):
    d = live.get(node)
    if not d:
        return None
    arms = [a for a in LIVE_ARMS if a in d]
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(arms) + 0.7))
    stats = {}
    for i, a in enumerate(arms):
        per = {s: [sum(t) for t in v] for s, v in d[a].items()}
        io = statistics.mean(statistics.mean(t[0] for t in v) for v in d[a].values())
        rc = statistics.mean(statistics.mean(t[1] for t in v) for v in d[a].values())
        st = statistics.mean(statistics.mean(t[2] for t in v) for v in d[a].values())
        m, lo, hi = boot_ci([statistics.mean(v) for v in per.values()])
        stats[a] = {"total": m, "lo": lo, "hi": hi, "io": io, "recompose": rc, "stall": st,
                    "schedules": len(per), **meta[node][a]}
        left = 0
        for val, c, lab, hatch in ((io, LIVE_ARMS[a][1], None, None), (rc, "#c3c2b7", "recomposition", None),
                                   (st, "#3d3d3a", "blocking decision stall", "////")):
            ax.barh(i, val, 0.62, left=left, color=c, edgecolor="white", linewidth=1, hatch=hatch,
                    label=lab if i == 0 else None)
            left += val
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.text(hi + 5, i, f"{m:.0f} s", va="center", fontsize=6.5)
    ax.set_yticks(range(len(arms)), [LIVE_ARMS[a][0] for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel("end-to-end time per schedule (s)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Live runs, {NODES.get(node, node)}", loc="left")
    save(fig, figdir, f"fig_live_{node}")
    return stats


# ------------------------------------------------------------------ output
def save(fig, figdir, name):
    if not name.startswith(("fig_main", "fig_live", "fig_learning")):
        figdir = os.path.join(figdir, "extra")      # backup / appendix figures
        os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=200)
    plt.close(fig)


def main():
    res, figdir = sys.argv[1], sys.argv[2]
    os.makedirs(figdir, exist_ok=True)
    off = load_offline(res)
    md = ["# SSC results (Delta, A100 node)\n"]
    for (node, app), d in sorted(off.items()):
        md.append(f"\n## Offline: {NODES.get(node, node)}, app {app}\n")
        md.append("| arm | score | 95% CI | slowdown vs best (geomean) | 95% CI | seeds | windows | latency med (s) | p95 (s) | stall per schedule (s) | zero-score windows | violations |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for v in sorted(d, key=lambda v: -summarize(d[v])["score"]):
            s = summarize(d[v])
            md.append(f"| {ARMS.get(v, (v,))[0]} `{v}` | {s['score']:.3f} | [{s['lo']:.3f}, {s['hi']:.3f}] | "
                      f"{s['slow']:.3f}x | [{s['slow_lo']:.3f}, {s['slow_hi']:.3f}] | "
                      f"{s['seeds']} | {s['windows']} | {s['lat_med']:.2f} | {s['lat_p95']:.2f} | "
                      f"{s['lat_sum_per_sched']:.0f} | {s['lost_or_violation']} | {s['violations']} |")
        # paired head-to-head on shared seeds: S1+SSC vs S2 without SSC
        for a, b in (("v10@qwen3:4b", "none@qwen3:4b+think"), ("v10@qwen3:4b", "none@qwen3:14b+think"),
                     ("v10@qwen3:4b", "none@qwen3:4b"), ("v10@qwen3:4b", "rulesh")):
            if a in d and b in d:
                seeds = common_seeds(d, [a, b])
                sa = [statistics.mean(r["score"] for r in d[a] if r["seed"] == s) for s in sorted(seeds)]
                sb = [statistics.mean(r["score"] for r in d[b] if r["seed"] == s) for s in sorted(seeds)]
                diff = boot_ci([x - y for x, y in zip(sa, sb)])
                la = statistics.mean(r["latency_s"] for r in d[a] if r["seed"] in seeds)
                lb = statistics.mean(r["latency_s"] for r in d[b] if r["seed"] in seeds)
                md.append(f"\npaired on {len(seeds)} shared schedules, `{a}` minus `{b}`: score "
                          f"{diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]; mean latency {la:.2f} s vs "
                          f"{lb:.2f} s ({lb / la if la else float('inf'):.0f}x faster)")
        fig_pareto(off, figdir, node, app)
        fig_pareto(off, figdir, node, app, metric="slow")
        fig_phase_heatmap(off, figdir, node, app)
    fig_main(off, figdir)
    for app in ("gs", "app2"):
        fig_quality_by_node(off, figdir, app)
    for node in NODES:
        fig_model_sweep(off, figdir, node)
        fig_guards(off, figdir, node)
    # noise sensitivity: rescore the key arms with each single repeat of the grid as ground truth
    key_arms = ["v10@qwen3:4b", "v10@llama3.1:8b", "none@qwen3:4b+think", "none@qwen3:4b", "rulesh", "guards"]
    md.append("\n## Noise sensitivity (slowdown vs best, geomean; ground truth from one grid repeat)\n")
    md.append("| node, app | arm | median of 3 | repeat 0 | repeat 1 | repeat 2 |")
    md.append("|---|---|---|---|---|---|")
    for (node, app), d in sorted(off.items()):
        gp = os.path.join(res, "grid", f"grid_{node}.jsonl")
        if not os.path.exists(gp):
            continue
        reps = [load_grid(gp, rep) for rep in range(3)]
        for v in key_arms:
            if v not in d:
                continue
            cols = [summarize(d[v])["slow"]]
            for g in reps:
                try:
                    cols.append(math.exp(statistics.mean(math.log(slowdown(r, g)) for r in d[v])))
                except KeyError:
                    cols.append(float("nan"))
            md.append(f"| {node}, {app} | `{v}` | " + " | ".join(f"{c:.3f}x" for c in cols) + " |")
    live, meta = load_live(res)
    for node in sorted(live):
        st = fig_live(live, meta, figdir, node)
        if st:
            md.append(f"\n## Live: {node}\n")
            md.append("| arm | total (s) | 95% CI | I/O | recompose | decision stall | schedules | lost puts | search misses |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for a, s in st.items():
                md.append(f"| {LIVE_ARMS[a][0]} | {s['total']:.0f} | [{s['lo']:.0f}, {s['hi']:.0f}] | {s['io']:.0f} | "
                          f"{s['recompose']:.1f} | {s['stall']:.1f} | {s['schedules']} | {s['lost']} | {s['miss']} |")
    open(os.path.join(figdir, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
