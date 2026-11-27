"""LaTeX tables and paper figures from the Delta results (host Python; reuses paper_plots.py).

    python3 paper_tables.py <results dir> <paper dir>

Writes <paper>/tables/*.tex (\\input by main.tex) and <paper>/figures/*.pdf. Every number in the
paper comes from here, so the draft updates when new results land. A table whose data is not in
yet is written with a visible "pending" row instead of being left out.
"""

import json
import math
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import paper_plots as pp  # noqa: E402

APPS = {"gs": "Gray-Scott", "app2": "DL/HDF5", "app3": "WfCommons"}
PHASE_NAMES = {
    "gs_output": "GS snapshot output", "gs_analysis": "GS analysis re-read", "gs_products": "GS data products",
    "gs_log": "GS logs (searched)", "gs_archive": "GS log archive", "gs_field_reread": "GS field re-read",
    "gs_events": "GS events (searched)", "dl_samples": "DLIO training samples", "dl_ckpt": "DLIO checkpoint",
    "h5_particles": "h5bench particles (searched)", "mt_project": "Montage mProject images",
    "mt_fit": "Montage mDiffFit fits", "ep_fastq": "Epigenomics FASTQ", "ep_map": "Epigenomics map (searched)",
    "kg_vcf": "1000Genome VCF", "kg_indiv": "1000Genome individuals"}
APP_PHASES = {"gs": ["gs_output", "gs_analysis", "gs_products", "gs_log", "gs_archive"],
              "app2": ["dl_samples", "dl_ckpt", "h5_particles"],
              "app3": ["mt_project", "mt_fit", "ep_fastq", "ep_map", "kg_vcf", "kg_indiv"]}
COMP_COLS = [(p, c, x) for p in ("dram", "disk") for c in (False, True) for x in ("none", "lz4", "zstd")]


def tex_escape(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write(path, body):
    with open(path, "w") as f:
        f.write(body)


def comp_name(k):
    p, c, i, x = k
    return f"{'DRAM' if p == 'dram' else 'SSD'}/{'cache' if c else '--'}/{x}"


# ------------------------------------------------------------------ motivation
def grid_cells(path):
    acc = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        c = r["composition"]
        acc[(r["phase"], (c["placement"], c["cache"], c["indexer"], c["compression"]))].append(
            (r["wall_s"], r.get("failed_puts", 0)))
    return {k: (statistics.median(x[0] for x in v), max(x[1] for x in v)) for k, v in acc.items()}


def fig_motivation(res, paper, panels):
    """Heatmap: each valid composition's time relative to the phase's best (indexer as required)."""
    panels = [(n, a, lab) for n, a, lab in panels if os.path.exists(os.path.join(res, "grid", f"grid_{n}.jsonl"))
              and any(k[0] in APP_PHASES[a] for k in grid_cells(os.path.join(res, "grid", f"grid_{n}.jsonl")))]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.6), squeeze=False)
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
    im = None
    for ax, (n, app, lab) in zip(axes[0], panels):
        cell = grid_cells(os.path.join(res, "grid", f"grid_{n}.jsonl"))
        phases = [p for p in APP_PHASES[app] if any(k[0] == p for k in cell)]
        m, txt = [], []
        for ph in phases:
            idx = ph in pp.SEARCHED
            row = [cell.get((ph, (p, c, idx, x))) for p, c, x in COMP_COLS]
            best = min(t for t, lost in (v for v in row if v) if not lost)
            m.append([float("nan") if (v is None or v[1]) else v[0] / best for v in row])
            txt.append(["lost" if (v and v[1]) else ("best" if v and v[0] == best else
                        f"{v[0] / best:.1f}" if v else "") for v in row])
        im = ax.imshow(m, cmap=cmap, norm=LogNorm(1, 32), aspect="auto")
        for i in range(len(phases)):
            for j in range(len(COMP_COLS)):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=4.6,
                        color="white" if (m[i][j] == m[i][j] and m[i][j] > 4) else pp.INK)
        ax.set_yticks(range(len(phases)), [PHASE_NAMES[p] for p in phases], fontsize=6)
        ax.set_xticks(range(len(COMP_COLS)), [f"{'D' if p == 'dram' else 'S'}{'+c' if c else ''}\n{x}"
                                              for p, c, x in COMP_COLS], fontsize=5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, loc="left", fontsize=7.5)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
    cb.set_ticks([1, 2, 4, 8, 16, 32], labels=["1x", "2x", "4x", "8x", "16x", "32x"])
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(paper, "figures", "fig_motivation.pdf"))
    plt.close(fig)


def table_best(res, paper, nodes, app):
    """Best composition per phase on each node and how much slower the worst valid one is."""
    cells = {n: grid_cells(os.path.join(res, "grid", f"grid_{n}.jsonl")) for n in nodes
             if os.path.exists(os.path.join(res, "grid", f"grid_{n}.jsonl"))}
    lines = ["\\begin{tabular}{l" + "rr" * len(cells) + "}", "\\toprule",
             "Phase & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{pp.NODES.get(n, n)}}}" for n in cells) + " \\\\",
             " & " + " & ".join("best & worst/best" for _ in cells) + " \\\\", "\\midrule"]
    for ph in APP_PHASES[app]:
        row = [PHASE_NAMES[ph]]
        for n, cell in cells.items():
            ok = {k: t for (p, k), (t, lost) in cell.items() if p == ph and not lost and k[2] == (ph in pp.SEARCHED)}
            if not ok:
                row += ["pending", ""]
                continue
            b = min(ok, key=ok.get)
            row += [comp_name(b), f"{max(ok.values()) / ok[b]:.1f}$\\times$"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(paper, "tables", f"best_{app}.tex"), "\n".join(lines) + "\n")


# ------------------------------------------------------------------ main offline table
ARMS_MAIN = [
    "Fixed I/O stacks (today)",
    ("fixed-default", "Default stack, as shipped"),
    ("fixed-default-idx", "Default stack + declared indexer"),
    ("fixed-best", "Best fixed stack (hand-tuned, hindsight)"),
    "Automatic composition (this work)",
    ("v12d@qwen3:4b", "S1 + SSC (qwen3-4B)"), ("v12d@llama3.1:8b", "S1 + SSC (llama3.1-8B)"),
    ("v10@qwen3:4b", "S1 + SSC, v10 (qwen3-4B)"), ("v10@llama3.1:8b", "S1 + SSC, v10 (llama3.1-8B)"),
    "Alternative deciders (design justification)",
    ("none@qwen3:4b+think", "S2, no SSC (qwen3-4B, thinking)"),
    ("none@qwen3:14b+think", "S2, no SSC (qwen3-14B, thinking)"),
    ("v10@qwen3:4b+think", "S2 + SSC (qwen3-4B, thinking)"),
    ("none@qwen3:4b", "S1, no SSC (qwen3-4B)"),
    ("v10x@qwen3:4b", "S1 + SSC, no node experience"),
    ("rules", "Expert rules"), ("rulesh", "Expert rules + capacity codec"),
    ("wknn10", "SSC lookup, no model")]


def acc(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["seed"]].append(r["slowdown"] <= pp.ACC_WITHIN)
    return pp.boot_ci([100 * statistics.mean(v) for v in by.values()])


def table_main(off, paper, scenarios):
    cols = [(k, lab.replace("\n", " ")) for k, lab in scenarios if k in off]
    lines = ["\\begin{tabular}{l" + "rrr" * len(cols) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in cols) + " \\\\",
             "Controller & " + " & ".join("acc.\\,\\% & slow. & s/dec." for _ in cols) + " \\\\"]
    for arm in ARMS_MAIN:
        if isinstance(arm, str):        # group heading
            lines += ["\\midrule", f"\\multicolumn{{{1 + 3 * len(cols)}}}{{l}}{{\\emph{{{arm}}}}} \\\\"]
            continue
        v, name = arm
        if not any(v in off[k] for k, _ in cols):
            continue
        row = [name]
        for k, _ in cols:
            d = off[k]
            if v not in d:
                row += ["--", "--", "--"]
                continue
            s = pp.summarize(d[v])
            a = acc(d[v])
            n = s["seeds"]
            row += [f"{a[0]:.0f}" + ("$^\\dagger$" if n < 10 else ""), f"{s['slow']:.2f}$\\times$",
                    f"{s['lat_med']:.1f}" if s["lat_med"] >= 0.05 else "0"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(paper, "tables", "main.tex"), "\n".join(lines) + "\n")


# ------------------------------------------------------------------ module usage
def table_modules(off, paper, key, arms=("v10@qwen3:4b", "none@qwen3:4b+think", "none@qwen3:4b")):
    """Per phase: how often each module was on in the arm's choices, and the measured best chain."""
    if key not in off:
        return
    d = off[key]
    node, app = key
    cell = grid_cells(f"{sys.argv[1]}/grid/grid_{node}.jsonl")
    lines = ["\\begin{tabular}{lll" + "l" * len(arms) + "}", "\\toprule",
             "Phase & best (place/cache/codec) & " + " & ".join(
                 {"v10@qwen3:4b": "S1 + SSC", "none@qwen3:4b+think": "S2, no SSC",
                  "none@qwen3:4b": "S1, no SSC"}.get(a, a) for a in arms if a in d) + " \\\\", "\\midrule"]
    lines[0] = "\\begin{tabular}{ll" + "l" * len([a for a in arms if a in d]) + "}"
    for ph in APP_PHASES[app]:
        ok = {k: t for (p, k), (t, lost) in cell.items() if p == ph and not lost and k[2] == (ph in pp.SEARCHED)}
        if not ok:
            continue
        b = min(ok, key=ok.get)
        row = [PHASE_NAMES[ph], comp_name(b) + ("/idx" if b[2] else "")]
        for a in arms:
            if a not in d:
                continue
            rs = [r for r in d[a] if r["phase"] == ph]
            if not rs:
                row.append("--")
                continue
            top = Counter((r["recommended"]["placement"], r["recommended"]["cache"], r["recommended"]["indexer"],
                           r["recommended"]["compression"]) for r in rs).most_common(1)[0]
            ok_pct = 100 * statistics.mean(r["slowdown"] <= pp.ACC_WITHIN for r in rs)
            row.append(f"{comp_name(top[0])}{'/idx' if top[0][2] else ''} ({ok_pct:.0f}\\%)")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(paper, "tables", f"modules_{node}_{app}.tex"), "\n".join(lines) + "\n")


# ------------------------------------------------------------------ ablation / sweep / live
def table_ablation(off, paper):
    rows = [("none@qwen3:4b", "No SSC (option list only)"), ("v10x@qwen3:4b", "SSC, no node experience"),
            ("v10@qwen3:4b", "Full SSC"), ("ng_v10@qwen3:4b", "Full SSC, guards off"),
            ("guards", "Guards only (no model)"), ("wknn10", "SSC lookup, no model")]
    cols = [k for k in [("1g", "gs"), ("1g", "app2"), ("1g", "app3"), ("05g", "app3"), ("05gp", "app3")] if k in off]
    lines = ["\\begin{tabular}{l" + "rr" * len(cols) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{APPS[a]}, {pp.NODES.get(n, n)}}}" for n, a in cols) + " \\\\",
             "Context & " + " & ".join("acc.\\,\\% & slow." for _ in cols) + " \\\\", "\\midrule"]
    for v, name in rows:
        if not any(v in off[k] for k in cols):
            continue
        r = [name]
        for k in cols:
            if v in off[k]:
                r += [f"{acc(off[k][v])[0]:.0f}", f"{pp.summarize(off[k][v])['slow']:.2f}$\\times$"]
            else:
                r += ["--", "--"]
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(paper, "tables", "ablation.tex"), "\n".join(lines) + "\n")


def table_sweep(off, paper, key=("1g", "gs")):
    if key not in off:
        return
    d = off[key]
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Model (thinking off) & acc.\\,\\% no SSC & acc.\\,\\% SSC & slow. no SSC & slow. SSC & s/dec. \\\\",
             "\\midrule"]
    for m, lab in (("qwen3:1.7b", "qwen3-1.7B"), ("qwen3:4b", "qwen3-4B"), ("qwen3:8b", "qwen3-8B"),
                   ("qwen3:14b", "qwen3-14B"), ("llama3.1:8b", "llama3.1-8B")):
        a, b = d.get(f"none@{m}"), d.get(f"v10@{m}")
        if not (a and b):
            continue
        lines.append(f"{lab} & {acc(a)[0]:.0f} & {acc(b)[0]:.0f} & {pp.summarize(a)['slow']:.2f}$\\times$ & "
                     f"{pp.summarize(b)['slow']:.2f}$\\times$ & {pp.summarize(b)['lat_med']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(paper, "tables", "sweep.tex"), "\n".join(lines) + "\n")


def common_schedules(live):
    """Keep, per node, only the schedules every arm ran (so arms are averaged over the same runs)."""
    out = {}
    for node, d in live.items():
        common = set.intersection(*(set(v) for v in d.values())) if d else set()
        out[node] = {a: {sc: v[sc] for sc in common} for a, v in d.items()} if common else d
    return out


def table_live(res, paper):
    live, meta = pp.load_live(res)
    live = common_schedules(live)
    order = ["oracle", "fixed-best", "rules", "s1-ssc", "s1-ssc-async", "s1-ssc-llama", "s1-nossc",
             "s2-nossc", "s2-nossc-async"]
    for node, d in live.items():
        lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
                 "Controller & sched. & I/O (s) & recomp. (s) & decision (s) & total (s) \\\\", "\\midrule"]
        for a in order:
            if a not in d:
                continue
            tots = [t for v in d[a].values() for t in v]
            io, rc, st = (statistics.mean(t[i] for t in tots) for i in range(3))
            lines.append(f"{pp.LIVE_ARMS[a][0]} & {len(d[a])} & {io:.0f} & {rc:.1f} & {st:.0f} & {io + rc + st:.0f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(os.path.join(paper, "tables", f"live_{node}.tex"), "\n".join(lines) + "\n")


def table_agreement(off, paper, keys=(("1g", "gs"), ("4g", "gs"), ("1g", "app2"), ("4g", "app2"), ("1g", "app3"),
                                      ("05g", "app3"), ("05gp", "app3")),
                    v="v10@qwen3:4b"):
    """How often the model's choice per axis equals the model-free lookup's (the option the SSC
    evidence ranks best), and who is right when they differ (both decisions scored on the grid)."""
    ax = ("placement", "cache", "compression")
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Scenario & place. & cache & codec & differ & model right / lookup right \\\\", "\\midrule"]
    for k in keys:
        d = off.get(k)
        if not d or v not in d or "wknn10" not in d:
            continue
        look = {(r["seed"], r["step"]): r for r in d["wknn10"]}
        per = {a: [] for a in ax}
        dev = []
        for r in d[v]:
            l = look.get((r["seed"], r["step"]))
            if not l or l["phase"] != r["phase"]:
                continue
            for a in ax:
                per[a].append(r["recommended"][a] == l["recommended"][a])
            if any(r["recommended"][a] != l["recommended"][a] for a in ax):
                dev.append((r["slowdown"] <= pp.ACC_WITHIN, l["slowdown"] <= pp.ACC_WITHIN))
        n = len(per["placement"])
        if not n:
            continue
        lines.append(f"{APPS[k[1]]}, {pp.NODES.get(k[0], k[0])} & " + " & ".join(
            f"{100 * statistics.mean(per[a]):.0f}\\%" for a in ax) +
            f" & {100 * len(dev) / n:.0f}\\% & {sum(x for x, _ in dev)} / {sum(y for _, y in dev)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(paper, "tables", "agreement.tex"), "\n".join(lines) + "\n")


def main():
    res, paper = sys.argv[1], sys.argv[2]
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(paper, sub), exist_ok=True)
    off = pp.load_offline(res)
    figtmp = os.path.join(paper, "figures", ".build")
    os.makedirs(figtmp, exist_ok=True)
    pp.fig_main(off, figtmp)
    pp.fig_learning_curve(res, figtmp)
    live, meta = pp.load_live(res)
    live = common_schedules(live)
    for node in live:
        pp.fig_live(live, meta, figtmp, node)
    for f in os.listdir(figtmp):
        if f.endswith(".pdf"):
            shutil.copy(os.path.join(figtmp, f), os.path.join(paper, "figures", f))
    fig_motivation(res, paper, [("1g", "app3", "WfCommons, 1 GB DRAM node"),
                                ("05g", "app3", "WfCommons, 0.5 GB DRAM node"),
                                ("05gp", "app3", "WfCommons, 0.5 GB DRAM + parallel FS")])
    for app in ("gs", "app3"):
        table_best(res, paper, ["1g", "05g", "05gp"], app)
    table_main(off, paper, pp.SCENARIOS)
    for key in (("1g", "app3"), ("05g", "app3"), ("05gp", "app3"), ("1g", "gs")):
        table_modules(off, paper, key)
    table_ablation(off, paper)
    table_sweep(off, paper)
    table_live(res, paper)
    table_agreement(off, paper)
    print("tables:", sorted(os.listdir(os.path.join(paper, "tables"))))
    print("figures:", sorted(f for f in os.listdir(os.path.join(paper, "figures")) if f.endswith(".pdf")))


if __name__ == "__main__":
    main()
