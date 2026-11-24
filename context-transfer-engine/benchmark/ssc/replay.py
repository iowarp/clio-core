"""Offline SSC ablation: replay the recorded live windows through AgentJev with SSC variants.

    python replay.py <loop_agentjev.jsonl> <grid.jsonl> <out_dir> [--variants base,local,...]

The workload features of each window (client trace, sampled ratio, declared intent) come from
the live run and do not depend on the composition, so the loop can be replayed without CLIO:
AgentJev answers on the SSC variant, the (fixed) gate applies, and the window outcome under the
composition in effect is looked up in the measured grid (mean MB/s and lost puts per phase and
composition). Decisions apply from the next window, as in the live loop.

Variants are cumulative:
    base      the SSC as used live (with the fixed gate)
    local     + question-local evidence: each question carries the few SSC facts it depends on
    derived   + the action is derived from the per-axis answers (recompose when the recommended
              composition differs from the current one and either the drift detector fired or the
              same recommendation was made last window) instead of asked as keep/recompose
    options   + measured cost of each ChiMod on this node (from the grid, aggregated over
              write-only and read-heavy phases, not per phase), attached to the option it
              describes rather than listed in shared text (the small model favours options
              that are merely named in the context)
    outcomes  + outcome memory: composition, MB/s and lost puts of past windows per workload class
    neutral   + the current composition is presented neutrally, with the measured switch cost

Policies static (never change) and oracle_reactive (best composition for the last window's
phase, one window late) bound the replay.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

import compose
import ssc_live
from controller import DECLARED

from bench_shared import to_agentjev, norm_agentjev  # noqa: E402

URL = "http://127.0.0.1:8149/api/evaluate"
VARIANTS = ["base", "local", "derived", "options", "facts", "verdicts", "outcomes", "neutral"]
PLACE = {"dram": "dram", "ssd": "disk"}
WRITE_PHASES = ("checkpoint", "random_ingest", "log_append", "log_archive")
READ_PHASES = ("hot_reread",)
RECOMPOSE_S = 2.4          # mean measured restart cost in the live loops


def key(c):
    return (c["placement"], c["cache"], c["indexer"], c["compression"])


def human(c):
    mods = [m for m, on in (("cache", c["cache"]), ("indexer", c["indexer"]),
                            (f"compressor ({c['compression']})", c["compression"] != "none")) if on]
    tier = "DRAM" if c["placement"] == "dram" else "SSD"
    return f"{' -> '.join(mods + ['core'])} on {tier}"


class Grid:
    def __init__(self, path):
        acc = defaultdict(list)
        for l in open(path):
            r = json.loads(l)
            acc[(r["phase"], key(r["composition"]))].append((r["mb_per_s"], r.get("failed_puts", 0)))
        self.cell = {k: (sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs))
                     for k, vs in acc.items()}
        self.best = {}
        for (ph, k), (mb, lost) in self.cell.items():
            need_idx = DECLARED[ph]["searched_later"] == "yes"
            if lost or k[2] != need_idx:
                continue
            if ph not in self.best or mb > self.best[ph][1]:
                self.best[ph] = (k, mb)

    def outcome(self, phase, c):
        return self.cell[(phase, key(c))]

    def profile(self):
        """Median throughput with a module relative to without it, all else equal."""
        def rel(phases, flip):
            out = []
            for (ph, k), (mb, _) in self.cell.items():
                if ph not in phases:
                    continue
                base = flip(k)
                if base and (ph, base) in self.cell and self.cell[(ph, base)][0] > 0:
                    out.append(mb / self.cell[(ph, base)][0])
            out.sort()
            return out[len(out) // 2] if out else None

        def on(i, val, off):
            return lambda k: (k[:i] + (off,) + k[i + 1:]) if k[i] == val else None
        mods = {"cache": on(1, True, False), "indexer": on(2, True, False),
                "lz4": on(3, "lz4", "none"), "zstd": on(3, "zstd", "none"),
                "SSD placement (vs DRAM)": on(0, "disk", "dram")}
        def eff(x):
            return f"slows them by {100 * (1 - x):.0f}%" if x < 0.97 else "has no measurable cost"
        self.slowdown = {kind: {("ssd" if m.startswith("SSD") else m): max(0.0, 1 - rel(ph, f))
                                for m, f in mods.items()}
                         for kind, ph in (("write", WRITE_PHASES), ("read", READ_PHASES))}
        return {m: f"Measured on this node: on write-only streams it {eff(rel(WRITE_PHASES, f))}; "
                   f"on read-heavy streams it {eff(rel(READ_PHASES, f))}."
                for m, f in mods.items()}


def tiers_from(text):
    m = re.search(r"DRAM buffer tier ([\d.]+) GB free, node-local SSD tier ([\d.]+) GB free", text)
    return {"dram": {"remaining_gb": float(m.group(1)), "score": 1.0},
            "disk": {"remaining_gb": float(m.group(2)), "score": 0.2}}


class VariantBuilder(ssc_live.SSCBuilder):
    def __init__(self, current, level, profile_text):
        super().__init__(current)
        self.level = level                 # index into VARIANTS
        self.profile_text = profile_text
        self.outcomes = []                 # (class, composition, mb/s, lost)

    def on(self, name):
        return self.level >= VARIANTS.index(name)

    def deployment(self, tiers):
        return (f"A single-node CLIO deployment with 16 cores and 15 GB of RAM. "
                f"CLIO manages a DRAM buffer tier ({tiers['dram']['remaining_gb']:.1f} GB free) and a "
                f"node-local SSD tier ({tiers['disk']['remaining_gb']:.1f} GB free). There is no shared "
                f"parallel file system and only one node.")

    def composition_text(self):
        txt = super().composition_text()
        if self.on("neutral"):
            txt = ("Currently deployed (this is not a recommendation; it was chosen for earlier windows): "
                   + txt + f"\nChanging the composition restarts the runtime and costs about "
                   f"{RECOMPOSE_S:.1f} s; a workload class in the history lasted "
                   f"{self.mean_run():.1f} windows on average.")
        return txt

    def mean_run(self):
        runs, prev = [], None
        for h in self.history:
            if h[0] == prev:
                runs[-1] += 1
            else:
                runs.append(1)
            prev = h[0]
        return sum(runs) / len(runs) if runs else 1.0

    def option_outcome(self, axis, value):
        """Outcome memory for one option: past windows of the active class run with it."""
        cls = self.history[-1][0]
        vs = [(mb, lost) for c0, c, mb, lost in self.outcomes if c0 == cls and c[axis] == value]
        if not vs:
            return ""
        return (f" Past windows of this workload run with it: {sum(v[0] for v in vs) / len(vs):.0f} MB/s, "
                f"{sum(v[1] for v in vs) / len(vs):.0f} writes lost per window ({len(vs)} windows).")

    def evidence(self, tiers):
        """Per-question facts, taken from the SSC sections (question-local evidence)."""
        cur, dom, shifted, novel = self.drift()
        w, d, r = self.history[-1][1], self.history[-1][2], self.history[-1][3]
        access, comp = cur
        cap = self.capacity(tiers, w, r)
        ev = {
            "shift": [f"active class: {access}", f"dominant class: {dom[0] if dom else 'none yet'}",
                      f"drift detector verdict, different in kind: {'yes' if shifted else 'no'}"],
            "placement": [f"latency critical: {d['latency_critical']}", f"durability: {d['durability']}",
                          f"bytes read per byte written: {w['reuse']:.1f}", cap],
            "cache": [f"reads: {w['n_get']} ops; bytes read per byte written: {w['reuse']:.1f}",
                      f"writes: {w['n_put']} ops, mean {w['mean_put'] / 1024:.0f} KB"],
            "compression": [f"sampled compressibility: {comp} ({r:.1f}x)",
                            f"latency critical: {d['latency_critical']}", cap],
            "indexer": [f"searched later: {d['searched_later']}",
                        f"writes: {w['n_put']} ops, mean {w['mean_put'] / 1024:.0f} KB"],
            "action": [f"drift detector verdict, different in kind: {'yes' if shifted else 'no'}", cap,
                       f"current composition: {self.composition_text().splitlines()[0]}"],
        }
        if self.on("neutral"):
            ev["action"].append(f"changing costs about {RECOMPOSE_S:.1f} s of restart; this class lasted "
                                f"{self.mean_run():.1f} windows on average")
        return ev

    def verdict_criteria(self, qs, tiers):
        """Each option carries a plain-word cost (measured on this node) and benefit (from the
        active stream's observed, sampled and declared facts); the model weighs the two."""
        w, d, r = self.history[-1][1], self.history[-1][2], self.history[-1][3]
        reads = w["reuse"] >= 2
        need = w["bytes_put"] / (1 << 30)
        total = sum(t["remaining_gb"] for t in tiers.values())
        fits_raw, fits_comp = need <= 0.8 * total, need / max(r, 1.0) <= 0.8 * total
        searched = d["searched_later"] == "yes"
        slow = self.slowdown["read" if reads else "write"]

        def cost(m):
            x = slow[m]
            lvl = "low" if x < 0.10 else "moderate" if x < 0.45 else "high"
            return f"Cost on this node: {lvl} (slows this kind of stream by {100 * x:.0f}%)."
        comp_benefit = ("high (the stream does not fit uncompressed, and fits compressed)" if not fits_raw and fits_comp
                        else "low (the stream already fits uncompressed)" if r >= 1.3
                        else f"none (the stream barely compresses, {r:.1f}x)")
        v = {
            ("placement", "dram"): "Cost: none, it is the fastest tier. Benefit: high for any stream that fits.",
            ("placement", "ssd"): cost("ssd") + " Benefit: only extra capacity, which the stream "
                                  + ("does not need." if need <= 0.8 * self.dram_free(tiers) else "needs."),
            ("cache", "enable"): cost("cache") + " Benefit: " + ("high (the stream re-reads its data)."
                                                                if reads else "none (the stream does not re-read its data)."),
            ("cache", "disable"): "Cost: none. Benefit: keeps the chain short.",
            ("indexer", "enable"): cost("indexer") + " Benefit: " + ("required (the stream will be searched later)."
                                                                    if searched else "none (the stream will not be searched)."),
            ("indexer", "disable"): ("Cost: the stream can no longer be searched, which it needs." if searched
                                     else "Cost: none.") + " Benefit: avoids indexing work.",
            ("compression", "none"): ("Cost: none." if fits_raw else
                                      "Cost: writes will be lost, the stream does not fit uncompressed.")
                                     + " Benefit: fastest, no CPU work.",
            ("compression", "lz4"): cost("lz4") + " Benefit: " + comp_benefit + ".",
            ("compression", "zstd"): cost("zstd") + " Benefit: " + comp_benefit + "; slightly higher ratio than lz4.",
        }
        for (ax, opt), txt in v.items():
            qs[ax]["criteria"][opt] += " " + txt

    def dram_free(self, tiers):
        return tiers["dram"]["remaining_gb"]

    def questions_for(self, tiers):
        qs = self.questions()
        if self.on("neutral"):
            pre = ("Use the deployment, the deployed ChiMods and both memories. Judge what is best for the "
                   "active data stream on its own merits; the current composition is only a starting point. ")
            for qid, q in qs.items():
                q["instructions"] = q["instructions"].replace(ssc_live.Q_PREAMBLE, pre)
        if self.on("verdicts"):
            self.verdict_criteria(qs, tiers)
        elif self.on("options"):
            p = self.profile_text
            crit = {ax: qs[ax]["criteria"] for ax in ("placement", "cache", "indexer", "compression")}
            crit["placement"]["dram"] += " The fastest tier on this node; its capacity is limited."
            crit["placement"]["ssd"] += " " + p["SSD placement (vs DRAM)"]
            crit["cache"]["enable"] += " Pays off only when data is re-read soon. " + p["cache"]
            crit["indexer"]["enable"] += " Required if the data will be searched later. " + p["indexer"]
            crit["compression"]["none"] += " Fastest; stores the stream at full size."
            crit["compression"]["lz4"] += " " + p["lz4"]
            crit["compression"]["zstd"] += " " + p["zstd"]
        if self.on("facts") and not self.on("verdicts"):
            w, d, r = self.history[-1][1], self.history[-1][2], self.history[-1][3]
            need = w["bytes_put"] / (1 << 30)
            total = sum(t["remaining_gb"] for t in tiers.values())

            def fits(n):
                return "fits in the free space" if n <= 0.8 * total else "does NOT fit in the free space"
            crit = {ax: qs[ax]["criteria"] for ax in ("placement", "cache", "indexer", "compression")}
            crit["indexer"]["enable"] += f" The active stream: will be searched later = {d['searched_later']}."
            crit["indexer"]["disable"] += f" The active stream: will be searched later = {d['searched_later']}."
            crit["cache"]["enable"] += f" The active stream re-reads {w['reuse']:.1f} bytes per byte written."
            crit["compression"]["none"] += f" The active stream uncompressed {fits(need)} of the tiers."
            for codec in ("lz4", "zstd"):
                crit["compression"][codec] += (f" Sampled ratio of the active stream: {r:.1f}x; compressed it "
                                               f"{fits(need / max(r, 1.0))} of the tiers.")
            crit["placement"]["dram"] += f" The active stream is latency critical = {d['latency_critical']}."
        if self.on("outcomes"):
            vals ={"placement": {"dram": "dram", "ssd": "disk"}, "cache": {"enable": True, "disable": False},
                    "indexer": {"enable": True, "disable": False},
                    "compression": {"none": "none", "lz4": "lz4", "zstd": "zstd"}}
            for ax, m in vals.items():
                for opt, v in m.items():
                    qs[ax]["criteria"][opt] += self.option_outcome(ax, v)
        if self.on("local"):
            ev = self.evidence(tiers)
            for qid, q in qs.items():
                q["instructions"] += "\nEvidence for this question:\n" + "\n".join(f"- {e}" for e in ev[qid])
        return qs


def ask(state_text, qs):
    body = json.dumps(to_agentjev({"state_text": state_text, "questions": qs})).encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return norm_agentjev(json.loads(r.read().decode()), qs)


def recommended(ans, current):
    return {"placement": PLACE.get(ans.get("placement", {}).get("pred"), current["placement"]),
            "cache": ans.get("cache", {}).get("pred") == "enable",
            "indexer": ans.get("indexer", {}).get("pred") == "enable",
            "compression": ans.get("compression", {}).get("pred") or current["compression"]}


def gate(ans, current, w, tiers, ratio, action=None):
    """Fixed gate: only overrides the axis it gates (compression when the stream cannot fit
    uncompressed); never rewrites the other axes. Placement is left to CLIO's own spill."""
    action = action or ans.get("action", {}).get("pred")
    if action == "recompose":
        prop = {"placement": PLACE.get(ans.get("placement", {}).get("pred"), current["placement"]),
                "cache": ans.get("cache", {}).get("pred") == "enable",
                "indexer": ans.get("indexer", {}).get("pred") == "enable",
                "compression": ans.get("compression", {}).get("pred") or current["compression"]}
    else:
        prop = dict(current)
    need = w["bytes_put"] / (1 << 30)
    gated = False
    if prop["compression"] == "none" and need > 0.8 * sum(t["remaining_gb"] for t in tiers.values()):
        prop["compression"] = "zstd" if (ans.get("compression", {}).get("pred") in (None, "none")) \
            else ans["compression"]["pred"]
        gated = True
    return (prop if prop != current else None), gated


def run(policy, rows, grid, level=None, log=None):
    c = dict(compose.DEFAULT)
    b = VariantBuilder(c, level or 0, grid.profile())
    b.slowdown = grid.slowdown
    out = []
    phases_seen = []
    last_rec = None
    for r in rows:
        ph = r["phase"]
        mb, lost = grid.outcome(ph, c)
        rec = {"step": r["step"], "phase": ph, "composition": dict(c), "mb_per_s": mb, "failed_puts": lost,
               "best_mb_per_s": grid.best[ph][1], "best": grid.best[ph][0]}
        dom = Counter(phases_seen).most_common(1)[0][0] if phases_seen else None
        rec["true_shift"] = dom is not None and ph != dom
        phases_seen.append(ph)
        new = None
        if policy == "oracle_reactive":
            bk = grid.best[ph][0]
            best = dict(zip(("placement", "cache", "indexer", "compression"), bk))
            new = best if best != c else None
        elif policy == "agentjev":
            tiers = tiers_from(r["state_text"])
            b.current = dict(c)
            b.observe(r["window"], r["ratio"], DECLARED[ph])
            b.outcomes.append((b.history[-1][0], dict(c), mb, lost))
            st = b.state(tiers)
            qs = b.questions_for(tiers)
            ans = ask(st["text"], qs)
            action = None
            if b.on("derived"):
                rec_c = recommended(ans, c)
                action = "recompose" if rec_c != c and (ans["shift"]["pred"] or rec_c == last_rec) else "keep_current"
                last_rec = rec_c
            new, gated = gate(ans, c, r["window"], tiers, r["ratio"], action)
            rec.update(answers={k: v.get("pred") for k, v in ans.items()}, gated=gated)
            if log is not None and r["step"] in (0, 4, 13):
                log.write(f"===== step {r['step']} {ph}\n{st['text']}\n\n"
                          + "\n\n".join(f"[{k}] {q['instructions']}" for k, q in qs.items()) + "\n\n")
        if new is not None:
            c = dict(new)
            rec["recomposed_to"] = dict(c)
        out.append(rec)
    return out


AX = ("placement", "cache", "indexer", "compression")


def score(out):
    eff = [o["mb_per_s"] / o["best_mb_per_s"] for o in out]
    srch = sum(1 for o in out if DECLARED[o["phase"]]["searched_later"] == "yes" and not o["composition"]["indexer"])
    s = {"eff": sum(eff) / len(eff), "lost": sum(o["failed_puts"] for o in out), "search_miss": srch,
         "recomposes": sum(1 for o in out if o.get("recomposed_to"))}
    # requirement-met efficiency: a window that loses writes or misses a required index scores 0
    req = []
    for e, o in zip(eff, out):
        need_idx = DECLARED[o["phase"]]["searched_later"] == "yes"
        req.append(0 if o["failed_puts"] or (need_idx and not o["composition"]["indexer"]) else min(e, 1.0))
    s["req_eff"] = sum(req) / len(req)
    ans = [o for o in out if "answers" in o]
    if ans:
        s["shift_acc"] = sum(o["answers"]["shift"] == o["true_shift"] for o in ans) / len(ans)
        s["recommend_match"] = {}
        for i, a in enumerate(AX):
            def pred(o):
                v = o["answers"].get(a)
                return {"placement": PLACE.get(v), "cache": v == "enable", "indexer": v == "enable",
                        "compression": v}[a]
            s["recommend_match"][a] = sum(pred(o) == o["best"][i] for o in ans) / len(ans)
        s["action_recompose"] = sum(o["answers"]["action"] == "recompose" for o in ans)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loop")
    ap.add_argument("grid")
    ap.add_argument("out_dir")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(a.loop)]
    grid = Grid(a.grid)
    results = {}
    for pol in ("static", "oracle_reactive"):
        results[pol] = score(run(pol, rows, grid))
    for v in a.variants.split(","):
        with open(os.path.join(a.out_dir, f"ssc_{v}.txt"), "w") as log:
            out = run("agentjev", rows, grid, VARIANTS.index(v), log)
        with open(os.path.join(a.out_dir, f"replay_{v}.jsonl"), "w") as f:
            f.writelines(json.dumps(o) + "\n" for o in out)
        results[v] = score(out)
        print(v, json.dumps(results[v]), flush=True)
    json.dump(results, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(f"\n{'policy':16s} {'eff':>5s} {'reqEff':>6s} {'lost':>5s} {'srch':>4s} {'n_rc':>4s} {'shift':>5s}  "
          "rec: place cache index compr")
    for k, s in results.items():
        rm = s.get("recommend_match")
        print(f"{k:16s} {100 * s['eff']:4.0f}% {100 * s['req_eff']:5.0f}% {s['lost']:5.0f} {s['search_miss']:4d} "
              f"{s['recomposes']:4d} " + (f"{100 * s['shift_acc']:4.0f}%  " if 'shift_acc' in s else "    -  ")
              + ("  ".join(f"{100 * rm[x]:4.0f}%" for x in AX) if rm else ""))


if __name__ == "__main__":
    main()
