"""SSC v8: experience-grounded storage system context (offline candidate, developed on seeds 1-3).

Design rules, fixed before any run (from the context-design literature: small models copy what
they see, ignore negation, are weak with numbers, and cannot learn machine costs from prose):
  1. No ChiMod descriptions in prose. Options appear only in the answer schema, and in the evidence
     every option of a question is listed in a fixed order with the same keys.
  2. Facts are positive, typed fields ("fits: yes / does not fit"), no advice, no "X does not help".
  3. Numbers are pre-digested: workload sizes are compared to the free space, measured costs are
     turned into relative words (fastest / a little slower / about 3x slower).
  4. Machine-specific costs come only from measurements on this node, as experience: the k most
     similar OTHER workloads measured on this node (leave-workload-out: the window's own workload is
     never in its memory, so the context never contains the answer for the window it describes).
  5. Order: node experience first, the current window last; question-local evidence.

Baseline for the same information without a model: knn_decide (nearest workload's fastest options,
restricted to codecs that fit).
"""

import math
import statistics
import zlib
from collections import defaultdict

import json
import workload

AXES = ("placement", "cache", "indexer", "compression")
OPTS = {"placement": ("dram", "disk"), "cache": (True, False), "compression": ("none", "lz4", "zstd")}
WORD = {"placement": {"dram": "dram", "disk": "ssd"}, "cache": {True: "enable", False: "disable"},
        "compression": {"none": "none", "lz4": "lz4", "zstd": "zstd"}}
GB = 1 << 30


# ---------------------------------------------------------------- node memory
def load_cells(path):
    """(phase, composition key) -> (mean wall seconds, lost writes) from a measured grid."""
    acc = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        c = r["composition"]
        acc[(r["phase"], (c["placement"], c["cache"], c["indexer"], c["compression"]))].append(
            (r["wall_s"], r.get("failed_puts", 0)))
    return {k: (statistics.median(v[0] for v in vs), max(v[1] for v in vs)) for k, vs in acc.items()}


def phase_facts(phase):
    """Machine-free description of one window of a workload (what the client trace shows)."""
    blob, n, reads, kind = workload.PHASES[phase]
    data = workload.payload(kind, blob, 0, 0)[:workload.MB]
    ratio = len(data) / max(1, len(zlib.compress(bytes(data), 1)))
    return {"blob": blob, "n_put": n, "n_get": reads, "ratio": ratio}


FIT_AWARE = False   # v11: similarity also compares how far raw / compressed data overflow DRAM
DRAM_GB = 1.0       # this node's DRAM tier size (set by the builder from the tiers)


def features(f):
    base = (math.log2(f["blob"]), math.log2(f["n_put"]), math.log2(1 + f["n_get"] / f["n_put"]),
            math.log2(max(f["ratio"], 1.0)))
    if not FIT_AWARE:
        return base
    gb = f["blob"] * f["n_put"] / GB
    over = lambda size: math.log2(max(size / DRAM_GB, 1.0))     # 0 while it fits in DRAM
    return base + (over(gb), over(gb / max(f["ratio"], 1.0)))


def neighbours(cur, memory, k):
    """The k most similar remembered workloads: (distance, phase, facts)."""
    fc = features(cur)
    d = sorted((math.dist(fc, features(f)), ph, f) for ph, f in memory.items())
    return d[:k]


def similarity(dist):
    return "high" if dist < 1.5 else "medium" if dist < 3.0 else "low"


def option_times(cells, phase, axis, idx):
    """Fastest measured time on `phase` for each option of `axis` (other axes at their best, the
    indexer as the window requires); None if every composition with that option lost writes."""
    out = {}
    for opt in OPTS[axis]:
        ts = [t for (ph, key), (t, lost) in cells.items()
              if ph == phase and not lost and key[2] == idx and key[AXES.index(axis)] == opt]
        out[opt] = min(ts) if ts else None
    return out


# v12 switches (defaults = the frozen v10 behaviour)
SHOW_RATIO = False  # v12: every cost word carries the measured ratio, e.g. "a little slower (1.4x)"
CAPACITY = 0.8      # share of the free space counted as usable; v12: 1.0, the physical capacity


def relative(t, best):
    if t is None:
        return "lost writes (did not fit)"
    r = t / best
    if r <= 1.15:
        word = "fastest"
    elif r <= 1.6:
        word = "a little slower"
    else:
        word = f"about {r:.0f}x slower" if r < 10 else "more than 10x slower"
    return f"{word} ({r:.1f}x)" if SHOW_RATIO else word


# ---------------------------------------------------------------- window facts
def fits(w, ratio, tiers):
    need = w["bytes_put"] / GB
    free = sum(t["remaining_gb"] for t in tiers.values())
    raw = need <= CAPACITY * free
    comp = need / max(ratio, 1.0) <= CAPACITY * free
    return {"none": raw, "lz4": comp, "zstd": comp}, need, free


def tier_fit(size_gb, tiers):
    """Where one window of `size_gb` lands on this node: DRAM first, overflow to the SSD."""
    dram, total = tiers["dram"]["remaining_gb"], sum(t["remaining_gb"] for t in tiers.values())
    if size_gb > CAPACITY * total:
        return "does not fit in the free space"
    if size_gb <= CAPACITY * dram:
        return f"fits in DRAM ({size_gb:.2f} GB)"
    return f"fills DRAM, about {size_gb - CAPACITY * dram:.1f} GB spills to the second tier"


def size_text(blob):
    return f"{blob // workload.MB} MB" if blob >= workload.MB else f"{blob // 1024} KB"


def facts_text(f):
    reads = (f"{f['n_get']} reads per window ({f['n_get'] / f['n_put']:.0f} reads per blob written)"
             if f["n_get"] else "write-only")
    return (f"writes: {f['n_put']} blobs of {size_text(f['blob'])} per window; reads: {reads}; "
            f"compressibility: {f['ratio']:.1f}x")


# ---------------------------------------------------------------- the SSC
PREAMBLE = ("Use the node experience (measured on this node) and the facts of the current window. "
            "Choose the option that is fastest for the current window on this node, among the options "
            "that fit in the free space. ")
QTEXT = {"placement": "Which storage tier should the current window's data be placed on?",
         "cache": "Should the cache be in the chain for the current window?",
         "compression": "Which compression setting should the chain use for the current window?",
         "indexer": "Should the indexer track the current window?"}
CRITERIA = {"placement": {"dram": "DRAM buffer tier.", "ssd": "Node-local SSD tier."},
            "cache": {"enable": "Cache in the chain.", "disable": "No cache in the chain."},
            "compression": {"none": "No compression.", "lz4": "Compress with lz4.", "zstd": "Compress with zstd."},
            "indexer": {"enable": "Indexer in the chain.", "disable": "No indexer in the chain."}}


def build(w, ratio, declared, tiers, phase, cells, k=3, experience=True):
    """state text and questions for one window. `phase` is used only to leave the window's own
    workload out of the memory (the model never sees the phase name)."""
    global DRAM_GB
    DRAM_GB = tiers["dram"]["remaining_gb"]
    cur = {"blob": w["mean_put"], "n_put": w["n_put"], "n_get": w["n_get"], "ratio": ratio}
    memory = {ph: phase_facts(ph) for ph in workload.remembered(phase)}
    nb = neighbours(cur, memory, k) if experience else []
    idx = declared["searched_later"] == "yes"
    fit, need, free = fits(w, ratio, tiers)

    lines = []
    if nb:
        lines.append("## NODE EXPERIENCE (other workloads measured on this node, most similar first)")
        for i, (d, ph, f) in enumerate(nb):
            lines.append(f"workload {chr(65 + i)} (similarity to the current window: {similarity(d)}): {facts_text(f)}")
        lines.append("")
    lines.append("## CURRENT WINDOW")
    lines.append(facts_text(cur))
    lines.append(f"data written per window: {need:.2f} GB; free space: {free:.1f} GB")
    if FIT_AWARE:
        lines.append(f"free space: DRAM {tiers['dram']['remaining_gb']:.1f} GB, SSD {tiers['disk']['remaining_gb']:.1f} GB "
                     "(data goes to DRAM first; what does not fit spills to the SSD)")
        lines.append("where one window lands: " + "; ".join(
            f"{o}: {tier_fit(need / (1.0 if o == 'none' else max(ratio, 1.0)), tiers)}" for o in OPTS["compression"]))
    else:
        lines.append("fits in the free space: " + "; ".join(
            f"{WORD['compression'][o]}: {'yes' if fit[o] else 'does not fit'}" for o in OPTS["compression"]))
    lines.append(f"searched later: {declared['searched_later']}; latency critical: {declared['latency_critical']}")
    text = "\n".join(lines)

    qs = {}
    for axis in AXES:
        ev = []
        if axis == "indexer":
            ev.append(f"searched later: {declared['searched_later']}")
        else:
            for i, (d, ph, f) in enumerate(nb):
                ot = option_times(cells, ph, axis, idx)
                best = min(t for t in ot.values() if t is not None)
                ev.append(f"workload {chr(65 + i)} measured: " + "; ".join(
                    f"{WORD[axis][o]}: {relative(ot[o], best)}" for o in OPTS[axis]))
            if axis == "compression":
                ev.append("current window, fits in the free space: " + "; ".join(
                    f"{o}: {'yes' if fit[o] else 'does not fit'}" for o in OPTS["compression"]))
        qs[axis] = {"type": "choice", "criteria": dict(CRITERIA[axis]),
                    "instructions": PREAMBLE + QTEXT[axis] + "\nEvidence for this question:\n"
                    + "\n".join(f"- {e}" for e in ev)}
    return text, qs


SPEED_ONLY = False   # v10: experience carries speed only (see digest)


def digest(nb, cells, axis, idx):
    """v9: one experience verdict per option, merged over the neighbours (closer workloads weigh
    more): similarity-weighted mean of log(time / fastest); lost writes count as 10x slower.
    v10 (SPEED_ONLY): a neighbour where an option lost writes did not FIT; that says nothing about
    the option's speed on the current window, whose fit is stated separately. Such a neighbour is
    left out of that option's average (10x only if the option lost writes on every neighbour)."""
    acc = {o: 0.0 for o in OPTS[axis]}
    wsum = {o: 0.0 for o in OPTS[axis]}
    for d, ph, _ in nb:
        ot = option_times(cells, ph, axis, idx)
        best = min(t for t in ot.values() if t is not None)
        wt = 1.0 / (1.0 + d)
        for o in OPTS[axis]:
            if ot[o] is None and SPEED_ONLY:
                continue
            wsum[o] += wt
            acc[o] += wt * math.log(10.0 if ot[o] is None else ot[o] / best)
    return {o: math.exp(acc[o] / wsum[o]) if wsum[o] else 10.0 for o in acc}


def build_v9(w, ratio, declared, tiers, phase, cells, k=3, experience=True):
    """v9: v8's context, but each question's evidence is one line per option (fixed order, same
    keys) with its feasibility and its merged measured cost on similar workloads of this node.
    experience=False is the cold start (a node with no measurements yet): the same context with the
    node-experience block and every "on similar workloads" line left out; only the window's own
    facts remain."""
    text, qs = build(w, ratio, declared, tiers, phase, cells, k, experience=experience)
    cur = {"blob": w["mean_put"], "n_put": w["n_put"], "n_get": w["n_get"], "ratio": ratio}
    memory = {ph: phase_facts(ph) for ph in workload.remembered(phase)}
    nb = neighbours(cur, memory, k)
    idx = declared["searched_later"] == "yes"
    fit, need, _ = fits(w, ratio, tiers)
    for axis in ("placement", "cache", "compression"):
        rel = digest(nb, cells, axis, idx)
        best = min(rel.values())
        lines = []
        for o in OPTS[axis]:
            parts = []
            if axis == "compression" and FIT_AWARE:
                parts.append(tier_fit(need / (1.0 if o == "none" else max(ratio, 1.0)), tiers))
            elif axis == "compression":
                parts.append("fits in the free space" if fit[o] else "does not fit in the free space")
            elif axis == "placement" and FIT_AWARE and o == "dram":
                parts.append("uncompressed window " + tier_fit(need, tiers))
            if experience:
                parts.append(f"on similar workloads on this node: {relative(rel[o] * 1.0, best)}")
            lines.append(f"{WORD[axis][o]}: " + ("; ".join(parts) if parts else "no measurements on this node yet"))
        qs[axis]["instructions"] = (PREAMBLE + QTEXT[axis] + "\nEvidence for each option:\n"
                                    + "\n".join(f"- {e}" for e in lines))
    qs["indexer"]["instructions"] = (PREAMBLE + QTEXT["indexer"] + "\nEvidence for each option:\n"
                                     f"- enable: {'required: the data is searched later' if idx else 'adds work to every write'}\n"
                                     f"- disable: {'the data is not searchable later' if idx else 'no extra work'}")
    return text, qs


def wknn_decide(w, ratio, declared, tiers, phase, cells, k=3):
    """No model, v9's information: per axis the option with the lowest merged cost, codecs
    restricted to those that fit."""
    global DRAM_GB
    DRAM_GB = tiers["dram"]["remaining_gb"]
    cur = {"blob": w["mean_put"], "n_put": w["n_put"], "n_get": w["n_get"], "ratio": ratio}
    memory = {ph: phase_facts(ph) for ph in workload.remembered(phase)}
    nb = neighbours(cur, memory, k)
    idx = declared["searched_later"] == "yes"
    fit, _, _ = fits(w, ratio, tiers)
    rec = {"indexer": idx}
    for axis in ("placement", "cache", "compression"):
        rel = digest(nb, cells, axis, idx)
        cand = {o: r for o, r in rel.items() if axis != "compression" or fit[o]} or {"zstd": 0.0}
        rec[axis] = min(cand, key=cand.get)
    return rec


def knn_decide(w, ratio, declared, tiers, phase, cells):
    """No model: the nearest other workload's fastest option per axis, codecs restricted to those
    that fit (the same information v8 gives the model)."""
    cur = {"blob": w["mean_put"], "n_put": w["n_put"], "n_get": w["n_get"], "ratio": ratio}
    memory = {ph: phase_facts(ph) for ph in workload.remembered(phase)}
    _, ph, _ = neighbours(cur, memory, 1)[0]
    idx = declared["searched_later"] == "yes"
    fit, _, _ = fits(w, ratio, tiers)
    rec = {"indexer": idx}
    for axis in ("placement", "cache", "compression"):
        ot = option_times(cells, ph, axis, idx)
        cand = {o: t for o, t in ot.items() if t is not None and (axis != "compression" or fit[o])}
        if axis == "compression" and not cand:
            cand = {"zstd": 0.0}
        rec[axis] = min(cand, key=cand.get)
    return rec


def decisive(w, ratio, declared, tiers, phase, cells, k=3):
    """v12d: per axis, the option the node experience ranks best when the evidence is decisive, i.e.
    every other option is outside the "fastest" bin (more than 1.15x slower); codecs restricted to
    those that fit. Axes where the evidence is a near-tie are left to the model (absent)."""
    global DRAM_GB
    DRAM_GB = tiers["dram"]["remaining_gb"]
    cur = {"blob": w["mean_put"], "n_put": w["n_put"], "n_get": w["n_get"], "ratio": ratio}
    memory = {ph: phase_facts(ph) for ph in workload.remembered(phase)}
    nb = neighbours(cur, memory, k)
    idx = declared["searched_later"] == "yes"
    fit, _, _ = fits(w, ratio, tiers)
    out = {}
    for axis in ("placement", "cache", "compression"):
        rel = digest(nb, cells, axis, idx)
        cand = {o: r for o, r in rel.items() if axis != "compression" or fit[o]}
        if len(cand) < 2:
            continue
        best = min(cand, key=cand.get)
        if all(r / cand[best] > 1.15 for o, r in cand.items() if o != best):
            out[axis] = best
    return out
