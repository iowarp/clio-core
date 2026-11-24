"""Live SSC builder for a running CLIO node.

Context comes from three sources (challenge C1):
    observed  CTE target stats (score, free space) and a client-side op trace; the CTE
              telemetry ring is kept as a cross-check only, because zero-IPC shared-memory
              reads never reach it and interposers (the indexer's read-backs) add ops of
              their own
    sampled   a recently written blob is read back and test-compressed (compressibility)
    declared  what the application states about its data (durability, searched later)

TelemetryWindow polls the ring in a background thread (it holds only 1024 entries) and
summarises one window. SSCBuilder keeps the window history (long-term memory), runs a
drift detector on the last window (short-term memory) and renders the five SSC sections
plus the controller summary in the same shape and wording as SSC-Bench.
"""

import os
import sys
import threading
import time
import zlib
from collections import Counter

from bench_shared import AXIS_Q, Q_PREAMBLE, ACTION_OPTS, URGENCY_LEVELS  # noqa: E402

MB = 1 << 20
TIER_KEY = {"dram": "dram", "disk": "ssd"}            # question option key per CLIO tier
TIER_TEXT = {"dram": "DRAM buffer tier", "disk": "node-local SSD tier"}
CODECS = ["lz4", "zstd"]


class TelemetryWindow:
    """Accumulates CTE telemetry entries between start() and stop()."""

    def __init__(self, cte, client, period_s=0.05):
        self.cte, self.client, self.period = cte, client, period_s
        self.last_lt = 0
        self.entries = []
        self._stop = threading.Event()
        self._t = None

    def _poll(self):
        got = self.client.AsyncPollTelemetryLog(self.last_lt).result()
        fresh = [e for e in got if e.logical_time_ > self.last_lt]
        if fresh:
            self.last_lt = max(e.logical_time_ for e in fresh)
            self.entries.extend(fresh)

    def _loop(self):
        while not self._stop.is_set():
            self._poll()
            time.sleep(self.period)

    def start(self):
        self.entries = []
        self._poll()          # advance past anything older than this window
        self.entries = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        self.t0 = time.time()

    def stop(self):
        self._stop.set()
        self._t.join()
        self._poll()
        self.t1 = time.time()
        return self.summary()

    def summary(self):
        put = self.cte.CteOp.kPutBlob
        get = self.cte.CteOp.kGetBlob
        puts = [e for e in self.entries if e.op_ == put]
        gets = [e for e in self.entries if e.op_ == get]
        bp, bg = sum(e.size_ for e in puts), sum(e.size_ for e in gets)
        return {"n_put": len(puts), "n_get": len(gets), "bytes_put": bp, "bytes_get": bg,
                "mean_put": bp / len(puts) if puts else 0, "mean_get": bg / len(gets) if gets else 0,
                "reuse": bg / bp if bp else 0.0, "seconds": self.t1 - self.t0}


def tier_state(client):
    """{"dram"|"disk": {"remaining_gb", "score"}} from the CTE target stats (observed)."""
    client.AsyncStatTargets().wait()
    out = {}
    for name in client.AsyncListTargets().result():
        info = client.AsyncGetTargetInfo(name).result()
        out["dram" if name.startswith("ram::") else "disk"] = {
            "remaining_gb": info.remaining_space / (1 << 30), "score": info.target_score}
    return out


def sample_compressibility(tag, blob_name, size):
    data = tag.GetBlob(blob_name, size)[:MB]
    return len(data) / max(1, len(zlib.compress(data, 1)))


def classify(w, ratio):
    """Rule-based workload class from observed features (the 'digested' label)."""
    if w["n_get"] and w["reuse"] >= 2:
        access = "small hot set read many times" if w["mean_get"] <= 2 * MB else "large blobs re-read"
    elif w["mean_put"] >= MB:
        access = "large sequential writes"
    elif w["mean_put"] and w["mean_put"] < 64 * 1024:
        access = "many small writes"
    else:
        access = "mixed reads and writes"
    comp = "high" if ratio >= 3 else "moderate" if ratio >= 1.3 else "none (already compressed or random)"
    return access, comp


class SSCBuilder:
    """Holds the window history across runtime restarts; needs no CLIO connection itself.
    Tier state is observed by the client process (tier_state) and passed to state()."""

    SPILL_AWARE = False              # SSC v3 capacity wording (see capacity)
    HIDE_CURRENT = False             # SSC v4: no current-composition section (see state)

    def __init__(self, current):
        self.current = dict(current)
        self.history = []                # list of (class tuple, window summary, declared)

    # sections ---------------------------------------------------------------
    def deployment(self, tiers):
        cpu = os.cpu_count()
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1 << 30)
        return (f"A single-node CLIO deployment with {cpu} cores and {mem:.0f} GB of RAM. "
                f"CLIO manages a DRAM buffer tier ({tiers['dram']['remaining_gb']:.1f} GB free) and a "
                f"node-local SSD tier ({tiers['disk']['remaining_gb']:.1f} GB free). There is no shared "
                f"parallel file system and only one node.")

    @staticmethod
    def composability():
        return ("- core (CTE): tag and blob store that places data on one of the deployed storage tiers. "
                "Always the last stage of the chain. The placement tier is chosen per data stream.\n"
                "- cache: sits in front of core and keeps recently written or read blobs in the fastest tier "
                "so repeat reads are served from memory. Only helps when the same data is read again soon; "
                "otherwise it wastes fast-tier capacity.\n"
                "- compressor: compresses blobs on the way in and decompresses on the way out. Adds CPU time "
                "and latency; useless on data that is already compressed. Deployed codecs:\n"
                "  - lz4: very fast lossless codec with a modest ratio. Adds little CPU time.\n"
                "  - zstd: slower lossless codec with a higher ratio. Best for cold data and text; too slow "
                "for high-rate ingest or latency-sensitive paths.\n"
                "- indexer: tracks every blob and tag mutation and maintains a searchable index. Costs extra "
                "work on every write; only worth it when people or agents will search this data later.\n"
                "Not in the chain on this system: replication, safe_bdev, summarizer. The core data "
                "organizer (tier movement ahead of use) is off.\n"
                "Composition rule: interceptor ChiMods form an ordered chain that ends in core, "
                "cache -> indexer -> compressor -> core.")

    def composition_text(self):
        c = self.current
        chain = [m for m, on in (("cache", c["cache"]), ("indexer", c["indexer"]),
                                 ("compressor (" + c["compression"] + ")", c["compression"] != "none")) if on]
        return (f"Chain for the main data stream: {' -> '.join(chain + ['core'])}. "
                f"Placement: {TIER_TEXT[c['placement']]}.")

    @staticmethod
    def window_text(cls, w, declared):
        access, comp = cls
        return (f"  class: {access}\n  compressibility (sampled): {comp}\n"
                f"  writes: {w['n_put']} ops, mean {w['mean_put'] / 1024:.0f} KB; "
                f"reads: {w['n_get']} ops; bytes read per byte written: {w['reuse']:.1f}\n"
                f"  durability: {declared['durability']}\n  searched later: {declared['searched_later']}\n"
                f"  latency critical: {declared['latency_critical']}")

    def observe(self, window, ratio, declared):
        cls = classify(window, ratio)
        self.history.append((cls, window, declared, ratio))
        return cls

    @staticmethod
    def capacity(tiers, w, ratio):
        """Will another window of this stream fit (80% headroom), raw and compressed at the sampled ratio?"""
        need = w["bytes_put"] / (1 << 30)
        dram, total = tiers["dram"]["remaining_gb"], sum(t["remaining_gb"] for t in tiers.values())

        def verdict(n):
            if n <= 0.8 * dram:
                return "fits in the DRAM buffer tier"
            if n <= 0.8 * total:
                # CTE spills to the next tier whatever the placement, so naming a tier here only
                # pulls the model toward that tier (SSC v3)
                return ("fits in the free space; what does not fit in the preferred tier spills over "
                        "automatically, so this does not require changing placement" if SSCBuilder.SPILL_AWARE
                        else "fits only across both tiers")
            return "does not fit in the free space of both tiers"
        return (f"capacity for one window of the active stream ({need:.2f} GB written): uncompressed "
                f"{verdict(need)}; compressed at the sampled ratio ({ratio:.1f}x) {verdict(need / max(ratio, 1.0))}")

    def drift(self):
        """Drift detector over the class history: dominant = most frequent past class."""
        past = [h[0] for h in self.history[:-1]]
        cur = self.history[-1][0]
        if not past:
            return cur, None, False, True
        dom = Counter(past).most_common(1)[0][0]
        return cur, dom, cur != dom, cur not in past

    def state(self, tiers):
        cur, dom, shifted, novel = self.drift()
        last_w, last_d, last_r = self.history[-1][1], self.history[-1][2], self.history[-1][3]
        cap = self.capacity(tiers, last_w, last_r)
        if dom is None:
            lt = "No history yet: this is the first observed window."
        else:
            n = sum(1 for h in self.history[:-1] if h[0] == dom)
            dw, dd = next((h[1], h[2]) for h in reversed(self.history[:-1]) if h[0] == dom)
            others = sorted({h[0][0] for h in self.history[:-1] if h[0] != dom})
            lt = (f"window: {len(self.history) - 1} past windows\ndominant ({n} of {len(self.history) - 1} windows):\n"
                  + self.window_text(dom, dw, dd)
                  + (f"\nalso seen: {'; '.join(others)}" if others else ""))
        st = (f"window: last {last_w['seconds']:.1f} s\n"
              f"drift detector: active class = {cur[0]}; matches dominant: {'no' if shifted else 'yes'}; "
              f"seen in history: {'no' if novel else 'yes'}\nactive workload:\n" + self.window_text(cur, last_w, last_d)
              + "\nCurrent fill levels: " + ", ".join(
                  f"{TIER_TEXT[t]} {v['remaining_gb']:.1f} GB free" for t, v in tiers.items()) + "\n" + cap)
        sections = {"deployment": self.deployment(tiers), "composability": self.composability(),
                    "current_composition": self.composition_text(), "long_term_memory": lt,
                    "short_term_memory": st}
        if SSCBuilder.HIDE_CURRENT:
            # SSC v4: the model judges what fits the workload; the controller, which knows the
            # deployed chain, decides whether that is a change (the model anchored on it otherwise)
            del sections["current_composition"]
        text = "\n\n".join(f"## {k.replace('_', ' ').upper()}\n{v}" for k, v in sections.items())
        summary = ("## CONTROLLER SUMMARY\n"
                   f"drift detector: active class = {cur[0]}; different in kind from the dominant workload: "
                   f"{'yes' if shifted else 'no'}; never seen in the long-term history: {'yes' if novel else 'no'}\n"
                   + cap + "\n"
                   "options this system can run: placement: dram, ssd | compression: none, lz4, zstd | "
                   "cache: enable, disable | indexer: enable, disable")
        return {"sections": sections, "text": summary + "\n\n" + text,
                "drift": {"class": cur, "shift": shifted, "novel": novel}, "capacity": cap}

    @staticmethod
    def questions():
        def ch(axis, opts):
            return {"type": "choice", "instructions": Q_PREAMBLE + AXIS_Q[axis], "criteria": opts}
        return {
            "shift": {"type": "noul", "instructions": Q_PREAMBLE +
                      "Is the workload active now different in kind from this system's dominant long-term "
                      "pattern? A change in rate, a short burst that has already ended, or a hardware event "
                      "does not change the kind of workload."},
            "action": {"type": "choice", "instructions": Q_PREAMBLE +
                       "What should the storage controller do with the ChiMod composition right now?",
                       "criteria": dict(ACTION_OPTS)},
            "placement": ch("placement", {"dram": "Place the active stream's data on the DRAM buffer tier.",
                                          "ssd": "Place the active stream's data on the node-local SSD tier."}),
            "cache": ch("cache", {"enable": "Include the cache ChiMod in the chain for the active stream.",
                                  "disable": "Leave cache out of the chain for the active stream."}),
            "compression": ch("compression", {"none": "No compression.", "lz4": "Compress with lz4.",
                                              "zstd": "Compress with zstd."}),
            "indexer": ch("indexer", {"enable": "Include the indexer ChiMod in the chain for the active stream.",
                                      "disable": "Leave indexer out of the chain for the active stream."}),
        }
