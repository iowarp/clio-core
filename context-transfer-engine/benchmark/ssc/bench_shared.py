"""Helpers shared with the SSC-Bench question set (copied from ssc_bench.generate / ssc_bench.run_eval
so this harness runs from a single clio-core clone)."""

AXIS_Q = {
    "placement": "Which storage tier should the active data stream be placed on?",
    "block_device": "Which block device type should back the active data stream?",
    "cache": "Should the cache ChiMod be in the chain for the active data stream?",
    "compression": "Which compression setting should the chain use for the active data stream?",
    "replication": "How many copies should the chain keep of the active data stream?",
    "indexer": "Should the indexer ChiMod track the active data stream?",
    "prefetch": "Should the prefetcher ChiMod be in the chain for the active data stream?",
    "aggregation": "Should the aggregator ChiMod be in the chain for the active data stream?",
    "dedup": "Should the dedup ChiMod be in the chain for the active data stream?",
    "encryption": "Should the encryption ChiMod be in the chain for the active data stream?",
    "tier_migration": "Should the tier_migrator ChiMod run for the active data stream?",
    "order_compress_encrypt": "If the chain for the active data stream uses both compressor and encryption, in which order should they run?",
    "order_dedup_compress": "If the chain for the active data stream uses both dedup and compressor, in which order should they run?",
}
Q_PREAMBLE = ("Use the deployment, the deployed ChiMods, the current composition and both memories. "
              "The active data stream is whatever workload the short-term memory says is running now; "
              "if it says the usual workload continues, that workload is described in the long-term memory. ")
URGENCY_LEVELS = [
    "No change to the composition is needed.",
    "A change would help, but it can wait for the next job boundary or maintenance window.",
    "The current composition clearly mismatches the active workload and should change soon.",
    "Act immediately: data is at risk of loss, confidentiality is violated, or a tier is about to run out of space.",
]
ACTION_OPTS = {
    "keep_current": "Leave the current ChiMod composition unchanged.",
    "recompose": "Change the composition now to fit the active workload.",
    "escalate": "Defer to a slower reasoning system: the active workload cannot be classified confidently "
                "(contradictory or incomplete signals), or it needs a capability or capacity this system "
                "does not have.",
}


def to_agentjev(record):
    qs = []
    for qid, q in record["questions"].items():
        if q["type"] == "noul":
            qs.append({"id": qid, "type": "boolean", "question": q["instructions"],
                       "criteria": {"true": "Yes", "false": "No"}})
        elif q["type"] == "choice":
            qs.append({"id": qid, "type": "choice", "question": q["instructions"],
                       "options": q["criteria"]})
        else:
            qs.append({"id": qid, "type": "score", "question": q["instructions"],
                       "levels": q["criteria"]})
    return {"state": record["state_text"], "questions": qs}


def norm_agentjev(resp, questions):
    out = {}
    answers = resp["results"][0]["answers"]
    for a in answers:
        qid = a["id"]
        q = questions.get(qid)
        if q is None:
            continue
        if q["type"] == "noul":
            dist = a.get("distribution") or {}
            p = float(dist.get("true", a.get("prob_true", a.get("probability", 0.0))))
            out[qid] = {"pred": p >= 0.5, "conf": max(p, 1 - p), "p_true": p}
        elif q["type"] == "choice":
            out[qid] = {"pred": a.get("value"), "conf": a.get("top_probability", a.get("probability")),
                        "dist": a.get("distribution")}
        else:
            # "level" is the argmax; "score" is the expected level, which can round to a level nobody picked
            out[qid] = {"pred": float(a["level"] if "level" in a else a.get("score", 0)), "conf": None}
    return out
