"""Client process for one controller step: run a phase window and report.

    python3 step.py <phase> <tag_name> <seed>

Prints one JSON line: throughput, the client-side trace summary (what the SSC uses), the
core telemetry summary (cross-check), the sampled compressibility of a blob written in
this window, and the tier state. The window's blobs are then deleted, as if the stream
had been drained, so windows do not accumulate in the tiers.
"""

import json
import os
import sys

import clio_cte_core_ext as cte

import ssc_live
import workload


def main():
    phase, tag_name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    assert cte.clio_init(cte.RuntimeMode.kClient, False), "clio_init failed"
    assert cte.initialize_cte(os.environ["CLIO_SERVER_CONF"], cte.PoolQuery.Dynamic()), "initialize_cte failed"
    client = cte.get_cte_client()
    tele = ssc_live.TelemetryWindow(cte, client)
    trace = workload.ClientTrace()
    should_pause = None
    if os.environ.get("SSC_WATERMARK"):
        # capacity watermark (controller reflex): pause the window once free space falls below
        # this fraction of what was free when the window started
        wm = float(os.environ["SSC_WATERMARK"])
        free0 = sum(t["remaining_gb"] for t in ssc_live.tier_state(client).values())
        should_pause = lambda i: sum(t["remaining_gb"] for t in ssc_live.tier_state(client).values()) < wm * free0
    tele.start()
    res = workload.run_phase(cte, phase, tag_name, seed=seed, trace=trace, start=start, should_pause=should_pause)
    core = tele.stop()
    if "paused_at" in res:
        # the controller recomposes and resumes the window in a new client; keep the blobs
        print(json.dumps({"result": res, "window": trace.summary(), "core_telemetry": core}), flush=True)
        return
    blob_bytes = workload.PHASES[phase][0]
    try:
        ratio = ssc_live.sample_compressibility(cte.Tag(tag_name), "b0", blob_bytes)
    except RuntimeError:
        ratio = 1.0
    tiers = ssc_live.tier_state(client)
    workload.delete_phase(cte, client, phase, tag_name)
    print(json.dumps({"result": res, "window": trace.summary(), "core_telemetry": core, "ratio": ratio,
                      "tiers": ssc_live.tier_state(client), "tiers_before_drain": tiers}), flush=True)


if __name__ == "__main__":
    main()
