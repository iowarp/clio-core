"""Client process: run workload phases against the running chain and report JSON.

    python3 measure.py '<composition json>' phase1,phase2,... [repeat]

For each phase: one unmeasured warm-up run (first-touch cost of a fresh runtime), then
`repeat` measured runs. Every run's blobs are deleted afterwards so each run starts from
the same tier state. Prints one JSON line per measured run: throughput, failed puts and
the bytes the run consumed on each tier (remaining-space delta).
"""

import json
import os
import sys
import time

import clio_cte_core_ext as cte

import workload


def tier_remaining(client):
    client.AsyncStatTargets().wait()
    time.sleep(1.2)     # target stats are re-scanned every poll_period_ms (1 s)
    return {n: client.AsyncGetTargetInfo(n).result().remaining_space for n in client.AsyncListTargets().result()}


def settle(client, baseline, timeout_s=20.0):
    """Wait until every tier has its baseline free space back (the previous run's deletes are
    really freed), so each run starts from the same tier state. Returns (seconds, clean)."""
    t0 = time.time()
    while True:
        now = tier_remaining(client)
        if all(now[n] >= baseline[n] - (8 << 20) for n in baseline):
            return time.time() - t0, True
        if time.time() - t0 > timeout_s:
            return time.time() - t0, False


def main():
    c = json.loads(sys.argv[1])
    phases = sys.argv[2].split(",")
    repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    assert cte.clio_init(cte.RuntimeMode.kClient, False), "clio_init failed"
    # binds the client to the chain top named by CLIO_CTE_POOL
    assert cte.initialize_cte(os.environ["CLIO_SERVER_CONF"], cte.PoolQuery.Dynamic()), "initialize_cte failed"
    client = cte.get_cte_client()
    baseline = tier_remaining(client)          # empty tiers, before any phase
    for p in phases:
        if p != "gs_archive":      # the archive fills the tiers; its first run is its own warm-up
            workload.run_phase(cte, p, f"{p}_warm", seed=99)
            workload.delete_phase(cte, client, p, f"{p}_warm")
        for r in range(repeat):
            settle_s, clean = settle(client, baseline)
            before = tier_remaining(client)
            res = workload.run_phase(cte, p, f"{p}_{r}", seed=r)
            after = tier_remaining(client)
            res["used_bytes"] = {n: before[n] - after[n] for n in before}
            used = sum(max(v, 0) for v in res["used_bytes"].values())
            ram = sum(max(v, 0) for n, v in res["used_bytes"].items() if "ram" in n)
            res["dram_fraction"] = round(ram / used, 3) if used else None
            res["clean_start"] = clean
            res["settle_s"] = round(settle_s, 2)
            res["composition"] = c
            res["rep"] = r
            print(json.dumps(res), flush=True)
            workload.delete_phase(cte, client, p, f"{p}_{r}")


if __name__ == "__main__":
    main()
