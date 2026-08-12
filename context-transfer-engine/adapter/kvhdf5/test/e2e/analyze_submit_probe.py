#!/usr/bin/env python3
"""Join the submit probe's device and host halves and emit the per-hop breakdown.

Usage:  analyze_submit_probe.py <sweep_dir> [--csv out_dir]

The join key is the task POD's address (kvhdf5 forces the task backend to kPinnedHost, so
both sides name the same slot by the same address).

Two device clocks are in play and they are NOT interchangeable:

  * clock64() cycles resolve the fine hops (the field stores, the fence, the push). Valid
    only within one kernel launch on one block. Converted to ns with a ratio derived PER
    RECORD from that record's own long enter->wait span, which both clocks bracket on the
    same block — so SM clock boost is divided out rather than assumed away.

  * %globaltimer is the only device clock comparable to the host's, but it ticks in ~1024 ns
    steps. It is used ONLY for the two cross-domain hops, and their uncertainty is floored
    at one tick. A cross-domain hop is never quoted tighter than that.

In the ASYNC arm the fire and the drain are separate kernel launches, so clock64 is not
comparable across them: no cycles->ns ratio can be derived per record. Those records fall
back to the measured SM clock, and the fallback is reported rather than hidden.
"""

import csv
import os
import statistics
import sys

NS_PER_US = 1000.0


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: int(v) if v.lstrip("-").isdigit() else v
                 for k, v in row.items()} for row in csv.DictReader(f)]


def read_meta(path):
    rows = read_csv(path)
    return rows[0] if rows else None


def load_run(rundir, arm):
    dev = read_csv(os.path.join(rundir, f"probe_dev_{arm}.csv"))
    host = read_csv(os.path.join(rundir, f"probe_host_{arm}.csv"))
    meta = read_meta(os.path.join(rundir, f"probe_meta_{arm}.csv"))
    if not dev or not host or not meta:
        return []

    tick = meta["globaltimer_tick_ns"] or 1024
    sm_ghz = float(meta["sm_ghz"]) or 2.5

    # The device<->host clock offset DRIFTS (~20 ppm here — tens of us over one run, which
    # is larger than the poll latency itself). Two anchors bracket the run; interpolate
    # linearly in host time. Without this, completion-visibility comes out NEGATIVE.
    o0, t0 = meta["clock_offset_ns"], meta["clock_at_ns"]
    o1, t1 = meta["clock_offset_end_ns"], meta["clock_at_end_ns"]
    drift = o1 - o0

    def offset_at(t_host):
        if t1 == t0:
            return o0
        return o0 + (o1 - o0) * (t_host - t0) / (t1 - t0)

    by_ptr = {}
    for h in host:
        by_ptr.setdefault(h["task_ptr"], []).append(h)

    out = []
    for d in dev:
        cands = by_ptr.get(d["task_ptr"])
        if not cands:
            continue
        h = cands.pop(0)  # submits to one slot are strictly ordered
        if not cands:
            by_ptr.pop(d["task_ptr"])

        # A record whose device half never closed (kernel killed, cap exhausted) is dropped
        # rather than counted as a zero-latency hop.
        if not (d["c_enter"] and d["c_pushed"] and d["c_wait_end"]):
            continue
        if not (h["t_pop"] and h["t_sendout_end"] and h["t_exec_start"]):
            continue

        # cycles -> ns.
        #
        # SYNC: the send and the wait happen in the SAME kernel launch on the same block, so
        # clock64 is continuous across them and the enter->wait span (tens of ms, i.e. tens
        # of thousands of globaltimer ticks) yields a per-record ratio that divides out SM
        # clock boost exactly. Use it.
        #
        # ASYNC: WriteAsync runs in the fire kernel and WriteWait in a LATER, SEPARATE drain
        # kernel. clock64 is a per-SM free-running counter with no cross-launch guarantee —
        # a span across those two launches is not a valid cycle count, and deriving a ratio
        # from it would silently corrupt hops 1 and 2. Fall back to the measured SM clock
        # and say so. (The sync arm validates that fallback: see ratio_check below.)
        span_ns = d["d_wait_end"] - d["d_enter"]
        span_cy = d["c_wait_end"] - d["c_enter"]
        same_kernel = arm == "sync"
        derived = same_kernel and span_cy > 0 and span_ns > 20 * tick
        ns_per_cy = span_ns / span_cy if derived else 1.0 / sm_ghz

        def cy(a, b):
            return (d[b] - d[a]) * ns_per_cy

        rec = {
            "arm": arm,
            "ratio_derived": derived,
            # --- device, fine (cycle-resolved) ---
            "h1_fields_ns": cy("c_enter", "c_fields"),
            "h1b_qentry_ns": cy("c_fields", "c_prefence"),
            "h2a_fence_ns": cy("c_prefence", "c_postfence"),
            "h2b_push_ns": cy("c_postfence", "c_pushed"),
            "h12_submit_ns": cy("c_enter", "c_pushed"),
            # --- cross-domain: device push -> CPU observes it ---
            "h3_poll_ns": (h["t_pop"] - offset_at(h["t_pop"])) - d["d_pushed"],
            # --- host ---
            "h4_recvin_prologue_ns": h["t_pre_route"] - h["t_pop"],
            "h5_route_ns": h["t_post_route"] - h["t_pre_route"],
            "h6_queue_wait_ns": h["t_exec_start"] - h["t_post_route"],
            "h7_exec_ns": h["t_sendout_begin"] - h["t_exec_start"],
            "h8_writeback_ns": h["t_sendout_end"] - h["t_sendout_begin"],
            # --- cross-domain: CPU flips the flag -> device sees it ---
            "h9_completion_vis_ns": (d["d_wait_end"]
                                     - (h["t_sendout_end"] - offset_at(h["t_sendout_end"]))),
            # Offset-INVARIANT cross-check: the two cross-domain hops share one unknown
            # offset, so their SUM is independent of it entirely. If h3+h9 computed this way
            # disagrees with h3+h9 from the interpolated offsets, the interpolation is wrong.
            "h3_plus_h9_exact_ns": ((d["d_wait_end"] - d["d_pushed"])
                                    - (h["t_sendout_end"] - h["t_pop"])),
            "clock_drift_ns": drift,
            # --- denominators ---
            "device_rtt_ns": cy("c_enter", "c_wait_end"),
            "device_spin_ns": cy("c_wait_begin", "c_wait_end"),
            "host_service_ns": h["t_sendout_end"] - h["t_pop"],
            # --- the hot/idle discriminator ---
            "hot": h["idle_iters_at_pop"] == 0 and h["sleep_us_at_pop"] == 0,
            "idle_iters": h["idle_iters_at_pop"],
            "sleep_us": h["sleep_us_at_pop"],
            "worker_pop": h["worker_pop"],
            "worker_exec": h["worker_exec"],
            "seq": d["seq"],
            "tick_ns": tick,
            "ns_per_cy": ns_per_cy,
            "ns_per_cy_smclock": 1.0 / sm_ghz,
        }
        # The second denominator, and the one the paper's claim actually needs.
        #
        # Against the full round trip every hop but execution rounds to 0.0%, because the
        # storage I/O is milliseconds and the machinery is microseconds. That comparison is
        # true but says nothing: it would make ANY submit path look free. So we also total
        # the RUNTIME OVERHEAD -- every hop except the transfer engine's actual I/O -- and
        # ask what share of THAT the device-side submission is. This is the number that
        # distinguishes a cheap submit path from an expensive one.
        rec["overhead_ns"] = (rec["h12_submit_ns"] + max(rec["h3_poll_ns"], 0)
                              + rec["h4_recvin_prologue_ns"] + rec["h5_route_ns"]
                              + rec["h6_queue_wait_ns"] + rec["h8_writeback_ns"]
                              + max(rec["h9_completion_vis_ns"], 0))
        out.append(rec)
    return out


def stats(vals):
    if not vals:
        return None
    vals = sorted(vals)
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "p10": vals[int(0.10 * (len(vals) - 1))],
        "p90": vals[int(0.90 * (len(vals) - 1))],
        "min": vals[0],
        "max": vals[-1],
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


HOPS = [
    ("h1_fields_ns", "1  device: task-field stores"),
    ("h1b_qentry_ns", "1b device: queue-entry setup"),
    ("h2a_fence_ns", "2a device: __threadfence_system()"),
    ("h2b_push_ns", "2b device: ring-buffer push"),
    ("h12_submit_ns", "== DEVICE SUBMIT (hops 1+2) =="),
    ("h3_poll_ns", "3  cross:  CPU poll latency until Pop"),
    ("h4_recvin_prologue_ns", "4  host:   RecvIn prologue"),
    ("h5_route_ns", "5  host:   forced dispatch hop (RouteTask)"),
    ("h6_queue_wait_ns", "6  host:   worker-queue wait"),
    ("h7_exec_ns", "7  host:   execution (transfer engine)"),
    ("h8_writeback_ns", "8  host:   completion writeback (SendOut)"),
    ("h9_completion_vis_ns", "9  cross:  completion visible on device"),
    ("device_spin_ns", "-- device spin (Wait), for reference"),
    ("device_rtt_ns", "== DEVICE ROUND TRIP (denominator) =="),
]


def report(name, recs, out_dir=None):
    if not recs:
        print(f"\n### {name}: NO RECORDS\n")
        return
    tick = recs[0]["tick_ns"]
    hot = [r for r in recs if r["hot"]]
    idle = [r for r in recs if not r["hot"]]
    fallback = sum(1 for r in recs if not r["ratio_derived"])

    print(f"\n### {name}   (n={len(recs)}: {len(hot)} hot-spin, {len(idle)} idle-wakeup)")
    if fallback:
        print(f"    NOTE: {fallback}/{len(recs)} records used the measured SM clock rather "
              f"than a per-record ratio (async: fire and drain are separate kernels).")
    else:
        # Validates the SM-clock fallback the async arm has to use: if the per-record ratio
        # (which divides boost out exactly) agrees with the calibrated SM clock here, the
        # fallback is sound there too.
        d_ratio = statistics.median([r["ns_per_cy"] for r in recs])
        s_ratio = recs[0]["ns_per_cy_smclock"]
        print(f"    cycles->ns: per-record ratio {d_ratio:.4f} ns/cy vs SM-clock "
              f"{s_ratio:.4f} ns/cy  ({100*(d_ratio-s_ratio)/s_ratio:+.2f}%)")
    funnel = {r["worker_exec"] for r in recs}
    print(f"    executing workers: {sorted(funnel)}   (S3 funnel: all GPU tasks -> one worker)")
    print(f"    cross-domain hops (3, 9) carry a +/-{tick} ns floor from the globaltimer tick.")

    # Validate the drift correction against the offset-invariant sum. These must agree; if
    # they don't, the interpolated offset is wrong and no cross-domain hop can be trusted.
    lerp_sum = statistics.median([r["h3_poll_ns"] + r["h9_completion_vis_ns"] for r in recs])
    exact_sum = statistics.median([r["h3_plus_h9_exact_ns"] for r in recs])
    resid = lerp_sum - exact_sum
    drift = statistics.median([r["clock_drift_ns"] for r in recs])
    print(f"    clock drift over the run: {drift/NS_PER_US:.1f}us  |  offset-invariant check: "
          f"median(h3+h9) lerp={lerp_sum/NS_PER_US:.3f}us vs exact={exact_sum/NS_PER_US:.3f}us "
          f"(residual {resid/NS_PER_US:+.3f}us)")
    neg = sum(1 for r in recs if r["h9_completion_vis_ns"] < -tick)
    if neg:
        print(f"    WARNING: {neg}/{len(recs)} records still have a NEGATIVE (unphysical) "
              f"completion-visibility time — the offset correction is not sufficient.")
    print()
    print(f"    {'hop':<42} {'median':>11} {'p10':>10} {'p90':>10}   {'% of RTT':>9} {'% of ovh':>9}")
    rtt = statistics.median([r["device_rtt_ns"] for r in recs])
    ovh = statistics.median([r["overhead_ns"] for r in recs])
    for key, label in HOPS:
        s = stats([r[key] for r in recs])
        if not s:
            continue
        pct = 100.0 * s["median"] / rtt if rtt else 0.0
        po = 100.0 * s["median"] / ovh if (ovh and key not in
                                           ("h7_exec_ns", "device_rtt_ns",
                                            "device_spin_ns")) else float("nan")
        pos = f"{po:>8.2f}%" if po == po else " " * 9
        print(f"    {label:<42} {s['median']/NS_PER_US:>9.3f}us {s['p10']/NS_PER_US:>8.3f}us "
              f"{s['p90']/NS_PER_US:>8.3f}us   {pct:>8.2f}% {pos}")
    print(f"    {'== RUNTIME OVERHEAD (all hops but exec) ==':<42} {ovh/NS_PER_US:>9.3f}us"
          f"{'':>21}   100.00%")
    dev_share = 100.0 * statistics.median([r["h12_submit_ns"] for r in recs]) / ovh if ovh else 0
    print(f"    -> device-side submission is {dev_share:.1f}% of the runtime overhead, "
          f"{100.0*statistics.median([r['h12_submit_ns'] for r in recs])/rtt:.3f}% of the write.")

    for tag, group in (("hot-spin", hot), ("idle-wakeup", idle)):
        if not group:
            continue
        s = stats([r["h3_poll_ns"] for r in group])
        print(f"    poll latency, {tag:<12} median={s['median']/NS_PER_US:8.3f}us  "
              f"p10={s['p10']/NS_PER_US:7.3f}us  p90={s['p90']/NS_PER_US:8.3f}us  n={s['n']}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"hops_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hop", "label", "n", "median_ns", "mean_ns", "p10_ns", "p90_ns",
                        "min_ns", "max_ns", "stdev_ns", "pct_of_device_rtt",
                        "pct_of_runtime_overhead"])
            for key, label in HOPS:
                s = stats([r[key] for r in recs])
                if not s:
                    continue
                skip = key in ("h7_exec_ns", "device_rtt_ns", "device_spin_ns")
                w.writerow([key, label.strip(), s["n"], f"{s['median']:.1f}",
                            f"{s['mean']:.1f}", f"{s['p10']:.1f}", f"{s['p90']:.1f}",
                            f"{s['min']:.1f}", f"{s['max']:.1f}", f"{s['stdev']:.1f}",
                            f"{100.0*s['median']/rtt:.3f}" if rtt else "",
                            "" if skip or not ovh else f"{100.0*s['median']/ovh:.3f}"])
            s = stats([r["overhead_ns"] for r in recs])
            w.writerow(["overhead_ns", "RUNTIME OVERHEAD (all hops but exec)", s["n"],
                        f"{s['median']:.1f}", f"{s['mean']:.1f}", f"{s['p10']:.1f}",
                        f"{s['p90']:.1f}", f"{s['min']:.1f}", f"{s['max']:.1f}",
                        f"{s['stdev']:.1f}", f"{100.0*s['median']/rtt:.3f}", "100.000"])


def main():
    sweep = sys.argv[1]
    out_dir = None
    if "--csv" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--csv") + 1]

    all_recs = {}
    for label in sorted(os.listdir(sweep)):
        run_root = os.path.join(sweep, label)
        if not os.path.isdir(run_root):
            continue
        arm = "async" if "async" in label else "sync"
        recs = []
        for rep in sorted(os.listdir(run_root)):
            recs += load_run(os.path.join(run_root, rep), arm)
        if recs:
            all_recs[label] = recs

    for label, recs in all_recs.items():
        report(label, recs, out_dir)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "all_submits_raw.csv"), "w", newline="") as f:
            cols = ["config", "arm", "seq", "hot", "idle_iters", "sleep_us", "worker_pop",
                    "worker_exec", "ratio_derived"] + [k for k, _ in HOPS]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for label, recs in all_recs.items():
                for r in recs:
                    r = dict(r, config=label)
                    w.writerow(r)
        print(f"\nwrote {out_dir}/all_submits_raw.csv and per-config hops_*.csv")


if __name__ == "__main__":
    main()
