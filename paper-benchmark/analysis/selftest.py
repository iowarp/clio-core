#!/usr/bin/env python3
"""Schema-conformance self-test for analyze_exploration.py.

Builds SYNTHETIC exploration logs -- clearly labelled as such, and never
presented as measurements -- whose only job is to drive the code paths that a
given real log happens not to exercise. The Nyx sweep this was developed
against is an OLD 25-column log with one error bound, so without this the
error-bound stage, the current 31-column header and the degraded paths would
ship untested.

    ./selftest.py            # build, run, check, clean up

Each case asserts on the pipeline's OUTPUT, not on its internals.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "analyze_exploration.py")

#: The header the current writer emits (neuropress_telemetry.cc).
CURRENT_SCHEMA = [
    "seq", "blob", "chunk_bytes", "role", "rank", "lib_name", "algo_idx",
    "preset", "quantize", "shuffle", "pred_ratio", "pred_ct_ms", "pred_dt_ms",
    "ratio", "ct_ms", "dt_ms", "psnr_db", "cost", "primary_cost", "adopted",
    "quality_measured", "meas_rmse", "meas_max_error", "meas_psnr_db",
    "meas_ssim", "meas_ssim_deviation", "entropy", "mad", "second_deriv",
    "eb_encoded",
]

CODECS = ["nvcomp-lz4", "nvcomp-zstd", "nvcomp-ans", "nvcomp-bitcomp"]


def synth(n_steps=6, n_fields=3, bounds=(1e-3, 1e-2), seed=7,
          anisotropy: float = 0.0) -> pd.DataFrame:
    """A synthetic sweep with SEVERAL error bounds and the current header.

    The generative model is deliberately simple and known: ratio falls with
    entropy, rises with the error bound, and each codec has a fixed offset.
    That gives the self-test something to assert against -- the pipeline must
    recover the sign of a relationship that was put there on purpose.

    `anisotropy` plants the failure the matched-control stage exists to find:
    two extra components `vy` and `vz` carrying byte-for-byte the SAME feature
    values as `vx` at every timestep, whose ratios differ by a fixed factor.
    Nothing computable from entropy, MAD and the second derivative can tell
    them apart, so a stage that reports them as explained is broken.
    """
    rng = np.random.default_rng(seed)
    rows = []
    seq = 0
    fields = ["density", "vx", "temp"][:n_fields]
    # The planted twins carry vx's features exactly and a scaled ratio.
    twins = {"vy": 1.0 + anisotropy, "vz": 1.0 + 2.0 * anisotropy} \
        if anisotropy > 0 and "vx" in fields else {}
    fields = fields + list(twins)
    for step in range(n_steps):
        for fi, fld in enumerate(fields):
            blob = f"plt{step:05d}/fab0000_comp{fi:02d}_{fld}/chunk_0"
            # A twin's features are vx's, to the bit -- no noise is drawn for
            # it, because the whole point is that they are indistinguishable.
            twin_gain = twins.get(fld)
            # vx is the base of the family: it must be noise-free as well, or
            # it would not be identical to its own twins and the fixture would
            # plant a weaker failure than the one being tested for.
            is_component = bool(twins) and (twin_gain or fld == "vx")
            feat_i = fields.index("vx") if is_component else fi
            # Entropy climbs with time, as a spreading front does.
            entropy = (1.0 + 0.5 * step + 0.3 * feat_i
                       + (0.0 if is_component else rng.normal(0, 0.05)))
            mad = 0.02 * (feat_i + 1) * (1 + 0.3 * step)
            sd = 0.5 * mad
            chunk_bytes = 1 << 20
            cands = []
            for ci, lib in enumerate(CODECS):
                for quant in (0, 1):
                    for shuf in (0, 4):
                        ebs = bounds if quant else (None,)
                        for eb in ebs:
                            base = 40.0 / max(entropy, 0.2)
                            r = base * (twin_gain or 1.0) \
                                * (1.0 + 0.15 * ci) \
                                * (3.0 if quant else 1.0) \
                                * (1.0 + (0.4 if shuf else 0.0)) \
                                * (1.0 + (200 * eb if eb else 0.0)) \
                                * float(rng.lognormal(0, 0.06))
                            ct = 0.5 * (1 + ci) * float(rng.lognormal(0, 0.1))
                            dt = 0.3 * (1 + ci) * float(rng.lognormal(0, 0.1))
                            cands.append((lib, ci, quant, shuf, eb, r, ct, dt))
            # Cost with the DOCUMENTED defaults, so the self-test also checks
            # that the recovery finds non-zero ct/dt weights (the real Nyx log
            # has them both at zero, so that path was never exercised).
            def cost(ct, dt, r):
                return (1.0 * max(1.0, ct) + 1.0 * max(1.0, dt)
                        + 1.0 * chunk_bytes / (min(100.0, r) * 5e6))
            costs = [cost(c[6], c[7], c[5]) for c in cands]
            best_i = int(np.argmin(costs))
            prim_i = int(rng.integers(0, len(cands)))
            primary_cost = costs[prim_i]
            for i, (lib, ci, quant, shuf, eb, r, ct, dt) in enumerate(cands):
                is_prim = i == prim_i
                rows.append({
                    "seq": seq, "blob": blob, "chunk_bytes": chunk_bytes,
                    "role": "primary" if is_prim else "alt",
                    "rank": -1 if is_prim else i,
                    "lib_name": lib, "algo_idx": ci, "preset": 2,
                    "quantize": quant, "shuffle": shuf,
                    "pred_ratio": r * float(rng.lognormal(0, 0.3)),
                    "pred_ct_ms": ct * float(rng.lognormal(0, 0.3)),
                    "pred_dt_ms": dt * float(rng.lognormal(0, 0.3)),
                    "ratio": r, "ct_ms": ct, "dt_ms": dt,
                    "psnr_db": (min(120.0, 10 * np.log10(1.0 / (eb ** 2 / 3)))
                                if quant else -1.0),
                    "cost": costs[i], "primary_cost": primary_cost,
                    "adopted": int(i == best_i),
                    "quality_measured": 1,
                    "meas_rmse": (eb / 2 if quant else 0.0),
                    "meas_max_error": (eb * 0.98 if quant else 0.0),
                    "meas_psnr_db": (80.0 if quant else 120.0),
                    "meas_ssim": 1.0 - (1e-6 if quant else 0.0),
                    "meas_ssim_deviation": (1e-6 if quant else 0.0),
                    "entropy": entropy, "mad": mad, "second_deriv": sd,
                    # The writer's own rule: input 3 is `quant ? eb : 1e-7`.
                    "eb_encoded": eb if quant else 1e-7,
                })
                seq += 1
    return pd.DataFrame(rows)[CURRENT_SCHEMA]


def run(args, cwd):
    p = subprocess.run([sys.executable, DRIVER] + args, cwd=cwd,
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
    return p


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="np_selftest_")
    failures = []

    def chk(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name
              + (f"  {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    try:
        # ---- case 1: current 31-column schema, several error bounds ------
        df = synth()
        p1 = os.path.join(tmp, "synthetic_current.csv")
        df.to_csv(p1, index=False)
        r = run([p1, "-o", os.path.join(tmp, "out1")], tmp)
        chk("current-schema log runs to completion", r.returncode == 0,
            r.stderr.strip().splitlines()[-1] if r.returncode else "")
        o1 = os.path.join(tmp, "out1", "synthetic_current")
        s1 = json.load(open(os.path.join(o1, "summary.json")))
        chk("features read from the log itself (no sidecar needed)",
            s1["dataset"]["usable_features"] ==
            ["entropy", "mad", "second_deriv"],
            str(s1["dataset"]["usable_features"]))
        chk("both error bounds detected",
            len(s1["dataset"]["error_bounds_lossy"] or []) == 2,
            str(s1["dataset"]["error_bounds_lossy"]))
        chk("the 1e-7 lossless sentinel is NOT read as an error bound",
            all(abs(b - 1e-7) > 1e-12
                for b in (s1["dataset"]["error_bounds_lossy"] or [])))
        chk("error-bound comparison produced rows",
            os.path.exists(os.path.join(o1, "paired",
                                        "error_bound_comparison.csv")))
        chk("error-bound stage NOT reported unavailable",
            "error_bound" not in s1["unavailable_analyses"])
        cm = s1["cost_model"]
        chk("cost weights recovered as the documented defaults",
            cm["identified"] and abs(cm["w_ct"] - 1.0) < 1e-6
            and abs(cm["w_dt"] - 1.0) < 1e-6,
            f"w_ct={cm['w_ct']:.4g} w_dt={cm['w_dt']:.4g} "
            f"cap={cm['cap']:g}")
        # The synthetic data was BUILT with ratio falling in entropy; the
        # pipeline must recover that sign or something is wrong upstream.
        wf = pd.read_csv(os.path.join(
            o1, "summary", "feature_correlations_within_field.csv"))
        ent = wf[wf.feature == "entropy"]["pooled_within_field_r"]
        chk("the planted entropy->ratio relationship is recovered, negative",
            len(ent) and (ent < 0).mean() > 0.9,
            f"{(ent < 0).mean() * 100:.0f}% of cells negative, "
            f"median r={ent.median():+.2f}")
        ebc = pd.read_csv(os.path.join(o1, "paired",
                                       "error_bound_comparison.csv"))
        chk("the planted bound->ratio relationship is recovered, positive",
            (ebc["d_ratio"] > 0).mean() > 0.9,
            f"{(ebc['d_ratio'] > 0).mean() * 100:.0f}% of pairs improve")
        eb_fig = os.path.join(o1, "figures", "error_bound_vs_ratio.png")
        chk("error-bound figure drawn when bounds vary",
            os.path.exists(eb_fig))

        # ---- case 2: features absent, no sidecar -> degrades, not crashes -
        d2 = df.drop(columns=["entropy", "mad", "second_deriv"])
        sub = os.path.join(tmp, "nofeat")
        os.makedirs(sub, exist_ok=True)
        p2 = os.path.join(sub, "nofeatures.csv")
        d2.to_csv(p2, index=False)
        r = run([p2, "-o", os.path.join(tmp, "out2"), "--no-figures"], tmp)
        chk("log with no features and no sidecar still completes",
            r.returncode == 0)
        s2 = json.load(open(os.path.join(tmp, "out2", "nofeatures",
                                         "summary.json")))
        chk("no features -> reported unavailable, not as a null result",
            s2["dataset"]["usable_features"] == []
            and "correlations" in s2["unavailable_analyses"])
        chk("REPORT.md still written without features",
            os.path.getsize(os.path.join(tmp, "out2", "nofeatures",
                                         "REPORT.md")) > 2000)

        # ---- case 3: dt never measured (the runtime default) --------------
        d3 = df.copy()
        d3["dt_ms"] = -1.0
        sub3 = os.path.join(tmp, "nodt")
        os.makedirs(sub3, exist_ok=True)
        p3 = os.path.join(sub3, "nodt.csv")
        d3.to_csv(p3, index=False)
        r = run([p3, "-o", os.path.join(tmp, "out3"), "--no-figures"], tmp)
        chk("log with dt_ms = -1 everywhere completes", r.returncode == 0)
        o3 = os.path.join(tmp, "out3", "nodt")
        s3 = json.load(open(os.path.join(o3, "summary.json")))
        chk("dt = -1 is treated as unmeasured, not as a fast decompression",
            "dt_prediction" in s3["unavailable_analyses"])
        w3 = pd.read_csv(os.path.join(o3, "summary", "winners.csv"))
        chk("no fastest-dt winner is invented from the -1 sentinel",
            w3["fastest_dt_value"].isna().all())

        # ---- case 4: an unreadable input must not lose the good one -------
        bad = os.path.join(tmp, "broken.csv")
        with open(bad, "w") as fh:
            fh.write("not,an,exploration,log\n1,2,3,4\n")
        r = run([p1, bad, "-o", os.path.join(tmp, "out4"), "--no-figures"],
                tmp)
        chk("a broken input is reported but does not lose the good one",
            r.returncode == 1
            and os.path.exists(os.path.join(tmp, "out4",
                                            "synthetic_current", "REPORT.md")),
            "exit 1 with the good workload still written")

        # ---- case 5: a PLANTED anisotropy must be found, not explained ---
        # vy and vz carry vx's features to the bit and compress 1.5x and 2x
        # better. Nothing computable from those features can separate them, so
        # a matched-control stage that reports the log as clean is broken --
        # and so is a ceiling that comes back at 1.0.
        d5 = synth(anisotropy=0.5)
        sub5 = os.path.join(tmp, "aniso")
        os.makedirs(sub5, exist_ok=True)
        p5 = os.path.join(sub5, "aniso.csv")
        d5.to_csv(p5, index=False)
        r = run([p5, "-o", os.path.join(tmp, "out5"), "--no-figures"], tmp)
        chk("log with planted anisotropy completes", r.returncode == 0,
            r.stderr.strip().splitlines()[-1] if r.returncode else "")
        o5 = os.path.join(tmp, "out5", "aniso")
        s5 = json.load(open(os.path.join(o5, "summary.json")))
        an = s5.get("anisotropy") or {}
        chk("the planted anisotropy is DETECTED", bool(an.get("anisotropic")),
            str(an.get("reasons"))[:160])
        fam = pd.read_csv(os.path.join(o5, "anisotropy",
                                       "component_families.csv"))
        chk("the vector family is detected from the field names alone",
            set(fam["axis"]) == {"x", "y", "z"} and set(fam["base"]) == {"v"},
            f"{sorted(set(fam['base']))} {sorted(set(fam['axis']))}")
        sp = pd.read_csv(os.path.join(o5, "anisotropy",
                                      "feature_blind_spread.csv"))
        tight = sp[sp["max_rel_feature_diff_at_most"] <= 1e-6]
        chk("pairs with IDENTICAL features still show the planted 2x gap",
            len(tight) and 1.7 <= float(tight.iloc[0]["max_ratio_fold"]) <= 2.4,
            f"max fold {float(tight.iloc[0]['max_ratio_fold']):.2f}x at "
            f"tol 1e-06" if len(tight) else "no pairs at tol 1e-06")
        ce = pd.read_csv(os.path.join(o5, "anisotropy",
                                      "explainable_ceiling.csv"))
        chk("the R^2 ceiling is pulled below 1 by identical-feature chunks",
            len(ce) and float(ce.iloc[0]["max_achievable_oof_r2"]) < 0.99,
            f"ceiling {float(ce.iloc[0]['max_achievable_oof_r2']):.4f}"
            if len(ce) else "no ceiling rows")
        od = pd.read_csv(os.path.join(o5, "anisotropy",
                                      "component_ordering.csv"))
        xz = od[(od.axis_a == "x") & (od.axis_b == "z")]
        chk("the ordering is recovered with the right winner",
            len(xz) and xz.iloc[0]["more_compressible_axis"] == "z",
            f"winner {xz.iloc[0]['more_compressible_axis']}, "
            f"fold {float(xz.iloc[0]['median_ratio_fold']):.2f}x, "
            f"p={float(xz.iloc[0]['sign_test_p']):.3g}" if len(xz) else "")
        # And the negative control: the ORIGINAL fixture has no twins, so the
        # stage must not manufacture a family or a finding out of it.
        fam1 = os.path.join(o1, "anisotropy", "component_families.csv")
        chk("no vector family is invented when the fields are unrelated",
            not os.path.exists(fam1),
            "component_families.csv absent for density/vx/temp")

        print("\n" + "=" * 60)
        print(f"{'ALL PASS' if not failures else 'FAILURES: ' + str(failures)}")
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
