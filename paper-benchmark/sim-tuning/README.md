# Simulation tuning: making per-chunk codec choice pay

Goal: simulation workloads where NeuroPress v2 (learning on, exploration off)
beats storing every chunk with the best single codec, which is picked by
exhaustive search under the balanced 4-tier cost model. The model is never
retrained on these data.

## Nyx: Sedov blasts in a layered multiphase medium

Patches (apply on top of `../nyx/patches/nyx-raw-field-dump.patch`; rebuild
`nyx_HydroTests`):

- `../nyx/patches/nyx-sedov-multiphase.patch`: new `prob.*` options for the
  Sedov problem in `Exec/HydroTests/Prob.cpp`, and `max_prob_param` raised
  to 32.
  - `nphase`: number of ambient phases.
  - `phase_rho`, `phase_p`: phase density and pressure contrast. With
    `phase_p=1` the phases are in pressure equilibrium, so they stay exactly
    as set until a blast reaches them.
  - `phase_block`: clump size in cells.
  - `phase_pow2`: phase densities are exact powers of two.
  - `nlayer`: alternate clumpy and smooth layers along z. z is the slowest
    index, so one 4 MiB chunk lies in one layer.
  - `smooth_amp`, `smooth_k`, `smooth_exp`: the smooth layers' density wave,
    `rho0 * exp(amp * wave)` when `smooth_exp=1`.
  - `nblast`: number of explosions.
  - `noise_dens`, `noise_vel`: per-cell noise in the ambient density and
    velocity.
- `../nyx/patches/nyx-dump-select-derived.patch`: two environment variables
  for the raw field dump.
  - `NYX_DUMP_STATE="density rho_E"` writes only those state fields, the way
    `amr.plot_vars` selects plotfile variables.
  - `NYX_DUMP_DERIVED="logden ..."` also writes those derived fields.

Chosen setup (`nyx_probe.sh final25g ...`, see `/mnt/nvme0/v2-work/runs/nyx_tuning.log`):

```
NCELL=256 STEPS=2600 PLOT_INT=20 STATE="density rho_E" DERIVED="logden"
prob.nphase=8 prob.phase_pow2=1 prob.phase_p=1 prob.nlayer=4
prob.smooth_exp=1 prob.smooth_amp=1.0 prob.smooth_k=1
prob.nblast=1 prob.exp_energy=0.05 prob.p_ambient=7.62939453125e-06
```

## Why this works

- **Clumpy layers favour `ans` with byte shuffle.** Each cell holds one of 8
  power-of-two densities, mixed at random, so the data has low byte entropy
  but no smoothness. `ans` with byte shuffle compresses it about 5.7×;
  `spratio` gets about 1.2×.
- **Smooth layers favour `spratio`.** The density wave is gentle, so
  `spratio`'s neighbour prediction works well there.
- **The model reads both correctly.** The trained model, unchanged, ranks
  both kinds well. `pattern_lab.py` finds such pairs from the training sweep
  tool and the model's predictions.
- **The data changes over time.** The blast sweeps through both kinds of
  layer, turning them into shocked flow.

## Tuning history (128³ unless noted; cost vs best single codec)

| Probe | Possible gain | NeuroPress, learning, 1 pass |
|---|---|---|
| Sedov, state fields only | 0.1% | +38% |
| + 12 derived fields | 3.8% | +43% |
| 8 phases with different pressures | 5.4% | +8.8% |
| 8 phases in pressure equilibrium | 5.6% | +11.5% |
| + clumpy / smooth layers, power-of-two levels, `exp` wave | 13.7% | +17.5% |
| same, fields `density`, `logden`, `rho_E` only (offline replay) | 19.3% | −7.0% |
| same at 256³, 4 layers | 26.8% | −10.1% (replay); **−11.4% through Clio** |

What hurt NeuroPress, so these were dropped from the output:

- **Momentum fields.** They are exactly zero outside the blast, and the model
  over-predicts the ratio of bit-shuffle codecs on mostly-zero chunks by up
  to 1,000×.
- **`Temp`, `soundspeed`, `divu`, `magvort`, `MachNumber`.** Their chunks are
  noisy, so no codec gains much, or the model mispredicts them.

## Scripts

- `nyx_probe.sh`: one probe run (generate, stage, exhaustive search, score).
- `probe_eval.py`: opportunity per field, plus NeuroPress's result from the
  learning replay.
- `np_mistakes.py`: where NeuroPress's extra cost over the best single codec
  comes from.
- `pattern_lab.py`: candidate chunk patterns, swept and scored offline.
- `tune_sims.sh`, `tune_report.py`: parameter sweeps for LAMMPS and Nyx.
