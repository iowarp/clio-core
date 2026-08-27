# The 1,000-timestep evolution study — raw record

The measurements behind each workload's "Default Evolving Benchmark
Configuration" section. **26 configurations**, every one run for 1,000
timesteps and sampled every 10, scored by `../evolution.py`.

```
<workload>/<config>.json            evolution.py's summary: mean/median/max/min,
                                    pct_active, pct_cells_same, and the full
                                    per-interval series for both metrics
<workload>/<config>.blocks.csv.gz   the RAW per-block values -- one row per
                                    (step_from, step_to, field, block) with its
                                    evolution and pct_cells_same
<workload>/run*.sh                  the exact sweep that produced them
warpx/FE_*.txt                      WarpX FieldEnergy, the evidence that
                                    do_moving_window=0 is a resonant cavity
```

Re-rank any workload without re-running anything:

```bash
mkdir -p /tmp/ev && for f in nyx/*.json; do
    d=/tmp/ev/$(basename $f .json); mkdir -p $d; cp $f $d/evolution.json; done
../evolution_rank.py /tmp/ev
```

`blocks.csv` is the file to go to for anything the summary does not answer —
per-field breakdowns, which blocks never moved, how a single field's activity
tracks the run. Both per-field tables in `vpic/README.md` and `nyx/README.md`
were computed from it, not from the summaries.

The dumps these were measured from are NOT kept: the four sweeps produced about
120 GB of field data between them (12.8 GB per VPIC configuration alone) and
each `run*.sh` deletes its own after measuring it. Re-run the script to
regenerate.
