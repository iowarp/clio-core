# External-baseline campaign

NeuroPress against fixed strategies and the external GPU compressors it cannot
select — cuSZ, cuSZp v3 and ndzip — on nyx, vpic, warpx and lammps.

## Why these codecs are baselines, not candidates

`kNeuroPressTrainedGpuBaseIds = {13,14,15,16,17,18,23,24}`
(`neuropress_bridge.cc`) filters cuSZ/cuSZp/ndzip out of the candidate set
before ranking, and the model has no output slot for them: its action space is
8 nvcomp algorithms x quantize x shuffle = 32. Upstream is the same — its own
trace campaign records ndzip as the per-chunk optimum on ~15% of chunks and
NeuroPress selecting it 0.0% of the time. So these arms measure what the
**action space** gives up, which is a separate question from whether the
**selector** is good.

## One-time setup

```bash
./install_codecs.sh --arch 80        # builds cuSZp v3 + ndzip, patches included
cp env.sh site.sh                    # then edit site.sh with your paths (gitignored)
```

Then configure clio-core with the codecs on `CMAKE_PREFIX_PATH` — see the
message `install_codecs.sh` prints. **Verify `CLIO_CTP_ENABLE_{CUSZ,CUSZP,NDZIP}`
are all `ON`.** If a codec is missing the arm still runs: `WireIdForName` falls
back to zstd and produces a plausible but wrong result. `run_static.sh` guards
against this with a codec census, but check the flags anyway.

`cmake` must be >= 3.28 — clio-core uses `set_tests_properties(DIRECTORY ...)`,
and an older cmake fails configure with hundreds of "Can not find test to add
properties to".

## Running

Three jobs per workload, in order. Add `-A <account>` — the `#SBATCH --account`
line is a placeholder.

```bash
sbatch -A acct --export=ALL,WL=nyx phase1.sbatch      # 32-action oracle sweep
sbatch -A acct --export=ALL,WL=nyx,TAG=-cc,W_CT=1,W_DT=0,W_IO=1,COST_BW=1.2e6 \
       phase1b.sbatch                                 # NeuroPress: adaptive vs frozen
sbatch -A acct --export=ALL,WL=nyx phase3.sbatch      # cusz / cuszp / ndzip
python3 analyze/perchunk_cost_all_workloads.py 1.2e6 5e6               # the cross-workload table
```

VPIC needs its dumps generated first (`sbatch vpic_gen.sbatch`, ~35 min); nyx
and warpx use `paper-benchmark/<wl>/gen_fields.sh`; lammps is in-situ.

## Cost models — read this before comparing numbers

Two different costs appear in this work and they give **opposite answers**:

| model | weights | what wins |
|---|---|---|
| ratio-only | `W_CT=0 W_DT=0 W_IO=1` | high-ratio slow codecs (`zstd\|q1\|s4`) |
| time-inclusive | `W_CT=1 W_DT=0 W_IO=1` | fast codecs (`bitcomp\|q1\|s4`) |

`perchunk_cost_all_workloads.py` scores `compress_ms + bytes/BW`, so NeuroPress must be **run**
under matching weights (`W_CT=1`) or it is graded on an objective it was never
given. That is what `TAG=-cc` above is for, and `perchunk_cost_all_workloads.py` prefers a
`<wl>-cc` run when one exists.

Compression time is measured; transfer time is modelled at `BW`. Pass several
bandwidths — cuSZ's viability is entirely a bandwidth question.

## Caveats that affect how results should be read

- **Sweep timings run ~5% high.** `ct_ms` in `explore.csv` is measured during a
  32x-per-chunk sweep, under contention a standalone run lacks (3.177 vs
  3.029 ms for the same action). Harmless for ratio-only cost, a real bias for
  time-inclusive cost — it inflates sweep-derived rows (optimum, best-fixed)
  relative to directly measured ones.
- **Online SGD is not deterministic.** Repeat runs of the same adaptive arm
  varied 1.2% in stored bytes; the frozen arm was bit-identical across a full
  rebuild. Do not report a margin under ~2% from a single run.
- **Verify fidelity before ranking a lossy codec.** cuSZ and cuSZp missed
  eb=1e-3 on 151-276 of 300 chunks wherever it was checked. The in-situ LAMMPS
  path has no `--verify`, so its lossy arms are unproven.
- **LAMMPS covers 162 of 300 chunks** — its `best_mode` sweep hits a CUDA OOM.
  Runs without the sweep reach 468 chunks, so this limits the oracle only.
