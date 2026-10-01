# Four-timestep temporal classifier

`classify_temporal.py` splits every field of four timesteps (T0 < T1 < T2 < T3)
into the same spatial blocks. It then labels each block **Exact dedup**,
**Near dedup**, **Cold** or **Hot** from how the block changes across
T0→T1, T1→T2 and T2→T3. It reports every metric the task asks for.

## Run

```bash
# Validate first: known exact / cold / hot blocks (including the task's three
# examples) must come out right, and the shares must sum to 100%.
python3 classify_temporal.py --self-test

# Nyx, three windows of four consecutive dumps (dump k ≈ step 10k):
for w in "50 51 52 53" "150 151 152 153" "450 451 452 453"; do
  set -- $w; L=plt$(printf %03d $1)-$(printf %03d $4)
  python3 classify_temporal.py \
    --root /projects/bekn/imuradli/np-dumps/nyx/fields --frames $w \
    --label "Nyx Sedov 128^3, dumps $1-$4" \
    --out /projects/bekn/imuradli/temporal-classifier/nyx/$L \
    --fig-dir ../figures/temporal-classifier/nyx/$L
done
```

Options (defaults in brackets):
- `--block` sets the block edge [16].
- `--block-sizes` sets the granularity sweep [8 16 32 64].
- `--rel-tol` sets the element-change tolerance [1e-6].
- `--near-frac` sets the changed fraction allowed for NEAR [0].
- `--score` picks the change score, `rms` or `l2` [rms].
- `--threshold` fixes the Cold/Hot split [derived].

One window of 6 fields × 128³ × 4 timesteps (201 MB) takes about 70–90 s on a login node, with about 370 MiB peak RSS.

## Method

Each block transition gets one state, checked in this order:

1. **EXACT**: the block is byte-identical. An xxh3-64 hash is compared first, then every byte is compared as well.
2. **NEAR**: not byte-identical, but at most `--near-frac` of the elements moved by more than `--rel-tol × max(|a|,|b|)`.
3. **COLD / HOT**: the change score is at most / above the threshold.

**Change score.** The default `rms` score is the block's RMS change divided by the whole field's RMS, so all blocks of a field share one scale. The task's block-normalized L2 (`l2`) divides by the block's own values instead. As a result, any block near zero reads about 1: on Nyx, that is 2.3% of transitions, mostly momentum at the blast centre. It is still reported for every block.

**Threshold.** The Cold/Hot threshold is derived from the data, not assumed. The script takes log10(score) over all non-dedup consecutive transitions and fits a two-component Gaussian mixture:
- If the two components are clearly separate (Ashman D ≥ 2), their posterior crossing is the threshold.
- Otherwise it uses Otsu's split and says the split is not a population boundary.

The Otsu, mixture, histogram-valley and P50/P75/P90 candidates are all printed.

**Window class.** A block's class is the majority of its three states at the top level (Dedup / Cold / Hot):
- Three different states resolve to the most severe one.
- A Dedup majority is EXACT only when two of the states are EXACT.
- Blocks that are Hot in any transition are also flagged `transient_hot`, and a stricter "most severe state" class is reported alongside.

## Outputs

Results go under `--out`:
- **`summary.txt`** holds:
  - thresholds and how they were chosen;
  - shares by bytes and by blocks, with the exact vs near breakdown;
  - a per-transition table, including the long-distance transitions;
  - distributions with percentiles;
  - the element-level dedup ceiling;
  - the state transition matrix and three-transition paths;
  - spatial clustering;
  - change concentration (share held by the top 1/5/10/20% of blocks);
  - the granularity sweep, with the threshold both re-derived and held fixed;
  - answers to the research questions;
  - runtime.
- **`blocks.csv`** and **`blocks.parquet`** have one row per field × block: hashes, the `exact_duplicate` relation, every metric for all six transitions, states and class.
- **`thresholds.json`**, **`granularity.csv`** and **`transition_matrix.csv`**.

Figures go under `--fig-dir`:
- `fig1_2_change_distributions` (the chosen score and changed fraction)
- `fig3_breakdown`
- `fig4_transitions`
- `fig5_spatial_map` (mid-z block slice of every field, beside the data)
- `fig6_transition_matrix`
- `fig7_change_concentration`
- `fig8_granularity`

## Other formats

Subclass `FieldSource` (`fields()`, `load(field, t)`) for HDF5, ADIOS2 or Zarr. `RawF32Source` reads the flat float32 dumps from `nyx/gen_fields.sh`, laid out as `<frame>/fab0000_compNN_<field>.f32` on the grid from `gen.json`.

## Caveats

- Blocks must divide the grid; there is no padding.
- A Nyx dump number is a counter of writes, not a recorded step. Step ≈ 10 × dump is inferred, not verified.
- The answer depends mostly on **which** four timesteps are chosen. Always state the window.
