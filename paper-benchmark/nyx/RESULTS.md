# Nyx workload — measurements

Measurements. Moved out of `README.md`, which is the operational document: how
to run the workload, how to get its data out, and the parameters that make that
data evolve.

Every number here was measured on one A100 at the settings its own section
states. Where a section predates the current defaults it says so.

### Seeing what a lossy bound costs

`../viz_lossy.py` puts the original, the decompressed copy and `|error|` on one
plate. It needs the decompressed bytes, which the replay driver will write on a
cold read:

```bash
# Tag explicitly. Deriving one by stripping '0' and '.' collapses 0.001, 0.01
# and 0.1 to the same string, and the second run silently overwrites the first.
for spec in 0.001:eb001 0.01:eb01 0.1:eb10; do
  EB=${spec%%:*}; T=${spec##*:}
  CLIO_NEUROPRESS_STAGE_H2D=1 ./run_config.sh dynamic --fields /tmp/nyx-quick \
      --chunk 1048576 --eb $EB --check-bound --results /tmp/nyx-lossy --tag $T
  $BUILD/bin/neuropress_field_replay --readback /tmp/nyx-lossy/$T/blobs.csv \
      --dump-decompressed /tmp/nyx-decomp/$T --tag nyx_$T \
      --dir /tmp/nyx-quick --ext .f32 --chunk 1048576 --check-bound
done
../viz_lossy.py --orig /tmp/nyx-quick --out /tmp/nyx-viz/lossy \
    --plt /tmp/nyx-quick-plotfiles --compare 0.001:/tmp/nyx-decomp/eb001 \
    --compare 0.01:/tmp/nyx-decomp/eb01 --compare 0.1:/tmp/nyx-decomp/eb10
../viz_actions.py --out /tmp/nyx-viz/actions --sel 0.001:/tmp/nyx-lossy/eb001 \
    --sel 0.01:/tmp/nyx-lossy/eb01 --sel 0.1:/tmp/nyx-lossy/eb10
```

`--compare EB:DIR` is repeatable. The plate is original on top, one decompressed
row per bound, then one `|error|` row per bound. Every field row shares one
color scale so the rows are comparable; each error row is scaled to ITS OWN
bound, so across bounds compare the decompressed rows, not the error rows.

Three things that will otherwise cost an hour:

- **`CLIO_NEUROPRESS_STAGE_H2D=1` is required for the offline sweep, and
  nothing here sets it.** `CLIO_NEUROPRESS_REQUIRE_DEVICE` defaults *on*
  (`compressor_runtime.cc`), NeuroPress preprocessing is CUDA-only, and a
  file-replay driver hands the compressor host memory by construction. Without
  the stage every chunk is refused and every blob fails `rc=11` with
  `0 B in -> 0 B on the tier`. `run_config_insitu.sh` sets residency
  deliberately; `run_config.sh` sets neither, so the offline path is broken
  out of the box on this build.
- **`--chunk` must equal one component** (`ncell^3 x 4`, so 1048576 at 64³),
  or a dumped `.bin` is a fragment spanning field boundaries and will not
  reshape.
- **The readback `--tag` must match the writer's**, which is `nyx_$NAME`, not
  the config name. A wrong tag finds no blobs and reports `rc=11` — the same
  symptom as the residency failure, from an unrelated cause.

Measured on the 64³ quick run, `dynamic`, over the same 156 MiB. Every chunk
landed inside its bound at both settings:

| | stored ratio | worst `max|err|` |
|---|---|---|
| lossless | 4.114× | 0 (bit-exact) |
| `--eb 0.001` | **6.470×** | 0.000949 |
| `--eb 0.01` | 6.198× | 0.00949 |
| `--eb 0.1` | 10.975× | 0.0950 |

**The ratio is not monotone in the bound.** `0.001` stores BETTER than `0.01`
(6.470× against 6.198×) on identical input. The cause is visible in
`viz_actions.py`: under the balanced cost model the two bounds pick different
codecs for the same chunk — on density, `0.01` sits on `zstd` for the first
four dumps and reaches a mean 471.7×, while `0.001` sits on `cascaded` at
26.2×, and the ordering reverses again once both settle on `ans`. Selection,
not quantization, is what moves the total. Do not assume a looser bound is a
smaller file.

The pictures say something the ratio does not. The bound is **absolute**, so
what it costs depends entirely on local magnitude:

| field | eb 0.001 | eb 0.01 | eb 0.1 | 0.1 as % of peak | outcome |
|---|---|---|---|---|---|
| `density` | 25/26 | 26/26 | 24/26 | 3.8% | flattened at 0.01, obliterated at 0.1 |
| `xmom` `ymom` `zmom` | 17–19/26 | 23–24/26 | 25/26 | 0.10% | shell intact at every bound |
| `rho_E` | **0/26** | **0/26** | **0/26** | 0.0057% | **bit-exact at all three** |
| `rho_e` | **0/26** | **0/26** | 9/26 | 0.0085% | **bit-exact below 0.1** |

(chunks quantized, out of 26)

On density the ambient is 1.0 and the evacuated centre reaches 0.002. At 0.001
the reconstruction is visually indistinguishable from the original. At 0.01 the
bound is a 1% perturbation outside the shock but already larger than the value
inside it, so the front is untouched while the interior gradient collapses to a
slab. At 0.1 the interior becomes a single flat level and the quantizer's
Cartesian grid shows through as a **diamond** — the clearest picture of the
effect in the whole set, and still invisible at the shock front. The three
bounds as a ladder are what makes that legible; any one of them alone is
ambiguous.

And the bound is a **ceiling, not an instruction**. NeuroPress decides per chunk
whether quantizing is worth it: `rho_E` declined at all three bounds, `rho_e`
declined twice and then accepted 0.1 on 9 of 26 chunks, and density actually
quantizes FEWER chunks at 0.1 (24) than at 0.01 (26). A "lossy run" is lossy
only where the selector took the offer.

### The action progression

`../viz_actions.py` draws what was selected per dump, and it draws the whole
action rather than just the codec: lane is the library, colour is the bound, a
filled marker means quantize was taken, a square means the 4-byte shuffle.

It lives one level up because it is workload-agnostic: it reads both blob-name
shapes, so the same script serves the LAMMPS benchmark.

The point is that **the selection moves as the physics does**. On density the
run walks cascaded/zstd/gdeflate → snappy → bitcomp → cascaded → ans, settling
only once the blast has filled enough of the box that the block stops changing
character. Measured switches over 26 dumps: density 7 (eb 0.001), 3 (0.01),
6 (0.1); `xmom` 5 / 10 / 4. `rho_e` never leaves `lz4` — its lane is a flat
line, and at eb=0.1 the marker simply fills in from dump 19 onward.

Preset is not drawn: it is 2 throughout these runs, and the script says so
rather than hiding it if that ever changes.

### Temporal redundancy is the number that explains all of it

`evolution.png` now carries a fourth panel: the share of cells **bit-identical
to the previous dump**.

| dump | 1 | 12 | 25 |
|---|---|---|---|
| `density` | 99.9% | 86.5% | 57.1% |
| `xmom` `ymom` `zmom` | 99.7% | 83.5% | 50.9% |
| `rho_E` `rho_e` | 99.9% | 86.3% | 56.7% |

It falls almost linearly across the run. That single column is the clearest
statement of why a fixed codec is the wrong answer here: the block genuinely
stops being the same kind of data, and every other curve on the page —
compressibility, the chosen action, the cost of a given bound — is downstream
of it. It is also the metric `gen_fields.sh` reasons about when it argues for
raising `--exp-energy`, now measured rather than asserted.

### `--exp-energy` is the knob that makes the data evolve — at a fixed `stop_time`

**Read this section together with "Default Evolving Benchmark Configuration"
above, which measures the same knob at a fixed STEP COUNT and finds it inert.**
Both are right. Everything below holds when `stop_time` ends the run, where a
stronger blast buys more steps as well as a faster shock. When `--steps` ends
the run, the extra speed and the shorter CFL timestep cancel exactly and `--cfl`
is the knob instead.

The Sedov shock radius goes as `R = 1.033 (E t^2 / rho)^(1/5)`, so at the deck's
`E=1` and `t=0.01` it reaches only **~0.16 of a unit box** — most of the domain
never changes and consecutive dumps are nearly identical.

Raising `E` fixes two things at once, which is not obvious: a wider shock also
means higher velocities, so the CFL timestep shrinks and the *same* `stop_time`
now takes **more** steps. Measured at 128³, `density` ratio range over the run:

| `--exp-energy` | steps | ratio range | spread |
|---|---|---|---|
| 1 (deck default) | ~300 | 12.7x .. 87.5x | 6.9x |
| **10** | ~675 | 4.6x .. 205.3x | **44.4x** |
| 100 | ~1575 | 1.8x .. 205.8x | 113.9x |

A validated 1 GB configuration:

```bash
./run_config_insitu.sh explore-balance   --ncell 128 --steps 2000 --int 32 --chunk 8388608   --exp-energy 10 --explore-k 3 --explore-thresh -1 --verify   --results /tmp/nyx1g --tag e10
```

21 frames, 671 steps, 126/126 blobs verified bit-exact, and **no frame
bit-identical to its predecessor on any field**. Compressibility falls
monotonically from ~86x to ~4.6x as the blast fills the domain, and the
selector moves cascaded -> bitcomp -> ans along the way. Both knobs are
recorded in `meta.json`, so a run stays self-describing.

`--steps 2000` there is deliberately generous: the run stops on `stop_time`, so
the step count is an outcome, not an input.

## Results

> **Measured before the evolution study changed the default `nyx.cfl` to 0.8.**
> These numbers are at the Sedov deck's `cfl = 0.5`. The new default makes the
> data evolve faster and therefore compress less: a quick-profile in-situ cell
> moves from 31.3x to 23.3x. The ORDERING of the policies is what this section
> is about and is unaffected; the magnitudes are not directly comparable.

Sedov blast, 128³, 21 frames, 1,008 MiB float32, 252 chunks, A100, lossless.
All 252 blobs verified bit-exact in every configuration, in process and on a
cold read from a separate process.

Regenerated after `775dd9b4` fixed the dump: it had been writing the first
`validbox().numPts()` elements of the GROWN FArrayBox — a mixture of valid and
ghost cells — instead of the valid box. Every number in this section is from
the corrected dump (each file is exactly 128³ × 4 B).

| config | ratio | stored | density | rho_E | rho_e | xmom | ymom | zmom | Σ ms | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| **`static-zstd-s4`** | **162.9×** | 6.2 MiB | 222× | 192× | 194× | 145× | 123× | 144× | 720 | 33.8 s |
| `static-zstd-s8` | 141.2× | 7.1 MiB | 192× | 166× | 167× | 127× | 107× | 124× | 813 | 32.0 s |
| `best` | 134.1× | 7.5 MiB | 139× | 192× | 184× | 118× | 103× | 115× | 628 | 111.9 s |
| `static-zstd` | 132.6× | 7.6 MiB | 184× | 159× | 160× | 109× | 104× | 117× | 895 | 13.8 s |
| `explore` | 126.9× | 7.9 MiB | 147× | 139× | 164× | 116× | 102× | 114× | 678 | 94.9 s |
| `dynamic-ratio` | 75.3× | 13.4 MiB | 109× | 52× | 51× | 96× | 90× | 100× | 791 | 45.8 s |
| `dynamic` | 23.7× | 42.6 MiB | 29× | 36× | 18× | 19× | 20× | 31× | 245 | 54.9 s |
| `learn` | 20.8× | 48.4 MiB | 29× | 14× | 14× | 26× | 31× | 25× | 151 | 93.8 s |

The mis-sliced data changed more than the magnitudes: `static-zstd` (no
shuffle) moves from 79.1× to 132.6×, from below `dynamic-ratio` to above
`explore`, and `dynamic`/`learn` roughly double. The ORDER of the top result is
unchanged — a fixed zstd with the 4-byte stride still wins — but any statement
about *how far* the others trail had to be re-derived, which is what the rest
of this section does.

### The shuffle stride is the headline, and it flips with the element width

Nyx can be built at either precision, so this is a **controlled** result:
identical problem, identical code, identical timesteps — only `AMReX_PRECISION`
changes. (`gen_fields.sh` with a double build and `NYX_DUMP_NATIVE=1` emits
`.f64`; replay with `--f64`.)

| fixed nvcomp-zstd | Nyx float32 (1,008 MiB) | Nyx float64 (2,016 MiB) |
|---|---|---|
| no shuffle | 132.6× | 94.3× |
| **4-byte** shuffle | **162.9×** | 104.4× |
| **8-byte** shuffle | 141.2× | **110.8×** |
| best stride | **4** | **8** |

(The float64 column was re-measured from the corrected dump too and came back
unchanged to three digits — 94.321 / 104.406 / 110.804 — so only the float32
column moved.)

The preference flips with the width, on the same physics. The cross-workload
comparison points the same way — LAMMPS float64 prefers 8 (1.198× against
1.159×) — but that one confounds data and width; this one does not.

The stride has to match the element width, and getting it wrong costs more
than any codec choice. NeuroPress encodes byte-shuffle as a **single bit**
meaning `GPUCOMPRESS_PREPROC_SHUFFLE_4` — a 4-byte stride. That is exactly
right for float32 Nyx, and it is why upstream's benchmarks never surfaced the
limitation: their recommended Nyx build is single precision. On float64 LAMMPS
the same bit is the wrong width, and no action in the searched space can ask
for 8.

Upstream is aware of the gap without having closed it. Its own Nyx bridge
(`examples/nyx_amrex_bridge.hpp:255`) asks for 8 on double data —
`shuffle_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4` — but the filter accepts
only 0 and 4 (`src/hdf5/H5Zgpucompress.c:258`), prints
`"only 0 and 4 are supported), ignoring shuffle"`, and falls back to **no
shuffle at all** rather than to 4.

### The default cost model is badly wrong on this data

`dynamic` — NeuroPress inference under the default balanced cost — reaches
**23.7× where 162.9× was available**, a 6.9× miss, and `learn` is worse still
at 20.8×. The balanced objective charges compression time, so it picks the fast
codecs (bitcomp ×168, ans ×26, cascaded ×20) on data whose whole value is in
the slow entropy coders; `best` and `explore`, which do not, pick zstd for
158–183 of the 252 chunks. Zeroing the two latency weights (`dynamic-ratio`)
recovers 75.3× immediately — still less than a fixed zstd, because the
ratio-only objective ranks on a predicted ratio that is itself wrong.

This is the same failure mode as in the LAMMPS benchmark but far larger:
there, inference reached 1.074× against 1.198× achievable (an 11% miss); here
it forfeits most of an order of magnitude. Highly compressible data punishes a
selector that optimises for speed.

### Exploration does not reach the fixed codec

`explore` and `best` measure every alternative and still land at 127–134×,
below a fixed zstd with the right stride (162.9×) that costs a third of the
wall clock — and below a fixed zstd with NO shuffle at all (132.6×). Both facts have the same cause as in the LAMMPS run: the shuffle the
search can request is chosen per action from a space that pairs it with the
codec, and the search is bounded by what the action space can express.


## The 1,000-timestep evolution study

Ranking behind `README.md`'s "Default Evolving Benchmark Configuration". Six configurations, 101 dumps
each, 4,800 block samples per configuration, 1 MiB blocks, sampled every 10
timesteps. Raw per-block values in `../evolution-study/nyx/`.

Average normalized block evolution: **0.1159**, with **76.7% of blocks active**
and **80.04% of cells bit-identical** to the previous dump (99.97% on the first
pair, 49.83% on the last).

| config | parameters | mean E | median | active blocks | cells bit-identical | p10 interval | last quarter | sim s |
|---|---|---|---|---|---|---|---|---|
| `e10_cfl09` | `exp_energy=10, cfl=0.9` | 0.1175 | 0.0729 | 78.6% | 76.44% | 0.0619 | 0.0635 | 27 |
| **`e10_cfl08`** | `exp_energy=10, cfl=0.8` | **0.1159** | 0.0697 | 76.7% | 80.04% | 0.0602 | 0.0625 | 25 |
| `e1_cfl08` | `exp_energy=1, cfl=0.8` | 0.1138 | 0.0693 | 76.5% | 80.15% | 0.0602 | 0.0625 | 28 |
| `e100` | `exp_energy=100, cfl=0.5` | 0.1040 | 0.0520 | 65.0% | 89.59% | 0.0425 | 0.1033 | 25 |
| `e10` | `exp_energy=10, cfl=0.5` | 0.1011 | 0.0513 | 64.7% | 89.66% | 0.0425 | 0.0993 | 25 |
| `e1` | `exp_energy=1, cfl=0.5` (deck default) | 0.0981 | 0.0504 | 64.4% | 89.73% | 0.0420 | 0.0938 | 25 |

`e10_cfl09` scores highest and is **not** the default: see "why not 0.9" below.

### Why the evolving configuration was chosen

**At a fixed step count `exp_energy` does nothing, and that is exact rather than
approximate.** The share of cells bit-identical to the previous dump, by decile
of the 1,000 steps:

| decile | 1 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|
| `e1` | 99.8 | 96.7 | 92.2 | 86.5 | 76.3 |
| `e10` | 99.8 | 96.7 | 92.1 | 86.4 | 76.2 |
| `e100` | 99.8 | 96.7 | 92.1 | 86.3 | 76.0 |
| `e1_cfl08` | 99.6 | 94.0 | 85.1 | 73.9 | 53.4 |
| `e10_cfl08` | 99.6 | 94.0 | 85.0 | 73.7 | 53.2 |
| `e10_cfl09` | 99.6 | 93.0 | 82.4 | 69.0 | 44.8 |

A hundredfold change in blast energy moves the last decile by 0.3 points; the
CFL number moves it by 23. And the E-independence reappears at the higher CFL —
`e1_cfl08` and `e10_cfl08` agree to a tenth of a point everywhere — so it is a
property of the problem, not a coincidence at one setting.

Sedov is self-similar. With `R(t) = 1.033 (E t^2 / rho)^(1/5)` the shock speed
is `u_s = (2/5) R/t`, and the timestep is CFL-limited to
`dt = cfl * dx / (c1 * u_s)` for a `c1` set by the post-shock sound speed. So

```
dR per step = u_s * dt = cfl * dx / c1
```

— **a fixed number of cells per timestep, independent of E, rho and t.** A
stronger blast moves the shock faster and shrinks the timestep by exactly the
same factor. `--exp-energy` therefore buys reach per unit *time* and nothing per
unit *step*.

This does not contradict the `--exp-energy` table further down, which is
measured at the deck's fixed `stop_time = 0.01`: there a stronger blast buys
more *steps* (~300 at E=1, ~675 at E=10, ~1575 at E=100) and the ratio spread
follows. Both are true and they answer different questions. If the run is ended
by `stop_time`, raise `--exp-energy`; if it is ended by `--steps`, raise
`--cfl`.

**Why not `cfl = 0.9`,** which scores 5% higher. Nyx's mass sum over the same
1,000 steps:

| `nyx.cfl` | mass drift |
|---|---|
| 0.5 | 0 (exact) |
| **0.8** | **-1.19e-07** — float32 roundoff |
| 0.9 | **-1.09e-04** — ~1000x worse |

At 0.9 the scheme produces negative densities that Nyx's `Enforce minimum
density` floor clamps, and the clamping shows up as mass that is not conserved.
The run still exits 0 and holds no NaN, so the evolution metric cannot see it —
this had to be checked separately. 0.8 is Nyx's own documented default, keeps
mass to float32 roundoff, and gives up 1.4% of the mean.
