# Figure 9 — what it measures, and what it does not

End-to-end wall-clock time per workload. Panel (a) is the ablation ladder,
panel (b) NeuroPress against the external codecs. `figure_9.sh` runs the
arms and `plot_fig9.py` draws them, both beside this file.

This file is the reference for the measurement, and it lives beside the
plotter: `figures/fig9/` now holds this figure's harness, its one plotter and
this note. What a run generates is its output, under `--out`, and is not kept
here.

## Delta only — do not run this on Chameleon

This figure is a *device* measurement, and its arms are pinned to Delta's two
storage classes: untiered arms write tier 1 to `/work/hdd/<acct>` (the PFS),
`+Tier` arms spill tier 2 to `/work/nvme/<acct>`. The account is discovered from
a `delta_<acct>` group with a matching `/work/hdd/<acct>`, so on Chameleon
`CLIO_ACCT` resolves empty, neither root exists, and `require_root` exits 4
before the first arm.

Pointing `PFS_ROOT` and `NVME_ROOT` at writable Chameleon paths does not fix it,
it hides it: on a node with one storage class both roots land on the same
device, `+Tier` stops being a change of tier and the ablation compares an arm
against itself. The panel (b) arms also need the GPU codecs (nvCOMP/bitcomp,
cuSZ, cuSZp3, ndzip) built for the A100s, and a full-size run replays ~30 GB per
arm across 17 arms.

What *is* portable is the offline model comparison, figure 8 panel (d)
(`../../model-accuracy/`): it scores a recorded trace, needs no codecs, no tiers
and no GPU, which is why that is the piece pushed for Chameleon.

## Running it

```bash
./figure_9.sh -w nyx -s full            # one workload, all arms
./figure_9.sh -w nyx -s smoke --dry-run # print the arm set and stop
```

Each run writes, under `--out`:

| file | what it is |
|---|---|
| `fig9.csv` | one row per arm: `panel,workload,strategy,compute_min,io_min,total_min,std_min,ratio,eb`. Its own workload is filled, the other four left blank so a single-workload CSV plots at the full layout. |
| `arms.csv` | the exact arm set this run built — label, panel, config, eb, tier, async_ms |
| `run.json` | the parameters behind it |

Merge per-workload runs by repeating the flag:
`plot_fig9.py --csv nyx.csv --csv vpic.csv ...`. A blank row never replaces a
measured one, so the order does not matter.

**17 arms.** Panel (a) is the ablation ladder ending in ONE lossy arm at
`EB_LOW` (1e-3); an earlier version swept a 1e-3/1e-2/1e-1 ladder across three
arms, which is three extra full-size runs for a comparison the figure does not
make. `PANEL_B_TIER=1` (default) runs every panel (b) codec twice, once writing
through to the PFS and once over the tier, so the tier is the only variable in
that comparison. `PANEL_B_TIER=0` restores the older five-arm shape in which the
external codecs ran untiered while NeuroPress kept its tier — that compares
NeuroPress-as-a-system against codecs alone, and is the shape the published
campaign was run in. `arms.csv` records whichever was used, and `plot_fig9.py`
takes its arms, workloads and bounds from the CSV, so a campaign of either shape
plots completely.

## Where the bytes land

One device class per arm class, so `+Tier` changes the tiering and nothing else.

| arm class | tier 1 | tier 2 | durable bytes end up on |
|---|---|---|---|
| untiered — Baseline, `nvCOMP`, `NP only`, and the un-suffixed panel (b) rows | file at `$PFS_ROOT/<arm>/cte_tier.dat_node<N>` | none | **the PFS** |
| `+Tier`, `+Tier+Async` | `ram` — shared memory, **no file** | file at `$NVME_ROOT/<arm>/cte_tier2.dat_node<N>` | **NVMe** |

Baseline is not special-cased: `--no-compress` calls `AsyncPutBlob` on the same
cte_core pool as every other arm (`neuropress_field_replay.cc:566`) and lands on
the same tier-1 device. It just skips the codec.

A `+Tier` arm's bytes reach tier 2 through the periodic flush worker when
`+Async` is on, and only at the final `FlushData` when it is not
(`BENCH_FLUSH_MS=0`). Nothing fsyncs; the tier file is opened `O_RDWR|O_CREAT`
with no preallocation, so it grows sparsely.

Both roots hold one subdirectory per arm, deleted as soon as the arm is
measured. Each run prints the filesystem each root resolved to, and logs the
durable file's size and allocated bytes per arm, so "the untiered arms went to
the PFS" is a measurement rather than a claim.

**Defaults and why.** `PFS_ROOT` is `/work/hdd/<acct>/<user>/fig9-pfs` and
`NVME_ROOT` is `/work/nvme/<acct>/<user>/fig9-nvme`; `<acct>` comes from the
`delta_<acct>` group that has a matching `/work/hdd/<acct>`, so nothing is
hardcoded to one user. Override either, or set `CLIO_ACCT`.

This was previously one root, with the untiered arms taking their tier from
`--out`. That default sits under the repo on `/u`, which is **NFS home, not the
PFS**. Measured on Delta, 3×512 MiB with `conv=fdatasync`: `/u` 587–875 MB/s,
`/work/nvme` 434–502 MB/s, `/work/hdd` comparable to `/work/nvme`. The untiered
arms therefore had the *fastest* durable device and the `+Tier` arms the
slowest — the opposite of the ablation's premise, and an understatement of what
tiering buys.

Note that `/work/nvme` is NVMe **media behind Lustre**, reached over the
network, not node-local NVMe. On a compute node `/tmp` is node-local NVMe, so
`NVME_ROOT=/tmp/fig9-$SLURM_JOB_ID` gives the sharper device contrast; the
default stays on `/work/nvme` because it survives the job.

## What the segments mean

- **Solid** = the write loop (`stage+compress`): stats, NN, quantize, codec,
  tier put, setup, scheduling.
- **Light** = `total_min - compute_min`, i.e. the input read plus the final
  flush.
- **Excluded from both**: H2D staging — the figure assumes the data is already
  on the GPU, as upstream's VOL write does — and, for LAMMPS, the simulate time.
  Staging is removed as the *union* of the per-chunk intervals, not their sum:
  chunks pipeline, and on AI the per-chunk walls overlap 52–77×, so summing them
  over-subtracted up to 3.8% of the total.
- `ratio` is input bytes / stored bytes. `eb` is the error bound; `0` is
  lossless.
- `std_min` is empty unless a caller merges repeats — each arm is run once.

## Why there is no I/O bar

`io_min` is `total_min - compute_min`. It is **not** an I/O measurement, and the
figure does not claim one.

Audited 2026-09-18: the runtime's per-chunk `io_ms` is a real awaited write
(`AsyncPutBlob` → `PutBlobImpl` → `ModifyExistingData` → bdev `AsyncWrite`,
polled to completion in `fs_bdev_transport.cc`) — buffered, not fsynced. On an
**untiered** arm that write lands on the file tier inside the loop, so 100% of
the arm's durable write is already inside the solid bar. On a `+Tier+Async` arm
the put goes to RAM at memcpy speed and only 0–38% is, with the rest moved later
by an untimed periodic `FlushData` worker. The same field therefore measures a
file write on some arms and a memcpy on others, and the per-chunk intervals
overlap. A wall-clock compute/IO split is not derivable from this
instrumentation.

Two alternative splits were tried and rejected, because a reader will ask:

- **`loop - Baseline's loop`** ("what compression adds"), floored at 0. It mixes
  any I/O difference into the codec term and collapses whenever compression pays
  for itself: it read 0.00 s for AI's 32× lossy arm, and 3.14 s for AI's nvCOMP
  arm whose codec kernel took 0.74 s.
- **The per-chunk phase sums** (`compress + preproc + ...`). Chunks pipeline, so
  those are per-chunk latencies, not shares of elapsed time — on AI they sum to
  52× (nvCOMP) and 77× (cuSZ) of the arm's total. They cannot stack into a bar.

The codec's own cost is reported per arm in the driver log as `codec=` instead:
a CUDA-event bracket around the compress call, the same quantity NeuroPress
measures and the one its model ranks on. Being a per-chunk cost rather than a
slice of wall clock, it is logged and never plotted. It also carries
verification-read latency — the phase sum does not filter `path=="write"`.

## What a full-size run costs

Measured 2026-09-19/20 on Delta, and worth knowing before submitting:

- **Reading the dump dominates if it sits on Lustre.** A full LAMMPS Baseline
  arm (30.01 GiB) measured `read 784 s / stage+compress 110 s / total 896 s` —
  87% of it the input read, at ~40 MB/s, metadata-bound over 3,933 files. Every
  arm pays that identically and compression cannot reduce it, which is why
  Baseline is never the slowest bar. **Stage the dump to node-local `/tmp`**
  (real NVMe, 1.5 TB, no quota) and replay from there; `warm_cache` does not
  help, it read 30 GiB for ~10 minutes and the arm still paid the 784 s.
- **Ask for memory.** Slurm's default is ~1 GB per CPU, so `--cpus-per-task=16`
  gives 16 GB and the run is OOM-killed. Use `--mem=0`. Note that the harness's
  own `node_state` reads `/proc/meminfo`, which reports the NODE's free memory
  and is blind to the cgroup limit — it will cheerfully print `memavail=196 GB`
  while the job dies at 16.
- **Watch the project quota.** `/work/hdd` and `/work/nvme` are one Lustre
  filesystem sharing one project quota. Each untiered arm writes a tier the size
  of the payload, and an arm that is SIGKILLed never reaches `run_arm`'s cleanup,
  so 30 GB orphans accumulate under `PFS_ROOT`. Setting `NVME_ROOT=/tmp/...`
  keeps the tiered arms off the quota entirely.

## Caveats to carry into any claim

- The durable writes of an untiered arm happen inside the write loop, so
  `io_min` near zero does not mean the arm did no I/O.
- Run-to-run spread on the lossless, heavy-writing arms is 18–34%. The lossy,
  high-ratio arms reproduce to 0.1–0.5% because they are compute-bound.
- `warm_cache` runs once before the arm loop, not per arm, so by the later arms
  the page cache is full of earlier arms' output and the input reads cold. The
  light segment therefore tracks an arm's **position in the sequence** as much
  as its codec: measured 67 s for arm 1 against 152–246 s for arms 2–7 on Nyx.
  Only the solid segment is comparable across arms.
