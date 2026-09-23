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
`+Tier` arms spill tier 2 to the compute node's own NVMe (`/tmp`). The account is discovered from
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
SMOKE_GB=2 ./figure_9.sh -w nyx -s smoke   # a smaller smoke payload
```

`-s smoke` replays `SMOKE_GB` GiB (default 4), `-s full` the whole dump. The
smoke size is a **byte budget, not a file count**: the five dumps differ 40x in
file size -- 8 MiB for VPIC/Nyx/WarpX/LAMMPS against a whole 327 MiB tensor for
AI -- so the older fixed `--max-files 60` replayed 480 MiB of one workload and
655 MiB of another, and AI needed a hardcoded override of its own. The budget is
filled in the order the driver replays (sorted by path), a dump smaller than it
replays whole, and `MAXF` still overrides either size. Each arm reads that
payload once, so a run reads `SMOKE_GB` x 17 arms, plus one untimed `warm_cache`
pass.

Each run writes, under `--out`:

| file | what it is |
|---|---|
| `fig9.csv` | one row per arm: `panel,workload,strategy,compute_min,io_min,total_min,std_min,ratio,eb,runs,bound,payload_mib,setup_min` — the MEDIAN run by total (mean of the middle two for an even count), `std_min` the sample std of the totals, `bound` `ok`/`exceeded` for a lossy arm. Rebuilt after every run. `TBD_OTHERS=1` adds blank rows for the other four workloads. |
| `fig9_runs.csv` | every recorded run: `panel,workload,strategy,rep,compute_min,io_min,total_min,ratio,bound,setup_min,eb` |
| `<arm>/r<N>/cusz_setup.csv` | cuSZ only (`CLIO_CUSZ_PHASE_LOG`): one STAMPED row per span, `phase,elems,ms,start_ns`, with `phase` in `stream` / `mgr` / `release` — the CUDA stream and the `psz_resource` that cuSZ's rev1 API makes the wrapper build and tear down for EVERY chunk. `setup_min` is the UNION of those spans clipped to the measured window, never their sum: chunks compress on several workers at once, and summing produced 14.307 s inside a 13.539 s arm. The build and the release stay separate spans because the codec kernel runs between them. The output copy is not logged: moving the compressed frame out of the manager's buffer is output work every codec does. An arm that never calls cuSZ writes no file and records 0; a NeuroPress arm that routes some chunks to cuSZ records exactly those. `plot_fig9.py --deduct-setup` subtracts it and relabels the y-axis — the default figure plots what was measured. |
| `arms.csv` | the exact arm set this run built — label, panel, config, eb, tier, async_ms |
| `run.json` | the parameters behind it |

Merge per-workload runs by repeating the flag:
`plot_fig9.py --csv nyx.csv --csv vpic.csv ...`. A blank row never replaces a
measured one, so the order does not matter.

**AI runs lossless only.** An AI checkpoint is model weights, not a simulation
field, so there is no error budget to spend and the figure offers it no lossy
point: its panel (a) ladder stops before the lossy rung and its panel (b) runs
at `eb=0`, which leaves it **12 arms** against the other workloads' 17. cuSZ and
cuSZp3 are dropped from it entirely rather than run at `eb=0` -- they are
error-bounded codecs with no lossless mode, so a "lossless cuSZ" arm would not
be the codec the label promises. The rule lives in `LOSSLESS_ONLY_WORKLOADS`
(default `AI`) and is applied per workload, including to the TBD rows a run
writes for the other four, so a merged CSV never shows AI a column of lossy bars
marked TBD that were never going to run. At `eb=0` panel (b)'s `Best fixed
nvCOMP` and `NeuroPress` repeat panel (a)'s `nvCOMP` and `NP only` -- the same
measurement drawn in both panels.

**Caveat for a merged figure.** `plot_fig9.py` takes an arm's legend bound from
the first non-zero `eb` it finds for that label across ALL workloads, so in a
merged plot `NeuroPress` is labelled with the simulation workloads' bound while
AI's bar of that name is lossless. Plot AI separately, or read its bounds from
`arms.csv`, until the legend is made per-workload.

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
`NVME_ROOT` is `/tmp/fig9-nvme-<jobid>`, the node's own NVMe; `<acct>` comes from the
`delta_<acct>` group that has a matching `/work/hdd/<acct>`, so nothing is
hardcoded to one user. Override either, or set `CLIO_ACCT`.

This was previously one root, with the untiered arms taking their tier from
`--out`. That default sits under the repo on `/u`, which is **NFS home, not the
PFS**. Measured on Delta, 3×512 MiB with `conv=fdatasync`: `/u` 587–875 MB/s,
`/work/nvme` 434–502 MB/s, `/work/hdd` comparable to `/work/nvme`. The untiered
arms therefore had the *fastest* durable device and the `+Tier` arms the
slowest — the opposite of the ablation's premise, and an understatement of what
tiering buys.

**Tier 2 is the node's own NVMe, never NVMe over the network.** `/work/nvme`
is SSD media behind Lustre (pool `ddn_ssd`, while `/work/hdd` is pool
`ddn_hdd`), reached over the network. On a Delta GPU node `/tmp` is a local
NVMe drive private to the job: `nvme0n1` (MZXL51T6HBJR), xfs, 1.5 TB, no quota,
**1.2 GB/s** durable write (3 × 2 GiB, `conv=fdatasync`, job 22274732) against
434–502 MB/s for `/work/nvme`. `/tmp` is wiped when the job ends, which costs
nothing: each tier image is deleted as soon as its arm is measured, and results
live under `--out`. `require_local_tier2` refuses a network filesystem
(Lustre, NFS, GPFS, ...) as tier 2 and exits 4; set `ALLOW_NETWORK_TIER2=1` only
to reproduce the older network-NVMe runs. Every job log's header names the
filesystem and device each root resolved to.

Campaigns before 2026-09-21 put tier 2 on `/work/nvme`; their `+Tier` bars
measure NVMe over the network, and `run.json`'s `nvme_root` records which.

## What the segments mean

- **Solid** = NON-I/O ELAPSED: `total_min - io_min`, i.e. stats, NN, quantize,
  codec, task hand-offs, per-chunk library setup and scheduling. It is
  deliberately NOT called "compute": on the cuSZ arm 82% of it is
  `psz_create_resource_manager`/`psz_release_resource`, not computation, and the
  word hid that for the whole first round of this figure.
- **Light** = MEASURED elapsed device I/O. Each bdev transport brackets its own
  transfer (`CLIO_IO_LOG`, `modules/bdev/include/clio_runtime/bdev/io_log.h`):
  for a file tier from the device-to-host copy of the image through the write,
  for the RAM tier around the copy into the segment. The harness takes the
  UNION of those intervals, never their sum, because writes overlap, and each
  interval is first CLIPPED to the measured window the driver prints
  (`window: start_ns N   end_ns M`, the same steady clock `io_log.h` stamps
  with), so no write outside the timed region can contribute to the bar. This is
  the span upstream's VOL also calls I/O (`vol_d2h_copy_ms` +
  `vol_io_queue_wait_ms` + drain), and it covers the final flush too.
  Campaigns before 2026-09-22 had no I/O measurement at all: the pale segment
  was the input read plus the flush.
- **Excluded from both**: the input read and H2D staging — the figure assumes
  the data is already in memory and on the GPU, as upstream's VOL write does —
  and, for LAMMPS, the simulate time. `EXCLUDE_READ=0` puts the read back.
  Inputs are copied once per job to node-local `/tmp` (`STAGE_INPUT=1`,
  16 parallel streams), so every arm reads the same local bytes instead of
  re-reading Lustre, whose cached pages expire after 600 s idle.
- **No selection log** (`SELECTION_LOG=0`): it hashes every input chunk and
  compressed payload byte by byte inside the compressor, ~19 ms per 8 MiB
  chunk that only the compressed arms paid (jobs 22300816, 22300826).
  Staging is removed as the *union* of the per-chunk intervals, not their sum:
  chunks pipeline, and on AI the per-chunk walls overlap 52–77×, so summing them
  over-subtracted up to 3.8% of the total.
- `ratio` is input bytes / stored bytes. `eb` is the error bound; `0` is
  lossless.
- **Repeated, rotated runs.** Each arm runs `REPS` times (default 3); rep *k*
  starts the arm list *k*/`REPS` of the way round, so no arm always runs first.
  The bar is the median run and the whisker ±1 std of the totals. One run per
  arm was the noisiest input: PFS write loops moved 25–30% between identical
  runs, and single Lustre stalls of 0.1–2 s decided which arm won.
- **Verification** runs once per arm, in rep 1 (it is untimed but a full
  read-back). A lossy arm is checked element-wise against its bound
  (`--check-bound`); one that exceeds it is still drawn, with `*` on its label.
  **cuSZ and cuSZp3 are not checked**: they quantize internally, in fp32, so
  their bound is the codec's contract rather than ours. Their `bound` column is
  empty. The check still covers every arm that uses our quantizer (nvCOMP `-q`,
  NeuroPress lossy).
- **RAM tier pre-faulted** (`CLIO_PREFAULT=0` on +Tier arms) during setup, so
  page faults no longer land in the timed loop (1.7 s of full Nyx's
  nvCOMP+Tier loop before).

## The I/O bar, and what it is not

Since 2026-09-22 the pale segment IS a measurement: the bdev transports time
their own transfers (`CLIO_IO_LOG`), and the harness takes the union of those
intervals. Solid is `total - io`, so the two always add to the bar.

What it does NOT use is the compressor's per-chunk `io_ms`. Audited 2026-09-18:
that field is the awaited put (`AsyncPutBlob` → `PutBlobImpl` →
`ModifyExistingData` → bdev `AsyncWrite`), so it carries task hand-offs and
scheduling, it sums to many times the elapsed loop because chunks pipeline, and
it measures a durable file write on an untiered arm but a memcpy on a tiered
one. The bdev timers avoid all three: they bracket the transfer itself, they
are unioned rather than summed, and each records which device class moved the
bytes (`file` or `ram`).

Note what each arm's I/O therefore means. On an **untiered** arm the whole
durable write happens inside the loop, so the pale segment is the PFS. On a
`+Tier` arm the puts land in RAM at memcpy speed and the NVMe write happens in
the flush, which the union also covers. The two bars are both measured I/O, but
they are I/O to different devices -- that is the ablation's point, not a flaw.

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
- **Watch the project quota.** `/work/hdd` sits on one project quota. Each
  untiered arm writes a tier the size of the payload, and an arm that is
  SIGKILLed never reaches `run_arm`'s cleanup, so 30 GB orphans accumulate under
  `PFS_ROOT`. The tiered arms are off the quota entirely: their tier 2 is the
  node's `/tmp`.

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
