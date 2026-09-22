# Evaluation on Aurora: enacting the plan

The plan is `external/eternia/eternia/sections/evaluation-plan.tex` (in the
main checkout). This file records what has actually run, in the order the
plan prescribes: 4-node baselines for every workload on every Aurora
substrate first, then the studies. Every number here is from a job log
under `build-spike/pbs/`.

Rules in force while the system is being brought up: 4 nodes, every job
under 5 minutes; the non-Eternia (baseline) workloads first, Eternia last;
then the decks grow towards 15-minute runs, never past.

## Substrates on Aurora

| substrate | binary | data plane |
|---|---|---|
| MPI | `clio_<wl>_mpi_bench` | MPICH (built with Level Zero), GPU-aware; device buffers straight to MPI |
| oneCCL | `clio_<wl>_ccl_bench` | the NCCL analogue: collectives and send/recv on device buffers |
| Intel SHMEM | `clio_<wl>_ishmem_bench` | the NVSHMEM analogue: symmetric heap, one-sided puts, fcollect, reductions |
| Eternia | `clio_<wl>_paged_newcoro_aot` | the system under test (clio runtime, coroutine editions) |

The three baseline editions share one SYCL source per workload
(`sycl_baseline/`), built by `build_baselines_aurora.sh`, run by
`pbs_baseline_aurora.sh` (one debug-scaling allocation carries all three,
one rank per node, one tile per rank).

## Stage 1: 4-node baselines, small decks (bring-up)

Small decks first, to prove every (workload, substrate) cell runs and
gates on 4 nodes. Times are the edition's own timed region; "comm" is the
time spent in the substrate's collectives and exchanges.

| workload | deck | MPI | oneCCL | Intel SHMEM |
|---|---|---|---|---|
| kmeans | 4 GB global (1 GB/rank), 4 iters | 2465 ms (comm 1026) PASS | 2540 ms (comm 1062) PASS | 2045 ms (comm 622) PASS |
| grayscott | 4 GB global, 4 steps, 1 MB page | 35 ms (comm 15) PASS | 43 ms (comm 22) PASS | 52 ms (comm 30) PASS; v_checksum identical on all three |
| gmx | K=512, 4 M atoms, 1 pass | 91 ms PASS | 82 ms PASS | 105 ms PASS; conservation exact, mesh checksum and gather energy identical on all three |
| lbann | 1024 -> 4096 -> 256, batch 64, 5 steps | 178 ms (comm 22) PASS | 215 ms (comm 104) PASS | 257 ms (comm 106) PASS; loss and weight digest bit-equal to the dense reference on all three |
| lammps_md | L=40 (256 k atoms), 20 steps, rebin 10, T=3.0 | 25 ms PASS | 28 ms PASS | 32 ms PASS; statics and resort exact on all three |

**Stage 1 is complete: every workload runs and gates on every substrate at 4 nodes.**

Findings on the way:

- Intel SHMEM's default symmetric heap refused a 1 GB allocation; the job
  script sets `ISHMEM_SYMMETRIC_SIZE` to 40 GB, and the editions keep bulk
  data no peer names in plain USM (`AllocLocal`).
- The debug queue caps at two nodes; 4-node jobs go to `debug-scaling`.
- oneCCL send/recv on an in-order stream deadlocks on a closed ring when
  every rank posts recv first (lammps_md hung for its whole cap; grayscott's
  open chain did not). Even ranks send first, odd ranks receive first now.
- lammps_md at the melt deck (`--temp 3.0`) drifts 4e-3 over 20 steps on
  every substrate, above the default 5e-4 tolerance; the CUDA editions'
  recipe for this deck sets `--drift-tol 5e-3`, and the baseline runs use
  it. The statics and resort gates are exact.
- Uneven plane counts (23 planes over 4 ranks) put each rank's halo_hi
  slot at a different offset. Under ISHMEM a put lands at the SENDER's
  offset on the peer, so it overwrote the peer's last owned plane (1.5% of
  pairs missing at step 0), and unequal symmetric allocation sizes
  corrupted the heap. lammps_md and grayscott now keep halo_hi at a fixed
  slot after the maximum plane count, so every rank's slab has one shape.
- lbann's editions inherited the CUDA default of 8 work-groups x 256; on a
  PVC tile that is 2048 work-items and even a quarter-size deck exceeded
  the cap. The grid-stride kernels keep bit-identical sums at any
  work-item count, so the baseline runs use `--blocks 1024`.

## Stage 2: 4-node baselines at the plan's anchor decks (32 GB/node)

**Stage 2 is complete: every baseline has a 4-node deck that fits 5 minutes on every substrate.**

Cap 240 s per rank. lammps_md's deck is bounded by memory before time: the
padded Verlet list is nb^3 x cap x maxneigh x 4 B, and at the plan's
L=406 that is ~66 GB per rank (cap 48 against ~18 atoms per bin), past a
tile; L=256 (16.5 GB of list per rank) is the stage-two deck and runs in
under 2 s, so the 15-minute stage can raise steps and checkpoint volume
rather than L. The anchor lbann deck (65536 -> 65536 -> 458752,
batch 1024) exceeded the cap on MPI: at 32 GB of parameters per node the
dense kernels need more than 4 minutes for 5 steps, so a smaller lbann
deck is measured first to size the largest one that fits.

| workload | deck | MPI | oneCCL | Intel SHMEM |
|---|---|---|---|---|
| kmeans | 128 GB global (32 GB/rank), 4 iters, 1024 work-groups | 79.1 s (19.8 s/iter, comm 2.3 s) PASS | 77.6 s (19.4 s/iter, comm 0.1 s) PASS | 66.9 s (16.7 s/iter, comm 8.3 s) PASS with a 16 GB heap (out of device memory at 40 GB) |
| grayscott | 128 GB global (32 GB/rank), 8 steps, 1 MB page, 40 GB symmetric heap | 1.34 s (168 ms/step, comm 28 ms) PASS | 1.34 s (168 ms/step, comm 26 ms) PASS | 1.36 s (170 ms/step, comm 31 ms) PASS; v_checksum identical on all three |
| gmx | K=2048 (17 GB/node), 20 M atoms, 1 pass | 1009 ms PASS | 990 ms PASS | 992 ms PASS; conservation exact, checksums identical; gather dominates (~880 ms) |
| lbann | 65536 -> 65536 -> 458752, batch 1024, 5 steps (anchor) | TIMEOUT (> 240 s at 8 work-groups) | -- | -- |
| lbann | 16384 -> 16384 -> 114688 (2 GB/node), batch 1024, 3 steps, 1024 work-groups | 27.7 s (9.2 s/step) PASS | 25.9 s (8.6 s/step) PASS | 29.1 s (9.7 s/step) PASS; digests identical |
| lbann | 32768 -> 32768 -> 229376 (8 GB/node), batch 1024, 5 steps, 1024 work-groups | 179.0 s (35.8 s/step, comm 8.8 s) PASS | 172.4 s (34.5 s/step, comm 1.7 s) PASS | 179.2 s (35.8 s/step, comm 8.4 s) PASS; digests identical. The 5-minute-stage lbann deck; the 32 GB anchor (~150 s/step) belongs to the 15-minute stage |
| lammps_md | L=256 (67 M atoms, 17 M/rank), 20 steps, rebin 10, ckpt every 5 (DRAM), cap 48 | 1.63 s (81.6 ms/step, halo 246 ms) PASS; 4 checkpoints x 1.3 GB staged at 51.5 GB/s | 1.59 s (79.3 ms/step, halo 201 ms) PASS | 1.76 s (87.9 ms/step, halo 378 ms) PASS. Each leg's 225 s of wall time is the host deck build and the double-precision statics reference at 67 M atoms, outside the timed run |

## Stage 3: Eternia at 4 nodes, stage-one decks

`pbs_newcoro_aurora_4n.sh` (the two-node script with the node count from
the allocation and the storage tiers sized from the deck through
`BENCH_TIER_HBM_MB` / `BENCH_TIER_RAM_MB`).

First attempt, finding: the four-node job reused the single-node config
unchanged, whose tiers total 640 MB per node (kmeans: 64 MB of HBM + 576
MB of RAM). A 1 GB/node deck filled them, every put was refused -- the
reduction's tiny blobs included -- and all four ranks died with "timed
out publishing" after the 120 s reduction timeout. The two-node runs fit
because they carried 128 MB per node. The tiers now follow the deck.

| workload | deck | Eternia (4 nodes) |
|---|---|---|
| kmeans | 4 GB global (1 GB/rank), 4 iters, 1 MB page, HBM 512 MB, tiers 2 GB + 2 GB | 2.76-2.78 s (1.45 GB/s), checksum identical on all ranks, 158-177 faults/rank. Baselines on the same deck: MPI 2.47 s, oneCCL 2.54 s, ISHMEM 2.05 s. Not yet resident (512 MB cache against 1 GB); E1 proper needs the cache to hold the deck |
| grayscott | 4 GB global, 4 steps, 1 MB page, HBM 512 MB, tiers 2 GB + 2 GB | 0.69-0.74 s (43-46 GB/s of paged traffic), v_checksum 3086389.535277 = the baselines' exactly; 520-528 faults and 2048 puts per rank per run. Baselines on the same deck: 35 / 43 / 52 ms. At a deck this small every step's full write-back through the transfer engine dominates; the ratio is the number to watch as the deck grows |
| gmx | default deck, 128 KB page | pending |
| lbann | default deck | refused by the edition's own check: at 4 nodes the output band is 2 rows per node against 4-row pages (nodes would share a page). Resubmitted on the stage-one deck 1024 -> 4096 -> 256, batch 64, 5 steps (16 pages per node in both bands) |
| lbann | 1024 -> 4096 -> 256, batch 64, 5 steps | pending |
| lammps_md | L=28, 10 steps, ballistic gate | pending |
| grayscott, resident (E1 configuration) | as above but HBM tier 2 GB/node holds the deck, tiers 4 GB + 2 GB | 0.61-0.81 s, checksum 3086389.535277 = the baselines'; 520-528 faults (frame cache, served from the HBM tier) and 2048 puts per rank; no storage traffic |
| kmeans, resident (E1 configuration) | as above but HBM tier 2 GB/node holds the whole deck, tiers 4 GB + 2 GB | 2.81-2.84 s (1.41 GB/s), checksum identical on all ranks, 162-174 faults/rank. The faults are the vector's per-block frame cache (8 pages/block, streaming by design), served from the HBM tier: no storage traffic, which is E1's condition |

## Plan coverage so far

| study | status |
|---|---|
| 4-node baseline (plan's step 1) | baselines: complete for all five workloads on MPI, oneCCL and Intel SHMEM at stage-one decks; stage-two (5-minute) decks complete for kmeans, gmx, lbann, lammps_md, grayscott anchor queued. Eternia: kmeans and grayscott pass at stage-one decks (checksums equal to the baselines'); gmx, lbann, lammps_md queued; the resident-deck pass (E1's configuration) queued for kmeans and grayscott |
| E1 scaling (8 -> 64 nodes) | not started; needs the 4-node rung closed first |
| E2 page size | not started |
| E3 persistence | not started; the baselines' `--ckpt-dir` arms (Lustre-direct, DAOS-direct) exist in the lammps_md edition |
| E4 tiering | partly exercised before this plan at 2 nodes: kmeans, grayscott and weights through HBM -> DAOS, HBM -> Flare, and DRAM -> DAOS -> Flare (AURORA.md); the 4-node five-composition sweep is not started |
| E5 memory reduction | not started |
| E6 organization (gnn) | not started; gnn has only the older single-node CTE edition and needs its OGB datasets staged |

Order of work in force: every cell at 4 nodes under 5 minutes first, then
the same cells grown toward 15 minutes, non-Eternia before Eternia.

## Stage 4: baselines toward 15-minute runs

Cap 840 s per rank, 15-minute walltime. lbann's 32 GB anchor is about
150 s per step at 1024 work-groups, and with the host-side seeding of
34 GB of parameters per rank 5 steps overran the cap on MPI; 3 steps is
the deck that fits 15 minutes, each substrate as its own job. The other four already fit 5 minutes at
the plan's anchor decks, so their 15-minute rows raise the work
(iterations, steps, passes, checkpoint volume) at the same size.

| workload | deck | MPI | oneCCL | Intel SHMEM |
|---|---|---|---|---|
| lbann | 65536 -> 65536 -> 458752 (32 GB/node), batch 1024, 5 steps, 1024 work-groups; one job per substrate | TIMEOUT at 840 s | (not run) | (not run) |
| lbann | the same at 3 steps | pending | pending | pending |
| kmeans | 128 GB global (32 GB/rank), 12 iters, 1024 work-groups | pending | pending | pending |
| grayscott | 128 GB global, 256 steps, 1 MB page | pending | pending | pending |
| gmx | K=2048, 20 M atoms, 200 passes | 190.1 s (spread 13.9 s, gather 173.1 s, comm 0.1 s) PASS | 190.1 s (spread 13.9 s, gather 173.0 s, comm 0.2 s) PASS | 190.2 s (spread 13.9 s, gather 173.1 s, comm 0.2 s) PASS; checksums identical to the single-pass anchor on all three |
| lammps_md | L=256, 300 steps, rebin 10, checkpoint every 5 (60 x 1.3 GB, DRAM); one job per substrate | pending | pending | pending |
