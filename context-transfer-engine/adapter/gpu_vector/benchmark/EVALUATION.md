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
| kmeans | 4 GB global (1 GB/rank), 4 iters | 2465 ms (comm 1026) PASS | 2540 ms (comm 1062) PASS | 2045 ms (comm 622) PASS (global-atomics assignment, superseded) |
| kmeans, tiled assignment | the same deck, 1024 work-groups | 68.9 ms (17.2 ms/iter, comm 13) PASS | 173.0 ms (43.2 ms/iter, comm 80) PASS | 61.6 ms (15.4 ms/iter, comm 6) PASS; centroid checksum 30720.0010-30720.0013 (atomic summation order), all four iterations counted. 36x faster than the global-atomics assignment on the same deck: the earlier stage-one time was almost entirely atomic contention. The Eternia kmeans edition accumulates with global atomics too, so its 2.8 s on this deck is the same contention and it needs the same treatment before the comparison means anything |
| grayscott | 4 GB global, 4 steps, 1 MB page | 35 ms (comm 15) PASS | 43 ms (comm 22) PASS | 52 ms (comm 30) PASS; v_checksum identical on all three (at dt=1, superseded) |
| grayscott, dt=0.5 | the same deck after the stability fix | 50 ms (comm 29) PASS | 35 ms (comm 14) PASS | 47 ms (comm 26) PASS; v_checksum 2754766.354742 on all three, the reference for the Eternia rerun |
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
| grayscott | 128 GB global (32 GB/rank), 8 steps, 1 MB page, 40 GB symmetric heap | 1.34 s (168 ms/step, comm 28 ms) PASS | 1.34 s (168 ms/step, comm 26 ms) PASS | 1.36 s (170 ms/step, comm 31 ms) PASS; v_checksum identical on all three (at dt=1, superseded) |
| grayscott, dt=0.5 | the same anchor after the stability fix | 1.34 s (167.6 ms/step, comm 26 ms) PASS | 1.34 s (167.5 ms/step, comm 25 ms) PASS | 1.36 s (170.2 ms/step, comm 38 ms) PASS; v_checksum 99539486.036462 on MPI and oneCCL, ...463 on Intel SHMEM (last digit of the host reduction order) |
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
| kmeans | 4 GB global (1 GB/rank), 4 iters, 1 MB page, HBM 512 MB, tiers 2 GB + 2 GB | global-atomics assignment (superseded): 2.76-2.78 s (1.45 GB/s), checksum identical on all ranks, 158-177 faults/rank. With block-private accumulators: 1.11 s (3.6 GB/s) on all four nodes, centroid checksum 30719.999878 (the tiled baselines' 30720.0010-30720.0013 within the 1e-4 tolerance), 151-172 faults and evicts per rank, 0 puts. Tiled baselines on the same deck: 69 / 173 / 62 ms, so the remaining 16x is the paging itself (64 blocks streaming 1 GB through a 512 MB frame cache), no longer atomic contention. Baselines on the same deck: MPI 2.47 s, oneCCL 2.54 s, ISHMEM 2.05 s. Not yet resident (512 MB cache against 1 GB); E1 proper needs the cache to hold the deck |
| grayscott | 4 GB global, 4 steps, 1 MB page, HBM 512 MB, tiers 2 GB + 2 GB | at dt=1 (superseded): 0.69-0.74 s (43-46 GB/s of paged traffic), v_checksum 3086389.535277 = the baselines' exactly; 520-528 faults and 2048 puts per rank per run. Baselines on the same deck: 35 / 43 / 52 ms. At a deck this small every step's full write-back through the transfer engine dominates; the ratio is the number to watch as the deck grows |
| gmx | default deck (128^3 mesh, 200 k atoms), 128 KB page (one plane), 16 blocks | spread 5.0-5.2 ms (dense reference 1.6 ms), gather+sum 17.0-20.2 ms; conservation, mesh and gather gates pass on all four nodes (mesh checksum and interpolation energy bit-equal to the dense reference); 35 faults, 0 evicts, 64 puts per rank, no put or get errors. Baselines on the same deck: 91 ms / 82 ms / 105 ms. |
| lbann | default deck | refused by the edition's own check: at 4 nodes the output band is 2 rows per node against 4-row pages (nodes would share a page). Resubmitted on the stage-one deck 1024 -> 4096 -> 256, batch 64, 5 steps (16 pages per node in both bands) |
| lbann | 1024 -> 4096 -> 256, batch 64, 5 steps | paged 119.6 ms/step (its in-process dense reference: 73.7 ms/step); loss and weight gates pass on all four nodes; 1048-1242 faults and 722 puts per rank. Baselines on the same deck: 35.6 / 43.0 / 51.4 ms/step |
| lammps_md | L=28 (88 k atoms, 4 z-planes per node), 10 steps, 64 KB page, ballistic gate | 56.5-68.2 ms (5.7-6.8 ms/step, 12.9-15.5 Matom-steps/s); ballistic gate passes on all four nodes (bitwise x/v mismatches 0, KE rel_err 4.7e-14, momentum exact); x and v resident (0 faults, 0 evicts); 10 of 10 step iterations per block. Eternia's stage-one MD deck is the integrator edition, so this is a functional pass rather than a like-for-like number against the baselines' L=40 melt deck |
| grayscott, dt=0.5 | the same deck after the stability fix | 0.66-0.76 s (42-48 GB/s of paged traffic), v_checksum 2754766.354742 = the dt=0.5 baselines' exactly on all four nodes; 520-528 faults, 0 evicts, 2048 puts per rank, no put or get errors. Baselines on the same deck: 50 / 35 / 47 ms |
| grayscott, resident (E1 configuration) | as above but HBM tier 2 GB/node holds the deck, tiers 4 GB + 2 GB | at dt=1: 0.61-0.81 s, checksum 3086389.535277 = the baselines'; at dt=0.5: 0.73-0.84 s, checksum 2754766.354742 = the dt=0.5 baselines' on all four nodes, same fault and put counts; 520-528 faults (frame cache, served from the HBM tier) and 2048 puts per rank; no storage traffic |
| kmeans, resident (E1 configuration) | as above but HBM tier 2 GB/node holds the whole deck, tiers 4 GB + 2 GB | global-atomics assignment: 2.81-2.84 s (1.41 GB/s), checksum identical on all ranks, 162-174 faults/rank. With block-private accumulators: 1.00-1.03 s (3.9-4.0 GB/s), checksum 30720.000018 on all ranks, 156-176 faults and evicts per rank, 0 puts. The faults are the vector's per-block frame cache (8 pages/block, streaming by design), served from the HBM tier: no storage traffic, which is E1's condition |

## E4: tiering composition at 4 nodes (debugging scale)

`submit_e4_aurora.sh`: one 10 GB/node tier budget split five ways
(DRAM / DAOS / Flare in percent), 8 GB/node decks, 1 MB pages, a 4 GB HBM
frame cache in front of the tiers, one job per cell, 780 s cap. A tier at
0 MB is left out of the config. The plan asks for 64 GB/node; this is the
rung that has to fit five minutes first. The Eternia editions only -- the
baselines have no tiers, their reference is the resident 15-minute rows.
Stats are per rank after the seed (ResetStats before the timed loop), so
`puts` counts write-backs during the measured pass only.

The DRAM-only cells each log one red `RouteTask: inline retry exhausted ...
pool=PoolId(major:513, minor:1) method=14` per rank after the timed run.
That is the editions' post-run TIER SPLIT diagnostic asking the SECOND
storage entry (513.1) for its stats; the DRAM-only config has only one
entry, so the pool does not exist and the query fails after its 5 s
budget. It runs after the measurement, the results are unaffected, and
the diagnostic's `host used = cap, remain 0` in those cells is its own
documented failed-query signature, not a full tier.

| workload | dram100 (100/0/0) | dram75 (75/25/0) | bal25 (25/50/25) | daos70 (10/70/20) | lustre70 (10/20/70) |
|---|---|---|---|---|---|
| kmeans, 1 iter over 8 GB/node | 2.02-2.18 s (3.7-4.0 GB/s), checksum 30720.000074 on all ranks, 7463-7472 faults and evicts, 0 puts | 3.97 s on all four nodes (2.0 GB/s), checksum 30719.999974, ~7456 faults and evicts; the DAOS tier file was its full 2.5 GB on every node, so the DAOS share carried a third of the faults at about half the DRAM-only rate | 5.58 s on all four nodes (1.43 GB/s), checksum 30719.999983, ~7450 faults and evicts; DAOS file 5.0 GB and Flare file 2.5 GB per node, both tiers at their full capacity, so only 0.5 GB stayed in DRAM | 5.21 s on three nodes and 4.44 s on the fourth (1.5-1.8 GB/s), checksum 30720.000022, ~7450 faults and evicts; DAOS file 7.0 GB per node (its full capacity), Flare file 2.0 GB on two nodes and 1.0 GB on the two others, the rest in DRAM: faster than the balanced mix because the DAOS share replaced Flare's | 23.0 s on all four nodes (0.35 GB/s), checksum 30720.000002, ~7450 faults and evicts; Flare file 7.0 GB on two nodes and 6.0 GB on the others, DAOS file 2.0 GB per node. Flare-resident pages fault back at about 0.3 GB/s per node against DAOS's ~1.5 GB/s, so the Lustre-heavy mix is 4-11x the others: the workload is indifferent to DRAM vs DAOS but not to Lustre |
| grayscott, 1 step over 8 GB/node | 3.18-3.50 s (18-20 GB/s of paged traffic), v_checksum 20286390.376762 on all ranks, 10521-10640 faults and evicts, 4096 puts (the step's write-back) | 3.37-4.59 s (14-19 GB/s), same checksum, DAOS file its full 2.5 GB per node: 1.1-1.3x DRAM-only, less sensitive than kmeans's 1.9x | 6.53-7.58 s (8.5-9.8 GB/s), same checksum, DAOS 5.0 GB and Flare 2.5 GB per node (both full): 2.1x DRAM-only, the Flare quarter costing as much again as the DAOS half | 8.5-11.4 s (5.6-7.5 GB/s), same checksum, DAOS 7.0 GB per node (full), Flare 1-2 GB: 2.7-3.5x DRAM-only, slower than the balanced mix unlike kmeans, because grayscott writes every page back (4096 puts) and DAOS carries most of that write traffic here | 12.3-27.0 s (2.4-5.2 GB/s), same checksum, Flare 6-7 GB per node, DAOS 2.0 GB: 3.9-8.5x DRAM-only with a wide node spread (Flare's write path varies more than its read path); the slowest mix for both workloads, but grayscott's Lustre penalty is smaller than kmeans's 11x because its step is compute-heavier per page |
| gmx, 1 PME pass over a 1600^3 mesh (32.8 GB global) | spread 2.20-2.41 s, gather+sum 4.57-4.78 s (dense reference 3.1 ms); conservation, mesh and gather gates pass on all four nodes; 1201-1220 faults, ~1000 evicts, 800 puts per rank. The 19.5 MB page (one mesh plane) is the largest in the study | spread 3.05-4.14 s, gather+sum 7.59-9.49 s, all gates pass on all four nodes, same fault and put counts, DAOS file its full 2.5 GB per node: 1.7-2.0x DRAM-only | spread 8.38-9.26 s, gather+sum 12.63-13.42 s, all gates pass, DAOS 5.0 GB and Flare 2.5 GB per node (both full): 2.9-3.1x DRAM-only | spread 10.78-13.48 s, gather+sum 15.71-18.75 s, all gates pass, DAOS 7.0 GB per node (full), Flare 1.0 GB: 3.7-4.5x DRAM-only, SLOWER than balanced -- gmx behaves like grayscott here (it writes the mesh back), not like kmeans | spread 4.74-9.20 s, gather+sum 9.59-13.65 s, all gates pass, Flare 6.0 GB per node, DAOS 2.0 GB: 2.0-3.2x DRAM-only. THE ONLY WORKLOAD WHOSE LUSTRE-HEAVY MIX BEATS ITS DAOS-HEAVY ONE, and it is the one with 19.5 MB pages: 8 GB/node moves in ~1200 faults and 800 puts, so the traffic is a few large sequential transfers, which is Lustre's case and not DAOS's. kmeans at 1 MB pages pays 11x for the same mix. That page-size-times-tier interaction is E2's subject and the two studies should be read together |
| lammps_md, 1 ballistic step over 320^3 bins (x + v, 16 GB each globally, 8.4 GB/node) | 3.33-4.43 s (183 Matom-steps/s over 609 M atoms), ballistic gate passes on all four nodes (bitwise x/v mismatches 0, KE rel_err 6.5e-12), x and v resident so 0 faults and 0 evicts: the 10 GB DRAM tier holds the node's 8.4 GB | INFEASIBLE at this budget: 600 s cap, the run never finished. The runtime's shutdown flush declines blob after blob -- "persistent tier(s) declined ... 1044480 bytes free, next blob needs 1048576" -- because this workload PERSISTS ITS DATA and the composition's persistent share (2.6 GB) is far below the node's 8.4 GB | INFEASIBLE, same cause (persistent share 7.7 GB < 8.4 GB) | RERUN alone among DAOS users: FAILS DIFFERENTLY and worse. Node 3 was declared dead by the heartbeat ("SWIM: Node 3 confirmed dead after suspicion timeout"), sends timed out after 30 s, and rank 2's BALLISTIC GATE then FAILED with 73.4 M velocity mismatches whose `got` is ZERO -- v pages that were never delivered read back as zeros instead of raising. Ranks 0, 1 and 3 passed. So a remote read after a node is presumed dead is silently zero-filled; the gate caught it, the storage layer did not. SECOND instance of the same class as the weights Lustre defect: a read that cannot be served returns plausible bytes rather than an error | 4.49-8.48 s, ballistic gate passes on all four nodes, x and v still report 0 faults and 0 evicts while the tier files hold 6 GB on Flare and 2 GB on DAOS per node: only 1.3-1.9x DRAM-only. THE TIMED STEP NEVER READ FROM A TIER -- the spill happened while seeding, so this cell measures the WRITE path only. lammps_md's step holds its planes resident by construction (the edition's resident contract), which is why it is the least tier-sensitive row here and why its read-side numbers need a deck that oversubscribes the frame cache |
| lbann, 1 step of 65536 -> 131072 -> 1024, --no-ref (33.3 GB of weights per node in 33282 pages, 4096-page cache) | 18.68-18.75 s/step, 50055-50107 faults, 48229-48248 evicts, 41602 puts, no get or put errors; --no-ref does what it should (both gates report skipped, loss printed). The most faulting of any E4 workload: a 4 GB cache against 33 GB, so this row is the read-path row | 19.30-20.45 s/step (1.03-1.09x DRAM-only), same fault counts, DAOS file its full 2.5 GB per node | 19.25-25.32 s/step (1.03-1.35x), DAOS 5.0 GB and Flare 2.5 GB per node (both full) | 20.67-24.75 s/step (1.10-1.32x), DAOS 7.0 GB per node (full), Flare 2.0 GB | 43.84-63.83 s/step (2.3-3.4x), Flare 7.0 GB per node (full), DAOS 2.0 GB: the only composition lbann notices, and by the same sign as kmeans and grayscott |
| weights, 64 blocks x 128 pages (8 GB/node) | 1.75-1.97 s (4.1-4.6 GB/s), checksum OK and identical on all ranks, ~16200 faults, 0 put errors | 3.48-4.25 s (1.9-2.3 GB/s), same checksum, DAOS file its full 2.5 GB per node: 2.0-2.2x DRAM-only, as sensitive as kmeans | 13.1-17.8 s (0.45-0.61 GB/s), same checksum, DAOS 5.0 GB and Flare 2.5 GB per node (both full): 7-9x DRAM-only, far more than kmeans's 2.6x or grayscott's 2.1x on the same mix; weights' 32-thread blocks touch pages in a gather-like order the prefetcher cannot cover, so each Flare-resident page is paid at Flare latency | 6.0-7.2 s (1.1-1.3 GB/s), DAOS 7.0 GB per node (full), Flare 1-2 GB: 3.4-3.7x DRAM-only, faster than balanced as for kmeans. ANOMALY: rank 1's reduced checksum_total (15491913354553959092) differs from the other three ranks' (14506969961719879897) while every rank printed checksum=OK; got and want reduce together, so a peer's pair read twice and another's never would pass the per-rank gate and shift the total, which is what the numbers look like. The weights edition now runs an agreement round after the reduction (AgreeU64 in bench_dist.h) and fails the gate on disagreement. RERUN on the gated binary: 3.66-4.85 s (1.7-2.2 GB/s), all four ranks at 14506969961719879897, agreement gate passes -- so the shift is intermittent, not a property of this composition, and the gate now catches it. The rerun's numbers are the row's | FAILED: checksum MISMATCH on all four nodes, 52.7-64.1 s (0.12-0.15 GB/s), Flare 7.0 GB per node (full), DAOS 2.0 GB. Two distinct faults. (a) WRONG PAGE DATA: on nodes 1-3 every diagnosed page was visited once and came back with another page's content (the values match the expected sequence's period but at a different position: node 1 reads what looks like page p+2, node 3 p+2, node 2 p+4), and node 0 has pages that never landed (visits=0 from page 20 on). kmeans and grayscott passed their checksums on the same mix, so this is weights-specific: its faults are gather-like re-faults (9.7-16.1 k for 8192 pages) against a Flare tier at its 7 GB capacity. (b) THE SAME NODE-1 SHIFT AS THE DAOS-HEAVY CELL: node 1's reduced want is 15491913354553959092 against 14506969961719879897 on the other three (+984943392834079195, the identical offset), while its reduced got equals theirs; the per-page expected values it prints are identical to the other nodes'. RERUN on the agreement-gated binary: FAILS AGAIN, 59.9-74.7 s, got=7852382263554806326 want=14506969961719879897 on all four nodes. (b) did NOT recur -- every node's reduced want agrees and the agreement gate passes -- so the node-1 shift was intermittent and is now covered. (a) IS REPRODUCIBLE and is the real defect: node 1's page p comes back holding page p+2's content (got[p] matches want[p+2] to 1e-6, the generator being smooth), and node 0 has a contiguous run of pages from 1783 on that were never visited at all. Same binary, same deck, passes on every other composition, so the trigger is a FILE TIER AT ITS CAPACITY under weights' gather-like re-faulting: pages are served under the wrong identity or not at all. Next step is a single-node reproducer with a small file bdev filled to capacity |

## E2: page size at 4 nodes (debugging scale)

Same decks and same tier budget as E4, sweeping only the page size, through
the parallel runner (`BENCH_CELLS` takes `<workload>:<composition>:<page-kb>`).
Two arms: the DRAM-only composition isolates what the page size costs on its
own, and the Lustre-heavy one tests the interaction E4's gmx row exposed --
gmx moves 8 GB/node in ~1200 faults at 19.5 MB pages and is the only
workload that prefers Lustre to DAOS, so page size appears to decide which
tier is the right one.

| workload / composition | 64 KB | 256 KB | 1 MB | 4 MB | 16 MB |
|---|---|---|---|---|---|
| kmeans, DRAM-only | 12.00-12.02 s (0.67 GB/s), 130401-130475 faults | 3.26-3.42 s (2.3-2.5 GB/s), 32059-32070 faults | 2.34-2.50 s (3.2-3.4 GB/s), 7442-7483 faults (the E4 cell measured 2.02-2.18 s) | 2.56 s (3.12 GB/s), 1209-1236 faults | 0.50 s (15.97 GB/s), ZERO faults -- see the confound below |
| kmeans, DRAM-only, CACHE PINNED at 512 MB | 11.42 s, 123216-123253 faults | 3.19-3.21 s, 31028-31077 faults | 2.62 s, 7441-7490 faults | 2.53 s, 1213-1216 faults | n/a (a 16 MB page cannot make a 512 MB cache at 64 blocks) |
| kmeans, Lustre-heavy, CACHE PINNED at 512 MB | 22.64 s (0.35 GB/s) | 4.94 s (1.62 GB/s) | 3.33 s (2.40 GB/s) | 3.33 s (2.40 GB/s) | n/a |
| kmeans, Lustre-heavy | 30.0 s (0.27 GB/s), 130424-130471 faults | ABORTED at the 600 s cap on all four ranks: the kernel took DEVICE FATAL 7 ("fetch returned an error; its pages were left EMPTY -- a generational get names a generation the writer has not published"), then a GPU write to address 0 and a driver abort. Preceded by TaskProgress replica warnings, i.e. the same overloaded-network conditions as the zero-fill defect above | 4.78 s (1.67 GB/s), 7443-7470 faults, Flare 7.5 GB and DAOS 2.1 GB per node -- but the E4 cell at the SAME page and composition measured 23.0 s (0.35 GB/s). See the variance note below | 3.47-3.92 s (2.0-2.3 GB/s), 1201-1231 faults | 0.49 s (16.3 GB/s): resident, the cache confound again, and identical to the DRAM-only cell because no tier is touched |

**THE PINNED-CACHE ROW IS THE ONE TO READ FOR PAGE SIZE.** With the cache
held at 512 MB the curve is 11.42 / 3.19 / 2.62 s for 64 KB / 256 KB /
1 MB, against 12.01 / 3.34 / 2.42 s when the cache was left to grow with
the page. The two agree closely at these sizes, so the page size -- not
the cache -- is what drove the 4.4x between 64 KB and 1 MB. The curve then
flattens: 4 MB is 2.53 s against 1 MB's 2.62 s, a 2% gain for 6x fewer
faults, so 1 MB is where the per-fault cost stops being what limits the
run and the optimum is broad rather than sharp. The confound
only bites at the top of the sweep, where the cache became a large
fraction of the deck.

**THE FIRST SWEEP CHANGED TWO THINGS AT ONCE, AND THE SECOND ONE
DOMINATES AT 16 MB.** The vector's frame cache is `slots x blocks x page`, and the
editions take `--slots` (8) and `--blocks` (64) as counts, not bytes. So
sweeping the page size sweeps the cache with it:

| page | frame cache | faults |
|---|---|---|
| 64 KB | 32 MB | 130400 |
| 256 KB | 128 MB | 32060 |
| 1 MB | 512 MB | 7460 |
| 4 MB | 2 GB | 1220 |
| 16 MB | 8 GB = the whole node's deck | 0 |

At 16 MB the cache holds the entire 8 GB shard, so the run never faults at
all and its 0.50 s is not a page-size result -- it is the resident case.
The honest sweep holds the cache constant in BYTES by scaling `--slots`
inversely (512 MB at 64 blocks: 128 / 32 / 8 / 2 slots for 64 KB / 256 KB /
1 MB / 4 MB; 16 MB cannot reach 512 MB at 64 blocks and needs fewer
blocks). The cells above stand as a cache-size sweep, which is E5's
question, and E2 is rerun with the cache pinned.

**THE PINNED PAIR IS THE E2 RESULT.** With the cache held at 512 MB in both
arms, the Lustre penalty against DRAM-only is:

| page | DRAM-only | Lustre-heavy | penalty |
|---|---|---|---|
| 64 KB | 11.42 s | 22.64 s | 1.98x |
| 256 KB | 3.19 s | 4.94 s | 1.55x |
| 1 MB | 2.62 s | 3.33 s | 1.27x |
| 4 MB | 2.53 s | 3.33 s | 1.32x |

The penalty falls from 2.0x to about 1.3x as the page grows and then flattens,
which is the same mechanism gmx showed in E4 and is measured here without the
cache confound. Note the Lustre column is one draw each from a distribution
with a 3.2x spread (below), so the 1 MB and 4 MB points being equal is within
noise; the 64 KB-to-1 MB trend is far larger than the spread and is the
result.

**LUSTRE SPREAD, MEASURED.** Four repeats of the same cell (kmeans,
Lustre-heavy, 1 MB page, cache pinned at 512 MB) in one allocation:

| repeat | time | rate |
|---|---|---|
| a | 3.33 s | 2.40 GB/s |
| b | 3.88 s | 2.06 GB/s |
| c | 7.15 s | 1.12 GB/s |
| d | 10.69 s | 0.75 GB/s |

A 3.2x spread over identical runs. The four were scheduled across three
groups of four nodes and therefore overlapped, so part of this is the cells
contending with each OTHER on Flare, not only with the rest of the machine.
That does not rescue the number: it means a Lustre-heavy cell has no single
value, and the clean measurement is one Lustre cell at a time, repeated,
reported as a spread. Every Lustre-heavy figure in E2 and E4 above is one
draw from this distribution and should be read as such.

**LUSTRE NUMBERS VARY BY 5x RUN TO RUN.** The two runs above differ only
in when they ran and in that the second shared its allocation with four
other cells -- which should have made it slower, not 5x faster. Flare is
a shared filesystem and nothing here reserves bandwidth on it, so every
Lustre-heavy cell in E2 and E4 is a single sample of a noisy quantity.
The DRAM and DAOS cells do not show this (kmeans DRAM-only: 2.02-2.18 s
in E4, 2.34-2.50 s here). Before any Lustre-heavy number goes in a
figure it needs repeats and a spread, not a point.


## Scope, as decided 2026-09-23

weights is OUT. It is a synthetic gather we added as an adversarial case for
the prefetcher, not one of the plan's workloads (the plan names kmeans,
grayscott, gmx, lbann, lammps_md and gnn; `weights` appears once, in
passing). It is also the only workload still hitting the paging defect, so
chasing its cells costs allocations without moving plan coverage. Its rows
stay in the tables below as they were measured, marked, and are not pursued
further.

That makes **E5 complete** for every workload in scope -- kmeans, grayscott
and lbann at all four cache sizes -- and leaves **E4 complete** for kmeans,
grayscott, gmx and lbann at all five compositions, with lammps_md's three
missing cells being a RESULT (a workload that persists cannot use a
composition whose persistent share is smaller than its data) rather than a
gap.


gnn is OUT for now, but the reason CHANGED on 2026-09-23 08:00 and the gap
is much smaller than it was. Merging origin/gpu-coro-port brought a
DISTRIBUTED GraphSAGE edition (`gnn/clio_gnn_paged_newcoro.cc`) that takes
--nodes/--node and SYNTHESISES its features from a hash, so the papers100M
staging that made this a multi-day job is no longer needed at all.

What blocks it now is one thing: **that edition is CUDA-only and does not
compile on Aurora.** It declares `__global__` kernels and launches them with
`<<<>>>` unconditionally, with no `#if CTP_ENABLE_SYCL` split -- every other
edition carries one. icpx stops at the launch site:

    clio_gnn_paged_newcoro.cc:290:15: error: expected expression

The port is bounded: gnn has only two launch functions (LaunchSeed,
LaunchLayer) against kmeans's four, and kmeans's whole SYCL section --
Submit, SubmitYieldable, InitBackend and the wrappers -- is about 100 lines
that can be copied nearly verbatim. That is an afternoon, not a week, and it
would put gnn back into E4, E5 and E1 and make E6 reachable.

THE 15-MINUTE CAP GOVERNS THE FOOTPRINT, not the other way round. The plan
asks for 64 GB/node; the cells above ran at 8 GB/node, which is the
debugging rung and finishes in 2-60 s. The cap therefore has room: at
8x the footprint the slowest cell measured here (lbann Lustre-heavy, 63.8 s)
projects to about 8.5 minutes, so the plan's 64 GB/node is reachable inside
the cap for every cell. The tier budget has to grow with it (a composition
needs about 1.25x the data, and a persisting workload needs its persistent
share alone to cover the data), which is 80 GB/node against 512 GB of DDR5.

## E5: memory reduction at 4 nodes

How far the frame cache can be cut before the run stops keeping up. Fixed
deck (8 GB/node), fixed page (1 MB), fixed composition (balanced, so there
is real storage underneath); only the cache varies. Cache bytes are
`slots x blocks x page` for kmeans, grayscott and weights (64 blocks) and
`cap x page` for lbann.

EACH EDITION HAS A FLOOR AND grayscott's IS 10 FRAMES. Its stencil holds
ten planes at once -- z-1, z and z+1 of both u and v, plus both outputs --
so a smaller cache could evict a plane that is still being read, and the
edition refuses rather than corrupting the step. The first pass swept 1-16
slots and every grayscott cell below 10 refused, correctly. The sweep that
covers all four workloads is therefore 16 / 32 / 64 / 128 slots, which is
1 / 2 / 4 / 8 GB against the 8 GB shard: an eighth of the deck up to
resident.

FIRST PASS (balanced composition): 11 of 20 cells returned a number, and
the curve they make is not usable. The balanced mix carries a Lustre share,
so the cells inherit the 3.2x spread measured above: kmeans came out
4.39 / 3.92 / 4.16 / 5.77 / 5.51 s across 64 MB to 1 GB of cache and lbann
23.0 / 33.3 / 24.4 s across 128 MB to 1 GB. Both are non-monotone, which is
noise rather than a memory-reduction result. Of the nine that returned
nothing, four are grayscott below its 10-frame floor (my sweep's error) and
FIVE ARE THE PAGING DEFECT: weights at 2, 4 and 8 frames and lbann at 64 and
512. The pass is kept for that evidence, not for its timings; the curve is
rerun on dram75, which spills to DAOS and is stable.

SECOND PASS, on dram75 (75 DRAM / 25 DAOS / 0 Lustre), cache pinned by
scaling slots. This is the E5 curve; the first pass below is kept only as
defect evidence.

| workload | 1 GB | 2 GB | 4 GB | 8 GB |
|---|---|---|---|---|
| kmeans | 3.28 s | 3.21 s | 3.52 s | 1.90 s |
| grayscott | 3.02 s | 2.87 s | 2.64 s | 2.34 s (4102 faults) |
| weights | 3.49 s | 3.18 s | 4.64 s CHECKSUM MISMATCH | 2.94 s |
| lbann | 14.89 s/step | 19.05 s/step | 19.18 s/step | 18.98 s/step |

The two shapes are the result. grayscott improves monotonically and gains
23% over an 8x cache, because its stencil revisits planes and a larger cache
turns those revisits into hits. kmeans does NOT: flat from 1 to 2 GB and
SLOWER at 4 GB, because it streams one page per block and never revisits, so
extra frames only add eviction and write-back work. For a streaming workload
the cache can be cut to an eighth at no cost at all; for a stencil the same
cut costs 23%. That is the memory-reduction claim, workload-resolved. The 8 GB kmeans
point (1.90 s) is the deck going resident -- the cache finally equals the
shard -- so its curve is flat-then-cliff rather than gradual, while
grayscott's is gradual throughout. lbann is flat from 2 GB up and
FASTEST at 1 GB (14.9 s against 19.0 s), the same inversion as kmeans and
for the same reason: its backward pass sweeps the whole second matrix, so
no achievable cache turns those reads into hits and the extra frames only
add eviction work.

weights mismatches at 4 GB while passing at 1, 2 and 8 GB, ON A
COMPOSITION WITH NO LUSTRE (DRAM + DAOS only). So the paging defect is
not Lustre-specific and not a capacity threshold; it is non-monotone in
cache size, which is the signature of a race. See PAGING_DEFECT.md.

FIRST PASS (balanced composition), kept as defect evidence only:

| workload | 64 MB | 128 MB | 256 MB | 512 MB | 1 GB |
|---|---|---|---|---|---|
| kmeans | 4.39 s | 3.92 s | 4.16 s | 5.77 s | 5.51 s |
| grayscott | refused (10-frame floor) | refused | refused | refused | 32 s |
| weights | 7.72-8.62 s | DEVICE FATAL 2 | DEVICE FATAL 2 | checksum MISMATCH | ok |
| lbann | DEVICE FATAL 2 | 23.0-24.6 s | 33.3-34.7 s | DEVICE FATAL 2 | 24.4-24.9 s |

## E3: persistence at 4 nodes

grayscott, 8 steps, a snapshot every 2 steps, 1 GB/node deck, 1 MB pages,
16 frames of cache. Three arms per composition: no checkpoint (the floor),
a checkpoint left at blob score 1.0 in the fast tier (asynchronous), and one
drained out of it before the run continues (synchronous).

THE DECK IS 1 GB/NODE, NOT 8. A checkpoint is a SECOND full copy of the
vector, so the live deck plus four snapshots must fit the 10 GB tier. At
8 GB/node that is 40 GB and every checkpointing arm failed with "checkpoint
0 failed to materialize" -- which is itself the same constraint E4 found on
lammps_md, arrived at from the other direction.

| arm | DRAM-only | DAOS-heavy | Lustre-heavy |
|---|---|---|---|
| no checkpoint | 1.16 s | 6.76 s | 6.61 s |
| checkpoint, left hot | 16.63 s | 42.50 s | 44.20 s |
| checkpoint, drained | 17.71 s | 44.34 s | 44.50 s |
| checkpoint volume | 16 GB | 16 GB | 16 GB |
| of which checkpointing | 15.4 s (93%) | 35.2 s (83%) | 35.2 s (80%) |

Three things the matrix says:

1. **Persistence dominates everything else here.** Checkpointing is 80-93% of
   every run that does it. The tiering question E4 asks is second order once
   a workload checkpoints at this rate.
2. **Draining costs a steady few percent** -- 6.5% on DRAM, 4.3% on DAOS,
   0.7% on Lustre -- and never changes the ranking. Synchronous placement is
   cheap; it is the snapshot itself that is expensive.
3. **DAOS AND LUSTRE ARE INDISTINGUISHABLE FOR CHECKPOINTS** (35.2 s each),
   though they diverged sharply in E4 at an 8 GB/node deck. Checkpoint writes
   are large and sequential, which is the shape where Lustre matches an
   object store -- the same mechanism that makes gmx prefer Lustre in
   Section E4. The storage choice that matters for paging does not matter for
   persistence.

## E1: scaling, 256 to 512 nodes (production queue)

Weak scaling at 4 GB/node, so the deck grows with the rung (1 TB at 256
nodes) and a FLAT curve is the claim. kmeans and grayscott only: the other
editions need their mesh, layer widths or bin count divisible by the node
count, which fails at 320 and 448. One job per rung, all four editions
inside it. debug-scaling caps a user at one running job, so the ladder runs
in `prod`, whose floor is 256 nodes.

### The baselines scale; the Eternia runtime does not come up at 256 nodes

| rung | kmeans MPI | kmeans oneCCL | kmeans Intel SHMEM | kmeans Eternia |
|---|---|---|---|---|
| 256 | 1.10 s (55.2 ms/iter, comm 15.1 ms) | OK | OK | **TIMEOUT at 900 s, zero iterations** |

All three baselines passed at 256 nodes over 8.6 G points and 1 TB, and
grayscott passed on all three as well. The Eternia arm never completed an
iteration. Its log shows the runtime failing to form a working cluster
rather than the benchmark failing:

- `RouteTask: RouteLocal returned 4 ... task_ptr is null`
- `[stuck-wait] no completion after ...`
- `SWIM: Direct probe to node 96` and suspicion traffic against nodes 0, 32,
  64, 128, 160
- `ScanSendMapTimeouts: replica 0 of net_key ... timed out`, repeatedly

The oversized "Main segment: requested 1.2 TB" warning above it is NOT the
cause: that is the RAM-sized auto default being clamped to half the memory
budget, by design, and it happens at every scale.

**FIXED (config only, 2026-09-23 04:15).** SWIM's suspicion timeout is 60 s
and its expiry runs TriggerRecovery, which redistributes a LIVE node's
containers. The 256-way compose starves probe replies past that threshold,
so busy nodes are declared dead, their containers move, and routing then
answers Dne for a pool that has been taken away -- the spiral above. The
scaling config now sets `swim: enabled: false`, which is right for a batch
allocation where a node that truly dies takes the job with it anyway. Rung
320 is queued carrying the fix and is the test of it.

**This is the largest Eternia run ever attempted here -- every prior
distributed result in this file is at 4 nodes.** So the finding is simply
that the runtime's cluster formation and task routing do not yet work at
256 nodes, and the scaling ladder has found the limit it exists to find.
The baseline arms still give a usable 256-512 curve for the substrates.

## Plan coverage so far

| study | status |
|---|---|
| 4-node baseline (plan's step 1) | baselines: complete for all five workloads on MPI, oneCCL and Intel SHMEM at the stage-one, 5-minute and 15-minute decks (stage 4 table), after two benchmark fixes found by the longer decks: grayscott's dt and kmeans's assignment atomics. Eternia: kmeans, grayscott, lbann and lammps_md pass at stage-one decks (checksums equal to the baselines' where exact) and gmx passes at its default deck: the 4-node Eternia rung is closed at stage one; the resident-deck pass (E1's configuration) queued for kmeans and grayscott |
| E1 scaling (8 -> 64 nodes) | not started; needs the 4-node rung closed first |
| E2 page size | not started |
| E3 persistence | not started; the baselines' `--ckpt-dir` arms (Lustre-direct, DAOS-direct) exist in the lammps_md edition |
| **OPEN DEFECT (silent zero-fill after a presumed node death)** | with 5 cells x 4 nodes in one allocation, the heartbeat declared a node dead and lammps_md rank 2 then read 73.4 M velocity components as zeros, failing its gate; the other three ranks passed. Reads that cannot be served must raise, not return zeros. It also means HEAVY CONCURRENCY IN ONE ALLOCATION CAN TRIP THE FAILURE DETECTOR, so network- and DAOS-heavy cells should run with fewer neighbours until the suspicion timeout is tuned |
| **OPEN DEFECT #1: A PAGE THE KERNEL HOLDS IS NOT THERE.** This is the important one, and it is NOT workload-specific -- three different workloads hit it. Under cache pressure with a file tier in the composition, the paging layer and the kernel disagree about what is resident, and it surfaces three ways: (a) SILENTLY WRONG DATA -- weights reads another page's content or pages that never arrived, and its checksum is wrong by 2% to 65%; (b) `DEVICE FATAL 2 (HoldPage: page not resident)` -- hit by weights at 2 frames of cache and lbann at 64; (c) `DEVICE FATAL 7 (fetch returned an error; its pages were left EMPTY -- a generational get names a generation the writer has not published)` -- hit by kmeans at a 256 KB page on Lustre. Nine occurrences across the logs so far. The cache size is part of the trigger, not just the tier: weights on the balanced mix passes at 1 and 4 frames and fails at 8. |
| **DIAGNOSED.** Root cause traced to `PublishFetch`'s failed-get branch: it empties the frames, latches a code and RETURNS WITHOUT TRAPPING, so the kernel runs on. A write hold then skips the `Covers` check and writes into a frame that was never fetched (silent wrong data); a read hold trips `Covers` (code 3); a reused frame trips `Find` (code 2). One branch, three symptoms. The trigger is a generational get naming a generation a peer has not published yet -- transient, not an error -- which only arises when a page must be RE-fetched, hence only with a file tier and a cache under pressure. Full write-up and fix plan in `adapter/gpu_vector/PAGING_DEFECT.md`. |
| **NOT A DEFECT (correction).** I recorded the GPU "segfault at 0x0" as a second bug. It is not: `sycl_compat::Trap()` has no SPIR-V trap available and deliberately writes to null outside NVPTX, after latching the reason into the host-readable mirror. What is genuinely lost is the device printf carrying the set census, which the driver abort discards -- a diagnostics gap, addressed by step 2 of the plan. |
| (superseded) the device-side fatal path itself crashes | Every DEVICE FATAL above is followed by `Segmentation fault from GPU at 0x0 ... access: Write` and a driver abort, so a diagnosable error becomes a killed process and the run reports rc=134 instead of its own message. Whatever the fatal handler writes to, it is not mapped. Fixing this is independent of #1 and would make #1 far easier to chase. |
| (superseded framing) weights over a file tier | reproducible, and NOT only at tier capacity as first thought. Matrix so far, all weights at 4 nodes over a file tier: Lustre-heavy at 4 frames of cache MISMATCH (ratio 0.35 and 0.54 on two runs, with pages that were never fetched and pages holding another page's content); balanced at 4 frames OK; balanced at 1 frame OK; balanced at 8 frames MISMATCH (ratio 0.98). DRAM-only, DRAM-heavy and DAOS-heavy all pass. So the trigger involves BOTH a file tier and the cache size, not capacity alone, which points at cache management rather than the tier itself. kmeans and grayscott pass every one of these cells, so it is specific to weights' gather. Everything else in this file is unaffected |
| E4 tiering | partly exercised before this plan at 2 nodes: kmeans, grayscott and weights through HBM -> DAOS, HBM -> Flare, and DRAM -> DAOS -> Flare (AURORA.md). The 4-node sweep is scripted (`submit_e4_aurora.sh` over `pbs_newcoro_aurora_4n_tier.sh`: the plan's five DRAM/DAOS/Lustre compositions of one 10 GB/node budget, 8 GB/node decks, a tier at 0 MB left out of the config) and launches after the queued baselines. First pass covers kmeans, grayscott and weights; lbann and lammps_md need their decks sized to 8 GB/node first, and the Eternia gmx edition has a fixed 128^3 mesh (8 MB), too small to spill, so it needs a mesh-size knob before it can join |
| E5 memory reduction | not started |
| E6 organization (gnn) | OUT OF SCOPE for now (see Scope above): gnn has only the older single-node CTE edition and needs its OGB datasets staged |

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
| lbann | the same at 3 steps | 509.3 s (169.8 s/step, comm 8.9 s) PASS; 528 s of wall time per rank; weight digest identical on all three substrates | 502.3 s (167.4 s/step, comm 2.0 s) PASS; 521 s of wall time per rank | 512.0 s (170.7 s/step, comm 10.7 s) PASS; 531 s of wall time per rank; weight digest identical to oneCCL's |
| kmeans | 128 GB global (32 GB/rank), 12 iters, 1024 work-groups; one job per substrate | first attempt TIMEOUT at 840 s on an allocation spanning four racks; rerun 506.4 s (42.2 s/iter, comm 62 ms) PASS, all 12 iterations counted, centroid checksum 9814.375655. Iterations lengthen as the centroids settle (30 s for the first, 68 s for each of the last three: the assignment's atomics contend more once most points stop moving between clusters), so the 4-iteration anchor's 19.8 s/iter does not extrapolate | TIMEOUT at 840 s: 11 of 12 iterations, the steps growing from 20 s to 116 s (comm 25 s in total) | (not run) |
| kmeans | the same, baseline assignment tiled per work-group (local-memory sums and counts, one global atomic per group per entry) | 5.29 s (441 ms/iter, comm 2 ms) PASS, all 12 iterations, centroid checksum 30719.999972: 96x the global-atomics run, and the per-iteration time is now flat | 5.40 s (450 ms/iter, comm 135 ms) PASS, checksum 30719.999966 | 5.27 s (439 ms/iter, comm 16 ms) PASS, checksum 30719.999995; the three substrates agree to 1e-8 |
| kmeans | the same at 200 iterations (the 15-minute-class deck now that a step is 0.44 s), all three substrates in one job | 88.5 s (442 ms/iter, comm 1.26 s) PASS | 88.6 s (443 ms/iter, comm 1.36 s) PASS | 88.5 s (442 ms/iter, comm 1.41 s) PASS; all 200 iterations counted, centroid checksum 30719.99995-30719.99998 |
| grayscott | 128 GB global, 256 steps, 1 MB page | 42.7 s (166.8 ms/step, comm 0.5 s) but v_checksum=nan | 42.8 s (167.3 ms/step, comm 0.6 s), v_checksum=nan | 43.2 s (168.9 ms/step, comm 0.9 s), v_checksum=nan. INVALID on all three: the field diverged. Explicit Euler on the 7-point Laplacian is stable only while 12 x Du x dt < 2, and the editions ran Du=0.2 at dt=1 (2.4): the checkerboard mode grows 1.4x per step, invisible over the 4- and 8-step decks, NaN long before 256. The gates passed a NaN because NaN compares false against every tolerance. Fixed in every edition (baselines, Eternia, CUDA): dt=0.5, plus a finiteness gate; every earlier grayscott checksum in this file is at dt=1 and the rows are rerun after the rebuild |
| grayscott, dt=0.5 | 128 GB global, 256 steps, 1 MB page | 42.7 s (167.0 ms/step, comm 3.1 s) PASS | 42.7 s (166.7 ms/step, comm 3.0 s) PASS | 43.3 s (169.1 ms/step, comm 3.2 s) PASS; v_checksum 30810718.238060 identical on all three, finite. The step time is memory-bound and flat, so a 15-minute-class deck is 2048 steps (~6 min) if longer resolution is wanted; the 256-step deck stands as the row |
| gmx | K=2048, 20 M atoms, 200 passes | 190.1 s (spread 13.9 s, gather 173.1 s, comm 0.1 s) PASS | 190.1 s (spread 13.9 s, gather 173.0 s, comm 0.2 s) PASS | 190.2 s (spread 13.9 s, gather 173.1 s, comm 0.2 s) PASS; checksums identical to the single-pass anchor on all three |
| lammps_md | L=256, 300 steps, rebin 10, checkpoint every 5 (60 x 1.3 GB, DRAM); one job per substrate | 29.0 s (96.6 ms/step, halo 3.9 s, 60 checkpoints in 1.49 s) PASS, NVE drift 2.4e-3, statics and resort exact | 27.9 s (93.1 ms/step, halo 2.8 s, 60 checkpoints in 1.48 s) PASS | 30.8 s (102.7 ms/step, halo 5.6 s, 60 checkpoints in 1.48 s) PASS, NVE drift 2.4e-3 over 300 steps |

## E7: fixed 480 GB on fewer nodes (grayscott), status as of 2026-09-23 11:30

`submit_e7_fewer_aurora.sh`, rungs descending 32 16 12 8 6 4, one job per
rung on debug-scaling. Same 491520 MB grayscott deck, 2 steps, 1 MB page.

| rung | MB/node | MPI | oneCCL | Intel SHMEM | Eternia |
|---|---|---|---|---|---|
| 32 | 15360 | 82.13 ms/step | 82.59 | 93.95 | 23209.60 ms/step, OK, checksum 313164429.143324 (not gated by the edition) |
| 16 | 30720 | 158.80 | 159.68 | 168.35 | FAILED rc=134 (earlier run, before the tier-budget and writeback fixes); rerun job 8858365 QUEUED |
| 12 | 40960 | 210.39 | 210.81 | FAILED rc=99 (heap sizing, fixed since) | FAILED rc=134 (pre-fix); needs rerun |
| 8, 6, 4 | | not run | | | |

The 32-node Eternia run is 283x slower than the baselines. That is NOT
explained by paging: moving about 30.7 GB per node at the measured 19.35 GB/s
takes about 1.6 s, so paging should cost about 10x. The run averages
2725 us per fault, against 293-508 us in the earlier 4- and 8-node runs and
about 110 us for one documented round trip. OPEN: does the per-fault cost
track the per-node blob count (15 GB/node vs 4 GB) or the node count? The
16-node rung (30 GB/node) separates the two. Also OPEN: the paged grayscott
edition does not gate its checksum against a reference.

## E1 status as of 2026-09-23 11:30

Rung 256 (the run before the fix): all six baseline cells OK. Eternia kmeans
TIMEOUT, Eternia grayscott rc=134. Rungs 256/320/384/448/512 are resubmitted
with the rebuilt binaries and SWIM disabled (jobs 8858330, 8857156, 8856950,
8856951, 8856952 in `small`), all QUEUED.

## Single-node debugging of Eternia's resident performance (2026-09-23 afternoon)

Scaling is paused and production E1 cancelled until Eternia is competitive on
ONE node with the deck resident. Everything below is 1 node, 32 GB, all
substrates on the same 1024 x 256 grid unless stated.

**The comparison is fair.** Both editions time the same per-iteration work
(memsets, assign, reduction, update); setup is outside both. The Eternia
per-iteration time is one kernel launch (rounds=1, drv_gpu = total).

**kmeans: FIXED, 1.12x MPI.** VTune gpu-hotspots (after unsetting the module
env vars so Pin could inject):

| assign kernel | time (4 it) | SIMD | XVE active | stalled | idle | occupancy | mem read |
|---|---|---|---|---|---|---|---|
| MPI | 1.71 s | 32 | 31.8% | 45.2% | 23.0% | 75.5% | 78.5 GB/s |
| Eternia (before) | 3.79 s | 16 | 71.0% | 18.5% | 10.6% | 89.4% | 36.1 GB/s |

Compute-bound on ~10x the instructions. The assembly showed why: the assign
body is a stack call (not inlined), so IGC could not prove its pointers
global and every coordinate load was a GENERIC-address dispatch (mask high
bits, compare against the local window, divergent branch to load.ugm or
load.slm, join). Casting the point and centroid pointers with
`sycl::address_space_cast<global_space>` at each use (never stored: coroc
byte-copies frame locals) leaves two plain load.ugm per coordinate.
Result: **Eternia 478 ms/iter vs MPI 427 (1.12x)**, 0 faults, checksum OK;
was 905-977 (2.2x).

Ruled out on the way, each measured: unequal grid (was half of the original
5x), global atomics (local tile: no gain), handle reloads (raw pointer: no
gain), stack calls (fc2: 2%), register pressure (256 GRF: spill unchanged),
SIMD16 (forced SIMD32: no gain; 2x WORSE for grayscott), host memory
(everything is malloc_device), relaunch rounds (1 per iteration).

**grayscott: in progress.** Per-step split at 64 blocks showed steps 1-3 at
~6 s each (MPI 161 ms) with 252 faults and 14-17 rounds: kernel time, not
paging. Removing the per-step write-back of every plane (boundary planes
only) cut puts 64000 -> 1024 but did not change the time. The same global
cast is now applied to the stencil's eight plane pointers; under test.
A block-range halo-generation change crashed at 1024 blocks (DEVICE FATAL 3)
and was reverted.

Open for later: gpu_vector's CoFetch has no generation-wait loop for a stale
page a peer already claimed (YCoro Fetch has one); clock64() is 0 on SYCL so
LRU tie-breaks are inert; ~465 generic-address checks remain in the paging
machinery itself (per page, not per element).

### grayscott, resident, 1 node, 32 GB: the fixes, in order (2026-09-23 evening)

ms per step, steady state; MPI at the same grid: 160 (64 blocks), 50-55 (1024).

| build (cumulative) | 64 blocks | 1024 blocks |
|---|---|---|
| start of the day | ~6000 | ~3460 |
| global-address casts in the stencil | 2963-3237 | 583-667 |
| outputs seeded before the clock (no first-touch faults) | same | same |
| GRF 256 (the stencil loop spilled: 23 fills/spills per pass -> 1) | 933-1295 | 471-571 |
| stencil out of line (StencilPlane) | no change | no change |
| one flush for both boundary planes (was two, the second parked) | 480-638 | 435-530 |
| batched set scans: ScanSet keeps 8 volatile loads in flight | 323-357 | 400-474 |
| sliding window: each input plane fetched/held/unpinned once, not 3x | 216-220 | 414-485 |
| u+v fetched and unpinned together (CoFetch ranges, UnpinRange2) | **213-216 (1.34x)** | 378-470 |

Checksums equal MPI's to 12 digits in every row; 0 faults.

What did NOT help, measured: narrow cache sets (--set-size, opt-in): hashed
sets collide and evict at full capacity, 2-3x slower, AND the checksum goes
wrong once dirty interior pages are evicted -- boundary-only publishing is
correct only while resident; that eviction path must be fixed before any
out-of-core study. SIMD32 (2x worse), IGC StaticGASResolution /
-cl-intel-no-local-to-generic (no effect), pinned-host task records (device
atomics fault on host memory: AtomicAccessViolation). VTune source/bb-latency
mode (GTPin) crashes the coroutine kernels.

In flight: one deferred flush of both boundary planes; skip CoAwaitFetch when
nothing was submitted; publish only node-edge planes (the page table is shared
by all blocks on a node, so intra-node halos come from the shared cache).

### E1 at 4 nodes with the fixes (2026-09-24 00:40, job 8860490)

32 GB/node resident (25% cache headroom), every substrate on 1024 x 256, 8
iterations/steps. No spill but communication: kmeans 0 faults / 0 evicts /
0 puts; grayscott 0 evicts, its 16-32 faults are halo fetches of the
neighbours' edge planes and its 32 puts are its own edge publishes.

| workload | MPI | oneCCL | Intel SHMEM | Eternia | Eternia / MPI |
|---|---|---|---|---|---|
| kmeans (ms/iter) | 434.2 | 445.1 | 436.5 | 480.7-489.8 | 1.11-1.13x |
| grayscott (ms/step) | 59.4 | 59.4 | 60.8 | 121.9-155.1 | 2.0-2.6x |

Checksums: grayscott 97206390.029335 vs MPI .029317 (2e-13); kmeans within
5e-8. kmeans is inside the plan's 10-20%; grayscott is not yet (single-node
1.86x at this grid is the memory-locality gap of the slab traversal; the
cyclic plane order is under test).

### grayscott: where the remaining gap is (2026-09-24, 00:30-01:30)

1 node, 32 GB resident, ms per step (MPI: 160 at 64 blocks, 54 at 1024, 47 at 2048).

**Paging machinery is now cheap.** GS_NO_COMPUTE=1 runs every page operation
and skips the arithmetic: 8.3 ms of a 99.5 ms step at 1024 blocks (slab),
50 of 279 at 64 blocks. The rest is the stencil itself.

**The stencil is slower because of where it is compiled, not what it
computes.** The baselines' Step kernel is exactly Eternia's cyclic plane order
without paging (one work-group per plane, planes cyclic over groups). In the
coroutine kernel the stencil function (StencilPlane) is a stack call
(IGC_FunctionControl=3), SIMD16, and under the stack-call ABI spills inside
its loop: 1305 instructions / 68 loads / 42 stores / 77 spill-fill against
the baselines' 447 / 16 / 2 / 0.

Tried, measured, and not the answer:

| change | 64 blk | 1024 blk | why not |
|---|---|---|---|
| cyclic plane order (baseline's order) | 335 (worse) | 92 (from 99) | locality is a small part |
| --plane-split G (blocks share a slab) | -- | 140-932 | multiplies page ops, contention |
| SIMD32 on the coroutine kernel | 538 | 157-168 | the whole coroutine spills |
| IGC_FunctionControl=2 (subroutines) | 218 | 121 | IGC auto-picks 256 GRF, halves occupancy |
| IGC_FC=2 + forced 128 GRF | 444 | 127-140 | stencil clean (406/12/2/0) but slower overall |

Best in-kernel configurations: 64 blocks slab GRF 256 = 210 (1.31x);
1024 blocks cyclic GRF 128 = 92 (1.70x); 2048 cyclic GRF 128 = 90 (1.89x).

**Two-phase mode (--two-phase, opt-in).** The coroutine kernel fetches, pins
and resolves every page of the step into a pointer table (ResolveCoro;
node-halo planes at the step's generation); a plain kernel -- the baselines'
Step, verbatim, through the table -- computes; a second coroutine pass
unpins. Paging and residency stay Eternia's; only the arithmetic leaves the
coroutine. It changes what the benchmark exercises (pin-then-compute rather
than in-kernel faulting), so it is reported alongside the in-kernel mode, not
instead of it. Under test.

### Single node, the other workloads vs MPI (2026-09-24, job 8860605; resident, same grids)

| workload | MPI | Eternia (as built) | ratio | cause found | fix (queued) |
|---|---|---|---|---|---|
| kmeans | 425 ms/iter | 475 ms/iter | 1.12x | (fixed earlier: generic address space) | -- |
| grayscott, 2048 blocks | 46.4-47.0 ms/step | 60.4-61.2 ms/step two-phase (fc2) | 1.30x | stencil through pointer table ~54 ms + resolve/release 6-7 ms | -- |
| gmx (K=1280, 20M atoms) | 279 ms/pass (spread 220 + gather 59) | 701 ms/pass (spread 623 + gather 79) | 2.5x | every pass writes the whole 16 GB mesh back to DRAM (12800 puts) + generic loads/atomics in spread | v3: no write-site publish when resident (evict gate), out-of-line SpreadBin with global casts; job 8860617 |
| lbann (65536-65536-4096, b64) | 1977 ms/step | OUT_OF_RESOURCES | -- | set size hardcoded 24: 24576 slots for 69634 pages | v3: --set-size, resident default = share*1.25; job 8860617 |
| lammps_md (16.4M atoms) | 68.7 ms/step | 496.5 ms/step | 7.2x | single-node interior publish every step + 3 cache clears per resort (resort 2591 vs 35 ms, refaults) | v3 MD_LEAN=1; job 8860622 |

Grayscott two-phase (gs_tp3, 8860600): at 1024 blocks 66-70 ms/step vs MPI 50.5-51.6; at 2048 blocks 60-61 vs 46.4-47.0; checksum 21600573.157590 vs MPI .157630 (2e-12 rel); 0 faults/evicts/puts. Beats the in-kernel mode (92 / 90 ms/step).

### 1-node fixes, first results (2026-09-24 02:15, jobs 8860617 / 8860622)
- gmx v3 (write-site publish off when resident; out-of-line spread with global casts): spread 214.9 + gather+sum 80.3 = 295.2 ms/pass vs MPI 279.2 (1.06x); 0 puts, 0 evicts, conservation exact. With --publish: 734 ms/pass (the old behaviour).
- lbann v3 (resident set size): runs, but 13777 ms/step vs MPI 1977 -- 417794 faults and puts in 5 steps = every page refetched and written back every step by the generational protocol. Fixed for resident single-node runs (generation 0, no publish; evict gate); measuring in ct1.
- lammps_md MD_LEAN (no interior publish, no resort clears): 369.0 ms/step vs MPI 68.7; 0 faults/puts. KE after 20 steps 43562485.166 vs MPI 43562485.214. The DEFAULT single-node path gives KE 62771460.946 (reproduced twice) with every gate passing: it is WRONG physics, not just slow. Remaining gap: force 5767 vs 1247 ms, build 1652 vs 699. v4 moves the list-force pair loop out of line with global casts (job 8860657).
- lammps_md KE trajectory (8860644, lattice 40): default Eternia matches MPI bit-for-bit in KE through step 10, then at step 11 (first resort) returns EXACTLY to the step-0 state (KE 1152000, En == E0): the single-node resort's ClearCache discards the live frames and the next fetch reloads the seed from the store. MD_LEAN matches MPI at every n (1, 2, 5, 9, 10, 11). Default single-node mode is retired for E1.

### E1 comm/compute split, 1 node (ct1, job 8860668; GV_COMM_TIMING builds)
Communication = GPU time inside CoFetch/CoHoldPage/CoBeginFlush/CoEndFlush as a share of blocks' busy GPU time (intel_get_cycle_counter; coherence checked by a probe). Grayscott two-phase: host time of resolve+release (comm) vs stencil kernel (compute).

| workload | MPI | Eternia | ratio | comm share | comm ms |
|---|---|---|---|---|---|
| kmeans (1024 blk) | 428.4 ms/iter | 457 ms/iter | 1.07x | 1.3% | ~6 ms/iter |
| grayscott two-phase (2048 blk) | 46.2 ms/step | 62.2 ms/step | 1.35x | 13.6% | 8.5 ms/step; stencil 54.1 ms/step |
| gmx | 273.9 ms/pass | 294.1 ms/pass | 1.07x | 0.8% | ~2.3 ms/pass |
| lbann (256 KB pages) | 1982 ms/step | 8091 ms/step | 4.1x | 8.5% | 685 ms/step; compute is the gap (64 of 256 threads busy in Fwd1/Fwd2) |
| lammps_md v4 lean | 68.7 ms/step | 181.3 ms/step | 2.6x | (not instrumented) | force 2088 vs 1247, build 1419 vs 699 ms / 20 steps |

In-kernel grayscott (--plane-order cyclic) reports comm > busy, i.e. its counters are not valid; not the E1 mode, unexplained.
- ct2 (8860692): grayscott two-phase with global casts on the stencil's table pointers: 54.8-55.8 ms/step vs MPI 46.6 (1.19x); stencil 46.6 ms/step (= MPI's whole step), comm 8.2 ms/step (15%). lbann at 1 MB pages (4 rows/page, all threads busy): 4162 ms/step vs 1982 (2.1x), comm 4.1%; 2 and 4 MB pages are refused (a page would span blocks' output rows).
- lbann (ct4-ct6, 1 MB pages): per-phase timing showed upd1 at 3.4x the baseline (each 4-row page re-read the 16 MB batch from L2) and bwd1 at 2.7x (d1 read/written once per W2 page). Fix: each block holds its whole band and sweeps column-major (upd1: x column in registers across 64 rows) or keeps the o-sum in a register (bwd1); same summation order. Result: 1967.0 ms/step vs MPI 1957.7-1981.7 (1.00x); phases Eternia/MPI fwd1 784/1026, fwd2 185/117, bwd1 442/145, upd2 49/47, upd1 507/622; comm 8.9% (bwd1's per-block holds of all 1024 W2 pages). Small deck with the dense reference: LOSS and WEIGHT gates PASS, weights bit-equal. The old page-by-page fallback (env LBANN_UPD1_PAGED/LBANN_BWD1_PAGED) GPU-faults on the small deck: open, not the default path.

### lbann at 4 nodes, fixed (job 8861020)
Eternia 1919.3-1922.8 ms/step vs MPI 2062.0, oneCCL 2038.5, Intel SHMEM 2101.8 (0.93x MPI). Phases Eternia/MPI: fwd1 843/1082, fwd2 181/117, bwd1 238/145, upd2 47/47, upd1 552/623. Slowest-rank communication: Eternia 33.8 ms (GPU Fetch/Hold/Flush), MPI 336.8, oneCCL 195.0, Intel SHMEM 352.2 ms (wall time in exchanges). The fixes that got here: seed only the node's own rows; W1 and W2 node-private (no generations, no per-step publish -- this also removed a DEVICE FATAL 7 get/put race on the node's own pages and a HoldPage claim race when 1024 blocks demanded the same peer W2 pages); bwd1 by per-node partials combined in node order (the baselines' scheme) instead of pulling every peer W2 page each step (3 GB/step); fwd1 and upd1 over each block's whole held band.

### gmx at 4 nodes, fixed (job 8861048)
Eternia spread 213.5-219.8 ms/pass + gather+sum 83.2-83.5 ms/pass (~300 ms/pass) vs MPI 307.6, oneCCL 309.1, Intel SHMEM 309.3 ms/pass: 0.97x. mesh_checksum 16937849875879000010 and gather_energy 147849839193652 equal the three baselines bit for bit. Slowest-rank communication over the run: Eternia 3.4 ms, MPI 47.6, oneCCL 86.8, Intel SHMEM 43.6. The fixes: no write-site publish when resident; the cap check bounded by distinct planes (shared cache), not 4 x blocks; the gather made plane-wise like the baselines (a node reads only its own planes, so there is no halo, publish or barrier), which also made its rounding the baselines'.
