# Scaling Campaign: Reproduction & Rationale

The benchmark compares snapshot-checkpoint I/O paths for a GPU Gray-Scott simulation. All
arms run the *identical* computation (verified by a cross-arm FNV checksum on the persisted
bytes); only the snapshot sink differs. The headline claim under test: a **GPU-initiated,
non-blocking** I/O path (`async`) beats the traditional host-mediated and synchronous paths,
with the margin depending on storage tier and compute intensity.

---

## 1. Environment & prerequisites

| item | value |
|---|---|
| Container | `iowarp-kvhdf5-gpu` (image `iowarp/clio-core-devcontainer`) |
| Repo mount | host repo ↔ container `/workspace` (bind mount) |
| Binary | `/workspace/build/bin/kvhdf5_e2e_tests` (built from repo HEAD) |
| Build target | `cmake --build /workspace/build --target kvhdf5_e2e_tests -j` |
| GPU | RTX 4090, PCIe 4.0 x16, CUDA 12.x |
| `/dev/shm` | must be ≥16 GiB (`--shm-size=16g`); RAM-tier bdev + tmpfs outputs live here |
| async VOL | `/opt/vol-async-nomemcpy/lib` (see §5, arm `hdf5_async`) + `/opt/hdf5ts`, `/opt/argobots` |

**Always rebuild from HEAD before a campaign.** A stale `/workspace/build` silently shifts
every number (a prior metadata-blob change moved the async baseline ~17%). Verify the rebuild
is a no-op: `cmake --build ... --target kvhdf5_e2e_tests` should report "ninja: no work to do".

**Quiescence.** The submit-overhead measurements (separate study 08) are timing-sensitive; the
throughput campaign here is more forgiving but still: run on an otherwise-idle GPU (compositor +
terminal only). GPU clock/thermal state moves absolute RAM numbers ~5–20% run-to-run — take
medians and, if possible, lock clocks (`nvidia-smi -pm 1 -lgc <max>`).

---

## 2. The arms — what each is and why it's included

The arms form a deliberate **innovation ladder** and a **baseline spectrum**, so a reviewer can
see async beat every class of prior art, not one strawman.

| arm | what it does | role |
|---|---|---|
| `raw_inline` | GPU idle, synchronous `pwrite`+fsync per snapshot | naive synchronous baseline (the "GPU stalls on I/O" case) |
| `raw_threaded` | background CPU writer thread overlaps compute | non-HDF5 best practice |
| `hdf5_naive` | **contiguous, high-level `H5Dwrite`, default cache, no early-alloc, synchronous** | **typical unoptimized HDF5 user** (most codes) |
| `hdf5_inline` | tuned HDF5 (chunked, direct chunk-write, early-alloc), synchronous | expert-tuned HDF5, synchronous |
| `hdf5_threaded` | tuned HDF5 with a background writer | expert-tuned HDF5, overlapped — the **tough bar** |
| `hdf5_async` | HDF5 **Asynchronous I/O VOL** connector | state-of-the-art async I/O competitor |
| `hostclio` | host copies D2H then `PutBlob` to the CLIO bdev, blocking | CLIO without GPU-initiation |
| `sync` | GPU-initiated `PutBlob`, producer **waits** for completion | GPU submit, blocking |
| `async` | GPU-initiated `PutBlob`, **fire-and-continue** | **ours** — the innovation |
| `async_pinned` | `async` with the blob payload in pinned host memory (`DATA_PINNED=1`) | placement variant of async (see §7) |
| `pooled` | bounded-buffer double-buffering (M resident buffers) | memory-vs-throughput study, not a speed rival |

**How to read them for the paper:** lead with `raw_inline → hostclio → sync → async` (each step
adds one capability: host offload → GPU-initiated submit → non-blocking) and the HDF5 spectrum
`naive → inline/threaded → async-VOL`. `async` is the star. `async_pinned` and `pooled` are
reported in **separate panels** (a placement ablation and a memory study), not as peer bars.

---

## 3. The two storage tiers — and why both

- **`ram`** (`GSBENCH_BDEV=ram`, raw/HDF5 write to `/dev/shm`): storage is effectively free, so
  the numbers isolate the **software I/O path**. This is where GPU-initiated submission's low
  overhead is visible and where async wins biggest (~2.8× over synchronous baselines). It is the
  headline tier and matches the framework's "up to 50 GB/s RAM-backed" target.
- **`file`** (`GSBENCH_BDEV=file`, O_DIRECT bdev; raw fdatasyncs): durable disk. Bandwidth
  (~1.1 GB/s local NVMe) caps every arm, so margins compress — the honest, representative
  checkpoint-to-storage case.

We run **both tiers in every study**. RAM is weighted more heavily (full compute range; file
drops its priciest point) because RAM is the headline **and** RAM runs are ~2–3× cheaper.

---

## 4. The studies and why these ranges

All studies fix: `num_threads=1` (required for concurrent-put safety on multi-chunk snapshots),
`CUDA_MODULE_LOADING=EAGER` (avoids a one-time device-wide module-load stall in the timed
region), `GSBENCH_PREWARM=1`, `GSBENCH_INCOMPRESSIBLE=1` (worst case for any compression path,
so bytes moved are honest), `GSBENCH_SUBMIT_BLOCKS=chunks` (one fill block per chunk — the
default of 1 serializes the fill and *handicaps* the GPU arms; see §6), 3 repeats (5 on headline
points), median reported.

### C — compute / snapshot-period  (the async-advantage curve)
`STEPS_PER ∈ {4,8,12,24,48,96,192,384,768}`, `N=6400, chunks=4, snaps=12`, both tiers (file
drops 768), **10 arms** (adds `async_pinned`).
- **Why sweep steps:** async's benefit is hiding I/O behind compute, so the advantage is a
  function of compute-per-snapshot. This axis *is* the story. Measured: async ÷ synchronous
  baselines runs from ~2.8× (I/O-bound, steps≤8) down toward parity as compute dominates.
- **Why steps 4–8 at the low end:** the I/O-bound regime (steps=8, RAM) is the reference
  "headline" operating point; steps=4 brackets it.
- **Why up to 768:** to reach the fully compute-bound regime where async's overlap is exhausted
  and the device/pinned crossover appears (see §7). File drops 768 only for cost (RAM keeps it).
- **Why `async_pinned` is here:** the device/pinned crossover (§7) lives on this axis.

### K — chunk-count (fixed N)   *(NEW — added after preliminary runs)*
`chunks ∈ {1,4,16,64,128}`, `N=6400, snaps=12, steps ∈ {48,96}`, both tiers, 9 arms.
- **Why:** the preliminary all-arm run at `chunks=4` showed `hdf5_naive` was *not* slow — a
  contiguous whole-array write is efficient for few large chunks. The naive penalty only appears
  at **many small chunks** (default cache thrash, no early-alloc fault-in, partial-chunk I/O).
  This sweep is where "typical-user HDF5 degrades" actually shows, and it doubles as the
  advisor's "number of GPU blocks" parallelism axis (distinct from weak scaling).
- **Why cap at 128:** `chunks×snaps` must stay under the verified-safe async in-flight ceiling
  of ~2048 (128×12 = 1536). 256×12 = 3072 risks the old ring-buffer hang.

### B — I/O size
`N ∈ {1600,3200,4800,6400,9024}` (≈10 → 310 MB/snapshot), `chunks=16, snaps=12, steps ∈ {48,96}`,
both tiers, 9 arms.
- **Why these N:** they give a 32× span of bytes/snapshot; 6400 is the reference (156 MB/snap).
- **Why two compute points (48, 96):** Study C covers the full compute axis; the size study is
  taken at two representative operating points (48 more I/O-bound, 96 the file sweet spot, §7) so
  the size behavior is shown to hold across compute regimes, not at a single slice.

### W — weak scaling
`chunks ∈ {1,2,4,8,16,32}` with `N = 2048·√chunks = {2048,2896,4096,5792,8192,11584}`,
`snaps=8, steps ∈ {48,96}`, both tiers, 9 arms.
- **Why this formula:** weak scaling means work-per-unit constant while total work grows. Here
  the "unit" is a chunk; `N=2048·√chunks` holds **bytes-per-chunk ≈ 16.8 MB constant** (since
  bytes/chunk = N²·4/chunks) while total data grows ∝ chunks. The table values keep `N` an exact
  multiple of `chunks` (N/chunks = 2048/√chunks ∈ {2048,1448,1024,724,512,362}).
- **Why cap at 32 chunks:** `N=11584` already means a 4.3 GB snapshot set at snaps=8; going
  higher is memory- and time-expensive with little added signal.

### P — pooled M-sweep (memory vs throughput)
`POOL(M) ∈ {0,64,32,16,8}`, `N=6400, chunks=128, G=8 (GSBENCH_SUBMIT_BLOCKS), snaps=12,
steps=96`, both tiers; runs the `pooled` arm at each M plus one `async` fire-all reference.
- **What it measures:** the bounded double-buffering path streams `chunks` through only `M`
  resident buffers at depth `D=M/G`, capping in-flight memory. `POOL=0` ⇒ `M=chunks` (unbounded
  control). The sweep charts memory savings (M/chunks = 1, ½, ¼, ⅛, 1/16) vs throughput cost.
- **Why G=8, chunks=128:** G must divide M and be large enough to saturate HBM; G=8 divides all
  M values and 128 chunks gives a meaningful pool range. `pooled` is a *memory* result, not a
  speed rival to `async`.
- **Pacing (leave it on):** the `pooled` arm bounds host run-ahead automatically via
  `GSBENCH_PACE` (auto = `max(2, 512/(steps_per+2))` snapshots), keeping producer submission and
  CUDA launch queueing balanced. The default needs no knob. **Do not set `GSBENCH_PACE=0`** at
  these settings — it disables pacing and the bounded-pool (M<N) runs deadlock.

### Repeats
3 everywhere; **5 on headline configs** — `C steps8_ram` (the headline regime), `C steps8_file`
(durable contrast), `C steps48_file`, `B N6400_steps96_file`, `W chunks16_steps96_file` — so the
figures that anchor the paper carry tight error bars.

---

## 5. Key findings that drove these choices

These are the measured results (from the exploratory probes in this directory —
`tenarm_summary.txt`, `filetier_summary.txt`, `pinned_summary.txt`) that justify the design.

1. **Storage tier is the dominant axis.** RAM/steps=8: `async` ≈ 15–16 GB/s, **2.8× over the
   synchronous baselines**, 2.0× over the async VOL. File/steps=8: everything is disk-bound at
   ~1.1 GB/s and async's margin shrinks to ~1.0–1.4×. ⇒ RAM is the headline; both tiers reported.

2. **Compute intensity is the second axis** and monotone: async's lead is largest I/O-bound and
   decays toward parity as compute grows. ⇒ Study C sweeps it end-to-end.

3. **The `async` device/pinned crossover.** `DATA_PINNED` chooses where the blob payload lives:
   `kDeviceMem` (default) keeps it on-GPU (fast HBM writes) but the bdev server's D2H readback
   does **not** overlap compute; `kPinnedHost` writes mapped host over PCIe (slower submit) but
   **removes that server D2H**, so disk writes pipeline under compute. Result on file:
   device wins for steps ≤ 96; **pinned wins for steps ≥ 192** and rescues async from a
   compute-bound collapse (steps=192: 734 → 921 MB/s, tying `hdf5_threaded`; steps=384 it beats
   it). On RAM device always wins (no disk to pipeline under). ⇒ report `async` at its
   regime-optimal placement (device default; pinned when compute-bound on disk), disclosed.

4. **Your reference `raw=5900` is the *inline* (GPU-idle) raw.** Probed at RAM/steps=8:
   `raw_inline`=5847 (matches), `raw_threaded`=7214. So the impressive 3.36× used the synchronous
   strawman. Against the *fair* threaded baseline async is ~2.1×; against the *synchronous*
   baselines (the honest "vs non-async" framing) it's ~2.6–2.8×. ⇒ both `raw_inline` and
   `raw_threaded` are separate arms so the framing is explicit, never silent.

5. **`hdf5_naive` is contiguous, so it's fast at few large chunks and only degrades at many
   small chunks.** ⇒ the K study exists specifically to expose that regime; don't judge naive at
   chunks=4 (where it even edges tuned inline).

6. **`hdf5_async` needs the nomemcpy VOL.** The stock `/opt/vol-async` (`ENABLE_WRITE_MEMCPY=ON`)
   livelocks in Argobots at ≥12 in-flight snapshot datasets (100% CPU spin, never returns).
   `/opt/vol-async-nomemcpy` fixes it with zero source change. ⇒ the arm points at the nomemcpy
   build + `GSBENCH_HDF5_ASYNC_FWAIT=0`, and must not run oversubscribed alongside other jobs.

7. **`DATA_PINNED=1` is a 3× regression I/O-bound** (RAM steps=8: 5.2 vs 15.6 GB/s) but a **win
   compute-bound** — a regime-dependent tradeoff, not a flat loss (see finding 3).

8. **O_DIRECT is a wash on this NVMe** (buffered+fsync ≈ O_DIRECT at every point), so the raw
   baseline uses the default `RAW_ODIRECT=0`; durability parity is maintained by fsync either way.

9. **File sweet spot is steps ≈ 96** (compute ≈ I/O ≈ 140 ms/snap): async beats the synchronous
   baselines 1.4–1.8× *and* ties/beats the threaded arms. ⇒ the scaling studies (K/B/W) are taken
   at steps=96 (the sweet spot) and steps=48 (a more I/O-bound point) so results span two regimes.

---

## 6. Fairness / representativeness statement

The conditions are chosen to be **representative and honest**, not to cherry-pick:
- Both a *typical-user* HDF5 baseline (`hdf5_naive`) and *expert-tuned* ones are included, so
  async is shown beating the whole spectrum, not one weak setup.
- The async innovation is compared against genuinely *non-async* paths (inline raw/HDF5,
  host-driven `hostclio`, blocking `sync`) — the correct comparison for an async contribution —
  **and** against the async-VOL and threaded best-practice arms as the tough bars.
- Durability is enforced on every durable arm (fsync / O_DIRECT bdev); the RAM tier is clearly
  labeled as the storage-free upper bound.
- Every arm computes byte-identical data (cross-arm checksum), so no arm "wins" by doing less.
- `async`'s buffer placement is selected per regime with the rule stated explicitly, not swapped
  silently.
- **Known limits (state in the paper):** absolute MB/s are this box's (RTX 4090 / local NVMe);
  the *structural* results transfer, absolutes must be re-measured on the target machine. A
  higher-latency / parallel filesystem (real checkpoint target) would widen async's file-tier
  lead further but was not available to measure.

---

## 7. How to run

```bash
# 0. (host) ensure the container is up and the binary is current
docker start iowarp-kvhdf5-gpu
docker exec iowarp-kvhdf5-gpu bash -lc 'cd /workspace/build && cmake --build . --target kvhdf5_e2e_tests -j'

# 1. launch the campaign detached (per-arm timeout 600s), streaming to progress.log
#    The SCRIPT is version-controlled (here, next to the bench it drives); the RESULTS are not
#    — they default to /workspace/bench_results/scaling-campaign, which is gitignored. So
#    invoke by absolute path and send the log to the results dir, NOT to the script's dir:
#    run_campaign.sh resolves nothing relative to itself, so it runs from anywhere.
docker exec iowarp-kvhdf5-gpu bash -lc '
  R=/workspace/bench_results/scaling-campaign; mkdir -p "$R"
  CAMPAIGN_TIMEOUT=600 nohup bash \
    /workspace/context-transfer-engine/adapter/kvhdf5/test/e2e/scaling_campaign/run_campaign.sh \
    > "$R/progress.log" 2>&1 &
  echo "pid $!"'

# 2. watch progress (config transitions / failures / done)
docker exec iowarp-kvhdf5-gpu bash -lc \
  "tail -f /workspace/bench_results/scaling-campaign/progress.log \
   | grep --line-buffered -E 'FAIL|RETRY|CAMPAIGN DONE|^===='"
```

**Expected wall-clock:** ~5–5.5 h on this box (K/B/W run at two compute points, 48 & 96). RAM is
cheap; the file high-end — Study C steps≥384, Study B N=9024, Study W chunks=32 — is the heavy
tail. Well under the 7 h ceiling; trim the tail live if it overruns (see below).

**Resume after a stop/crash:** just relaunch step 1. `run_arm` skips any (study, config, arm, rep)
whose per-arm log already contains a result, so completed work is not repeated. For a clean slate
instead, delete `logs/`, `raw/master_raw.log`, and `progress.log` first.

**Trim live if it overruns 7 h:** stop the driver (`kill -9 <pid>; pkill -9 -x kvhdf5_e2e_test`),
edit the ranges (e.g. drop Study W's top chunk point or Study B's N=9024), relaunch — the resume
logic keeps everything already done.

**Outputs:**
- `raw/master_raw.log` — every result line, tagged `SWEEP=<study> CFG=<cfg> REP=<n> ARM=<arm>`.
  This is the recoverable source of truth; build CSV/plots from it.
- `logs/<study>__<cfg>__<arm>__rep<n>.log` — full stdout per arm run.
- `progress.log` — human-readable narration.

---

## 8. Robustness details & footguns

- **Killing the binary:** use `pkill -9 -x kvhdf5_e2e_test` — the comm name truncates to 15 chars
  (no trailing `s`). **Never `pkill -f <binary-path>`**: the path appears in the driver's own
  command line, so `-f` SIGKILLs the driver itself (observed as exit 137).
- **Zombie reaping:** this container's pid 1 does not reap children, so timed-out runs leave `Z`
  (defunct) entries. Liveness checks must ignore them:
  `ps -o pid,stat -C kvhdf5_e2e_tests | awk '$2!~/Z/'`. Bare `pgrep` counts zombies and would
  stall every reset on its full timeout. (The campaign's `hard_reset` already does this.)
- **Bind-mount truncation:** host `build/` *is* container `/workspace/build/`. Never
  `docker exec ... > build/x.csv` from the host — the host shell truncates the file before the
  container writes it. The campaign writes from *inside* the container, so it is safe; outputs
  live under `bench_results/` (persistent, on the bind mount) while bulky scratch (bdev/raw/HDF5 data)
  goes to `/workspace/build/gsbench_campaign_scratch` (gitignored, wiped per-arm).
- **bdev capacity:** the file/ram bdev cap is sized to `1.5×` total bytes + 1 GB. An undersized
  cap causes *silent* `PutBlob` loss (a known hazard) — do not shrink it.
- **Stale build:** see §1. Always rebuild from HEAD; a stale lib silently shifts every number.
- **`submit_blocks`:** the bench default is 1 (single fill block), which serializes the fill and
  understates the GPU arms. The campaign sets `GSBENCH_SUBMIT_BLOCKS=chunks` everywhere.
- **Pooled descriptor ≠ result:** the `pooled` arm prints a `GSBENCH_POOLED N=… M=… G=… D=…`
  descriptor line *before* it does the work. The resume/ok detector keys on `GSBENCH_RESULT`
  only — do not add `GSBENCH_POOLED` to the result pattern, or a run that dies after the
  descriptor but before finishing is miscounted as a completed result.
