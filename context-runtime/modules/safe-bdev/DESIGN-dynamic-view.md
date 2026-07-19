# Safe BDev — Dynamic View-Group Design

**Status:** implemented (per-chunk round-robin + banded addressing + per-slot
variable-width parity; all safe-bdev ctests pass, incl. the full-capacity test)
**Supersedes:** the append-only, physically-row-aligned "stripe groups" layout.
**Goal:** full capacity utilization when drives are added to a partially-filled
array, keeping erasure-coded reliability. Priority is **space + reliability over
speed**.

---

## 1. Why change

The current layout stripes data in fixed-width rows across a *group* of data
drives. Adding a data drive **freezes the current group at its high-water mark**
and opens a new, wider group over only the **remaining physical rows** — it
cannot backfill the new drive's low rows. A drive filled heavily while narrow
therefore **strands the added drive's capacity**.

Measured by `safe_bdev_capacity_add_data_then_parity`: fill ~84% of a 1-drive
array, add a 2nd drive, write the same again → only **5 of 13 blocks fit** in
phase 2 (the rest of the added drive is unreachable). All *stored* data verifies,
but you don't get the new drive's space.

## 2. Model

Two logical groups:

- **View group** — the *data* bdevs. Blocks are allocated **round-robin over the
  non-full members**, each filling to its **own** capacity. Total data capacity
  = **Σ (per-drive capacity)**. Not physically row-locked → no stranding.
- **Parity group** — the *parity* bdevs (0..M, `M = max_failures`).

Primitives: **1 MB chunk** (the EC unit a bdev is split into — configurable),
64 KiB **superblock** at member offset 0 (array identity + role + index),
Reed–Solomon codec, async parity builder, member-manifest WAL.

Two distinct notions of "block":
- **AllocateBlocks region** — an arbitrary logical extent `(off, len)` handed to
  the CTE.
- **EC chunk** — the fixed 1 MB unit used for parity. The chunks a region touches
  are a plain integer op: `chunk = off / 1MB` (a region may span several).

### 2.1 Choosing `k` / `m` (data vs parity split)

For `N` total drives, to survive up to `f` simultaneous drive failures with
dedicated parity, dedicate `m = f` drives to parity and use `k = N − f` for data
(`RS(k, m)`). That is the maximum-capacity split that still meets the target:
usable capacity = `k` drives, storage overhead = `m / N`. `f = N − 1` collapses
to a `k = 1` mirror; `f ≥ N` is impossible (at least one data drive must survive,
so `f ≤ N − 1`).

`k/m` to maximize safety at the target — columns = total drives, rows = max
drive failures to survive:

| max failures ↓ / total drives → | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| **1** | 1/1 | 3/1 | 5/1 | 7/1 |
| **2** | —   | 2/2 | 4/2 | 6/2 |
| **3** | —   | 1/3 | 3/3 | 5/3 |
| **4** | —   | —   | 2/4 | 4/4 |

`—` = infeasible (`f > N − 1`: can't survive that many failures with `N` drives).
Each cell satisfies `k + m = N`, `m = f`. E.g. 8 drives surviving 2 failures =
`RS(6, 2)` (6 data + 2 parity, 25% overhead); 4 drives surviving 3 failures =
`RS(1, 3)` (a 4-way mirror). In this dynamic array these are the *steady-state*
targets — a stripe reaches width `k` only once that many data drives have written
into it (narrower stripes are protected as mirrors meanwhile, §4.1).

## 3. Addressing (banding: the offset encodes member + slot)

Each data member owns 1 MB **slots** `0, 1, 2, …`. A chunk's logical offset
**encodes its physical location directly** — pure integer math, no lookup table:

```
chunk_index = off / 1MB
member  d   = chunk_index / SLOTS_PER_MEMBER      # SLOTS_PER_MEMBER = fixed cap, e.g. 2^32
slot    s   = chunk_index % SLOTS_PER_MEMBER
physical byte offset on member d = superblock + s · 1MB
```

- Per-chunk **round-robin** picks the member `d` for each new chunk; that
  member's own allocator picks the slot `s`; the returned offset is
  `(d · SLOTS_PER_MEMBER + s) · 1MB`.
- Decode is a plain `div`/`mod` (this is the "`off / 1MB` = chunk" op). There is
  **nothing extra to persist for addressing** — only each member's slot
  allocator (WAL, §7). No chunk map.
- `d` is the member's **stable positional index** (never rotates, even across
  remove/recover — a recovered member keeps its index), so held offsets stay
  valid across membership changes.
- Offsets are **sparse** (each member occupies a band of the address space), but
  the CTE stores bdev offsets opaquely, so nothing external depends on them being
  dense/contiguous.
- `SLOTS_PER_MEMBER` caps a member at `SLOTS_PER_MEMBER · 1MB`; with `2^32` slots
  that's 4 PiB/member, and `num_members · SLOTS_PER_MEMBER · 1MB` stays within
  u64 for any realistic member count.

## 4. Stripes & parity (offset-aligned, variable width)

A **stripe `s`** = the live data chunks at physical slot `s`, across the data
members deep enough to have one. Parity for stripe `s` lives at the **same** slot
`s` on each parity member:

```
parity[s] = RS(k_s, m) over { data_d[s] : member d has a LIVE chunk at slot s }
```

- `k_s` (stripe width) varies by slot — deep slots on smaller/late drives are
  narrower. The RS codec already supports per-width codes; cache codec objects
  keyed by `k_s` (few distinct widths).
- **Stripe membership is derived**, not stored: "member `d` is in stripe `s`" ⇔
  "`d` has a live chunk at slot `s`", which the per-member allocator knows.
- **Full capacity**: each member fills independently 0..cap; nothing is locked to
  the narrowest/oldest drive.

### 4.1 Narrow (sub-`m+1`-width) stripes — "mirror early, widen later"

`RS(k_s, m)` protects against `m` failures for **any** `k_s ≥ 1`. The degenerate
case `k_s = 1` (a single data chunk in the stripe) with `m = 1` is just a
**mirror** — the parity chunk is a copy of the data chunk. So:

- Data written while the array is narrow (e.g. a lone data drive, `m = 1`) is
  protected **immediately** as a mirror (`RS(1, 1)`) — 100% overhead, but safe.
- As more data drives are added and their round-robin writes fill slot `s`, that
  stripe widens to `RS(2, 1)`, `RS(3, 1)`, … and the overhead falls toward
  `m/(k+m)`. The parity is recomputed lazily when the widening write dirties the
  stripe (§6).

So you never trade away protection to get capacity: early writes are always
covered, just less space-efficiently until the array grows. (If you'd rather
*forbid* writes below some minimum width, that's a policy knob we can add.)

## 5. Data paths

**Allocate** — **per-chunk round-robin**: each 1 MB logical chunk in the request
is assigned to the next non-full data member (cursor advances per chunk), taking
that member's next free slot; the returned offset bands member+slot (§3). A
multi-chunk request therefore spreads across members and comes back as a block
list (adjacent same-member chunks coalesced).

**Write** — for each region, walk its chunks (`chunk = off / 1MB`), look up
`chunk_map[chunk] = (d, s)`, write to member `d` slot `s`, and mark slot `s`
**dirty** (parity stale).

**Read (healthy)** — per chunk, look up `(d, s)`, read from member `d` slot `s`.

**Read (degraded)** — if member `d` is down, reconstruct chunk `(d, s)` from the
other live data chunks at slot `s` + parity[s], via `RS(k_s, m)` decode.
Requires ≤ `m` members down at that stripe.

**Async parity** — a write dirties slot `s`; `BuildParity` drains dirty slots:
gather the present data chunks at `s`, `RS(k_s, m)` encode, write parity at `s`,
clear. A stripe is reconstructable only once its parity is current (not dirty).

## 6. Membership changes

**Add data drive (no data movement).** Append a data member (fresh index),
empty. The round-robin cursor now includes it, so **new** writes land on it; the
offsets it newly occupies become wider stripes and are **dirtied on write** →
`BuildParity` re-derives their parity. Existing data on other members never
moves; a stripe is only recomputed when the new drive actually adds a chunk to
it. → "parity may need recalculation" falls out of the normal dirty-stripe path;
zero bytes are copied. New drive's full capacity is available to new writes.

**Add parity drive.** Append a parity member (≤ M), bump `m`, re-dirty all
written stripes so `BuildParity` recomputes at the higher parity level.

**Remove drive.** Mark the member faulty/removed (positional; slot kept).
Degraded reads reconstruct from stripe-mates.

**Recover drive.** For each offset `s` where the failed member had a live chunk,
reconstruct from stripe `s` and write to the replacement. Idempotent → resumable
after a crash (as today). Keyed by offset instead of group row.

## 7. Crash restart

- **Per-member allocator WAL** (replaces the per-group alloc log): journal each
  `(member d, offset p)` alloc/free; replay to rebuild every member's live-chunk
  set + high-water; periodic compaction (append + compact, same shape as today).
- **Superblocks**: array identity + role + index (unchanged).
- **Member-manifest WAL**: membership + recovery state (unchanged; the WAL +
  roll-forward work already landed).
- Addressing needs nothing persisted — the offset bands member+slot (§3).
  Each member's slot allocator (high-water + free list + live set) is rebuilt by
  replaying the allocator WAL; stripe membership and parity are then re-derived
  from the per-member live sets.

## 8. Migration from the current layout

The on-disk format changes (per-member chunk allocation + banded addressing vs
group/row records). Safe-bdev is still WIP (PR #663), so the plan is to **bump
the superblock/WAL format version and NOT migrate in place** — pre-existing
arrays are recreated, and a version mismatch is refused rather than
mis-interpreted. (A one-shot converter can be added later if any array is worth
preserving.)

## 9. Metadata overheads

| Source | Overhead | Scaling |
|---|---|---|
| Superblock | 64 KiB / member | fixed per member |
| Parity storage | `m` parity drives | `m/(k+m)` of raw; the EC cost |
| Per-member allocator WAL | 1 record / live 1 MB chunk (member + slot) | grows with live chunks; 1 MB granularity keeps it 16× smaller than 64 KiB; WAL-derived + compacted |
| Addressing map | **none** | banded encoding is formulaic (§3) |
| Member-manifest WAL | tens of bytes / member | bounded by member count |
| RS codec cache | one codec per distinct stripe width | ≤ num data members |

## 10. Decisions (resolved)

- **Allocation granularity** — **per-chunk round-robin** over 1 MB EC chunks; a
  region touches `off / 1MB` chunks. *(settled)*
- **Addressing** — **banding**: the offset encodes (member, slot), decoded by
  `div`/`mod` (§3). No lookup map; only per-member slot allocators are persisted
  (via the WAL). Round-robin decides the member at alloc time; the offset carries
  it thereafter, so held offsets survive add/remove/recover. *(settled)*
- **Sub-`m+1`-width stripes** — allowed; narrow writes are protected as a mirror
  (`RS(1, 1)`) and widen to `RS(k, m)` as drives join (see §4.1). *(accepted)*
- **Parity placement** — **dedicated** parity members. *(confirmed)*

Remaining knobs (not blocking): the 1 MB chunk size (configurable), and whether
to add a policy that forbids writes below a minimum stripe width (§4.1).

## 11. Test plan

- **Capacity**: repeat the 1-drive→add-drive→write scenario; assert the 2nd
  write now **fully fits** (Σ capacity), and all data verifies.
- **Degraded read**: fault one data member; all data reconstructs (per-stripe,
  variable width).
- **Add-then-protect**: write with 1 data drive (mirror), add data + parity,
  verify old + new data intact and a subsequent fault still reconstructs.
- **Restart**: write, reboot, verify; membership + allocator state restored.
- **Interrupted recovery**: keep the existing consistency test (recover onto a
  spare, interrupt, restart, resume) — now keyed by offset.
- **Round-robin balance**: assert data spreads across members and full members
  are skipped.
