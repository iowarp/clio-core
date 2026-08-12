#include "arms.h"

#include <sstream>

namespace gsbench {

namespace {

std::vector<Arm> BuildRegistry() {
    // Selectors verified against gray_scott_threeway_bench.cu TEST_CASE tags (~line 2306+).
    // Env overrides verified against run_campaign.sh's tag_of()/case block (the superset --
    // it covers all 11 arms; run_threeway_bench.sh's run_arm calls agree on the 7 they share).
    return {
        // raw arm: no CLIO server. GSBENCH_RAW_INLINE selects the writer structure.
        {"raw_inline",    "[gsbench_raw]",        {{"GSBENCH_RAW_INLINE", "1"}}, false, false},
        {"raw_threaded",  "[gsbench_raw]",        {{"GSBENCH_RAW_INLINE", "0"}}, false, false},

        // hdf5: same TEST_CASE, GSBENCH_RAW_INLINE picks inline (GPU-idle) vs threaded
        // (background writer, overlaps compute) -- PrintResult names the row accordingly.
        {"hdf5_inline",   "[gsbench_hdf5]",       {{"GSBENCH_RAW_INLINE", "1"}}, false, false},
        {"hdf5_threaded", "[gsbench_hdf5]",       {{"GSBENCH_RAW_INLINE", "0"}}, false, false},

        // hdf5_naive: typical-user baseline -- always inline structure, naive dataset config.
        {"hdf5_naive",    "[gsbench_hdf5_naive]", {{"GSBENCH_RAW_INLINE", "1"},
                                                     {"GSBENCH_HDF5_NAIVE", "1"}}, false, false},

        // hdf5_async: needs the async-VOL LD_LIBRARY_PATH/HDF5_PLUGIN_PATH/HDF5_VOL_CONNECTOR
        // env, applied specially by the runner (see runner.cpp ApplyArmEnv), plus
        // GSBENCH_HDF5_ASYNC_FWAIT=0 to avoid the H5VL_async_file_wait busy-spin livelock.
        // POOL=0 (unbounded: one buffer per snapshot) is also the .cu default, but it is pinned
        // here explicitly so this baseline reads 0 even if something upstream sets the var.
        {"async_VOL",     "[gsbench_hdf5_async]", {{"GSBENCH_HDF5_ASYNC_FWAIT", "0"},
                                                     {"GSBENCH_HDF5_ASYNC_POOL", "0"}}, true, false},

        // async_VOL_reuse: same TEST_CASE and VOL toolchain as async_VOL, bounded to M reused
        // pinned buffers instead of one per snapshot. M defaults to 2 to mirror GPUH5's own 2
        // reused buffer groups, making it the direct bounded-memory comparison; override with
        // --hdf5-async-pool (which reaches this arm ONLY -- see reads_hdf5_async_pool).
        // FWAIT=1 here, unlike async_VOL: bounding the pool turns one end-of-run drain into a
        // drain per reuse, and FWAIT=0's ~2s hard-coded mutex backoff per wait -- harmless once,
        // at the end -- would otherwise be paid on every single buffer reuse.
        {"async_VOL_reuse", "[gsbench_hdf5_async]", {{"GSBENCH_HDF5_ASYNC_FWAIT", "1"},
                                                       {"GSBENCH_HDF5_ASYNC_POOL", "2"}}, true, false, true},

        {"hostclio",      "[gsbench_hostclio]",   {}, false, false},

        // GPUH5 arms. NAMING: `gpuh5` / `gpuh5_sync` are the PERSISTENT (resident cooperative)
        // arms -- the canonical design (see the bottom of this list). The RELAUNCHED variants
        // carry the _relaunch suffix. Both are GPU-initiated and bounded (2 reused groups for
        // the overlapped arms, 1 for the synchronous ones); gpuh5_noreuse is the old
        // one-dataset-per-snapshot async path, whose memory is linear in snaps.
        //
        // gpuh5_sync_relaunch: relaunched fused submit-AND-wait (1 buffer, no double
        // buffering). Two data backends as separate arms so the sync-vs-sync comparison
        // against gpuh5_sync (which is FORCED pinned) isolates the data-backend cost from the
        // cooperative-kernel cost.
        {"gpuh5_sync_relaunch",  "[gsbench_gpuh5_sync]",    {{"GSBENCH_DATA_PINNED", "0"}}, false, false},
        {"gpuh5_sync_relaunch_pinned", "[gsbench_gpuh5_sync]",    {{"GSBENCH_DATA_PINNED", "1"}}, false, false},
        {"gpuh5_noreuse",        "[gsbench_gpuh5_noreuse]", {{"GSBENCH_DATA_PINNED", "0"}}, false, false},
        {"gpuh5_noreuse_pinned", "[gsbench_gpuh5_noreuse]", {{"GSBENCH_DATA_PINNED", "1"}}, false, false},

        // pooled: GSBENCH_POOL=<M> is config-supplied (the campaign's Study P sweeps M), so no
        // fixed override here -- the runner sets it per work-unit when is_pooled is true.
        {"pooled",        "[gsbench_pooled]",     {}, false, true},

        // gpuh5_relaunch: the RELAUNCHED design -- async's shape but memory constant in
        // snapshots (2 reused buffer groups, snap % 2, drain-before-refill + device tag-stamp;
        // DESIGN §7 "Option B"). Checksum must equal the other GPUH5 arms. _pinned = the same
        // arm with kPinnedHost data (the backend gpuh5/gpuh5_sync are forced onto), so the
        // persistent-vs-relaunch comparison can be made on an equal data backend.
        {"gpuh5_relaunch", "[gsbench_gpuh5]",      {{"GSBENCH_DATA_PINNED", "0"}}, false, false},
        {"gpuh5_relaunch_pinned", "[gsbench_gpuh5]",      {{"GSBENCH_DATA_PINNED", "1"}}, false, false},

        // gpuh5 / gpuh5_sync: THE canonical design -- the WHOLE snapshot loop in ONE resident
        // cooperative kernel (grid.sync() between steps), DESIGN §7 "Option A", measured
        // head-to-head against the _relaunch variants. Bounded memory: gpuh5 keeps 2 groups
        // (overlap 1 snapshot ahead), gpuh5_sync keeps 1 (it submits-AND-waits, so only one
        // snapshot is ever in flight). Both FORCE kPinnedHost data for deadlock safety -- a
        // resident cooperative grid would starve the server's D2H -- which is why they report
        // pinned=1 regardless of GSBENCH_DATA_PINNED.
        {"gpuh5",            "[gsbench_persistent]",      {}, false, false},
        {"gpuh5_sync",       "[gsbench_persistent_sync]", {}, false, false},

        // gpuh5_floor: COMPUTE-ONLY isolation of GsPersistentKernel (GsPersistentComputeOnlyKernel,
        // every CLIO/I/O call stripped) -- a direct measured compute floor for the gpuh5/
        // gpuh5_sync bars in fig_write_decomp2.tex, replacing that figure's single
        // GsStepKernel-only floor (from [gsbench_reuse]) for those two bars specifically. Read
        // ms from the GSBENCH_PERSISTENT_FLOOR line, not the GSBENCH_RESULT MB/MBps fields
        // (meaningless here -- no I/O happens).
        {"gpuh5_floor",      "[gsbench_persistent_floor]", {}, false, false},
    };
}

}  // namespace

const std::vector<Arm>& AllArms() {
    static const std::vector<Arm> kArms = BuildRegistry();
    return kArms;
}

const Arm* FindArm(const std::string& name) {
    for (const auto& a : AllArms()) {
        if (a.name == name) return &a;
    }
    return nullptr;
}

std::vector<std::string> DefaultArmNames() {
    return {"raw_inline", "raw_threaded", "hostclio",
            "gpuh5", "gpuh5_noreuse", "gpuh5_sync",
            "hdf5_inline", "hdf5_threaded"};
}

std::optional<std::vector<std::string>> ParseArmList(const std::string& csv, std::string& err) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        if (!FindArm(tok)) {
            err = "unknown arm: " + tok;
            return std::nullopt;
        }
        out.push_back(tok);
    }
    if (out.empty()) {
        err = "empty --arms list";
        return std::nullopt;
    }
    return out;
}

}  // namespace gsbench
