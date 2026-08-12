#include "studies.h"

#include <algorithm>

namespace gsbench {

namespace {

EnvOverride E(std::string k, std::string v) { return EnvOverride{std::move(k), std::move(v)}; }
EnvOverride EU(std::string k, unsigned long long v) { return E(std::move(k), std::to_string(v)); }

// GSBENCH_N/CHUNKS/SUBMIT_BLOCKS/SNAPS/STEPS_PER for one config point.
std::vector<EnvOverride> ConfigEnv(unsigned N, unsigned chunks, unsigned snaps,
                                    unsigned steps_per, unsigned submit_blocks) {
    return {EU("GSBENCH_N", N), EU("GSBENCH_CHUNKS", chunks),
            EU("GSBENCH_SUBMIT_BLOCKS", submit_blocks), EU("GSBENCH_SNAPS", snaps),
            EU("GSBENCH_STEPS_PER", steps_per)};
}

void Append(std::vector<EnvOverride>& dst, const std::vector<EnvOverride>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

bool TierRequested(const std::vector<std::string>& tiers, const std::string& tier) {
    return tiers.empty() || std::find(tiers.begin(), tiers.end(), tier) != tiers.end();
}

bool StudyRequested(const std::vector<std::string>& studies, const std::string& s) {
    return studies.empty() || std::find(studies.begin(), studies.end(), s) != studies.end();
}

// Canonical tier iteration order used by every study loop in run_campaign.sh: file then ram.
const std::vector<std::string> kTierOrder = {"file", "ram"};

}  // namespace

std::vector<EnvOverride> CampaignBaseEnv() {
    return {E("GSBENCH_INCOMPRESSIBLE", "1"), E("GSBENCH_RAW_ODIRECT", "0"),
            E("GSBENCH_RAW_FSYNC", "1"), E("GSBENCH_PREWARM", "1")};
}

unsigned long long TotalMb(unsigned N, unsigned snaps) {
    // bash: $1 * $1 * 4 * $2 / 1000000, integer arithmetic throughout.
    return (unsigned long long)N * N * 4ull * snaps / 1000000ull;
}

unsigned long long BdevCap(unsigned N, unsigned snaps) {
    // bash: t * 3 / 2 + 1024, integer division.
    return TotalMb(N, snaps) * 3ull / 2ull + 1024ull;
}

std::vector<EnvOverride> TierEnv(const std::string& tier, unsigned N, unsigned snaps,
                                  const std::string& scratch) {
    const unsigned long long cap = BdevCap(N, snaps);
    if (tier == "file") {
        return {E("GSBENCH_BDEV", "file"), EU("GSBENCH_BDEV_CAP_MB", cap),
                E("GSBENCH_BDEV_PATH", scratch + "/clio_bdev.dat"),
                E("GSBENCH_DISK_DIR", scratch + "/raw_out"),
                E("GSBENCH_HDF5_DIR", scratch + "/hdf5_out")};
    }
    // ram: storage-free for every arm (/dev/shm).
    return {E("GSBENCH_BDEV", "ram"), EU("GSBENCH_BDEV_CAP_MB", cap),
            E("GSBENCH_BDEV_PATH", scratch + "/clio_bdev.dat"),
            E("GSBENCH_DISK_DIR", "/dev/shm/gsbench_raw"),
            E("GSBENCH_HDF5_DIR", "/dev/shm/gsbench_hdf5")};
}

unsigned Isqrt2048(unsigned chunks) {
    switch (chunks) {
        case 1: return 2048;
        case 2: return 2896;
        case 4: return 4096;
        case 8: return 5792;
        case 16: return 8192;
        case 32: return 11584;
        default: return 0;
    }
}

int RepsFor(const std::string& study, const std::string& cfg_key) {
    static const std::vector<std::string> kHeadline = {
        "C__steps8_ram",       // the async I/O-bound headline (reference regime)
        "C__steps8_file",      // its durable-disk contrast
        "C__steps48_file",
        "B__N6400_steps96_file",
        "W__chunks16_steps96_file",
    };
    const std::string key = study + "__" + cfg_key;
    for (const auto& h : kHeadline) {
        if (key == h) return 5;
    }
    return 3;
}

const std::vector<std::string>& ArmsC() {
    static const std::vector<std::string> kArms = {
        "raw_inline", "raw_threaded", "hdf5_naive", "hdf5_inline", "hdf5_threaded",
        "hdf5_async", "hostclio",     "sync",       "async",       "async_pinned"};
    return kArms;
}

const std::vector<std::string>& ArmsScale() {
    static const std::vector<std::string> kArms = {
        "raw_inline", "raw_threaded", "hdf5_naive", "hdf5_inline", "hdf5_threaded",
        "hdf5_async", "hostclio",     "sync",       "async"};
    return kArms;
}

std::vector<WorkConfig> BuildStudyC(const std::vector<std::string>& tiers,
                                     const std::string& scratch) {
    std::vector<WorkConfig> out;
    const unsigned N = 6400, chunks = 4, snaps = 12;
    for (const auto& tier : kTierOrder) {
        if (!TierRequested(tiers, tier)) continue;
        for (unsigned steps : {4u, 8u, 12u, 24u, 48u, 96u, 192u, 384u, 768u}) {
            if (tier == "file" && steps == 768) continue;  // file drops the priciest point
            WorkConfig w;
            w.study = "C";
            w.cfg_key = "steps" + std::to_string(steps) + "_" + tier;
            w.env = ConfigEnv(N, chunks, snaps, steps, /*submit_blocks=*/chunks);
            Append(w.env, CampaignBaseEnv());
            Append(w.env, TierEnv(tier, N, snaps, scratch));
            w.arms = ArmsC();
            w.reps = RepsFor("C", w.cfg_key);
            w.description = "C  " + w.cfg_key + "  (N=" + std::to_string(N) +
                             " chunks=" + std::to_string(chunks) +
                             " snaps=" + std::to_string(snaps) + " reps=" +
                             std::to_string(w.reps) + ")";
            out.push_back(std::move(w));
        }
    }
    return out;
}

std::vector<WorkConfig> BuildStudyK(const std::vector<std::string>& tiers,
                                     const std::string& scratch) {
    std::vector<WorkConfig> out;
    const unsigned N = 6400, snaps = 12;
    for (const auto& tier : kTierOrder) {
        if (!TierRequested(tiers, tier)) continue;
        for (unsigned steps : {48u, 96u}) {
            for (unsigned chunks : {1u, 4u, 16u, 64u, 128u}) {
                WorkConfig w;
                w.study = "K";
                w.cfg_key = "chunks" + std::to_string(chunks) + "_steps" +
                            std::to_string(steps) + "_" + tier;
                w.env = ConfigEnv(N, chunks, snaps, steps, /*submit_blocks=*/chunks);
                Append(w.env, CampaignBaseEnv());
                Append(w.env, TierEnv(tier, N, snaps, scratch));
                w.arms = ArmsScale();
                w.reps = RepsFor("K", w.cfg_key);
                w.description = "K  " + w.cfg_key + "  (N=" + std::to_string(N) +
                                 " chunks=" + std::to_string(chunks) +
                                 " snaps=" + std::to_string(snaps) +
                                 " steps=" + std::to_string(steps) +
                                 " reps=" + std::to_string(w.reps) + ")";
                out.push_back(std::move(w));
            }
        }
    }
    return out;
}

std::vector<WorkConfig> BuildStudyB(const std::vector<std::string>& tiers,
                                     const std::string& scratch) {
    std::vector<WorkConfig> out;
    const unsigned chunks = 16, snaps = 12;
    for (const auto& tier : kTierOrder) {
        if (!TierRequested(tiers, tier)) continue;
        for (unsigned steps : {48u, 96u}) {
            for (unsigned N : {1600u, 3200u, 4800u, 6400u, 9024u}) {
                WorkConfig w;
                w.study = "B";
                w.cfg_key = "N" + std::to_string(N) + "_steps" + std::to_string(steps) + "_" + tier;
                w.env = ConfigEnv(N, chunks, snaps, steps, /*submit_blocks=*/chunks);
                Append(w.env, CampaignBaseEnv());
                Append(w.env, TierEnv(tier, N, snaps, scratch));
                w.arms = ArmsScale();
                w.reps = RepsFor("B", w.cfg_key);
                w.description = "B  " + w.cfg_key + "  (chunks=" + std::to_string(chunks) +
                                 " snaps=" + std::to_string(snaps) +
                                 " steps=" + std::to_string(steps) +
                                 " reps=" + std::to_string(w.reps) + ")";
                out.push_back(std::move(w));
            }
        }
    }
    return out;
}

std::vector<WorkConfig> BuildStudyW(const std::vector<std::string>& tiers,
                                     const std::string& scratch) {
    std::vector<WorkConfig> out;
    const unsigned snaps = 8;
    for (const auto& tier : kTierOrder) {
        if (!TierRequested(tiers, tier)) continue;
        for (unsigned steps : {48u, 96u}) {
            for (unsigned chunks : {1u, 2u, 4u, 8u, 16u, 32u}) {
                const unsigned N = Isqrt2048(chunks);
                WorkConfig w;
                w.study = "W";
                w.cfg_key = "chunks" + std::to_string(chunks) + "_steps" +
                            std::to_string(steps) + "_" + tier;
                w.env = ConfigEnv(N, chunks, snaps, steps, /*submit_blocks=*/chunks);
                Append(w.env, CampaignBaseEnv());
                Append(w.env, TierEnv(tier, N, snaps, scratch));
                w.arms = ArmsScale();
                w.reps = RepsFor("W", w.cfg_key);
                w.description = "W  " + w.cfg_key + "  (N=" + std::to_string(N) +
                                 " bytes/chunk~16.8MB snaps=" + std::to_string(snaps) +
                                 " steps=" + std::to_string(steps) +
                                 " reps=" + std::to_string(w.reps) + ")";
                out.push_back(std::move(w));
            }
        }
    }
    return out;
}

// Study P sweeps GSBENCH_POOL per work-unit. That is safe only because each WorkConfig here
// lists a single arm (w.arms = {"pooled"}) AND no other arm reads GSBENCH_POOL.
//
// FOOTGUN for whoever adds a bounded-async sweep: GSBENCH_HDF5_ASYNC_POOL does NOT have that
// second property. async_VOL and async_VOL_reuse are the same TEST_CASE reading that same var,
// and WorkConfig::env is applied to every arm in w.arms (ComputeChildEnv appends it after the
// per-arm registry env, so it WINS). A unit with w.arms = {"async_VOL", "async_VOL_reuse"} and
// GSBENCH_HDF5_ASYNC_POOL in w.env would silently pool the unbounded baseline. Keep such a
// sweep to w.arms = {"async_VOL_reuse"} and put the unbounded baseline in its own unit (the
// async_ref_<tier> pattern below), or teach the runner to filter that key per-arm.
std::vector<WorkConfig> BuildStudyP(const std::vector<std::string>& tiers,
                                     const std::string& scratch) {
    std::vector<WorkConfig> out;
    const unsigned N = 6400, chunks = 128, G = 8, snaps = 12, steps = 96;
    for (const auto& tier : kTierOrder) {
        if (!TierRequested(tiers, tier)) continue;
        for (unsigned M : {0u, 64u, 32u, 16u, 8u}) {
            WorkConfig w;
            w.study = "P";
            w.cfg_key = "M" + std::to_string(M) + "_" + tier;
            w.env = ConfigEnv(N, chunks, snaps, steps, /*submit_blocks=*/G);
            Append(w.env, CampaignBaseEnv());
            Append(w.env, TierEnv(tier, N, snaps, scratch));
            w.env.push_back(EU("GSBENCH_POOL", M));
            w.arms = {"pooled"};
            w.reps = 3;
            w.description = "P  " + w.cfg_key + "  (chunks=" + std::to_string(chunks) +
                             " G=" + std::to_string(G) + " snaps=" + std::to_string(snaps) +
                             " reps=3)";
            out.push_back(std::move(w));
        }
        // one async fire-all reference per tier
        WorkConfig ref;
        ref.study = "P";
        ref.cfg_key = "async_ref_" + tier;
        ref.env = ConfigEnv(N, chunks, snaps, steps, /*submit_blocks=*/G);
        Append(ref.env, CampaignBaseEnv());
        Append(ref.env, TierEnv(tier, N, snaps, scratch));
        ref.arms = {"async"};
        ref.reps = 3;
        ref.description = "P  " + ref.cfg_key + "  (async fire-all reference)";
        out.push_back(std::move(ref));
    }
    return out;
}

std::vector<WorkConfig> BuildCampaign(const std::vector<std::string>& studies,
                                       const std::vector<std::string>& tiers,
                                       const std::string& scratch) {
    std::vector<WorkConfig> out;
    if (StudyRequested(studies, "C")) {
        auto c = BuildStudyC(tiers, scratch);
        out.insert(out.end(), std::make_move_iterator(c.begin()), std::make_move_iterator(c.end()));
    }
    if (StudyRequested(studies, "K")) {
        auto k = BuildStudyK(tiers, scratch);
        out.insert(out.end(), std::make_move_iterator(k.begin()), std::make_move_iterator(k.end()));
    }
    if (StudyRequested(studies, "B")) {
        auto b = BuildStudyB(tiers, scratch);
        out.insert(out.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
    }
    if (StudyRequested(studies, "W")) {
        auto w = BuildStudyW(tiers, scratch);
        out.insert(out.end(), std::make_move_iterator(w.begin()), std::make_move_iterator(w.end()));
    }
    if (StudyRequested(studies, "P")) {
        auto p = BuildStudyP(tiers, scratch);
        out.insert(out.end(), std::make_move_iterator(p.begin()), std::make_move_iterator(p.end()));
    }
    return out;
}

}  // namespace gsbench
