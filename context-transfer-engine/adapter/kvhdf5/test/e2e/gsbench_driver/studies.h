// The campaign study definitions (C/K/B/W/P): value lists, per-study arm sets, tiers, reps,
// HEADLINE configs. Ported from scaling_campaign/run_campaign.sh (the STUDY loops, set_tier,
// tag_of/case block, reps_for/HEADLINE) and cross-checked against
// scaling_campaign/README.md section 4.
#pragma once

#include <string>
#include <vector>

#include "arms.h"

namespace gsbench {

// One (study, cfg, rep) work unit: a fully-resolved GSBENCH_* environment plus the arms to run
// at it, in order. Single-config mode also uses this type with study/cfg_key left empty.
struct WorkConfig {
    std::string study;                  // "C","K","B","W","P", or "" (single-config mode)
    std::string cfg_key;                // e.g. "steps8_ram", "M64_ram", "" in single-config mode
    std::vector<EnvOverride> env;        // GSBENCH_N/CHUNKS/SNAPS/STEPS_PER/BDEV/... for this config
    std::vector<std::string> arms;       // arm names to run at this config (arm order = run order)
    int reps = 1;
    std::string description;             // human-readable, for --dry-run / progress narration
};

// campaign's fixed BASE_ENV, applied to every campaign work unit.
std::vector<EnvOverride> CampaignBaseEnv();

// bash: total_mb(N,snaps) = N*N*4*snaps/1000000 (integer division, matches run_campaign.sh).
unsigned long long TotalMb(unsigned N, unsigned snaps);
// bash: bdev_cap(N,snaps) = total_mb*3/2 + 1024 (integer division at each step).
unsigned long long BdevCap(unsigned N, unsigned snaps);

// Ports set_tier(): per-tier storage env (bdev/cap/path/disk_dir/hdf5_dir). `scratch` is the
// campaign SCRATCH dir (disk-backed, wiped per-arm); ram tier always uses /dev/shm regardless.
std::vector<EnvOverride> TierEnv(const std::string& tier, unsigned N, unsigned snaps,
                                  const std::string& scratch);

// isqrt2048(chunks): N = round(2048*sqrt(chunks)) via the exact lookup table in
// run_campaign.sh (keeps N an exact multiple of chunks). Returns 0 for unlisted chunk counts.
unsigned Isqrt2048(unsigned chunks);

// Number of reps for a given "STUDY__cfgkey" (5 on HEADLINE configs, else 3). Mirrors
// run_campaign.sh's reps_for()/HEADLINE array.
int RepsFor(const std::string& study, const std::string& cfg_key);

// Builders for each study. `tiers` is the requested tier subset (e.g. {"ram"} or {"ram","file"});
// each function filters/loops accordingly, matching the shell's per-study tier loop
// (Study C drops steps=768 on the file tier; everything else runs both tiers requested).
std::vector<WorkConfig> BuildStudyC(const std::vector<std::string>& tiers, const std::string& scratch);
std::vector<WorkConfig> BuildStudyK(const std::vector<std::string>& tiers, const std::string& scratch);
std::vector<WorkConfig> BuildStudyB(const std::vector<std::string>& tiers, const std::string& scratch);
std::vector<WorkConfig> BuildStudyW(const std::vector<std::string>& tiers, const std::string& scratch);
std::vector<WorkConfig> BuildStudyP(const std::vector<std::string>& tiers, const std::string& scratch);

// The 9-arm scaling set (Studies K/B/W) and 10-arm compute set (Study C), matching
// ARMS_SCALE / ARMS_C in run_campaign.sh.
const std::vector<std::string>& ArmsScale();
const std::vector<std::string>& ArmsC();

// Builds the full requested campaign: `studies` selects a subset of {C,K,B,W,P} (empty = all,
// in canonical C,K,B,W,P order), `tiers` a subset of {ram,file} (empty = both).
std::vector<WorkConfig> BuildCampaign(const std::vector<std::string>& studies,
                                       const std::vector<std::string>& tiers,
                                       const std::string& scratch);

}  // namespace gsbench
