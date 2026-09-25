/**
 * Phase hints for the CTE data organizer.
 *
 * Split out of bench_dist.h deliberately. bench_dist.h carries the CTE
 * COLLECTIVES (peer gets, reductions, slice exchange) and pulls in the vector
 * types they operate on; a benchmark that only wants to announce a phase
 * should not have to compile any of that, and lammps_md in fact cannot -- it
 * fails inside bench_dist.h's own machinery. A phase hint needs nothing but
 * the CTE client.
 */
#ifndef CLIO_CTE_GPU_VECTOR_BENCH_PHASE_H_
#define CLIO_CTE_GPU_VECTOR_BENCH_PHASE_H_

#include <clio_cte/core/core_client.h>

#include <cstdio>

namespace clio_bench_dist {

/**
 * PHASE HINTS FOR THE DATA ORGANIZER.
 *
 * A paged workload knows things about its own future that blob statistics
 * cannot show: that the pass it is about to run reads what the last pass
 * wrote, that a checkpoint is next, that a sweep is starting over. The CTE
 * carries that as ONE OPAQUE INTEGER, broadcast to every container by
 * Client::ReorganizeHint (PoolQuery::Broadcast, so all nodes agree) and read
 * by the configured DataOrganizer on its next round.
 *
 * The integer is opaque to the CORE but not to the organizers, so the
 * benchmarks need a shared vocabulary or each one invents its own and no
 * organizer can be reused. This is that vocabulary. It is deliberately tiny:
 *
 *   kPhaseScatter -- producing. Pages are being WRITTEN and will not be read
 *                    until the phase turns, so promoting them buys a
 *                    migration for a page with no reuse ahead of it.
 *   kPhaseGather  -- consuming. The pages written during scatter are about to
 *                    be READ, so an organizer may promote before the first
 *                    fault instead of one page-read late.
 *
 * kPhaseGather MUST stay 2: ScatterDataOrganizer::kPhaseGather is the
 * consumer and the core does not validate the value, so a mismatch is silent.
 * GrayScottDataOrganizer uses its own scheme (step index + 1) because its
 * policy is per-step rather than per-phase; it does not go through here.
 */
enum Phase : int {
  kPhaseNone = 0,    /**< no phase announced */
  kPhaseScatter = 1, /**< writing; reads come later */
  kPhaseGather = 2,  /**< reading what scatter produced */
};

/**
 * Announce a phase to the data organizer on every node.
 *
 * Broadcast and WAITED: the call is cheap next to a pass, and an organizer
 * must not still see the previous phase while this one is already running.
 * A failure is reported once and then ignored -- a missed hint degrades the
 * policy (organizers are written to work without one) rather than the run,
 * so it must not fail a benchmark that is otherwise correct.
 *
 * Uses the process-wide CLIO_CTE_CLIENT rather than a passed client: the
 * hint is pool-wide and broadcast, so it has nothing to do with the
 * per-run reduction client, which several benches only create when
 * nodes > 1 and which is therefore null exactly where organizers are most
 * often measured (single node).
 *
 * @param phase phase value; one of Phase, or a policy-specific integer
 * @param enabled false makes the call a no-op, for the --organizer-hint flag
 * @return true if the hint reached every container (or was disabled)
 */
inline bool SetPhase(int phase, bool enabled = true) {
  if (!enabled) return true;
  const clio::run::u32 rc = CLIO_CTE_CLIENT->ReorganizeHint(
      static_cast<clio::run::i32>(phase));
  if (rc != 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr,
                   "bench_dist: ReorganizeHint(%d) rc=%u; phase hints are "
                   "off for the rest of this run\n", phase, rc);
    }
    return false;
  }
  return true;
}

}  // namespace clio_bench_dist

#endif  // CLIO_CTE_GPU_VECTOR_BENCH_PHASE_H_
