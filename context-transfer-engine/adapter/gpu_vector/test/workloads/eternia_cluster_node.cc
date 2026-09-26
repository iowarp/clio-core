/* ----------------------------------------------------------------------
   A storage-serving cluster node for the workload cluster tests.

   The workload cluster tests put LAMMPS, GROMACS or LBANN on node 1 and this
   binary on nodes 2..N. There is no standalone Clio daemon -- every node in a
   cluster runs a binary that joins via CLIO_INIT and an entry in the hostfile
   -- so the peers need SOMETHING to run, and what they need to do is join,
   compose their local RAM tier, and stay alive while node 1 works.

   That is the whole point of the exercise. A vector's pages are CTE blobs and
   blobs hash across the cluster, so with peers present most of the workload's
   page faults are served from another node's memory rather than this one's.
   Running the workloads that way is what turns "the paging path is correct"
   into "the paging path is correct when the page is on a different machine".

   It exits when node 1 drops a sentinel file in the shared workspace, or when
   ETERNIA_NODE_TIMEOUT seconds have passed -- so a crashed or hung workload
   cannot leave a cluster of idle containers behind.
------------------------------------------------------------------------- */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>

namespace {

int EnvInt(const char* name, int dflt)
{
  const char* v = std::getenv(name);
  return (v != nullptr) ? std::atoi(v) : dflt;
}

std::string EnvStr(const char* name, const std::string& dflt)
{
  const char* v = std::getenv(name);
  return (v != nullptr) ? std::string(v) : dflt;
}

}  // namespace

int main()
{
  const int node = EnvInt("NODE_ID", 2);
  const int timeout_s = EnvInt("ETERNIA_NODE_TIMEOUT", 1800);
  const std::string done = EnvStr("ETERNIA_DONE_FILE", "/workspace/.eternia_cluster_done");

  std::fprintf(stderr, "[node%d] joining cluster\n", node);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "[node%d] runtime init failed\n", node);
    return 1;
  }
  // Membership has to settle before anyone creates cluster-spanning pools;
  // the same two seconds the gpu_vector distributed test waits.
  std::this_thread::sleep_for(std::chrono::seconds(2));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[node%d] cte client init failed\n", node);
    return 1;
  }
  // Announce readiness on the shared filesystem. The workload node waits for
  // these before it starts: a workload that begins before its peers are
  // serving creates all of its blobs locally, and then a "cluster" run is one
  // that never left the node.
  const std::string ready =
      EnvStr("ETERNIA_READY_FILE", "/workspace/.eternia_peer_ready.") +
      std::to_string(node);
  if (std::FILE* rf = std::fopen(ready.c_str(), "w")) {
    std::fprintf(rf, "%d\n", node);
    std::fclose(rf);
  } else {
    std::fprintf(stderr, "[node%d] could not write %s\n", node, ready.c_str());
    return 1;
  }
  std::fprintf(stderr, "[node%d] serving\n", node);

  const auto t0 = std::chrono::steady_clock::now();
  while (true) {
    if (std::FILE* f = std::fopen(done.c_str(), "r")) {
      std::fclose(f);
      std::fprintf(stderr, "[node%d] done file seen, exiting\n", node);
      std::remove(ready.c_str());
      return 0;
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (elapsed > timeout_s) {
      // Not a success: a peer outliving the workload by the full timeout means
      // the workload never finished, and reporting 0 here would let the
      // harness call that a pass.
      std::fprintf(stderr, "[node%d] timed out after %ds with no done file\n",
                   node, timeout_s);
      return 2;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}
