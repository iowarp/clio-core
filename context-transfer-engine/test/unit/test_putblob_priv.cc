/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Private-memory AsyncPutBlob (issue #830).
 *
 * CoreClient::AsyncPutBlob(const char*) writes a blob region straight from a
 * caller-owned PRIVATE buffer instead of making the caller hand-manage an SHM
 * buffer. It is the write-side mirror of the private-memory AsyncGetBlob
 * (issue #823), and like that path it resolves differently per configuration,
 * so this test runs in BOTH:
 *
 *   - Runtime started INLINE (co-located): the daemon shares this address
 *     space, so the private pointer is wrapped as a null-allocator ShmPtr and
 *     the write reads directly out of the caller's buffer — no staging copy.
 *     `CLIO_PUTBLOB_PRIV_MODE` unset / "inline".
 *   - Runtime started SEPARATE (its own process): a pure client cannot expose
 *     private memory, so the payload is staged through an SHM buffer copied in
 *     before Send, and the task is marked TASK_DATA_OWNER so ~PutBlobTask
 *     reclaims the staging buffer. `CLIO_PUTBLOB_PRIV_MODE=separate`.
 *
 * Read-back verification deliberately uses the ordinary SHM GetBlob, so a bug
 * in the private-memory READ path cannot mask a bug in the write path.
 *
 * The buffers handed to the API here are genuine private memory
 * (std::vector<char> / stack), never SHM — that is the whole point of the API.
 */

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_cte/core/core_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"
#include "runtime_server.h"
#include "simple_test.h"

using namespace std::chrono_literals;

namespace {

constexpr clio::run::u64 kRamTargetBytes = 256ULL * 1024 * 1024;
constexpr unsigned kPort = 10624;
const char *kTargetName = "putblob_priv_target";

/** True when CLIO_PUTBLOB_PRIV_MODE=separate: bring up a daemon in its own
 *  process and attach as a pure client (exercises the SHM staging path).
 *  Otherwise the runtime is started inline (exercises the direct path). */
bool SeparateMode() {
  const char *m = std::getenv("CLIO_PUTBLOB_PRIV_MODE");
  return m != nullptr && std::string(m) == "separate";
}

/** Run `clio_run <args>` to completion, returning its exit code. */
int RunCli(const std::vector<std::string> &args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char *> argv;
  for (auto &a : full) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    execv(argv[0], argv.data());
    _exit(127);
  }
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -3;
    }
    std::this_thread::sleep_for(100ms);
  }
}

clio::run::test::RuntimeServer *g_server = nullptr;

/** Deterministic byte at logical position i of the test pattern. */
char PatternByte(size_t i) { return static_cast<char>((i * 131 + 7) & 0xFF); }

class Fixture {
 public:
  bool initialized_ = false;
  bool separate_ = false;

  Fixture() {
    separate_ = SeparateMode();
    if (separate_) {
      initialized_ = InitSeparate();
    } else {
      initialized_ = InitInline();
    }
  }

  /** Inline: this process IS the runtime. Register a RAM target by hand. */
  /** Register a RAM target into the pool CLIO_CTE_CLIENT actually talks to
   *  (the default clio_cte_core pool). Shared by the inline and separate
   *  fixtures so both place the target where the client's PutBlobs look for
   *  it — same fix as test_getblob_priv: the compose in separate mode
   *  registers its storage into the COMPOSED pool (516.0) while the client
   *  talks to the default pool (512.0), so without this the setup put fails
   *  with rc 11 ("no targets") on the deps-cpu/Boost CI configs. */
  static bool RegisterRamTarget() {
    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(916, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                          kTargetName, bdev_pool_id,
                                          clio::run::bdev::BdevType::kRam,
                                          kRamTargetBytes);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(kTargetName,
                                        clio::run::bdev::BdevType::kRam,
                                        kRamTargetBytes,
                                        clio::run::PoolQuery::Local(),
                                        bdev_pool_id);
    reg.Wait();
    return reg->GetReturnCode() == 0;
  }

  bool InitInline() {
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
      return false;
    }
    std::this_thread::sleep_for(300ms);
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
      return false;
    }
    std::this_thread::sleep_for(200ms);

    if (!RegisterRamTarget()) return false;
    // Best-effort: attach the SHM metadata cache so the zero-IPC read-back
    // path can run where available.
    (void)CLIO_CTE_CLIENT->AttachShmCache();
    return true;
  }

  /** Separate: compose a real daemon in its own process, attach as a client. */
  bool InitSeparate() {
    const std::string work = "/tmp/clio_putblob_priv_test";
    std::filesystem::remove_all(work);
    std::filesystem::create_directories(work);
    const std::string yaml = work + "/compose.yaml";
    {
      std::ofstream f(yaml);
      f << "compose:\n"
           "  - mod_name: clio_cte_core\n"
           "    pool_name: \"putblob_priv_cte\"\n"
           "    pool_query: local\n"
           "    pool_id: \"516.0\"\n"
           "    storage:\n"
           "      - path: "
        << work << "/ram_dev\n"
           "        bdev_type: ram\n"
           "        capacity_limit: "
        << (kRamTargetBytes / (1024 * 1024)) << "mb\n"
           "    dpe:\n"
           "      dpe_type: random\n";
    }

    setenv("CLIO_WAIT_SERVER", "15", 1);
    setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

    static clio::run::test::RuntimeServer server;
    g_server = &server;
    if (!server.Start(kPort) || !server.WaitForReady()) {
      return false;
    }
    if (RunCli({"compose", "start", yaml}, 60) != 0) {
      return false;
    }
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
      return false;
    }
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
      return false;
    }
    // Register the RAM target into the pool this client actually uses — the
    // daemon services the bdev create + RegisterTarget exactly as inline mode
    // does. Without this the composed pool holds the only target and every
    // put from the client's default pool fails rc 11 (see RegisterRamTarget).
    if (!RegisterRamTarget()) {
      return false;
    }
    (void)CLIO_CTE_CLIENT->AttachShmCache();
    return true;
  }
};

Fixture *g_fixture = nullptr;

/** Read [off, off+n) of `blob` through an ordinary SHM GetBlob, so the
 *  verification path is independent of the private-memory READ path. */
std::vector<char> GetViaShm(clio::cte::core::TagId tag_id,
                            const std::string &blob, clio::run::u64 off,
                            size_t n) {
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(n);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0, n);
  auto g = cte->AsyncGetBlob(tag_id, blob, off, n, /*flags=*/0,
                             ctp::ipc::ShmPtr<>(buf.shm_));
  g.Wait();
  REQUIRE(g->GetReturnCode() == 0);
  std::vector<char> out(buf.ptr_, buf.ptr_ + n);
  ipc->FreeBuffer(buf);
  return out;
}

/** Get-or-create a tag and return its id. Deliberately uses the client API
 *  rather than the clio::cte::core::Tag wrapper, which is on its way out
 *  (issue #826). */
clio::cte::core::TagId ResolveTag(const std::string &name) {
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

/** A private (heap) buffer holding `n` bytes of the pattern from `base`. */
std::vector<char> PrivatePattern(size_t n, size_t base) {
  std::vector<char> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = PatternByte(base + i);
  return v;
}

}  // namespace

// The heart of the issue: write FROM private memory and have the right bytes
// land in the blob, under whichever runtime configuration the fixture chose.
TEST_CASE("PutBlob from private memory stores correct bytes", "[cte][830]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::TagId tag_id = ResolveTag("putblob_priv_tag");

  const std::string blob = "priv_blob";
  const size_t kN = 64 * 1024;  // 64 KiB

  // Full write from a genuinely private buffer (heap, not SHM).
  std::vector<char> priv = PrivatePattern(kN, 0);
  auto p = cte->AsyncPutBlob(tag_id, blob, /*offset=*/0, kN, priv.data());
  p.Wait();
  REQUIRE(p->GetReturnCode() == 0);

  std::vector<char> got = GetViaShm(tag_id, blob, 0, kN);
  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(got[i] == PatternByte(i));
  }
}

// An offset write must land at the right place and leave its neighbours alone —
// the in-place-overwrite shape the issue cares about for repeated writes.
TEST_CASE("Private PutBlob at an offset overwrites only its range",
          "[cte][830]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::TagId tag_id = ResolveTag("putblob_priv_offset_tag");

  const std::string blob = "offset_blob";
  const size_t kN = 32 * 1024;
  std::vector<char> base = PrivatePattern(kN, 0);
  auto p0 = cte->AsyncPutBlob(tag_id, blob, 0, kN, base.data());
  p0.Wait();
  REQUIRE(p0->GetReturnCode() == 0);

  // Overwrite a middle slice with a different pattern (base 500).
  const size_t kOff = 4096, kLen = 8192;
  std::vector<char> patch = PrivatePattern(kLen, 500);
  auto p1 = cte->AsyncPutBlob(tag_id, blob, kOff, kLen, patch.data());
  p1.Wait();
  REQUIRE(p1->GetReturnCode() == 0);

  std::vector<char> got = GetViaShm(tag_id, blob, 0, kN);
  for (size_t i = 0; i < kN; ++i) {
    const char want = (i >= kOff && i < kOff + kLen)
                          ? PatternByte(500 + (i - kOff))
                          : PatternByte(i);
    REQUIRE(got[i] == want);
  }
}

// Once the Future completes the caller owns its buffer again: in client mode
// the payload was copied into SHM before Send, and in runtime mode the daemon
// read straight from this buffer but is done by the time Wait() returns.
// Mutating it afterwards must not disturb what was stored.
TEST_CASE("Private PutBlob buffer is reusable after Wait", "[cte][830]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::TagId tag_id = ResolveTag("putblob_priv_reuse_tag");

  const std::string blob = "reuse_blob";
  const size_t kN = 16 * 1024;
  std::vector<char> priv = PrivatePattern(kN, 77);
  auto p = cte->AsyncPutBlob(tag_id, blob, 0, kN, priv.data());
  p.Wait();
  REQUIRE(p->GetReturnCode() == 0);

  // Scribble over the caller's buffer, then confirm the blob is unchanged.
  std::memset(priv.data(), 0xAB, kN);
  std::vector<char> got = GetViaShm(tag_id, blob, 0, kN);
  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(got[i] == PatternByte(77 + i));
  }
}

// Repeated writes must not leak or hang: the client path allocates a staging
// buffer per call and relies on TASK_DATA_OWNER to reclaim it. A leak here
// exhausts the main SHM segment and later writes fail; a missed free would trip
// the allocator's leak checker at shutdown. This is also the repeated-write
// workload issue #830 wants to optimize, so it must stay correct.
TEST_CASE("Private PutBlob is leak-free under repetition", "[cte][830]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::TagId tag_id = ResolveTag("putblob_priv_loop_tag");

  const std::string blob = "loop_blob";
  const size_t kN = 32 * 1024;

  for (int iter = 0; iter < 256; ++iter) {
    // Vary the pattern per iteration so a stale-buffer bug cannot pass by
    // coincidence: the read-back must see THIS iteration's bytes.
    const size_t base = static_cast<size_t>(iter) * 7 + 3;
    std::vector<char> priv = PrivatePattern(kN, base);
    auto p = cte->AsyncPutBlob(tag_id, blob, 0, kN, priv.data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
  }

  // Spot-check the final iteration's bytes survived (a full compare every
  // iteration would dominate runtime).
  const size_t last_base = static_cast<size_t>(255) * 7 + 3;
  std::vector<char> got = GetViaShm(tag_id, blob, 0, kN);
  REQUIRE(got[0] == PatternByte(last_base));
  REQUIRE(got[kN / 2] == PatternByte(last_base + kN / 2));
  REQUIRE(got[kN - 1] == PatternByte(last_base + kN - 1));
}

// A degenerate request returns an EMPTY future rather than submitting a task.
// Wait() must return immediately instead of hanging, and — the part that
// matters — no blob may be created as a side effect.
TEST_CASE("Private PutBlob rejects degenerate requests", "[cte][830]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::TagId tag_id = ResolveTag("putblob_priv_degenerate_tag");

  std::vector<char> priv = PrivatePattern(1024, 5);
  auto p_null = cte->AsyncPutBlob(tag_id, "null_blob", 0, 1024, nullptr);
  p_null.Wait();
  auto p_zero = cte->AsyncPutBlob(tag_id, "zero_blob", 0, 0, priv.data());
  p_zero.Wait();

  // Neither name may exist: a rejected request must not reach the runtime.
  auto listed = cte->AsyncGetContainedBlobs(tag_id);
  listed.Wait();
  REQUIRE(listed->GetReturnCode() == 0);
  for (const std::string &name : listed->blob_names_) {
    REQUIRE(name != "null_blob");
    REQUIRE(name != "zero_blob");
  }
}

int main(int argc, char **argv) {
  static Fixture fixture;
  g_fixture = &fixture;
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}
