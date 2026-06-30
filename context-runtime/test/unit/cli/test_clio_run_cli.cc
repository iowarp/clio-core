/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core (BSD 3-Clause license, see COPYING).
 */

/**
 * Unit tests for the clio_run CLI command implementations
 * (context-runtime/util/clio_run_cmd_*.cc and clio_run.cc).
 *
 * Two strategies are used:
 *  1. RefreshRepo() is called directly (linked from clio_run_commands).
 *     It is a pure local code generator, so we drive it against synthetic
 *     chimod repositories created in temp directories and verify both the
 *     error paths and the generated header/source content.
 *  2. The clio_run binary is invoked as a subprocess for argument
 *     parsing / dispatch / usage paths and for the "no runtime available"
 *     client-init failure paths. CLIO_WAIT_SERVER=0 guarantees client-mode
 *     commands fail immediately instead of waiting for a daemon, so no
 *     runtime is ever started and nothing is left running.
 */

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_run_commands.h"
#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

/** RAII temp directory under /tmp, removed (recursively) on destruction. */
struct TempDir {
  fs::path path;

  TempDir() {
    std::string tmpl = (fs::temp_directory_path() / "clio_cli_test_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* result = mkdtemp(buf.data());
    if (result == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path = result;
  }

  ~TempDir() {
    std::error_code ec;
    // Restore write permission everywhere so remove_all can't fail on
    // intentionally read-only directories created by tests.
    if (fs::exists(path, ec)) {
      for (auto it = fs::recursive_directory_iterator(
               path, fs::directory_options::skip_permission_denied, ec);
           it != fs::recursive_directory_iterator(); ++it) {
        fs::permissions(it->path(), fs::perms::owner_all,
                        fs::perm_options::add, ec);
      }
      fs::remove_all(path, ec);
    }
  }
};

void WriteFile(const fs::path& p, const std::string& content) {
  fs::create_directories(p.parent_path());
  std::ofstream ofs(p);
  ofs << content;
}

std::string ReadAll(const fs::path& p) {
  std::ifstream ifs(p);
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

bool Contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

/** Call RefreshRepo() with a writable argv built from strings. */
int CallRefreshRepo(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& a : args) {
    argv.push_back(a.data());
  }
  return RefreshRepo(static_cast<int>(argv.size()),
                     argv.empty() ? nullptr : argv.data());
}

/**
 * Make sure client-mode commands cannot block waiting for a daemon and
 * cannot collide with any runtime another test may have running on the
 * default port.
 */
void SetSafeCliEnv() {
  setenv("CLIO_WAIT_SERVER", "0", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);
  setenv("CLIO_PORT", "10599", 1);
}

/** Run the clio_run binary with the given args; return its exit code. */
int RunCli(const std::string& args) {
  SetSafeCliEnv();
  std::string cmd = std::string(CLIO_RUN_EXE) + " " + args + " >/dev/null 2>&1";
  int rc = std::system(cmd.c_str());
  if (rc == -1) {
    return -1;
  }
  if (WIFEXITED(rc)) {
    return WEXITSTATUS(rc);
  }
  return -2;  // killed by signal — always a test failure
}

/**
 * Run the clio_run binary against a LIVE runtime with a hard kill deadline.
 * fork/execv instead of std::system so a hung client can never outlive the
 * test (GNU `timeout` is not available on macOS). Returns the child's exit
 * code, -2 if it died from a signal, or -3 if it had to be killed on timeout.
 */
int RunCliTimed(const std::vector<std::string>& args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char*> argv;
  argv.reserve(full.size() + 1);
  for (auto& a : full) {
    argv.push_back(a.data());
  }
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) {
    return -1;
  }
  if (pid == 0) {
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, 1);
      dup2(devnull, 2);
      close(devnull);
    }
    execv(argv[0], argv.data());
    _exit(127);
  }

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) {
      if (WIFEXITED(status)) return WEXITSTATUS(status);
      return -2;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -3;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

/** Minimal valid module yaml with the standard inherited methods. */
const char* kAlphaModYaml =
    "# alpha module config\n"
    "module_name: alpha_mod\n"
    "namespace: test::ns\n"
    "version: 1.0.0\n"
    "kCreate: 0\n"
    "kDestroy: 1\n"
    "kNodeFailure: -1\n"
    "kMonitor: 9\n"
    "kRestart: 2\n"
    "kGetOrCreateThing: 10\n"
    "kCustomOp: 11\n"
    "kBadValue: hello\n"
    "kMapValue:\n"
    "  nested: 1\n"
    "plain_key: 5\n";

}  // namespace

//==============================================================================
// RefreshRepo — direct calls (pure code generator, no runtime involved)
//==============================================================================

TEST_CASE("RefreshRepo - usage and bad repo path", "[cli][refresh_repo]") {
  SECTION("no arguments prints usage and fails");
  REQUIRE(CallRefreshRepo({}) == 1);

  SECTION("too many arguments prints usage and fails");
  REQUIRE(CallRefreshRepo({"a", "b"}) == 1);

  SECTION("nonexistent repository path fails");
  REQUIRE(CallRefreshRepo({"/nonexistent/clio_cli_test_repo"}) == 1);

  SECTION("directory without clio_repo.yaml fails");
  TempDir tmp;
  REQUIRE(CallRefreshRepo({tmp.path.string()}) == 1);
}

TEST_CASE("RefreshRepo - malformed repository yaml", "[cli][refresh_repo]") {
  SECTION("unparsable repo yaml fails");
  {
    TempDir tmp;
    WriteFile(tmp.path / "clio_repo.yaml", "modules: [unclosed\n  - bad\n");
    REQUIRE(CallRefreshRepo({tmp.path.string()}) == 1);
  }

  SECTION("repo yaml without modules key fails");
  {
    TempDir tmp;
    WriteFile(tmp.path / "clio_repo.yaml", "namespace: foo\nname: x\n");
    REQUIRE(CallRefreshRepo({tmp.path.string()}) == 1);
  }

  SECTION("repo yaml with non-sequence modules fails");
  {
    TempDir tmp;
    WriteFile(tmp.path / "clio_repo.yaml", "modules: 5\n");
    REQUIRE(CallRefreshRepo({tmp.path.string()}) == 1);
  }
}

TEST_CASE("RefreshRepo - generates methods header and lib_exec source",
          "[cli][refresh_repo]") {
  TempDir tmp;
  // Repo with one full module plus a missing module dir and a module with
  // malformed yaml — the latter two must produce warnings, not failures.
  WriteFile(tmp.path / "clio_repo.yaml",
            "name: testrepo\n"
            "namespace: test::ns\n"
            "modules:\n"
            "  - alpha\n"
            "  - missing_mod\n"
            "  - badmod\n");
  WriteFile(tmp.path / "alpha" / "clio_mod.yaml", kAlphaModYaml);
  WriteFile(tmp.path / "badmod" / "clio_mod.yaml",
            "module_name: [unterminated\n  : :\n");

  REQUIRE(CallRefreshRepo({tmp.path.string()}) == 0);

  // module_name overrides the directory name in the include path
  fs::path header = tmp.path / "alpha" / "include" / "test::ns" / "alpha_mod" /
                    "autogen" / "alpha_methods.h";
  fs::path source = tmp.path / "alpha" / "src" / "autogen" / "alpha_lib_exec.cc";
  REQUIRE(fs::exists(header));
  REQUIRE(fs::exists(source));

  const std::string h = ReadAll(header);
  SECTION("header include guard uses namespace with :: mapped to _");
  REQUIRE(Contains(h, "#ifndef TEST__NS_ALPHA_AUTOGEN_METHODS_H_"));

  SECTION("header opens the repo namespace");
  REQUIRE(Contains(h, "namespace test::ns::alpha {"));

  SECTION("inherited methods (id < 10) are emitted");
  REQUIRE(Contains(h, "// Inherited methods"));
  REQUIRE(Contains(h, "kCreate = 0;"));
  REQUIRE(Contains(h, "kDestroy = 1;"));
  REQUIRE(Contains(h, "kMonitor = 9;"));

  SECTION("custom methods (id >= 10) are emitted");
  REQUIRE(Contains(h, "alpha_mod-specific methods"));
  REQUIRE(Contains(h, "kGetOrCreateThing = 10;"));
  REQUIRE(Contains(h, "kCustomOp = 11;"));

  SECTION("kMaxMethodId is one past the highest method id");
  REQUIRE(Contains(h, "kMaxMethodId = 12;"));

  SECTION("unimplemented (-1), non-integer, non-scalar, and non-k keys skipped");
  REQUIRE_FALSE(Contains(h, "kNodeFailure"));
  REQUIRE_FALSE(Contains(h, "kBadValue"));
  REQUIRE_FALSE(Contains(h, "kMapValue"));
  REQUIRE_FALSE(Contains(h, "plain_key"));

  SECTION("kRestart is lifecycle-only and never a dispatch method");
  REQUIRE_FALSE(Contains(h, "kRestart"));

  SECTION("GetMethodNames maps ids to names");
  REQUIRE(Contains(h, "v[0] = \"Create\";"));
  REQUIRE(Contains(h, "v[11] = \"CustomOp\";"));

  const std::string s = ReadAll(source);
  SECTION("source includes runtime + autogen headers via module_name");
  REQUIRE(Contains(s, "#include \"test::ns/alpha_mod/alpha_runtime.h\""));
  REQUIRE(Contains(s, "test::ns/alpha_mod/autogen/alpha_methods.h"));

  SECTION("kRestart >= 0 generates Runtime::Restart");
  REQUIRE(Contains(s, "void Runtime::Restart("));
  REQUIRE(Contains(s, "is_restart_ = true;"));

  SECTION("special task type names: Create/Destroy/GetOrCreate/regular");
  REQUIRE(Contains(s, "Cast<CreateTask>"));
  REQUIRE(Contains(s, "Cast<DestroyTask>"));
  REQUIRE(Contains(s, "alpha::GetOrCreateThingTask<alpha::CreateParams>"));
  REQUIRE(Contains(s, "Cast<CustomOpTask>"));

  SECTION("all Container virtual APIs are generated");
  REQUIRE(Contains(s, "clio::run::TaskResume Runtime::Run("));
  REQUIRE(Contains(s, "void Runtime::SaveTask("));
  REQUIRE(Contains(s, "void Runtime::LoadTask("));
  REQUIRE(Contains(s, "Runtime::AllocLoadTask("));
  REQUIRE(Contains(s, "void Runtime::LocalLoadTask("));
  REQUIRE(Contains(s, "Runtime::LocalAllocLoadTask("));
  REQUIRE(Contains(s, "void Runtime::LocalSaveTask("));
  REQUIRE(Contains(s, "Runtime::NewCopyTask("));
  REQUIRE(Contains(s, "Runtime::NewTask("));
  REQUIRE(Contains(s, "void Runtime::AggregateOut("));
  // DelTask is no longer generated — tasks are clio::run::shared_ptr handles
  // freed via RAII. NewTask now returns a shared_ptr.
  REQUIRE_FALSE(Contains(s, "void Runtime::DelTask("));
  REQUIRE(Contains(s, "clio::run::shared_ptr<clio::run::Task> Runtime::NewTask("));
  REQUIRE(Contains(s, "case Method::kGetOrCreateThing:"));

  SECTION("missing module directory produced no output but no failure");
  REQUIRE_FALSE(fs::exists(tmp.path / "missing_mod" / "src"));

  SECTION("malformed module yaml produced no output but no failure");
  REQUIRE_FALSE(fs::exists(tmp.path / "badmod" / "src"));
}

TEST_CASE("RefreshRepo - default namespace and module with no methods",
          "[cli][refresh_repo]") {
  TempDir tmp;
  // No namespace key -> defaults to "clio"; module yaml with no method
  // keys and no module_name -> empty method table, kMaxMethodId = 0.
  WriteFile(tmp.path / "clio_repo.yaml",
            "modules:\n"
            "  - beta\n");
  WriteFile(tmp.path / "beta" / "clio_mod.yaml", "version: 1.0.0\n");

  REQUIRE(CallRefreshRepo({tmp.path.string()}) == 0);

  fs::path header = tmp.path / "beta" / "include" / "clio" / "beta" /
                    "autogen" / "beta_methods.h";
  fs::path source = tmp.path / "beta" / "src" / "autogen" / "beta_lib_exec.cc";
  REQUIRE(fs::exists(header));
  REQUIRE(fs::exists(source));

  const std::string h = ReadAll(header);
  REQUIRE(Contains(h, "namespace clio::beta {"));
  REQUIRE(Contains(h, "kMaxMethodId = 0;"));
  REQUIRE_FALSE(Contains(h, "// Inherited methods"));

  const std::string s = ReadAll(source);
  // Without kRestart the Restart override must not be generated.
  REQUIRE_FALSE(Contains(s, "void Runtime::Restart("));
  REQUIRE(Contains(s, "namespace clio::beta {"));
}

TEST_CASE("RefreshRepo - unwritable output directory fails",
          "[cli][refresh_repo]") {
  if (geteuid() == 0) {
    INFO("running as root; read-only directory not enforceable - skipping");
    return;
  }

  TempDir tmp;
  WriteFile(tmp.path / "clio_repo.yaml", "modules:\n  - gamma\n");
  WriteFile(tmp.path / "gamma" / "clio_mod.yaml",
            "module_name: gamma\nkCreate: 0\n");

  // Pre-create the methods-header autogen dir read-only so the ofstream
  // open fails and GenerateChiModFiles throws.
  fs::path autogen_dir =
      tmp.path / "gamma" / "include" / "clio" / "gamma" / "autogen";
  fs::create_directories(autogen_dir);
  fs::permissions(autogen_dir,
                  fs::perms::owner_read | fs::perms::owner_exec,
                  fs::perm_options::replace);

  REQUIRE(CallRefreshRepo({tmp.path.string()}) == 1);

  fs::permissions(autogen_dir, fs::perms::owner_all,
                  fs::perm_options::replace);
}

//==============================================================================
// CliDispatch — clio_run binary argument parsing / usage paths (subprocess).
// None of these start a daemon: they exercise help text, unknown commands,
// and argument validation errors that return before any runtime init.
//==============================================================================

TEST_CASE("CliDispatch - top level usage and unknown commands", "[cli][dispatch]") {
  SECTION("no arguments prints usage, exit 1");
  REQUIRE(RunCli("") == 1);

  SECTION("--help and -h exit 0");
  REQUIRE(RunCli("--help") == 0);
  REQUIRE(RunCli("-h") == 0);

  SECTION("unknown command exits 1");
  REQUIRE(RunCli("frobnicate") == 1);
}

TEST_CASE("CliDispatch - legacy nested forms", "[cli][dispatch]") {
  SECTION("runtime with no subcommand exits 1");
  REQUIRE(RunCli("runtime") == 1);

  SECTION("runtime with unknown subcommand exits 1");
  REQUIRE(RunCli("runtime bogus") == 1);

  SECTION("runtime start/restart --help exit 0");
  REQUIRE(RunCli("runtime start --help") == 0);
  REQUIRE(RunCli("runtime restart --help") == 0);

  SECTION("repo with no subcommand exits 1");
  REQUIRE(RunCli("repo") == 1);

  SECTION("repo with unknown subcommand exits 1");
  REQUIRE(RunCli("repo bogus") == 1);

  SECTION("repo refresh with no args prints usage, exit 1");
  REQUIRE(RunCli("repo refresh") == 1);
}

TEST_CASE("CliDispatch - command help and argument errors", "[cli][dispatch]") {
  SECTION("start/restart help exits 0; unknown arg exits 1");
  REQUIRE(RunCli("start --help") == 0);
  REQUIRE(RunCli("start -h") == 0);
  REQUIRE(RunCli("start --bogus") == 1);
  REQUIRE(RunCli("restart --help") == 0);
  REQUIRE(RunCli("restart --bogus") == 1);

  SECTION("refresh dispatch: no args / bad repo exit 1");
  REQUIRE(RunCli("refresh") == 1);
  REQUIRE(RunCli("refresh /nonexistent/clio_cli_test_repo") == 1);

  SECTION("migrate help exits 0; missing required args exit 1");
  REQUIRE(RunCli("migrate --help") == 0);
  REQUIRE(RunCli("migrate -h") == 0);
  REQUIRE(RunCli("migrate") == 1);
  REQUIRE(RunCli("migrate --pool-id 200.0") == 1);
  REQUIRE(RunCli("migrate --pool-id 200.0 --container-id 1") == 1);

  SECTION("monitor parse errors return 0 (help) without contacting runtime");
  REQUIRE(RunCli("monitor --help") == 0);
  REQUIRE(RunCli("monitor -h") == 0);
  REQUIRE(RunCli("monitor --interval 0") == 0);   // interval < 1 rejected
  REQUIRE(RunCli("monitor --interval") == 0);     // missing argument
  REQUIRE(RunCli("monitor --bogus") == 0);        // unknown option

  SECTION("compose usage paths");
  REQUIRE(RunCli("compose") == 1);                // no args
  REQUIRE(RunCli("compose --help") == 0);
  REQUIRE(RunCli("compose -h") == 0);
  REQUIRE(RunCli("compose --unregister") == 1);   // missing config path
}

//==============================================================================
// CliClientError — client-mode commands with no runtime available.
// CLIO_WAIT_SERVER=0 (set by RunCli) makes WaitForLocalServer fail
// immediately, so each command takes its "Failed to initialize Clio
// client" path without ever waiting on or spawning a daemon.
//==============================================================================

TEST_CASE("CliClientError - commands fail fast without a runtime",
          "[cli][client_error]") {
  SECTION("stop fails when no runtime is up");
  REQUIRE(RunCli("stop") == 1);

  SECTION("legacy runtime stop fails the same way");
  REQUIRE(RunCli("runtime stop") == 1);

  SECTION("monitor --once fails after successful arg parse");
  REQUIRE(RunCli("monitor --once --json --verbose --interval 2") == 1);

  SECTION("migrate with full args fails at client init");
  REQUIRE(RunCli("migrate --pool-id 200.0 --container-id 1 --node-id 2") == 1);

  SECTION("compose with a config path fails at client init");
  TempDir tmp;
  WriteFile(tmp.path / "compose.yaml", "compose:\n  pools: []\n");
  REQUIRE(RunCli("compose " + (tmp.path / "compose.yaml").string()) == 1);
  REQUIRE(RunCli("compose --unregister " +
                 (tmp.path / "compose.yaml").string()) == 1);
}

//==============================================================================
// CliRuntime — monitor/compose/migrate/stop against a LIVE runtime daemon.
// The daemon is spawned via clio::run::test::RuntimeServer (posix_spawn of the
// clio_run binary) and is guaranteed to be reaped: RuntimeServer::Stop()
// runs from the destructor even when a REQUIRE throws, and every client
// subprocess has a hard RunCliTimed kill deadline so nothing can hang.
//==============================================================================

TEST_CASE("CliRuntime - client commands against a live runtime",
          "[cli][runtime_cmds]") {
  constexpr unsigned kPort = 10601;  // off 10500/10599 used by other tests

  TempDir tmp;
  WriteFile(tmp.path / "no_compose.yaml", "runtime:\n  num_threads: 1\n");
  WriteFile(tmp.path / "compose.yaml",
            "compose:\n"
            "  - mod_name: clio_bdev\n"
            "    pool_name: \"ram::cli_test_bdev\"\n"
            "    pool_query: local\n"
            "    pool_id: \"320.0\"\n"
            "    bdev_type: ram\n"
            "    capacity: 16MB\n"
            "    restart: true\n");

  // Clients must actually wait for the daemon (RunCli uses 0 = fail fast).
  setenv("CLIO_WAIT_SERVER", "15", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));  // also exports CLIO_PORT / CLIO_REPO_PATH
  REQUIRE(server.WaitForReady());

  SECTION("monitor --once renders the worker stats table");
  REQUIRE(RunCliTimed({"monitor", "--once"}, 60) == 0);

  SECTION("monitor --once --json --verbose renders JSON");
  REQUIRE(RunCliTimed({"monitor", "--once", "--json", "--verbose"}, 60) == 0);

  SECTION("compose: missing config file fails after client init");
  REQUIRE(RunCliTimed({"compose", (tmp.path / "nope.yaml").string()}, 60) == 1);

  SECTION("compose: config without compose section completes");
  // The config manager may fall back to previously saved pool configs, so
  // this can legitimately exit 0 (pools found) or 1 (none) — what matters
  // for this test is that the command completes without crashing.
  int no_compose_rc =
      RunCliTimed({"compose", (tmp.path / "no_compose.yaml").string()}, 60);
  REQUIRE((no_compose_rc == 0 || no_compose_rc == 1));

  SECTION("compose: creates the pool (and saves its restart config)");
  REQUIRE(RunCliTimed({"compose", (tmp.path / "compose.yaml").string()},
                      60) == 0);

  SECTION("compose --unregister destroys the pool again");
  REQUIRE(RunCliTimed({"compose", "--unregister",
                       (tmp.path / "compose.yaml").string()}, 60) == 0);

  SECTION("migrate runs the full admin-client path");
  // Pool 320.0 was just destroyed; the admin handler treats the unknown
  // container as a no-op migration. Either exit code is fine — what matters
  // is that the command completes (no hang, no crash).
  int migrate_rc = RunCliTimed({"migrate", "--pool-id", "320.0",
                                "--container-id", "0", "--node-id", "1"}, 60);
  REQUIRE((migrate_rc == 0 || migrate_rc == 1));

  SECTION("stop shuts the runtime down cleanly");
  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);

  // The daemon should now be gone; Stop() also reaps it (and would kill a
  // survivor, so nothing is ever left running).
  // Coverage-instrumented builds flush .gcda files on exit, which can take
  // tens of seconds for the daemon — wait generously before declaring it hung.
  for (int i = 0; i < 600 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  REQUIRE_FALSE(server.IsRunning());
  server.Stop();
}

SIMPLE_TEST_MAIN()
