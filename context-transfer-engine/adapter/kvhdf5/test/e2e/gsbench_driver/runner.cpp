#include "runner.h"

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <dirent.h>
#include <fcntl.h>
#include <glob.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace gsbench {

namespace {

// The async-VOL toolchain, ported verbatim from run_campaign.sh's VOL_LD/VOL_PLUGIN/
// VOL_CONNECTOR. Only the hdf5_async arm gets these on its LD path -- the
// ENABLE_WRITE_MEMCPY=OFF ("nomemcpy") build, because the stock /opt/vol-async livelocks in
// Argobots at scale (see scaling_campaign/README.md finding 6).
constexpr const char* kVolLd = "/opt/vol-async-nomemcpy/lib:/opt/hdf5ts/lib:/opt/argobots/lib";
constexpr const char* kVolPlugin = "/opt/vol-async-nomemcpy/lib";
constexpr const char* kVolConnector = "async under_vol=0;under_info={}";

bool ProcIsLiveKvhdf5(const std::string& pid_dir) {
    std::ifstream comm(pid_dir + "/comm");
    std::string name;
    if (!comm || !std::getline(comm, name)) return false;
    if (name != "kvhdf5_e2e_test") return false;  // comm truncates to 15 chars, no trailing 's'
    std::ifstream stat(pid_dir + "/stat");
    std::string line;
    if (!stat || !std::getline(stat, line)) return false;
    // Format: "<pid> (<comm>) <state> ...". The comm field can itself contain ')', so find the
    // LAST ')' before reading the state char that follows it.
    auto rparen = line.rfind(')');
    if (rparen == std::string::npos || rparen + 2 >= line.size()) return false;
    const char state = line[rparen + 2];
    return state != 'Z';  // ignore zombies: pid 1 in this container does not reap them
}

bool AnyLiveKvhdf5Proc() {
    DIR* d = opendir("/proc");
    if (!d) return false;
    bool found = false;
    struct dirent* de;
    while ((de = readdir(d)) != nullptr) {
        if (de->d_name[0] < '0' || de->d_name[0] > '9') continue;
        if (ProcIsLiveKvhdf5(std::string("/proc/") + de->d_name)) { found = true; break; }
    }
    closedir(d);
    return found;
}

void SleepMs(long ms) {
    struct timespec ts{ms / 1000, (ms % 1000) * 1'000'000L};
    nanosleep(&ts, nullptr);
}

std::string DirName(const std::string& path) {
    auto pos = path.find_last_of('/');
    return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}

}  // namespace

ChildEnvPlan ComputeChildEnv(const Arm& arm, const std::vector<EnvOverride>& extra_env,
                              const std::string& bin_dir) {
    ChildEnvPlan plan;
    // async-VOL isolation: HDF5_VOL_CONNECTOR is process-global, so every arm except
    // hdf5_async explicitly unsets it (else it silently attaches to the plain hdf5 arm's
    // file too -- see run_threeway_bench.sh's run_arm comment).
    if (arm.is_hdf5_async) {
        plan.set.push_back({"LD_LIBRARY_PATH", std::string(kVolLd) + ":" + bin_dir});
        plan.set.push_back({"HDF5_PLUGIN_PATH", kVolPlugin});
        plan.set.push_back({"HDF5_VOL_CONNECTOR", kVolConnector});
    } else {
        plan.unset.push_back("HDF5_VOL_CONNECTOR");
        plan.set.push_back({"LD_LIBRARY_PATH", bin_dir});
    }
    for (const auto& e : arm.env) plan.set.push_back(e);
    for (const auto& e : extra_env) plan.set.push_back(e);
    return plan;
}

void StateReset(bool hard, const std::string& scratch_dir) {
    std::system("pkill -9 -x kvhdf5_e2e_test >/dev/null 2>&1");
    for (int i = 0; i < 60 && AnyLiveKvhdf5Proc(); ++i) SleepMs(250);

    glob_t g{};
    if (glob("/dev/shm/chi_*", 0, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; ++i) unlink(g.gl_pathv[i]);
    }
    globfree(&g);

    if (hard && !scratch_dir.empty()) {
        const std::string bdev = scratch_dir + "/clio_bdev.dat";
        const std::string raw_out = scratch_dir + "/raw_out";
        const std::string hdf5_out = scratch_dir + "/hdf5_out";
        unlink(bdev.c_str());
        std::system(("rm -rf '" + raw_out + "' '" + hdf5_out + "' >/dev/null 2>&1").c_str());
        mkdir(raw_out.c_str(), 0755);
        mkdir(hdf5_out.c_str(), 0755);
    }
}

ChildOutcome RunOneChild(const RunnerOptions& opts, const Arm& arm, const ChildEnvPlan& plan,
                          const std::string& log_path) {
    ChildOutcome out;
    out.arm = arm.name;
    out.log_path = log_path;

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        out.exited_ok = false;
        return out;
    }

    const std::string bin_dir = DirName(opts.bin_path);
    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return out;
    }

    if (pid == 0) {
        // ---- child ----
        setsid();  // own process group, so a timeout can killpg() any grandchildren too
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);

        for (const auto& k : plan.unset) unsetenv(k.c_str());
        for (const auto& e : plan.set) setenv(e.key.c_str(), e.value.c_str(), 1);

        if (!bin_dir.empty()) chdir(bin_dir.c_str());

        char bin_arg[PATH_MAX];
        std::snprintf(bin_arg, sizeof(bin_arg), "%s", opts.bin_path.c_str());
        char sel_arg[256];
        std::snprintf(sel_arg, sizeof(sel_arg), "%s", arm.selector.c_str());
        char* child_argv[] = {bin_arg, sel_arg, nullptr};
        execvp(opts.bin_path.c_str(), child_argv);
        // execvp only returns on failure.
        std::fprintf(stderr, "gsbench_run: execvp(%s) failed: %s\n", opts.bin_path.c_str(),
                     std::strerror(errno));
        _exit(127);
    }

    // ---- parent ----
    close(pipefd[1]);
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::seconds(opts.timeout_sec);
    const auto kill_grace = std::chrono::seconds(15);  // hard cap on drain-after-SIGKILL
    bool killed = false;
    std::chrono::steady_clock::time_point kill_time{};
    std::string captured;
    char buf[65536];

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (!killed && now >= deadline) {
            killed = true;
            kill_time = now;
            out.timed_out = true;
            killpg(pid, SIGKILL);
        }
        if (killed && now - kill_time > kill_grace) break;  // give up draining

        struct pollfd pfd{pipefd[0], POLLIN, 0};
        const int poll_ms = 200;  // recheck the deadline at this granularity
        const int rc = poll(&pfd, 1, poll_ms);
        if (rc > 0 && (pfd.revents & (POLLIN | POLLHUP))) {
            ssize_t n;
            while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) captured.append(buf, n);
            if (n == 0) break;  // EOF: child (and any process-group siblings) closed the pipe
            if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) break;
        } else if (rc < 0 && errno != EINTR) {
            break;
        }
    }
    close(pipefd[0]);

    int status = 0;
    // The child may already be reaped by the time we get here if it exited on its own; a
    // timed-out/killed child is waited here too (setsid() means no other reaper races us).
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        out.exit_code = WEXITSTATUS(status);
        out.exited_ok = (out.exit_code == 0) && !out.timed_out;
    } else if (WIFSIGNALED(status)) {
        out.exit_code = -WTERMSIG(status);
        out.exited_ok = false;
    }

    out.result = ExtractResultLine(captured);

    if (!log_path.empty()) {
        std::ofstream log(log_path, std::ios::trunc);
        if (log) log << captured;
    }

    return out;
}

}  // namespace gsbench
