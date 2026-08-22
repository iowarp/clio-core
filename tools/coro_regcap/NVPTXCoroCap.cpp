// NVPTXCoroCap -- automatic, per-kernel register ceiling for GPU coroutine
// kernels. Nothing in this tree writes a register count or a launch bound by
// hand, and no ceiling is shared between two kernels that did not measure the
// same.
//
// WHY A CEILING IS NEEDED AT ALL. NVPTX has no tail calls, so CoroSplit cannot
// give each resume segment its own function: they are merged, and the register
// allocator then takes the LIVENESS UNION across every suspend point. The
// result is a register count that does not depend on the body -- an empty
// co_return-only coroutine measures the same ~193 registers as the full paging
// machinery. It is allocator laziness, not demand: under any explicit ceiling
// ptxas compresses the same code spill-free (64 regs, LOCAL:0, +1.4%
// instructions). Every other knob is a null result -- ptxas -O0/-O1/-O2/-O3,
// clang -O1/-O2/-Os, machine scheduler off, __noinline__ fences. Only a
// ceiling moves it. The real fix is per-segment allocation in the NVPTX
// backend; until then, this.
//
// WHY THE NUMBER IS MEASURED, NOT CHOSEN. A single constant is wrong for the
// same reason -maxrregcount is wrong: it clamps every kernel in the module to
// whatever the worst one needed. So the ceiling comes from a PROBE BUILD -- the
// same target compiled once with the pass disabled -- whose per-kernel register
// counts are read back out of the cubin with cuobjdump. A kernel that already
// fits the occupancy budget is left completely alone and keeps its natural
// allocation; only kernels above the budget are stamped, and each is stamped
// against its own measurement. See cmake/ClioCoroRegCap.cmake for the wiring
// and tools/coro_regcap/derive_caps.py for the derivation.
//
// WHAT IT STAMPS. At pipeline start -- before CoroSplit erases the evidence --
// it finds functions carrying `presplitcoroutine`, propagates "executes a
// coroutine" up the call graph, and stamps "nvvm.maxnreg" on exactly the
// ptx_kernel entries that reach one. Plain kernels are never touched, even when
// the probe measured them over budget: their allocation is honest. A kernel
// that already declares an explicit nvvm.maxnreg keeps its own setting.
//
// Options (set by the CMake module, never by a user):
//   -mllvm -clio-coro-cap-file=<path>   per-kernel caps, "<mangled> <n>" lines
//   -mllvm -clio-coro-maxnreg=<n>       fallback for kernels absent from the
//                                       file (0 = leave them alone)
//
// Usage: -fpass-plugin=/path/to/libNVPTXCoroCap.so (applies to the device
// compilation of every -x cuda TU; host modules are skipped by triple).

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

// The llvm-22-dev snapshot packages omit llvm/Passes/PassPlugin.h, so the
// (ABI-stable since LLVM 8) plugin entry contract is declared here verbatim.
#define LLVM_PLUGIN_API_VERSION 2
extern "C" {
struct PassPluginLibraryInfo {
  uint32_t APIVersion;
  const char *PluginName;
  const char *PluginVersion;
  void (*RegisterPassBuilderCallbacks)(llvm::PassBuilder &);
};
}

#include <cstdlib>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>

using namespace llvm;

namespace {

cl::opt<std::string> CoroCapFile(
    "clio-coro-cap-file", cl::Hidden, cl::init(""),
    cl::desc("Per-kernel register caps measured by the probe build: one "
             "'<mangled name> <maxnreg>' pair per line."));

cl::opt<unsigned> CoroMaxNReg(
    "clio-coro-maxnreg", cl::Hidden, cl::init(0),
    cl::desc("Register cap for coroutine kernels not named in the cap file. "
             "0 (default) leaves them at their natural allocation."));

/// Parse the probe build's derived caps. Malformed lines are skipped rather
/// than fatal: a stale or truncated cap file must degrade to "no ceiling",
/// which is slow, and never to a wrong ceiling, which is a silent miscompile
/// of the occupancy the kernel was validated at.
StringMap<unsigned> loadCaps(StringRef Path) {
  StringMap<unsigned> Caps;
  if (Path.empty())
    return Caps;
  std::ifstream In(Path.str());
  if (!In) {
    errs() << "NVPTXCoroCap: cannot open cap file '" << Path
           << "'; coroutine kernels keep their natural allocation\n";
    return Caps;
  }
  std::string Line;
  while (std::getline(In, Line)) {
    if (Line.empty() || Line[0] == '#')
      continue;
    std::istringstream LS(Line);
    std::string Name;
    unsigned Cap = 0;
    if (!(LS >> Name >> Cap) || Cap == 0)
      continue;
    Caps[Name] = Cap;
  }
  return Caps;
}

struct NVPTXCoroCapPass : PassInfoMixin<NVPTXCoroCapPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    Triple T(M.getTargetTriple());
    if (!T.isNVPTX())
      return PreservedAnalyses::all();

    const StringMap<unsigned> Caps = loadCaps(CoroCapFile);
    const unsigned Fallback = CoroMaxNReg;
    if (Caps.empty() && Fallback == 0)
      return PreservedAnalyses::all();

    // Seed with every function the front end marked as a coroutine. This
    // runs before CoroEarly/CoroSplit, so the attribute is still present.
    SmallPtrSet<const Function *, 32> Tainted;
    std::deque<const Function *> Work;
    for (const Function &F : M) {
      if (F.hasFnAttribute(Attribute::PresplitCoroutine)) {
        Tainted.insert(&F);
        Work.push_back(&F);
      }
    }
    if (Work.empty())
      return PreservedAnalyses::all();

    // Propagate "executes a coroutine" to transitive callers. Direct calls
    // only: at this pipeline point the coroutine ramp is always invoked
    // directly (the resume-via-frame-pointer indirection is created later,
    // by CoroSplit).
    while (!Work.empty()) {
      const Function *F = Work.front();
      Work.pop_front();
      for (const User *U : F->users()) {
        if (const auto *CB = dyn_cast<CallBase>(U)) {
          const Function *Caller = CB->getFunction();
          if (Caller && Tainted.insert(Caller).second)
            Work.push_back(Caller);
        }
      }
    }

    bool Changed = false;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      if (F.getCallingConv() != CallingConv::PTX_Kernel)
        continue;
      if (!Tainted.count(&F))
        continue;         // plain kernel: its allocation is honest, leave it
      if (F.hasFnAttribute("nvvm.maxnreg"))
        continue;         // an explicit per-kernel setting wins

      // Per-kernel measurement first; the fallback only covers kernels the
      // probe build did not report (a new kernel, or a renamed one).
      unsigned Cap = Fallback;
      auto It = Caps.find(F.getName());
      if (It != Caps.end())
        Cap = It->second;
      if (Cap == 0)
        continue;         // measured to already fit the budget

      F.addFnAttr("nvvm.maxnreg", std::to_string(Cap));
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "NVPTXCoroCap", "2.0",
          [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(NVPTXCoroCapPass());
                });
          }};
}
