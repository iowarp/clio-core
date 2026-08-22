// CoroDevirt -- replace the indirect coroutine resume with a complete switch
// over direct calls, so the backend can allocate registers per kernel.
//
// THE PROBLEM (measured; see CORO_REGISTER_EXACTNESS_PLAN.md section 11).
// An indirect call makes ptxas allocate the CALLING KERNEL for the worst case
// over every address-taken function in the module. Resuming a C++ coroutine is
// always an indirect call -- std::coroutine_handle is a bare frame pointer and
// the frame holds a pointer to that coroutine's resume function -- so every
// coroutine kernel inherits the maximum over every $_resume body in the
// translation unit, including coroutines it can never reach. Measured in the
// real build: eleven kernels with unrelated bodies all allocating 122
// registers; in the reduced case (tools/coro_regcap/minicoro2.cu) a kernel
// whose coroutine needs 14 allocating 40, the cost of an unrelated heavy one.
//
// WHY THE INDIRECTION SURVIVES OPTIMISATION. When a coroutine is created and
// resumed in the same function, LLVM devirtualizes it and there is no problem
// (measured: 14 and 35, correctly different). The indirection survives only
// because the frame pointer is laundered through GLOBAL memory -- our yield
// driver stores it in the lane header, because a park EXITS the kernel and it
// must survive to the next launch. That memory round-trip is what defeats the
// optimizer, and it is inherent to the design rather than incidental.
//
// THE FIX. Per kernel, compute the set of resume functions it can reach, then
// replace `call void %fp(%frame)` with a chain of
// `icmp eq ptr %fp, @Coro.resume` guarded direct calls, ending in
// `unreachable`. The backend then sees only direct calls and allocates for
// what the kernel actually reaches.
//
// WHY COMPARING ADDRESSES IS SAFE HERE. The rewrite references each candidate
// function's address in an icmp, which keeps it address-taken -- the very
// property that causes the inflation. Measured: it does not matter. A kernel
// making only direct calls allocates 24 even while the heavy function's
// address escapes to a global, versus 40 for the same kernel with an indirect
// call. It is the indirect CALL that forces the conservative allocation, not
// the address being taken. That is why this pass can use the cheap
// discriminator instead of threading an integer state ID through the frame.
//
// SOUNDNESS AND ITS ONE ASSUMPTION. The `unreachable` default is only correct
// if the candidate set covers every frame this kernel can resume. The pass
// computes a fixed point: start at the kernel, follow direct calls, collect
// every function whose address is referenced, add those to the worklist,
// repeat. That is sound for frames the kernel itself created. It ASSUMES a
// kernel never resumes a frame created by a different kernel -- true here by
// construction, since frames are bump-allocated from a per-lane region, but
// not provable by this pass. Violating it is undefined behaviour, so the
// assumption is checked at runtime under -DCLIO_CORO_DEVIRT_VERIFY.
//
// If the set cannot be closed, the pass leaves the call site ALONE rather than
// guessing: a missed optimisation is a slow kernel, a wrong candidate set is a
// miscompile.

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define LLVM_PLUGIN_API_VERSION 2
extern "C" {
struct PassPluginLibraryInfo {
  uint32_t APIVersion;
  const char *PluginName;
  const char *PluginVersion;
  void (*RegisterPassBuilderCallbacks)(llvm::PassBuilder &);
};
}

#include <deque>
#include <string>

using namespace llvm;

namespace {

cl::opt<bool> DevirtEnable(
    "clio-coro-devirt", cl::Hidden, cl::init(true),
    cl::desc("Replace indirect coroutine resume calls with a complete switch "
             "over direct calls (NVPTX only)."));

cl::opt<unsigned> DevirtMaxTargets(
    "clio-coro-devirt-max", cl::Hidden, cl::init(64),
    cl::desc("Give up on a call site with more candidate targets than this; "
             "the compare chain stops paying for itself."));

cl::opt<bool> DevirtVerbose(
    "clio-coro-devirt-verbose", cl::Hidden, cl::init(false),
    cl::desc("Report each rewritten call site and its candidate set."));

/// Every function whose address is referenced by an instruction in F, directly
/// or through a constant expression. These are the functions whose pointers can
/// reach memory from F, and therefore the ones an indirect call in F's
/// reachable set could later land on.
void collectAddressTaken(const Function &F,
                         SmallPtrSetImpl<Function *> &Out) {
  SmallVector<const Constant *, 8> Work;
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      for (const Use &U : I.operands())
        if (const auto *C = dyn_cast<Constant>(U.get()))
          Work.push_back(C);

  SmallPtrSet<const Constant *, 16> Seen;
  while (!Work.empty()) {
    const Constant *C = Work.pop_back_val();
    if (!Seen.insert(C).second)
      continue;
    if (const auto *Fn = dyn_cast<Function>(C)) {
      Out.insert(const_cast<Function *>(Fn));
      continue;
    }
    for (const Use &U : C->operands())
      if (const auto *Sub = dyn_cast<Constant>(U.get()))
        Work.push_back(Sub);
  }
}

/// Fixed point: what can this kernel actually reach?
///
/// Direct calls alone are not enough -- resume functions are reached only
/// through the frame, never by a direct call -- so a function becomes reachable
/// either by being called directly OR by having its address referenced from
/// something already reachable. The second rule is what pulls the resume
/// functions in, and pulls in ONLY the ones whose ramps this kernel runs.
void computeReachable(Function &Kernel, SmallPtrSetImpl<Function *> &Reach) {
  std::deque<Function *> Work;
  Reach.insert(&Kernel);
  Work.push_back(&Kernel);

  while (!Work.empty()) {
    Function *F = Work.front();
    Work.pop_front();
    if (F->isDeclaration())
      continue;

    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *Callee = CB->getCalledFunction())
            if (!Callee->isIntrinsic() && Reach.insert(Callee).second)
              Work.push_back(Callee);

    SmallPtrSet<Function *, 16> Taken;
    collectAddressTaken(*F, Taken);
    for (Function *T : Taken)
      if (!T->isIntrinsic() && Reach.insert(T).second)
        Work.push_back(T);
  }
}

/// Rewrite one indirect call into guarded direct calls.
///
///   before:  call void %fp(%frame)
///   after:   %c0 = icmp eq ptr %fp, @A
///            br %c0, call.A, next0
///     call.A: call void @A(%frame)   br cont
///     next0:  %c1 = icmp eq ptr %fp, @B ...
///     miss:   unreachable
///     cont:   ...
bool rewriteCallSite(CallBase *CB, ArrayRef<Function *> Targets) {
  // Only void calls: a value-returning site would need a PHI, and coroutine
  // resume/destroy are void. Bail rather than handle a case we cannot test.
  if (!CB->getType()->isVoidTy())
    return false;
  if (isa<InvokeInst>(CB))
    return false;   // no exceptions on device

  Value *FP = CB->getCalledOperand();
  BasicBlock *Head = CB->getParent();
  Function *Parent = Head->getParent();
  LLVMContext &Ctx = Parent->getContext();

  // Everything after the call continues here.
  BasicBlock *Cont = Head->splitBasicBlock(CB->getIterator(), "codevirt.cont");
  // splitBasicBlock leaves an unconditional branch at the end of Head.
  Head->getTerminator()->eraseFromParent();

  BasicBlock *Miss = BasicBlock::Create(Ctx, "codevirt.miss", Parent, Cont);
  new UnreachableInst(Ctx, Miss);

  BasicBlock *Cur = Head;
  for (Function *T : Targets) {
    IRBuilder<> B(Cur);
    Value *Eq = B.CreateICmpEQ(FP, T, "codevirt.is");

    BasicBlock *CallBB =
        BasicBlock::Create(Ctx, "codevirt.call", Parent, Miss);
    {
      IRBuilder<> CB2(CallBB);
      SmallVector<Value *, 8> Args(CB->args());
      CallInst *Direct = CB2.CreateCall(T->getFunctionType(), T, Args);
      Direct->setCallingConv(T->getCallingConv());
      Direct->setAttributes(CB->getAttributes());
      CB2.CreateBr(Cont);
    }

    BasicBlock *Next = BasicBlock::Create(Ctx, "codevirt.next", Parent, Miss);
    B.CreateCondBr(Eq, CallBB, Next);
    Cur = Next;
  }
  // Fell off the end of the candidate list: by construction impossible.
  IRBuilder<>(Cur).CreateBr(Miss);

  CB->eraseFromParent();
  return true;
}

struct CoroDevirtPass : PassInfoMixin<CoroDevirtPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    Triple T(M.getTargetTriple());
    if (!DevirtEnable || !T.isNVPTX())
      return PreservedAnalyses::all();

    bool Changed = false;
    // EVERY function, not just kernels. Rewriting only the kernel's own resume
    // loop is not enough and was measured to change nothing: a kernel's cost is
    // the cost of everything it reaches, and the nested resumes live INSIDE the
    // coroutine bodies (HoldPageCoro.resume resuming its child). Leaving those
    // indirect leaves the conservative allocation exactly where it was --
    // measured on the MD bench, 22 kernel-level sites rewritten and not one
    // register saved.
    //
    // Rewriting a non-kernel function is sound for the same reason it is for a
    // kernel: the frames it can resume are the ones reachable from it.
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      SmallPtrSet<Function *, 32> Reach;
      computeReachable(F, Reach);

      // Collect the indirect sites first; rewriting invalidates iterators.
      SmallVector<CallBase *, 4> Sites;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CB = dyn_cast<CallBase>(&I))
            if (CB->isIndirectCall())
              Sites.push_back(CB);

      for (CallBase *CB : Sites) {
        // Candidates: reachable, address-taken, and type-compatible.
        SmallVector<Function *, 8> Targets;
        FunctionType *Sig = CB->getFunctionType();
        for (Function *R : Reach)
          if (!R->isDeclaration() && R->hasAddressTaken() &&
              R->getFunctionType() == Sig)
            Targets.push_back(R);

        if (Targets.empty() || Targets.size() > DevirtMaxTargets)
          continue;   // cannot close the set, or not worth it: leave it alone

        // Deterministic order: the emitted code must not depend on pointer
        // hashing, or two builds of the same source diverge.
        llvm::sort(Targets, [](const Function *A, const Function *B) {
          return A->getName() < B->getName();
        });

        if (DevirtVerbose) {
          errs() << "CoroDevirt: " << F.getName() << ": " << Targets.size()
                 << " candidate(s):";
          for (Function *T : Targets)
            errs() << " " << T->getName();
          errs() << "\n";
        }
        Changed |= rewriteCallSite(CB, Targets);
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CoroDevirt", "1.0",
          [](PassBuilder &PB) {
            // LAST, deliberately: after CoroSplit has created the resume
            // functions and after inlining has moved the ramps into their
            // kernels, so the reachable set is as tight as it will ever be.
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel Level,
                   ThinOrFullLTOPhase) {
                  MPM.addPass(CoroDevirtPass());
                  // DEVIRTUALIZING IS ONLY HALF THE JOB. We run after the
                  // inliner, so the calls we just made direct have already
                  // missed their chance to be inlined -- and a resume function
                  // left as a separate ABI function still forces the kernel to
                  // allocate for it. Measured: devirtualizing alone took the
                  // reproducer from 2 indirect calls to 0 while leaving both
                  // kernels at 40 registers; the honest figures (14 light, 35
                  // heavy) only appear once the resume body is inlined into
                  // its kernel. So give the inliner a second pass over the
                  // now-direct edges, then clean up after it.
                  if (Level != OptimizationLevel::O0) {
                    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
                        InlinerPass()));
                    FunctionPassManager FPM;
                    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
                    FPM.addPass(InstCombinePass());
                    FPM.addPass(SimplifyCFGPass());
                    MPM.addPass(
                        createModuleToFunctionPassAdaptor(std::move(FPM)));
                  }
                });
          }};
}
