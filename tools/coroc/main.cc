/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * clio-coroc -- the CO_AWAIT transpiler.
 *
 * Reads ordinary C++ containing CO_AWAIT(...) and writes ordinary C++
 * containing none: each suspending function becomes a group-scoped state
 * machine over clio::co::Frame. The output has no coroutines, no function
 * pointers, no recursion and no vendor tokens, so nvcc, hipcc, clang-CUDA and
 * DPC++ all compile it identically.
 *
 * Design: $HOME/coroutines.md. The five edits (section 3.2):
 *
 *   E1  append `clio::co::Ctx &_cy` to every suspending function's signature
 *   E2  append `_cy` to every call to a suspending function
 *   E3  insert the Frame, the hoisted declarations and the dispatching switch
 *   E4  insert `case N:` + replay + park guard at each CO_AWAIT
 *   E5  close the switch and call Done() before every return
 *
 * WHY A REWRITER AND NOT AN AST PRINTER. The output is the input plus patches,
 * so the parse only has to be good enough to answer questions -- if clang
 * mismodels an attribute or a vendor builtin, the output has still lost
 * nothing. That is also what lets the tool parse with a small prelude instead
 * of CUDA, HIP or SYCL headers, and be coupled to none of their versions.
 */
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

llvm::cl::OptionCategory g_cat("clio-coroc");
llvm::cl::opt<std::string> g_root(
    "rewrite-root", llvm::cl::desc("only rewrite files under this prefix"),
    llvm::cl::cat(g_cat));
llvm::cl::opt<std::string> g_mirror(
    "mirror-to", llvm::cl::desc("write rewritten files under this directory"),
    llvm::cl::cat(g_cat));
llvm::cl::opt<bool> g_verbose("verbose", llvm::cl::desc("report what changed"),
                              llvm::cl::cat(g_cat));

constexpr char kAwaitMarker[] = "clio::co::AwaitMark";

/* ===================================================================== */
/* Source-text helpers                                                   */
/* ===================================================================== */

/** The text the user actually wrote for this range.
 *
 *  Spelling locations, not expansion: the operand of CO_AWAIT is a MACRO
 *  ARGUMENT, and its expansion range is the whole `CO_AWAIT(...)` invocation.
 *  Asking for the expansion here yields the marker back instead of the thing
 *  being awaited -- which is how this first showed up, as generated code
 *  containing `CO_AWAIT(Fetch(c, page), _cy)`. A macro argument's spelling
 *  location points at the token in the invocation, which is exactly the text
 *  we want to copy. */
std::string Text(const SourceManager &sm, const LangOptions &lo,
                 SourceRange sr) {
  SourceLocation b = sr.getBegin();
  SourceLocation e = sr.getEnd();
  if (b.isMacroID()) b = sm.getSpellingLoc(b);
  if (e.isMacroID()) e = sm.getSpellingLoc(e);
  return Lexer::getSourceText(CharSourceRange::getTokenRange(b, e), sm, lo)
      .str();
}

/** Rewriter refuses macro-body locations, and this design touches a macro on
 *  nearly every edit, so every edit goes through here and is checked. A silent
 *  no-op is the commonest way to lose a day in a LibTooling tool. */
bool Replace(Rewriter &rw, SourceRange sr, const std::string &text) {
  return !rw.ReplaceText(rw.getSourceMgr().getExpansionRange(sr), text);
}

bool InsertBefore(Rewriter &rw, SourceLocation loc, const std::string &text) {
  return !rw.InsertTextBefore(rw.getSourceMgr().getExpansionLoc(loc), text);
}

bool InsertAfterTok(Rewriter &rw, SourceLocation loc, const std::string &text) {
  return !rw.InsertTextAfterToken(rw.getSourceMgr().getExpansionLoc(loc), text);
}

/** End of the token at `loc`. A SourceRange ends at the START of its last
 *  token, which is the second thing that bites in a rewriting tool. */
SourceLocation TokEnd(const SourceManager &sm, const LangOptions &lo,
                      SourceLocation loc) {
  return Lexer::getLocForEndOfToken(sm.getExpansionLoc(loc), 0, sm, lo);
}

/** The `;` closing a statement, if there is one.
 *
 *  The end must come from getExpansionRange, not from getExpansionLoc of the
 *  end token: a statement ending inside a macro argument has an end location in
 *  the macro body, and mapping that token alone lands on the START of the
 *  expansion. Every CO_AWAIT statement ends inside a macro, so this is not an
 *  edge case here -- it is every case. */
SourceLocation SemiAfter(const SourceManager &sm, const LangOptions &lo,
                         const Stmt *s, const Stmt *anchor) {
  // Scan forward a few tokens rather than assuming the semicolon is the very
  // next one: findNextToken lexes from the END of the token AT the location,
  // so whether csr.getEnd() is the last token's start or one past it changes
  // which token comes back. Scanning is indifferent to that.
  // Anchor on the CO_AWAIT itself, not on the enclosing statement. A
  // statement whose range ENDS inside a macro has an unreliable expansion end
  // -- observed skipping past its own semicolon and landing the park guard in
  // the middle of the next line's for-init. In every accepted form the
  // CO_AWAIT is the last thing before the semicolon, so it is the right anchor.
  (void)s;
  SourceLocation p = sm.getExpansionRange(anchor->getSourceRange()).getEnd();
  for (int i = 0; i < 8; ++i) {
    auto tok = Lexer::findNextToken(p, sm, lo);
    if (!tok) break;
    if (tok->is(tok::semi)) return tok->getLocation();
    if (tok->getLocation() == p) break;
    p = tok->getLocation();
  }
  return SourceLocation();
}

/** A path with every separator turned into a forward slash.
 *
 * Clang reports paths with the host's native separator. On Windows that makes
 * a raw path two different bugs at once: a prefix comparison against a
 * slash-spelled --rewrite-root never matches, and inside an emitted `#line`
 * directive the path is a string literal full of escape sequences, where
 * a drive-relative "\\U" is a hard error rather than a warning. Forward
 * slashes are accepted everywhere we target, so one normalisation fixes both.
 *
 * @param path the path as clang spelled it
 * @return the same path with every native separator replaced by '/'
 */
std::string Slashed(llvm::StringRef path) {
  std::string out = path.str();
  std::replace(out.begin(), out.end(), '\\', '/');
  return out;
}

std::string LineDirective(const SourceManager &sm, SourceLocation loc) {
  const SourceLocation e = sm.getExpansionLoc(loc);
  std::string out = "\n#line ";
  out += std::to_string(sm.getSpellingLineNumber(e));
  out += " \"";
  out += Slashed(sm.getFilename(e));
  out += "\"\n";
  return out;
}

/* ===================================================================== */
/* Per-function facts                                                    */
/* ===================================================================== */

struct AwaitSite {
  const BinaryOperator *marker = nullptr;  /**< the whole (AwaitMark(), e) */
  const Expr *operand = nullptr;           /**< e */
  const Expr *awaiter = nullptr;           /**< the awaiter, for a leaf await */
  const Stmt *stmt = nullptr;
  const FunctionDecl *callee = nullptr;  /**< the child, when it resolved */
  /** True iff this await is a call to another suspending function rather
   *  than a leaf awaiter. Separate from `callee` because inside a class
   *  template the callee does not resolve to a declaration at all, and the
   *  emitter only ever needs the distinction -- never the decl. */
  bool nested = false;
  unsigned state = 0;
};

struct FnInfo {
  const FunctionDecl *fn = nullptr;
  std::vector<AwaitSite> awaits;
  std::vector<const VarDecl *> hoisted;
  std::map<const VarDecl *, const DeclStmt *> decl_stmt;
  std::vector<const ReturnStmt *> returns;
};

/** A CO_AWAIT is a comma whose left operand calls the marker and nothing else.
 *  Fully qualified, so a user function named AwaitMark cannot be mistaken. */
const BinaryOperator *AsAwaitMarker(const Stmt *s) {
  const auto *bo = dyn_cast<BinaryOperator>(s);
  if (bo == nullptr || bo->getOpcode() != BO_Comma) return nullptr;
  const auto *lhs = dyn_cast<CallExpr>(bo->getLHS()->IgnoreImplicit());
  if (lhs == nullptr) return nullptr;
  const FunctionDecl *fd = lhs->getDirectCallee();
  if (fd == nullptr) return nullptr;
  if (fd->getQualifiedNameAsString() != kAwaitMarker) return nullptr;
  return bo;
}

/** Strip parens/implicit nodes down to the expression the user wrote. */
const Expr *Bare(const Expr *e) {
  return e == nullptr ? nullptr : e->IgnoreParenImpCasts();
}

/**
 * The expression a variable is really initialized FROM.
 *
 * `auto h = CO_AWAIT(v.Hold(...))` does not give the marker back from
 * getInit(): copy-initializing a class type wraps it in a construct
 * expression, a temporary binding and a cleanup scope, none of which
 * IgnoreParenImpCasts removes. Only a scalar init is bare, which is why the
 * demo workload -- whose awaits return a scalar or are explicitly typed --
 * never showed this, and the first real benchmark did, as a hoisted
 * declaration spelled `decltype((AwaitMark(), (__VA_ARGS__)))`.
 *
 * @param e the initializer as written on the VarDecl
 * @return the underlying expression, with those wrappers peeled off
 */
const Expr *Initializer(const Expr *e) {
  const Expr *cur = Bare(e);
  for (int guard = 0; guard < 8 && cur != nullptr; ++guard) {
    if (const auto *ewc = dyn_cast<ExprWithCleanups>(cur)) {
      cur = Bare(ewc->getSubExpr());
      continue;
    }
    if (const auto *bt = dyn_cast<CXXBindTemporaryExpr>(cur)) {
      cur = Bare(bt->getSubExpr());
      continue;
    }
    if (const auto *mt = dyn_cast<MaterializeTemporaryExpr>(cur)) {
      cur = Bare(mt->getSubExpr());
      continue;
    }
    // An elidable copy/move around the real initializer.
    if (const auto *ce = dyn_cast<CXXConstructExpr>(cur)) {
      if (ce->getNumArgs() == 1) {
        cur = Bare(ce->getArg(0));
        continue;
      }
    }
    break;
  }
  return cur;
}

/* --------------------------------------------------------------------- */
/* Pass 1: which functions suspend. Local and explicit -- a property of    */
/* writing CO_AWAIT in your own body, never inferred transitively, so      */
/* there is no whole-program fixed point to compute.                       */
/* --------------------------------------------------------------------- */

class MarkerScan : public RecursiveASTVisitor<MarkerScan> {
 public:
  explicit MarkerScan(std::map<const FunctionDecl *, FnInfo> *out) : out_(out) {}

  bool TraverseFunctionDecl(FunctionDecl *fd) { return Enter(fd); }
  bool TraverseCXXMethodDecl(CXXMethodDecl *fd) { return Enter(fd); }

  bool VisitBinaryOperator(BinaryOperator *bo) {
    if (fn_ == nullptr) return true;
    const BinaryOperator *m = AsAwaitMarker(bo);
    if (m == nullptr) return true;
    FnInfo &fi = (*out_)[fn_];
    fi.fn = fn_;
    AwaitSite site;
    site.marker = m;
    site.operand = Bare(m->getRHS());
    fi.awaits.push_back(site);
    return true;
  }

 private:
  bool Enter(FunctionDecl *fd) {
    if (!fd->doesThisDeclarationHaveABody()) return true;
    FunctionDecl *saved = fn_;
    fn_ = fd;
    TraverseStmt(fd->getBody());
    fn_ = saved;
    return true;
  }
  std::map<const FunctionDecl *, FnInfo> *out_;
  FunctionDecl *fn_ = nullptr;
};

/* --------------------------------------------------------------------- */
/* Pass 2: the hoisting rule.                                             */
/*                                                                        */
/*   A declaration is hoisted iff it is in a block -- or a for/if/while/   */
/*   switch init -- that lexically contains a CO_AWAIT.                    */
/*                                                                        */
/* It moves the DECLARATION, not the variable: the hoisted variable is an  */
/* ordinary automatic, so loop counters stay in registers. It exists only  */
/* so the dispatch can jump past them without entering the scope of a      */
/* variable with non-vacuous initialization, which is ill-formed.          */
/* --------------------------------------------------------------------- */

class ScopeWalk : public RecursiveASTVisitor<ScopeWalk> {
 public:
  explicit ScopeWalk(FnInfo *fi) : fi_(fi) {}

  bool TraverseCompoundStmt(CompoundStmt *s) {
    Open();
    const bool r = RecursiveASTVisitor::TraverseCompoundStmt(s);
    Close();
    return r;
  }
  bool TraverseForStmt(ForStmt *s) {
    Open();
    const bool r = RecursiveASTVisitor::TraverseForStmt(s);
    Close();
    return r;
  }
  bool TraverseIfStmt(IfStmt *s) {
    Open();
    const bool r = RecursiveASTVisitor::TraverseIfStmt(s);
    Close();
    return r;
  }
  bool TraverseWhileStmt(WhileStmt *s) {
    Open();
    const bool r = RecursiveASTVisitor::TraverseWhileStmt(s);
    Close();
    return r;
  }
  bool TraverseSwitchStmt(SwitchStmt *s) {
    Open();
    const bool r = RecursiveASTVisitor::TraverseSwitchStmt(s);
    Close();
    return r;
  }

  /* A NESTED FUNCTION BODY IS NOT THIS FUNCTION'S BODY.
   *
   * A local class or a lambda declared inside a suspending function has
   * returns and declarations of its own, and they belong to IT. Walking into
   * them put `_cy_f.Done();` inside a local struct's operator[], where
   * `_cy_f` is not in scope -- which nvcc answered with a segfault and clang
   * with "reference to local variable declared in enclosing function".
   * kmeans has exactly that shape: a PageAt shim closing over the held page.
   */
  bool TraverseLambdaExpr(LambdaExpr *) { return true; }
  bool TraverseCXXRecordDecl(CXXRecordDecl *) { return true; }
  bool TraverseFunctionDecl(FunctionDecl *) { return true; }
  bool TraverseCXXMethodDecl(CXXMethodDecl *) { return true; }

  bool VisitDeclStmt(DeclStmt *s) {
    if (stack_.empty()) return true;
    for (const Decl *d : s->decls()) {
      if (const auto *vd = dyn_cast<VarDecl>(d)) {
        stack_.back().decls.push_back(vd);
        fi_->decl_stmt[vd] = s;
      }
    }
    return true;
  }

  bool VisitReturnStmt(ReturnStmt *s) {
    fi_->returns.push_back(s);
    return true;
  }

  /** An await marks every enclosing scope, so every declaration in any of
   *  them is hoisted -- the rule, verbatim. */
  bool VisitBinaryOperator(BinaryOperator *bo) {
    if (AsAwaitMarker(bo) == nullptr) return true;
    for (Scope &s : stack_) s.awaits = true;
    return true;
  }

 private:
  struct Scope {
    std::vector<const VarDecl *> decls;
    bool awaits = false;
  };
  void Open() { stack_.emplace_back(); }
  void Close() {
    if (stack_.back().awaits) {
      for (const VarDecl *vd : stack_.back().decls) fi_->hoisted.push_back(vd);
    }
    stack_.pop_back();
  }
  FnInfo *fi_;
  std::vector<Scope> stack_;
};

/* ===================================================================== */
/* The transpiler                                                        */
/* ===================================================================== */

class Transpiler {
 public:
  Transpiler(ASTContext &ctx, Rewriter &rw, DiagnosticsEngine &de)
      : ctx_(ctx), rw_(rw), de_(de), sm_(ctx.getSourceManager()),
        lo_(ctx.getLangOpts()) {}

  void Run(TranslationUnitDecl *tu) {
    MarkerScan scan(&fns_);
    scan.TraverseDecl(tu);
    for (auto &kv : fns_) {
      suspending_.insert(Canonical(kv.first));
      suspending_names_.insert(kv.first->getNameAsString());
    }
    for (auto &kv : fns_) Prepare(kv.second);
    for (auto &kv : fns_) RenameCollidingHoists(kv.second);
    CheckUnwrappedCalls(tu);
    // Emitting over a rejected program dereferences the very fields the
    // diagnostic said were missing -- an unusable await site has neither a
    // callee nor an awaiter -- so a source error must stop the pass, not
    // merely be counted alongside a crash.
    if (errors_ != 0) return;
    for (auto &kv : fns_) Emit(kv.second);
  }

  unsigned Errors() const { return errors_; }
  std::size_t FunctionCount() const { return fns_.size(); }

 private:
  /** The one declaration that stands for `fd` and every instantiation of it.
   *
   * A suspending function that is a member of a class template is SCANNED as
   * the template pattern -- that is the only place its body, and so its
   * CO_AWAIT, is written -- but every call to it resolves to an
   * instantiation, whose canonical decl is a different node. Without the hop
   * through the pattern the call looks like a call to an ordinary function
   * and E2 never appends the context, which is how `DeviceVector<T>::Fetch`
   * defeated the first version of this tool.
   *
   * Rewriting is unaffected: the pattern is the only text on disk, so one
   * patch to it serves all instantiations.
   *
   * @param fd any declaration of the function
   * @return the canonical declaration of its template pattern, or of itself
   */
  const FunctionDecl *Canonical(const FunctionDecl *fd) const {
    if (const FunctionDecl *pattern = fd->getTemplateInstantiationPattern()) {
      fd = pattern;
    }
    return fd->getCanonicalDecl();
  }

  void Error(SourceLocation loc, const std::string &msg) {
    const unsigned id =
        de_.getCustomDiagID(DiagnosticsEngine::Error, "clio-coroc: %0");
    de_.Report(sm_.getExpansionLoc(loc), id) << msg;
    ++errors_;
  }

  bool InRoot(SourceLocation loc) const {
    if (g_root.empty()) return true;
    return llvm::StringRef(Slashed(sm_.getFilename(sm_.getExpansionLoc(loc))))
        .starts_with(Slashed(g_root));
  }

  /* ---- preparation -------------------------------------------------- */

  void Prepare(FnInfo &fi) {
    ScopeWalk walk(&fi);
    walk.TraverseStmt(const_cast<Stmt *>(fi.fn->getBody()));

    unsigned next = 1;
    for (AwaitSite &s : fi.awaits) {
      s.state = next++;
      s.stmt = EnclosingStatement(s.marker);
      if (s.stmt == nullptr) {
        Error(s.marker->getBeginLoc(),
              "CO_AWAIT must be in statement position (v1 restriction)");
        continue;
      }
      if (const auto *mc = dyn_cast<CXXMemberCallExpr>(s.operand)) {
        // A leaf await is spelled `CO_AWAIT(<awaiter>.Take())`: the awaiter is
        // the object the Take() is called on, which is exactly what the
        // generated code needs for Ready() and Tag().
        if (mc->getMethodDecl() != nullptr &&
            mc->getMethodDecl()->getName() == "Take") {
          s.awaiter = Bare(mc->getImplicitObjectArgument());
          continue;
        }
      }
      if (const auto *call = dyn_cast<CallExpr>(s.operand)) {
        const FunctionDecl *cal = call->getDirectCallee();
        if (cal != nullptr && suspending_.count(Canonical(cal)) != 0) {
          s.callee = cal;
          s.nested = true;
          continue;
        }
      }
      // DEPENDENT CONTEXTS. Inside a class template nothing above resolves:
      // `w.Take()` and `CoBeginFetch(...)` are both parsed against a type
      // that does not exist yet, so there is no CXXMemberCallExpr and no
      // direct callee -- only a name. Matching on the name is enough, and
      // is all the emitter needs, because it uses `callee` purely as a
      // boolean and `awaiter` purely for its source text.
      //
      // This is what DeviceVector<T> is made of: every verb it exposes is a
      // member of a class template calling other members of the same class
      // template, so without this branch the transpiler cannot process the
      // one header the benchmarks are written against.
      if (DependentAwait(s)) continue;
      Error(s.operand->getBeginLoc(),
            "CO_AWAIT operand must be a call to a suspending function or "
            "'<awaiter>.Take()'");
    }
  }

  /**
   * Classify an await whose operand is dependent, by name.
   *
   * @param s the await site, filled in on success
   * @return true if the operand was recognised
   */
  bool DependentAwait(AwaitSite &s) {
    const auto *call = dyn_cast<CallExpr>(s.operand);
    if (call == nullptr) return false;
    const Expr *callee = call->getCallee()->IgnoreParenImpCasts();

    // `<awaiter>.Take()` or `this->Verb(...)` on a dependent object.
    if (const auto *dm = dyn_cast<CXXDependentScopeMemberExpr>(callee)) {
      const std::string name = dm->getMember().getAsString();
      if (name == "Take") {
        if (dm->isImplicitAccess()) return false;  // no object to name
        s.awaiter = Bare(dm->getBase());
        return true;
      }
      if (suspending_names_.count(name) != 0) {
        s.nested = true;
        return true;
      }
      return false;
    }

    // An unqualified call to a member of the current instantiation, which
    // clang leaves as an unresolved overload set until the type is known.
    std::string name;
    if (const auto *um = dyn_cast<UnresolvedMemberExpr>(callee)) {
      name = um->getMemberName().getAsString();
    } else if (const auto *ul = dyn_cast<UnresolvedLookupExpr>(callee)) {
      name = ul->getName().getAsString();
    } else {
      return false;
    }
    if (name == "Take") return false;  // a leaf await needs its object
    if (suspending_names_.count(name) == 0) return false;
    s.nested = true;
    return true;
  }


  /**
   * Give every hoisted declaration a name unique within its function.
   *
   * Hoisting FLATTENS scopes. Two declarations of `p` in sibling blocks
   * shadow nothing -- neither is in scope where the other is -- and become
   * a redefinition once both are lifted to function scope. The source is
   * not wrong; the transform is, so the transform fixes it.
   *
   * Renaming is exact rather than textual: clang knows which DeclRefExprs
   * resolve to this VarDecl, so each is rewritten and nothing that merely
   * shares the spelling is touched. lammps_md needed it for p, pg and
   * pub_need; grayscott for two page holds both called h.
   */
  void RenameCollidingHoists(FnInfo &fi) {
    std::map<std::string, int> seen;
    for (const VarDecl *vd : fi.hoisted) {
      const std::string name = vd->getName().str();
      const int n = ++seen[name];
      if (n == 1) continue;
      const std::string fresh = name + "_cy" + std::to_string(n);
      renamed_[vd] = fresh;
      // NOT at the declaration: EmitHoistStrips already rewrites the range
      // that covers the name, and two Rewriter edits over one range leave
      // whichever lost silently undone.
      struct V : RecursiveASTVisitor<V> {
        const VarDecl *target;
        std::string fresh;
        Transpiler *t;
        bool VisitDeclRefExpr(DeclRefExpr *e) {
          // Once only. A declaration can be reached through more than one
          // function's body -- a lambda, a local class -- and renaming an
          // already-renamed use produced `pub_need_cy2_cy2`.
          if (e->getDecl() == target && t->renamed_uses_.insert(e).second) {
            Replace(t->rw_, SourceRange(e->getLocation(), e->getLocation()),
                    fresh);
          }
          return true;
        }
      } v;
      v.target = vd;
      v.fresh = fresh;
      v.t = this;
      v.TraverseStmt(const_cast<Stmt *>(fi.fn->getBody()));
    }
  }

  /** The name a hoisted declaration ended up with. */
  std::string HoistName(const VarDecl *vd) const {
    auto it = renamed_.find(vd);
    return it == renamed_.end() ? vd->getName().str() : it->second;
  }

  /** Climb to the statement that directly contains this expression. */
  const Stmt *EnclosingStatement(const Stmt *s) {
    const Stmt *cur = s;
    for (int guard = 0; guard < 64; ++guard) {
      const auto parents = ctx_.getParents(*cur);
      if (parents.empty()) return nullptr;
      const Stmt *ps = parents[0].get<Stmt>();
      if (ps == nullptr) {
        // A DeclStmt's initializer hangs off a Decl, not a Stmt: go up through
        // the VarDecl to the DeclStmt that holds it.
        if (const auto *vd = parents[0].get<VarDecl>()) {
          const auto pd = ctx_.getParents(*vd);
          if (!pd.empty()) {
            if (const Stmt *ds = pd[0].get<Stmt>()) return ds;
          }
        }
        return nullptr;
      }
      if (isa<CompoundStmt>(ps)) return cur;
      cur = ps;
    }
    return nullptr;
  }

  /** R1: a call to a suspending function that is not wrapped in CO_AWAIT has
   *  no resume point, so its park would be silently lost. This is the check
   *  that does not exist in a hand-decorated world. */
  void CheckUnwrappedCalls(TranslationUnitDecl *tu) {
    struct V : RecursiveASTVisitor<V> {
      Transpiler *t;
      std::set<const CallExpr *> ok;
      bool VisitBinaryOperator(BinaryOperator *bo) {
        if (const BinaryOperator *m = AsAwaitMarker(bo)) {
          if (const auto *inner = dyn_cast<CallExpr>(Bare(m->getRHS())))
            ok.insert(inner);
        }
        return true;
      }
      bool VisitCallExpr(CallExpr *c) {
        const FunctionDecl *fd = c->getDirectCallee();
        if (fd == nullptr) return true;
        if (t->suspending_.count(t->Canonical(fd)) == 0) return true;
        if (ok.count(c) != 0) return true;
        pending.push_back(c);
        return true;
      }
      std::vector<const CallExpr *> pending;
    } v;
    v.t = this;
    v.TraverseDecl(tu);
    for (const CallExpr *c : v.pending) {
      if (v.ok.count(c) != 0) continue;
      if (!InRoot(c->getBeginLoc())) continue;
      Error(c->getBeginLoc(),
            "call to suspending function '" +
                c->getDirectCallee()->getNameAsString() +
                "' must be wrapped in CO_AWAIT(...)");
    }
  }

  /* ---- type spelling, without ever naming a type --------------------- */

  /** The type to give a hoisted declaration. Either the written type, copied
   *  verbatim, or a decltype built from the initializer -- so a dependent type
   *  never has to be printed, which is what makes templates a non-issue. */
  std::string HoistedType(const FnInfo &fi, const VarDecl *vd) {
    const TypeSourceInfo *tsi = vd->getTypeSourceInfo();
    const bool written_auto =
        tsi != nullptr && !tsi->getTypeLoc().getAs<AutoTypeLoc>().isNull();
    if (!written_auto) {
      // AN ARRAY DECLARATOR'S TypeLoc SPANS THE NAME. `T x[4]` is written
      // with the extent after the identifier, so the source range of the
      // type runs from `T` to `]` and copying it verbatim re-emits the name
      // -- observed as `T hz[4] hz[4]{};`. Descend to the element type and
      // let ArrayExtent put the brackets back.
      // EVERYTHING BEFORE THE NAME. A TypeLoc's source range leaves out
      // leading qualifiers, so `const float **s_sp0` came back as
      // `float **` and the hoist silently dropped the const -- which nvcc
      // then reported at the assignment, not at the declaration. Taking
      // the characters from the declaration's start up to the identifier
      // keeps the qualifiers and the stars, and leaves the extent to
      // ArrayExtent.
      auto it = fi.decl_stmt.find(vd);
      if (it != fi.decl_stmt.end()) {
        const CharSourceRange csr = CharSourceRange::getCharRange(
            sm_.getExpansionLoc(it->second->getBeginLoc()),
            sm_.getExpansionLoc(vd->getLocation()));
        std::string txt = Lexer::getSourceText(csr, sm_, lo_).str();
        // A TOP-LEVEL const has to go. The hoist declares the variable
        // above the switch and the original site becomes an assignment to
        // it, which a const object will not accept -- "expression must be
        // a modifiable lvalue", pointing at the assignment rather than at
        // the declaration that caused it. A const POINTEE is untouched:
        // `const float *p` stays exactly that, and only `const u64 n`
        // loses its qualifier.
        // For an ARRAY the qualifier sits on the ELEMENT type, so
        // `const double vals[4]` is not locally const-qualified and the
        // drop below has to be told to look inside. Its elements are
        // assigned one at a time by the strip, so the const must go.
        bool drop = vd->getType().isLocalConstQualified();
        if (const auto *at = ctx_.getAsArrayType(vd->getType())) {
          drop = drop || at->getElementType().isConstQualified();
        }
        if (drop) {
          // the LAST one: in `const float *const p` the top-level
          // qualifier is the second, and dropping the first leaves
          // `float *const p`, still not assignable.
          const std::size_t c = txt.rfind("const");
          if (c != std::string::npos) {
            txt.erase(c, 5);
            while (c < txt.size() && txt[c] == ' ') txt.erase(c, 1);
          }
        }
        if (!txt.empty()) return txt;
      }
      TypeLoc tl = tsi->getTypeLoc();
      for (;;) {
        ArrayTypeLoc atl = tl.getAs<ArrayTypeLoc>();
        if (atl.isNull()) break;
        tl = atl.getElementLoc();
      }
      return Text(sm_, lo_, tl.getSourceRange());
    }
    const Expr *init = vd->getInit();
    if (init == nullptr) {
      Error(vd->getLocation(), "'auto' declaration without an initializer");
      return "int";
    }
    // If the initializer is an await, decltype the *rewritten* form, because
    // the written form (CO_AWAIT(...)) will not exist in the output.
    const Expr *from = Initializer(init);
    for (const AwaitSite &s : fi.awaits) {
      if (s.marker == from || s.marker == init->IgnoreImplicit() ||
          s.marker == init->IgnoreImpCasts()) {
        return s.nested
                   ? "decltype(" + CallWithCtx(s) + ")"
                   : "clio::co::AwaiterResult<decltype(" +
                         Text(sm_, lo_, s.awaiter->getSourceRange()) + ")>";
      }
    }
    return "decltype(" + Text(sm_, lo_, init->getSourceRange()) + ")";
  }

  /**
   * The `[N]...` part of an array declarator, or empty.
   *
   * A declaration's TYPE and its DECLARATOR are different things, and the
   * hoist rebuilds the declaration from the type text plus the name -- which
   * silently drops the extent, turning `PageRef<T> hz[4]` into
   * `PageRef<T> hz`. The element type is what HoistedType returns, so the
   * extents have to come back here.
   *
   * @param vd the hoisted declaration
   * @return "[4]", "[2][3]", ... or "" for a non-array
   */
  std::string ArrayExtent(const VarDecl *vd) const {
    std::string out;
    QualType t = vd->getType();
    while (const auto *at = ctx_.getAsConstantArrayType(t)) {
      out += "[" + llvm::toString(at->getSize(), 10, false) + "]";
      t = at->getElementType();
    }
    return out;
  }

  /** The child call with the context threaded in (edit E2). */
  std::string CallWithCtx(const AwaitSite &s) {
    std::string t = Text(sm_, lo_, s.operand->getSourceRange());
    const std::size_t open = t.find('(');
    if (open == std::string::npos) {
      Error(s.operand->getBeginLoc(), "cannot find the call's opening paren");
      return t;
    }
    const auto *call = cast<CallExpr>(s.operand);
    t.insert(open + 1, call->getNumArgs() == 0 ? "_cy" : "_cy, ");
    return t;
  }

  /* ---- the save list ------------------------------------------------- */

  /** Conservative by design: every parameter and every hoisted variable. A
   *  superset of the live set is always correct, and costs frame bytes at a
   *  park and nothing on the fast path, because "locals stay in registers" is
   *  a property of the emission rather than of liveness precision. */
  std::vector<std::string> SaveList(const FnInfo &fi) {
    std::vector<std::string> names;
    for (const ParmVarDecl *p : fi.fn->parameters()) {
      if (p->getName().empty()) continue;
      // A PARAMETER PACK IS NOT ONE NAME. `Args... args` saved as `args`
      // is a use of an unexpanded pack, which is a hard error -- so it
      // carries its ellipsis into both the Push/Pop argument list and the
      // PackBytes type list, where the fold does the right thing for each.
      // DeviceVector's multi-range Fetch is variadic and lammps_md calls it
      // with two ranges, so this is not a corner case.
      names.push_back(p->isParameterPack() ? p->getName().str() + "..."
                                           : p->getName().str());
    }
    for (const VarDecl *vd : fi.hoisted) {
      // A re-derived reference is not carried across the park; see EmitBody.
      if (vd->getType()->isReferenceType()) continue;
      names.push_back(HoistName(vd));
    }
    return names;
  }

  static std::string Join(const std::vector<std::string> &v,
                          const std::string &prefix = "") {
    std::string out = prefix;
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (i != 0 || !prefix.empty()) out += ", ";
      out += v[i];
    }
    return out;
  }

  std::string AwaiterName(const AwaitSite &s) {
    return "_cy_a" + std::to_string(s.state);
  }

  std::string ParkReturn(const FnInfo &fi) {
    return fi.fn->getReturnType()->isVoidType() ? "return;" : "return {};";
  }

  /* ---- emission ------------------------------------------------------ */

  void Emit(FnInfo &fi) {
    if (!InRoot(fi.fn->getBeginLoc())) return;
    const auto *body = dyn_cast_or_null<CompoundStmt>(fi.fn->getBody());
    if (body == nullptr) return;

    /* A REFERENCE IS RE-DERIVED, NOT SAVED.
     *
     * It cannot be hoisted the usual way: a hoist declares the variable
     * value-initialized above the switch and assigns it later, and a
     * reference can be neither. But it does not need saving either -- it
     * is a NAME for something else, and the something else is still there
     * after the park. So its declaration moves above the switch WITH its
     * initializer and is re-executed on every entry.
     *
     * That is exactly right for the case that forced it: lammps_md's
     * CLIO_SHARED_PERSIST binds a reference into the block's shared arena,
     * and CLIO_COROC_RUN has already called PersistRestore by the time the
     * resume gets here -- so re-deriving the name is not merely allowed,
     * it is the only correct thing, because the arena is at a fresh
     * address each launch.
     *
     * The initializer is re-evaluated, so it must not depend on a hoisted
     * local, whose value at re-entry is whatever Pop last restored. Every
     * use so far derives from a parameter or a global.
     */
    for (const VarDecl *vd : fi.hoisted) {
      if (!vd->getType()->isReferenceType()) continue;
      if (vd->getInit() == nullptr) {
        Error(vd->getLocation(), "a reference across a CO_AWAIT needs an "
                                 "initializer to be re-derived from");
      }
    }

    EmitSignature(fi);
    EmitPrologue(fi, body);
    EmitHoistStrips(fi);
    for (const AwaitSite &s : fi.awaits) EmitAwait(fi, s);
    EmitEpilogue(fi, body);
    ++rewritten_;
  }

  /** E1
   *
   * THE CONTEXT GOES FIRST, not last.
   *
   * Appending it is the obvious choice and it is wrong, because a variadic
   * suspending function has its pack last too: `Fetch(u64 gen, Args... args)`
   * called as `Fetch(gen, off, count, _cy)` binds the context INTO the pack
   * and forwards it to GatherRanges as if it were a range bound. nvcc
   * reports only "no instance matches the argument list", well away from the
   * cause. DeviceVector's multi-range fetch is exactly that shape and
   * lammps_md calls it with two planes.
   *
   * First is unambiguous for every arity, variadic or not.
   */
  void EmitSignature(const FnInfo &fi) {
    const FunctionTypeLoc ftl =
        fi.fn->getFunctionTypeLoc();
    if (ftl.isNull()) {
      Error(fi.fn->getLocation(), "cannot locate the parameter list");
      return;
    }
    const std::string add =
        fi.fn->getNumParams() == 0 ? "clio::co::Ctx &_cy" : "clio::co::Ctx &_cy, ";
    if (!InsertAfterTok(rw_, ftl.getLParenLoc(), add)) {
      Error(fi.fn->getLocation(), "could not rewrite the signature");
    }
  }

  /** E3 */
  void EmitPrologue(const FnInfo &fi, const CompoundStmt *body) {
    std::string out = "\n/* clio-coroc: generated */\n";
    // ORDER MATTERS AND DECLARATION ORDER IS THE WRONG ONE. A hoisted `auto`
    // gets a decltype of its initializer, and that initializer can name
    // another hoisted variable -- `auto h = CO_AWAIT(v.Hold(z * plane, ...))`
    // inside `for (u64 z = ...)` produces `decltype(... z ...) h;` above
    // `u64 z;`. Emitting the ones whose type is written first, then the
    // deduced ones, fixes every case that can arise: a written type names no
    // hoisted variable, and a deduced one can only name variables the source
    // had already declared, which are written-type or deduced-from-earlier.
    std::vector<const VarDecl *> written;
    std::vector<const VarDecl *> deduced;
    for (const VarDecl *vd : fi.hoisted) {
      const TypeSourceInfo *tsi = vd->getTypeSourceInfo();
      const bool is_auto =
          tsi != nullptr && !tsi->getTypeLoc().getAs<AutoTypeLoc>().isNull();
      (is_auto ? deduced : written).push_back(vd);
    }
    // Re-derived names come first: they are what the value hoists may be
    // initialised from, never the other way round.
    for (const VarDecl *vd : fi.hoisted) {
      if (!vd->getType()->isReferenceType() || vd->getInit() == nullptr) {
        continue;
      }
      // A macro-expanded declaration must be moved as the INVOCATION, not
      // as the declarator: the declarator's spelling lives in the macro
      // DEFINITION, where the type and the name are still parameters.
      // lammps_md writes CLIO_SHARED_PERSIST(MdTables, s_tbl), whose
      // declarator spells `Type &name = ...`.
      SourceRange r = vd->getSourceRange();
      if (r.getBegin().isMacroID()) {
        const CharSourceRange csr = sm_.getExpansionRange(r);
        out += Lexer::getSourceText(csr, sm_, lo_).str() + ";\n";
      } else {
        out += Text(sm_, lo_, r) + ";\n";
      }
    }
    for (const std::vector<const VarDecl *> *group : {&written, &deduced}) {
      for (const VarDecl *vd : *group) {
        if (vd->getType()->isReferenceType()) continue;
        // Value-initialized, not vacuous: the dispatch shape defeats
        // definite-assignment analysis, so a bare declaration draws
        // -Wmaybe-uninitialized even where it is provably assigned. Legal
        // here because nothing jumps over these -- they sit above the switch.
        out += HoistedType(fi, vd) + " " + HoistName(vd) +
               ArrayExtent(vd) + "{};\n";
      }
    }
    std::vector<std::string> packs;
    const std::vector<std::string> base = SaveList(fi);
    for (const AwaitSite &s : fi.awaits) {
      if (s.nested) continue;
      const std::string aw = AwaiterName(s);
      out += "decltype(" + Text(sm_, lo_, s.awaiter->getSourceRange()) + ") " +
             aw + "{};\n";
    }
    // Frame size: the max over this function's suspend points of its live set.
    for (const AwaitSite &s : fi.awaits) {
      std::vector<std::string> types;
      if (!s.nested) types.push_back("decltype(" + AwaiterName(s) + ")");
      for (const std::string &n : base) {
        // `args...` becomes `decltype(args)...`: the ellipsis has to end up
        // OUTSIDE the decltype, or it expands nothing.
        if (n.size() > 3 && n.compare(n.size() - 3, 3, "...") == 0) {
          types.push_back("decltype(" + n.substr(0, n.size() - 3) + ")...");
        } else {
          types.push_back("decltype(" + n + ")");
        }
      }
      packs.push_back("clio::co::PackBytes<" + Join(types) + ">()");
    }
    if (packs.empty()) packs.push_back("0u");
    out += "constexpr clio::co::u32 _cy_fb = clio::co::MaxOf(" + Join(packs) +
           ");\n";
    out += "clio::co::Frame _cy_f(_cy, _cy_fb);\n";
    out += "switch (_cy_f.Resume()) {\ncase 0:";
    out += LineDirective(sm_, body->getLBracLoc());
    InsertAfterTok(rw_, body->getLBracLoc(), out);
  }

  /** Strip the hoisted declarations at their original sites, leaving the
   *  assignment behind. The declaration moves; the variable does not. */
  void EmitHoistStrips(const FnInfo &fi) {
    for (const VarDecl *vd : fi.hoisted) {
      auto it = fi.decl_stmt.find(vd);
      if (it == fi.decl_stmt.end()) continue;
      const DeclStmt *ds = it->second;
      if (!ds->isSingleDecl()) {
        Error(vd->getLocation(),
              "declare one variable per statement across a CO_AWAIT");
        continue;
      }
      const std::string name = HoistName(vd);
      // Same reason as in HoistedType: the extent sits AFTER the name, so a
      // strip that ends at the name leaves `[4];` behind as a statement.
      // An array has nowhere to put an initializer in the hoisted form
      // anyway, so the whole declaration goes.
      if (vd->getType()->isReferenceType()) {
        // Moved wholesale, initializer and all, so nothing stays behind.
        Replace(rw_, ds->getSourceRange(), "(void)" + name + ";");
        continue;
      }
      // AN ARRAY IS NOT ASSIGNABLE. The strip normally turns
      // `T x = init;` into `x = init;`, but `double vals[4] = {a,b,c,d}`
      // becomes `vals = {a,b,c,d}`, which is neither a modifiable lvalue
      // nor a valid initializer list. Element-wise assignment preserves
      // the semantics exactly and keeps the array in the save list, which
      // re-deriving it above the switch would not.
      if (vd->getType()->isArrayType() && vd->hasInit()) {
        if (const auto *il = dyn_cast<InitListExpr>(vd->getInit())) {
          std::string out;
          for (unsigned i = 0; i < il->getNumInits(); ++i) {
            out += (i != 0 ? " " : "") + name + "[" + std::to_string(i) +
                   "] = " + Text(sm_, lo_, il->getInit(i)->getSourceRange()) +
                   ";";
          }
          Replace(rw_, ds->getSourceRange(), out);
          continue;
        }
      }
      if (vd->getType()->isArrayType() && !vd->hasInit()) {
        Replace(rw_, ds->getSourceRange(), "(void)" + name + ";");
        continue;
      }
      const SourceRange strip(ds->getBeginLoc(), vd->getLocation());
      // With an initializer this leaves `x = init;`. Without one it leaves
      // `(void)x;`, which is a valid statement and keeps the line count.
      Replace(rw_, strip, vd->hasInit() ? name : "(void)" + name);
    }
  }

  /** E4 */
  void EmitAwait(const FnInfo &fi, const AwaitSite &s) {
    if (s.stmt == nullptr) return;
    const std::vector<std::string> base = SaveList(fi);
    const std::string N = std::to_string(s.state);
    const std::string ret = ParkReturn(fi);
    std::string pro;
    std::string epi;

    if (s.nested) {
      const std::string save = Join(base);
      pro = "\n[[fallthrough]];\ncase " + N + ":\n" +
            "if (_cy_f.Replaying()) _cy_f.Pop(" + save + ");\n";
      pro += LineDirective(sm_, s.stmt->getBeginLoc());
      epi = "\nif (_cy.Parked()) { _cy_f.Push(" + N +
            (save.empty() ? "" : ", " + save) + "); " + ret + " }\n";
      epi += LineDirective(sm_, s.stmt->getBeginLoc());
      Replace(rw_, s.marker->getSourceRange(), CallWithCtx(s));
    } else {
      const std::string aw = AwaiterName(s);
      std::vector<std::string> with = base;
      with.insert(with.begin(), aw);
      const std::string save = Join(with);
      pro = "\n" + aw + " = " +
            Text(sm_, lo_, s.awaiter->getSourceRange()) + ";\n" +
            "[[fallthrough]];\ncase " + N + ":\n" +
            "if (_cy_f.Replaying()) _cy_f.Pop(" + save + ");\n" +
            // The label sits BEFORE the test, so a resume re-evaluates
            // readiness and parks again if the host has not finished yet.
            "if (_cy.Any(!" + aw + ".Ready())) { _cy_f.Push(" + N + ", " +
            save + "); _cy.SetWaitTag(" + aw + ".Tag()); " + ret + " }\n";
      pro += LineDirective(sm_, s.stmt->getBeginLoc());
      Replace(rw_, s.marker->getSourceRange(), aw + ".Take()");
    }

    // A DeclStmt whose variable was hoisted is already being replaced starting
    // at its first token, so the prologue is folded into that same edit rather
    // than inserted at the same location (two edits at one point are not
    // ordered by the Rewriter).
    const VarDecl *hoisted_here = HoistedDeclOf(fi, s.stmt);
    if (hoisted_here != nullptr) {
      const auto it = fi.decl_stmt.find(hoisted_here);
      const DeclStmt *ds = it->second;
      const std::string name = hoisted_here->getName().str();
      Replace(rw_, SourceRange(ds->getBeginLoc(), hoisted_here->getLocation()),
              pro + (hoisted_here->hasInit() ? name : "(void)" + name));
    } else {
      InsertBefore(rw_, s.stmt->getBeginLoc(), pro);
    }

    if (!epi.empty()) {
      const SourceLocation semi = SemiAfter(sm_, lo_, s.stmt, s.marker);
      if (semi.isValid()) {
        InsertAfterTok(rw_, semi, epi);
      } else {
        Error(s.stmt->getEndLoc(),
              "CO_AWAIT statement must end in a semicolon (v1 restriction)");
      }
    }
  }

  const VarDecl *HoistedDeclOf(const FnInfo &fi, const Stmt *s) {
    const auto *ds = dyn_cast<DeclStmt>(s);
    if (ds == nullptr || !ds->isSingleDecl()) return nullptr;
    const auto *vd = dyn_cast<VarDecl>(ds->getSingleDecl());
    if (vd == nullptr) return nullptr;
    for (const VarDecl *h : fi.hoisted) {
      if (h == vd) return vd;
    }
    return nullptr;
  }

  /** E5: close the switch, and release the frame before every return so that
   *  the caller's cursor still reflects the live chain (invariant I1). */
  void EmitEpilogue(const FnInfo &fi, const CompoundStmt *body) {
    for (const ReturnStmt *r : fi.returns) {
      InsertBefore(rw_, r->getBeginLoc(), "_cy_f.Done(); ");
    }
    // The trailing `return {}` is unreachable when the body always returns --
    // which it does, from inside the switch -- but the compiler cannot see
    // that through the dispatch, and -Wreturn-type would fire on every
    // non-void suspending function. If control ever DID reach here, Done()
    // still runs exactly once, because the inner return path took its own.
    std::string tail = "\n}\n_cy_f.Done();\n";
    if (!fi.fn->getReturnType()->isVoidType()) tail += "return {};\n";
    InsertBefore(rw_, body->getRBracLoc(), tail);
  }

  ASTContext &ctx_;
  Rewriter &rw_;
  DiagnosticsEngine &de_;
  const SourceManager &sm_;
  const LangOptions &lo_;
  std::map<const FunctionDecl *, FnInfo> fns_;
  std::set<const FunctionDecl *> suspending_;
  /** Names of the above, for calls that never resolve to a decl
   *  because they are written inside a class template. */
  std::set<std::string> suspending_names_;
  /** Hoists that had to be renamed to survive the flattening. */
  std::map<const VarDecl *, std::string> renamed_;
  std::set<const DeclRefExpr *> renamed_uses_;
  unsigned errors_ = 0;
  unsigned rewritten_ = 0;
};

/* ===================================================================== */
/* Frontend plumbing                                                     */
/* ===================================================================== */

class Consumer : public ASTConsumer {
 public:
  Consumer(CompilerInstance &ci, Rewriter &rw) : ci_(ci), rw_(rw) {}
  void HandleTranslationUnit(ASTContext &ctx) override {
    Transpiler t(ctx, rw_, ci_.getDiagnostics());
    t.Run(ctx.getTranslationUnitDecl());
    if (g_verbose) {
      llvm::errs() << "clio-coroc: " << t.FunctionCount()
                   << " suspending function(s)\n";
    }
  }

 private:
  CompilerInstance &ci_;
  Rewriter &rw_;
};

class Action : public ASTFrontendAction {
 public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &ci,
                                                 llvm::StringRef) override {
    rw_.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());
    return std::make_unique<Consumer>(ci, rw_);
  }

  /** Write every file the Rewriter actually touched into the mirror tree, at
   *  the same relative path, so a consumer's include path can shadow the
   *  originals without a single source edit anywhere. */
  void EndSourceFileAction() override {
    SourceManager &sm = rw_.getSourceMgr();
    for (auto it = rw_.buffer_begin(); it != rw_.buffer_end(); ++it) {
      auto fe = sm.getFileEntryRefForID(it->first);
      if (!fe) continue;
      const std::string path = Slashed(fe->getName());
      const std::string root = Slashed(g_root);
      if (!root.empty() && path.rfind(root, 0) != 0) continue;
      std::string rel = path;
      if (!root.empty()) {
        rel = path.substr(root.size());
        while (!rel.empty() && rel.front() == '/') rel.erase(rel.begin());
      } else {
        rel = llvm::sys::path::filename(path).str();
      }
      llvm::SmallString<256> out(g_mirror.empty() ? "." : g_mirror.c_str());
      llvm::sys::path::append(out, rel);
      llvm::SmallString<256> dir(out);
      llvm::sys::path::remove_filename(dir);
      if (auto ec = llvm::sys::fs::create_directories(dir)) {
        llvm::errs() << "clio-coroc: cannot create " << dir << ": "
                     << ec.message() << "\n";
        continue;
      }
      std::error_code ec;
      llvm::raw_fd_ostream os(out, ec, llvm::sys::fs::OF_None);
      if (ec) {
        llvm::errs() << "clio-coroc: cannot write " << out << ": "
                     << ec.message() << "\n";
        continue;
      }
      it->second.write(os);
      if (g_verbose) llvm::errs() << "clio-coroc: wrote " << out << "\n";
    }
  }

 private:
  Rewriter rw_;
};

}  // namespace

int main(int argc, const char **argv) {
  auto parser = tooling::CommonOptionsParser::create(argc, argv, g_cat);
  if (!parser) {
    llvm::errs() << toString(parser.takeError());
    return 2;
  }
  tooling::ClangTool tool(parser->getCompilations(),
                          parser->getSourcePathList());
  // A LibTooling binary is not clang, so it does not find clang's builtin
  // headers or the host C++ library by itself. Baking in the paths of the LLVM
  // and GCC it was built against means callers pass only -I and -D -- which is
  // the point: the tool's toolchain and the project's are unrelated.
  tool.appendArgumentsAdjuster(tooling::getInsertArgumentAdjuster(
      {
#ifdef CLIO_COROC_RESOURCE_DIR
          "-resource-dir=" CLIO_COROC_RESOURCE_DIR,
#endif
#ifdef CLIO_COROC_GCC_TOOLCHAIN
          "--gcc-toolchain=" CLIO_COROC_GCC_TOOLCHAIN,
#endif
          "-Wno-unused-value",
      },
      tooling::ArgumentInsertPosition::BEGIN));
  return tool.run(tooling::newFrontendActionFactory<Action>().get());
}
