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

std::string LineDirective(const SourceManager &sm, SourceLocation loc) {
  const SourceLocation e = sm.getExpansionLoc(loc);
  std::string out = "\n#line ";
  out += std::to_string(sm.getSpellingLineNumber(e));
  out += " \"";
  out += sm.getFilename(e).str();
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
  const FunctionDecl *callee = nullptr;  /**< non-null iff a child call */
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
    for (auto &kv : fns_) suspending_.insert(Canonical(kv.first));
    for (auto &kv : fns_) Prepare(kv.second);
    CheckUnwrappedCalls(tu);
    for (auto &kv : fns_) Emit(kv.second);
  }

  unsigned Errors() const { return errors_; }
  std::size_t FunctionCount() const { return fns_.size(); }

 private:
  const FunctionDecl *Canonical(const FunctionDecl *fd) const {
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
    return sm_.getFilename(sm_.getExpansionLoc(loc)).starts_with(g_root);
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
          continue;
        }
      }
      Error(s.operand->getBeginLoc(),
            "CO_AWAIT operand must be a call to a suspending function or "
            "'<awaiter>.Take()'");
    }
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
        if (t->suspending_.count(fd->getCanonicalDecl()) == 0) return true;
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
      return Text(sm_, lo_, tsi->getTypeLoc().getSourceRange());
    }
    const Expr *init = vd->getInit();
    if (init == nullptr) {
      Error(vd->getLocation(), "'auto' declaration without an initializer");
      return "int";
    }
    // If the initializer is an await, decltype the *rewritten* form, because
    // the written form (CO_AWAIT(...)) will not exist in the output.
    for (const AwaitSite &s : fi.awaits) {
      if (s.marker == init->IgnoreImplicit() ||
          s.marker == init->IgnoreImpCasts()) {
        return s.callee != nullptr
                   ? "decltype(" + CallWithCtx(s) + ")"
                   : "clio::co::AwaiterResult<decltype(" +
                         Text(sm_, lo_, s.awaiter->getSourceRange()) + ")>";
      }
    }
    return "decltype(" + Text(sm_, lo_, init->getSourceRange()) + ")";
  }

  /** The child call with the context threaded in (edit E2). */
  std::string CallWithCtx(const AwaitSite &s) {
    std::string t = Text(sm_, lo_, s.operand->getSourceRange());
    const std::size_t close = t.rfind(')');
    if (close == std::string::npos) {
      Error(s.operand->getBeginLoc(), "cannot find the call's closing paren");
      return t;
    }
    const auto *call = cast<CallExpr>(s.operand);
    t.insert(close, call->getNumArgs() == 0 ? "_cy" : ", _cy");
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
      if (!p->getName().empty()) names.push_back(p->getName().str());
    }
    for (const VarDecl *vd : fi.hoisted) names.push_back(vd->getName().str());
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

    for (const VarDecl *vd : fi.hoisted) {
      if (vd->getType()->isReferenceType()) {
        Error(vd->getLocation(),
              "a reference declared across a CO_AWAIT cannot be hoisted");
      }
    }

    EmitSignature(fi);
    EmitPrologue(fi, body);
    EmitHoistStrips(fi);
    for (const AwaitSite &s : fi.awaits) EmitAwait(fi, s);
    EmitEpilogue(fi, body);
    ++rewritten_;
  }

  /** E1 */
  void EmitSignature(const FnInfo &fi) {
    const FunctionTypeLoc ftl =
        fi.fn->getFunctionTypeLoc();
    if (ftl.isNull()) {
      Error(fi.fn->getLocation(), "cannot locate the parameter list");
      return;
    }
    const std::string add =
        fi.fn->getNumParams() == 0 ? "clio::co::Ctx &_cy" : ", clio::co::Ctx &_cy";
    if (!InsertBefore(rw_, ftl.getRParenLoc(), add)) {
      Error(fi.fn->getLocation(), "could not rewrite the signature");
    }
  }

  /** E3 */
  void EmitPrologue(const FnInfo &fi, const CompoundStmt *body) {
    std::string out = "\n/* clio-coroc: generated */\n";
    for (const VarDecl *vd : fi.hoisted) {
      // Value-initialized, not vacuous: the dispatch shape defeats definite-
      // assignment analysis, so a bare declaration draws -Wmaybe-uninitialized
      // even where it is provably assigned. Legal here because nothing jumps
      // over these -- they sit above the switch.
      out += HoistedType(fi, vd) + " " + vd->getName().str() + "{};\n";
    }
    std::vector<std::string> packs;
    const std::vector<std::string> base = SaveList(fi);
    for (const AwaitSite &s : fi.awaits) {
      if (s.callee != nullptr) continue;
      const std::string aw = AwaiterName(s);
      out += "decltype(" + Text(sm_, lo_, s.awaiter->getSourceRange()) + ") " +
             aw + "{};\n";
    }
    // Frame size: the max over this function's suspend points of its live set.
    for (const AwaitSite &s : fi.awaits) {
      std::vector<std::string> types;
      if (s.callee == nullptr) types.push_back("decltype(" + AwaiterName(s) + ")");
      for (const std::string &n : base) types.push_back("decltype(" + n + ")");
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
      const std::string name = vd->getName().str();
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

    if (s.callee != nullptr) {
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
      const std::string path = fe->getName().str();
      if (!g_root.empty() && path.rfind(g_root, 0) != 0) continue;
      std::string rel = path;
      if (!g_root.empty()) {
        rel = path.substr(g_root.size());
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
