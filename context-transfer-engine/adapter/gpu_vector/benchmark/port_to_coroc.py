"""Port one gpu_vector paged benchmark from co_await onto clio-coroc.

Assembles <dir>/clio_<name>_paged_newcoro.cc out of four pieces that already
exist, so the science is copied rather than retyped:

    prologue   clio_<name>_paged_bench.cc, up to the device-pass guard
    workload   <name>_kernels.h, its CLIO_HAS_YCORO regions, transformed
    launches   cuda/<name>_launch_cuda.cc, its body, transformed
    driver     clio_<name>_paged_bench.cc, the device-pass-guarded main

Usage:  python port.py <benchmark-dir> <name> <namespace>
"""
import io
import os
import re
import sys

VERBS = ("Fetch", "HoldPage", "BeginFlush", "EndFlush", "Flush", "BeginFetch",
         "AwaitFetch", "UpdateRange")


def read(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def write(p, s):
    io.open(p, "w", encoding="utf-8", newline="").write(s)


def match_parens(s, i):
    """Index just past the ')' closing the '(' at s[i]."""
    depth = 0
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced")


def split_args(s):
    """Top-level comma split of an argument list (no surrounding parens)."""
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return out


def rewrite_awaits(src):
    """`co_await <recv>.<Verb>(a)` -> `CO_AWAIT(<recv>.Co<Verb>(a))`, and
    `co_await <Fn>(a)` -> `CO_AWAIT(<Fn>(a))` for a nested suspending call."""
    # A bare suspend on the old leaf awaiter, brace-initialised and with no
    # call at all: `co_await ::clio::run::gpu::YCoroSuspend{0ull};`. Its
    # coroc equivalent is an awaiter with Ready/Tag/Take, so it is rewritten
    # first and the general path below never sees it.
    src = re.sub(r"co_await\s+[\w:]*YCoroSuspend\s*\{([^{}]*)\}",
                 r"CO_AWAIT(::clio::co::YieldOnce{\1}.Take())", src)
    out, i = "", 0
    pat = re.compile(r"co_await\s+")
    while True:
        m = pat.search(src, i)
        if not m:
            out += src[i:]
            return out
        # SKIP PROSE. These headers discuss co_await in their comments --
        # "every co_await on a", "no co_await follows, so shared survives"
        # -- and a match there sends the paren scan below off into the next
        # real statement, which comes back as a stray ')' several lines
        # away. md_kernels.h has two such sentences.
        bol = src.rfind("\n", 0, m.start()) + 1
        if src[bol:m.start()].lstrip().startswith(("//", "*", "/*")):
            out += src[i:m.end()]
            i = m.end()
            continue
        out += src[i:m.start()]
        j = m.end()
        # the callee expression, up to its argument list
        k = src.find("(", j)
        callee = src[j:k]
        end = match_parens(src, k)
        args = src[k + 1:end - 1]
        if "." in callee:
            recv, verb = callee.rsplit(".", 1)
            if verb in VERBS:
                # HoldPage's `write` lost its default: coroc appends the
                # context last, so a defaulted parameter before it is
                # ill-formed. Spell it at the two-argument read sites.
                if verb == "HoldPage" and len(split_args(args)) == 2:
                    args += ", /*write=*/false"
                callee = recv + ".Co" + verb
        out += "CO_AWAIT(" + callee + "(" + args + "))"
        i = end


def rewrite_types(s):
    s = s.replace("gy::YCoroMain ", "void ")
    # YCoroTaskT<V> -> V
    s = re.sub(r"gy::YCoroTaskT<([^<>]*(?:<[^<>]*>)?[^<>]*)>\s*", r"\1 ", s)
    s = s.replace("gy::YCoroTask ", "void ")
    # a value live across a suspend must be trivially copyable (R6)
    s = s.replace("gv::Held<", "gv::PageRef<")
    s = re.sub(r"\bco_return\s*;", "return;", s)
    s = re.sub(r"\bco_return\s+", "return ", s)
    return s


def one_declarator(s):
    """`T a[N], b[N];` and `T a = 1, b = 2;` -> one statement each. The
    transpiler hoists a declaration, and a hoist can only move one name."""
    def fix(m):
        indent, ty, rest = m.group(1), m.group(2), m.group(3)
        parts = split_args(rest)
        if len(parts) < 2:
            return m.group(0)
        return "".join("%s%s %s;\n" % (indent, ty, p.strip()) for p in parts)[:-1]
    # Each declarator may carry its own `*`, and it belongs to the
    # DECLARATOR, not the type: `const float *rp0[9], *rp1[9];` splits into
    # two statements that each keep their star. lammps_md is full of these.
    decl = r"\**\s*[A-Za-z_]\w*(?:\[[^\]]*\])*(?:\s*=\s*[^;,]+)?"
    return re.sub(
        r"^([ \t]+)((?:const )?(?:unsigned |signed )?"
        r"(?:[A-Za-z_][\w:]*(?:<[^;]*?>)?))[ \t]+"
        r"(" + decl + r"(?:\s*,\s*" + decl + r")+)\s*;$",
        fix, s, flags=re.M)


def brace_init(s):
    """`T x{...};` -> `T x = T{...};`. The hoist strip rewrites
    `T x = init;` into `x = init;`; a brace-init becomes `x{init};`, which
    is not an expression."""
    return re.sub(r"^([ \t]+)([A-Za-z_][\w:]*(?:<[^;{]*>)?)\s+([A-Za-z_]\w*)\{([^;{}]*)\};$",
                  r"\1\2 \3 = \2{\4};", s, flags=re.M)


def dedupe_hoists(s):
    """Rename `auto h = CO_AWAIT(...)` when the name repeats in one function.

    Two such declarations in SIBLING blocks are ordinary C++ -- each shadows
    nothing, because neither is in scope where the other is. The hoist
    flattens both to function scope and they collide, which nvcc reports as
    `"h" has already been declared in the current scope`. grayscott's SeedCoro
    holds a u-plane and then a v-plane, both called `h`.
    """
    out = s.split("\n")
    depth, seen, i = 0, {}, 0
    while i < len(out):
        line = out[i]
        if depth == 0 and line.startswith("CTP_GPU_FUN"):
            seen = {}
        m = re.search(r"^\s*auto (\w+) = CO_AWAIT", line)
        if m:
            name = m.group(1)
            if name in seen:
                seen[name] += 1
                new = "%s%d" % (name, seen[name])
                d, j = depth, i
                while j < len(out):
                    out[j] = re.sub(r"\b%s\b" % re.escape(name), new, out[j])
                    d += out[j].count("{") - out[j].count("}")
                    if d < depth:
                        break
                    j += 1
            else:
                seen[name] = 1
        depth += line.count("{") - line.count("}")
        i += 1
    return "\n".join(out)


def extract_ycoro_regions(src):
    """The CLIO_HAS_YCORO blocks, each with the comment that introduces it."""
    lines = src.split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "#if CLIO_HAS_YCORO":
            # walk back over the doc comment attached to this block
            start = i
            j = i - 1
            while j >= 0 and (lines[j].strip().startswith(("*", "/*", "//")) or
                              lines[j].strip() == ""):
                if lines[j].strip().startswith("/*"):
                    start = j
                    break
                j -= 1
            else:
                start = i
            if lines[j].strip().startswith("/*"):
                start = j
            end = i
            while end < len(lines) and not lines[end].startswith("#endif  // CLIO_HAS_YCORO"):
                end += 1
            body = [l for l in lines[start:end]
                    if l.strip() != "#if CLIO_HAS_YCORO"]
            out += body + [""]
            i = end + 1
            continue
        i += 1
    return "\n".join(out)


def suspending_names(work):
    """The suspending functions the workload defines, by name.

    SUSPENDING MEANS CONTAINS CO_AWAIT, which is clio-coroc's own rule --
    local and explicit, never inferred. Matching the DECLARATION shape
    instead is not the same thing: lammps_md has YCoroMain functions with
    no await in them at all (HaloUnpinCoro), and the tool rightly leaves
    those alone, so a call site that adds the context to them passes one
    argument too many.
    """
    names = set()
    pat = re.compile(
        r"(?:CTP_GPU_FUN\s+(?:inline|CLIO_COROC_INLINE)|__device__)"
        r"\s+[\w:<>,\s*&]+?\s(\w+)\s*\(")
    for m in pat.finditer(work):
        brace = work.find("{", m.end())
        if brace < 0:
            continue
        depth, i = 0, brace
        while i < len(work):
            if work[i] == "{":
                depth += 1
            elif work[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if "CO_AWAIT(" in work[brace:i]:
            names.add(m.group(1))
    return names


def add_ctx_at_calls(body, names):
    """Append `, _cy` to every call to a suspending function.

    Done by NAME rather than inside CLIO_YCORO_RUN, because lbann and md
    generate their kernels from a macro: the coroutine call is an ARGUMENT to
    LB_KERNEL, not the operand of the run macro, so there is no single place
    to patch. clio-coroc does this itself for calls it can see; the launcher
    is a separate translation unit to it.
    """
    for n in sorted(names, key=len, reverse=True):
        out, i = "", 0
        while True:
            m = re.compile(r"\b" + re.escape(n) + r"\s*\(").search(body, i)
            if not m:
                out += body[i:]
                break
            op = m.end() - 1
            end = match_parens(body, op)
            inner = body[op + 1:end - 1]
            if inner.lstrip().startswith("_cy"):
                out += body[i:end]
            elif not inner.strip():
                out += body[i:op + 1] + "_cy)"
            else:
                # FIRST, matching E1: a pack sits last, so the context cannot.
                out += body[i:op + 1] + "_cy, " + inner + ")"
            i = end
        body = out
    return body


def run_macro_names(body, at):
    """The View and StackView parameter names in scope at offset `at`.

    CLIO_COROC_RUN needs them by name, and the CUDA launchers call them
    yv/ys while the SYCL ones call them vw/sv. Taking the nearest preceding
    declaration is enough: each is a parameter of the function the run macro
    sits in.
    """
    view, stack = "yv", "ys"
    for m in re.finditer(r"\bView\s+(\w+)", body[:at]):
        view = m.group(1)
    for m in re.finditer(r"\bStackView\s+(\w+)", body[:at]):
        stack = m.group(1)
    return view, stack


def rewrite_sycl_launcher(src, names):
    """The SYCL launcher, transformed the same way as the CUDA one.

    Its shape is uniform across all six benchmarks: a Submit wrapping
    parallel_for, a SubmitYieldable that publishes the stack and runs the
    entry, and one lambda per launch that RETURNS the coroutine. Under coroc
    there are no task types, so the lambda takes the context and CALLS the
    workload instead.
    """
    lines = src.split("\n")
    last = max(i for i, l in enumerate(lines) if l.startswith("#include"))
    body = "\n".join(lines[last + 1:])
    body = re.sub(r"^[ \t]*::?clio::run::gpu::YieldTlsPublish\([^;]*\);[ \t]*\n",
                  "", body, flags=re.M)
    body = re.sub(r"^[ \t]*gy::YieldTlsPublish\([^;]*\);[ \t]*\n", "",
                  body, flags=re.M)
    # the entry: `CLIO_YCORO_RUN(make(dev, vw.Block()))`
    out, i = "", 0
    while True:
        k = body.find("CLIO_YCORO_RUN(", i)
        if k < 0:
            out += body[i:]
            break
        view, stack = run_macro_names(body, k)
        op = k + len("CLIO_YCORO_RUN")
        end = match_parens(body, op)
        inner = body[op + 1:end - 1]
        inner = re.sub(r"\bmake\s*\(", "make(_cy, ", inner, count=1)
        out += body[i:k] + "CLIO_COROC_RUN(%s, %s, %s)" % (view, stack, inner)
        i = end
    body = out
    # each launch's lambda: take the context, and call rather than return
    body = re.sub(r"\[=\]\((Dev\w+\s+\w+,\s*u32\s+\w+)\)",
                  r"[=](clio::co::Ctx &_cy, \1)", body)
    body = re.sub(r"\breturn\s+(\w+Coro)\s*\(", r"\1(", body)
    return add_ctx_at_calls(body, names)


def rewrite_launcher(src, ns, names):
    """The launcher, from after its includes to EOF.

    NOT just its namespace block: lammps_md declares its kernels at global
    scope and opens clio::gv_bench::md only later, for the Launch wrappers.
    Taking everything after the includes keeps whatever namespace structure
    the file already has, and the prologue has already provided the headers.
    """
    lines = src.split("\n")
    last = max(i for i, l in enumerate(lines) if l.startswith("#include"))
    body = "\n".join(lines[last + 1:])
    body = re.sub(r"^[ \t]*gy::YieldTlsPublish\([^;]*\);[ \t]*\\?\n", "",
                  body, flags=re.M)
    out, i = "", 0
    while True:
        k = body.find("CLIO_YCORO_RUN(", i)
        if k < 0:
            out += body[i:]
            break
        view, stack = run_macro_names(body, k)
        out += body[i:k] + "CLIO_COROC_RUN(%s, %s, " % (view, stack)
        i = k + len("CLIO_YCORO_RUN(")
    return add_ctx_at_calls(out, names)


def keep_if_branch(src, opener):
    """Resolve `opener ... [#else ...] #endif` in favour of the true branch.

    The benchmarks gate their lane size on CLIO_YIELD_CORO, which this build
    does not define -- but the coroc edition wants the coroutine-sized arm,
    not the macro one, so the condition is resolved here rather than left to
    the preprocessor."""
    out, i = "", 0
    while True:
        k = src.find(opener, i)
        if k < 0:
            return out + src[i:]
        out += src[i:k]
        lines = src[k:].split("\n")
        depth, taken, j = 0, [], 0
        for j, l in enumerate(lines):
            t = l.strip()
            if j == 0:
                depth = 1
                continue
            if t.startswith("#if"):
                depth += 1
            elif t.startswith("#endif"):
                depth -= 1
                if depth == 0:
                    break
            elif t.startswith("#else") and depth == 1:
                # skip to the matching #endif
                d2 = 1
                for jj in range(j + 1, len(lines)):
                    tt = lines[jj].strip()
                    if tt.startswith("#if"):
                        d2 += 1
                    elif tt.startswith("#endif"):
                        d2 -= 1
                        if d2 == 0:
                            j = jj
                            break
                break
            taken.append(l)
        out += "\n".join(taken) + "\n"
        i = k + len("\n".join(lines[:j + 1])) + 1


def main():
    d, name, ns = sys.argv[1], sys.argv[2], sys.argv[3]
    # lammps_md names its bench after the directory and its kernels after the
    # namespace, so the two can differ.
    bname = sys.argv[4] if len(sys.argv) > 4 else name
    bench = os.path.join(d, "clio_%s_paged_bench.cc" % bname)
    kern = os.path.join(d, "%s_kernels.h" % name)
    launch = os.path.join(d, "cuda", "%s_launch_cuda.cc" % name)

    b = read(bench)
    # lammps_md wraps its whole coroutine driver in GV_MD_CORO. The
    # coroc edition IS that driver, so resolve it in favour of the
    # true branch before splitting -- otherwise the prologue ends
    # inside an open conditional and everything after it vanishes.
    b = keep_if_branch(b, "#if defined(GV_MD_CORO)")
    # ...and the driver refuses to run without it. The coroc edition
    # IS the suspending build, so define it rather than resolve the
    # negative guard, which has an #else arm carrying the real main.
    if "GV_MD_CORO" in b:
        b = "#define GV_MD_CORO 1\n" + b
    guard = "\n#if !CTP_IS_DEVICE_PASS\n"
    cut = b.index(guard) + 1
    prologue = b[:cut]
    driver = b[cut:]

    prologue = prologue.replace('#include "%s_kernels.h"' % name,
                                '#include "../gv_launch_bounds.h"\n'
                                '#include "%s_kernels.h"' % name, 1)
    # the lane arena: coroc packs a frame to the live set at each suspend
    # point rather than to the whole coroutine, so it needs far less
    prologue = re.sub(r"(static constexpr u32 kYieldLaneBytes = )\d+",
                      r"\g<1>1024", prologue)
    prologue = keep_if_branch(prologue, "#if defined(CLIO_YIELD_CORO)")
    # lammps_md's driver never includes the kernels header -- the LAUNCHER
    # did, and the launcher's includes are stripped here. Without it the
    # workload loses every helper the guarded regions do not define.
    # the SYCL arm needs the header the stripped launcher included
    prologue += "#if CTP_ENABLE_SYCL\n#include <sycl/sycl.hpp>\n#endif\n"
    need = '#include "%s_kernels.h"' % name
    if need not in prologue:
        prologue += ('\n#include "../gv_launch_bounds.h"\n' + need + "\n")

    work = extract_ycoro_regions(read(kern))
    work = rewrite_awaits(work)
    work = rewrite_types(work)
    work = one_declarator(work)
    work = brace_init(work)
    work = dedupe_hoists(work)
    # a transpiled function must be inlined into its kernel; see
    # CLIO_COROC_INLINE in co/yield_backend.h
    work = work.replace('CTP_GPU_FUN inline ', 'CTP_GPU_FUN CLIO_COROC_INLINE ')
    # lammps_md spells its coroutines `__device__ gy::YCoroMain Fn(`
    work = re.sub(r'(?m)^__device__ ', '__device__ CLIO_COROC_INLINE ', work)

    wrap = ("namespace " + ns + " {") in read(kern)
    names = suspending_names(work)
    lau = rewrite_launcher(read(launch), ns, names)
    sycl_src = os.path.join(d, "sycl", "%s_launch_sycl.cc" % name)
    if os.path.exists(sycl_src):
        syc = rewrite_sycl_launcher(read(sycl_src), names)
        lau = ("\n/* TWO BACKENDS, ONE WORKLOAD. Everything above this"
               " line is compiled for both: the transpiled state machine"
               " contains no vendor token. What differs is only how a grid"
               " is submitted. */\n"
               "#if CTP_ENABLE_SYCL\n" + syc +
               "\n#else  /* CUDA */\n" + lau +
               "\n#endif  /* CTP_ENABLE_SYCL */\n")

    # The SYCL launchers open with this, BEFORE any include: it tells
    # yield_stack.h and device_vector.h that this TU submits kernels,
    # which is what declares g_yield_smem_dg among other things. It
    # has to lead the file, so it cannot ride in the prologue.
    lead = ("#if CTP_ENABLE_SYCL" + chr(10) +
            "#define CLIO_SYCL_KERNEL_TU 1" + chr(10) + "#endif" + chr(10))
    out = (lead + prologue
           + "\n/* =====================================================\n"
             " * THE WORKLOAD, on the new coroutine API. Ordinary return\n"
             " * types, ordinary locals, one marker per suspending call.\n"
             " * The `, clio::co::Ctx &_cy` on each signature and at each\n"
             " * call site is appended by clio-coroc, not written here.\n"
             " * ===================================================== */\n"
           # WRAP ONLY IF THE SOURCE DID. gmx and the rest define their
           # coroutines inside clio::gv_bench::<name>; lammps_md defines
           # them at GLOBAL scope, and its launcher's kernels are global
           # too. Wrapping those moves the workload out of the launcher's
           # reach -- "identifier ReadProbeCoro is undefined".
           + (("namespace " + ns + " {\n\n") if wrap else "\n")
           + work
           # the launcher brings its OWN namespaces -- lbann and md both
           # reopen clio::run::gpu at file scope for their stack helpers --
           # so the workload's wrapper closes before it, not around it
           + (("\n}  // namespace " + ns + "\n") if wrap else "\n")
           + lau
           + "\n"
           + driver)

    dst = os.path.join(d, "clio_%s_paged_newcoro.cc" % bname)
    write(dst, out)
    print("%-10s -> %s  (%d lines, %d CO_AWAIT, %d CLIO_COROC_RUN)"
          % (name, os.path.basename(dst), out.count("\n"),
             out.count("CO_AWAIT("), out.count("CLIO_COROC_RUN(")))


main()
