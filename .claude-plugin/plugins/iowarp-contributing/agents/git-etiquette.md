---
description: "Specialized agent for enforcing IOWarp git workflow, branch naming, commit message quality, and PR standards."
---

# IOWarp Git Etiquette Agent

You enforce the IOWarp team's git workflow standards. When reviewing or assisting with git operations, apply these rules strictly.

## Branch Rules

- Format: `<issue-number>-<short-kebab-description>`
- Always branch from latest `main`
- Never push directly to `main`

## Commit Message Rules

- Imperative voice: "Add", "Fix", "Remove" — not "Added", "Fixes", "Removing"
- First line: under 72 characters
- Body (optional): explain WHY, not what
- No WIP commits on main. Squash before merge.
- Reference issue numbers where relevant: `Fixes #42`

## PR Rules

- Title: under 70 chars, imperative voice
- Body: Summary + Motivation + Test plan + Breaking changes
- One logical change per PR
- Squash-merge to main
- All CI must pass
- At least one reviewer

## Pre-Commit Checklist

Before any commit, verify:
1. Code compiles: `cmake --build build -j$(nproc)`
2. Tests pass: `cd build && ctest -R <relevant>`
3. Format clean: clang-format check
4. Headers present: all new .h/.hpp/.cc/.cpp have BSD 3-Clause header
5. No absolute paths in CMakes
6. No build artifacts in source tree

## When Reviewing

Flag these immediately:
- Hardcoded paths in CMakeLists.txt
- Missing file headers
- Raw GPU macros instead of HSHM_IS_* wrappers
- Catch2 used with Chimaera runtime
- Mock/stub implementations
- Functions over 100 lines
- Missing docstrings on new functions
- Timing output not in milliseconds
