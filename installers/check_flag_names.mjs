#!/usr/bin/env node
// Regression test for #363: installer scripts passing -D<FLAG>=... to CMake
// must name a flag CMakeLists.txt actually declares. A renamed/typo'd flag
// (e.g. the WRP_ -> CLIO_ rebrand leaving stale WRP_CORE_ENABLE_* references)
// is silently ignored by CMake, so the installer's override never takes
// effect -- exactly #363's "optional features block setup" failure mode.
//
// Usage: node installers/check_flag_names.mjs
// Exit 0 if every -D<FLAG>= in the checked installer files names a flag
// declared by an option()/set(... CACHE ...) in the root CMakeLists.txt (or
// a well-known external prefix: CMAKE_, HSHM_, BUILD_SHARED_LIBS). Exit 1
// and print each offending flag otherwise.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..");

const cmakeListsPath = join(repoRoot, "CMakeLists.txt");
const cmakeLists = readFileSync(cmakeListsPath, "utf8");

const declared = new Set();
// option(NAME "desc" DEFAULT)
for (const m of cmakeLists.matchAll(/option\(\s*([A-Za-z0-9_]+)\b/g)) {
  declared.add(m[1]);
}
// set(NAME ... CACHE ...)
for (const m of cmakeLists.matchAll(/set\(\s*([A-Za-z0-9_]+)\s+[^\n]*\bCACHE\b/g)) {
  declared.add(m[1]);
}

const wellKnownPrefixes = ["CMAKE_", "HSHM_"];
const wellKnownExact = new Set(["BUILD_SHARED_LIBS"]);

function isKnown(name) {
  if (declared.has(name)) return true;
  if (wellKnownExact.has(name)) return true;
  return wellKnownPrefixes.some((p) => name.startsWith(p));
}

const filesToCheck = [
  join(repoRoot, "installers/conda/build.sh"),
  join(repoRoot, "installers/vcpkg/portfile.cmake"),
  join(repoRoot, "installers/pip/README.md"),
];

let failures = [];
for (const file of filesToCheck) {
  const text = readFileSync(file, "utf8");
  for (const m of text.matchAll(/-D([A-Za-z0-9_]+)=/g)) {
    const name = m[1];
    if (!isKnown(name)) {
      failures.push({ file, name });
    }
  }
}

if (failures.length > 0) {
  console.error("Stale/unknown CMake flag(s) referenced by installer scripts:");
  for (const f of failures) {
    console.error(`  ${f.file}: -D${f.name}=...  (not declared in CMakeLists.txt)`);
  }
  console.error(`\n${failures.length} failure(s). See CMakeLists.txt for the current option() names.`);
  process.exit(1);
}

console.log(`OK: all -D<FLAG>= references in ${filesToCheck.length} installer file(s) are declared in CMakeLists.txt.`);
process.exit(0);
