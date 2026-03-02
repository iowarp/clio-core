#!/usr/bin/env python3
"""
Validate the iowarp-clio Claude Code plugin marketplace.

Checks:
  - marketplace.json structure and required fields
  - Plugin manifests (plugin.json) for each listed plugin
  - SKILL.md files have proper YAML frontmatter
  - Agent .md files have proper YAML frontmatter
  - settings.json references the correct marketplace
  - Contributing skill covers Jaime's full workflow steps
  - Dev-setup skill covers Docker devcontainer path

Exit 0 = all checks pass, exit 1 = failures found.
"""

import json
import os
import sys
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERRORS = []
WARNINGS = []


def error(msg):
    ERRORS.append(msg)
    print(f"  FAIL: {msg}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"  WARN: {msg}")


def ok(msg):
    print(f"  OK: {msg}")


def check_file_exists(path, label):
    full = os.path.join(ROOT, path)
    if os.path.isfile(full):
        ok(f"{label} exists")
        return True
    error(f"{label} not found at {path}")
    return False


def check_json(path, label):
    full = os.path.join(ROOT, path)
    try:
        with open(full) as f:
            data = json.load(f)
        ok(f"{label} is valid JSON")
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        error(f"{label} invalid: {e}")
        return None


def check_frontmatter(path, label):
    full = os.path.join(ROOT, path)
    try:
        with open(full) as f:
            content = f.read()
        if content.startswith("---"):
            # Find closing ---
            end = content.find("---", 3)
            if end > 0:
                fm = content[3:end].strip()
                if "description:" in fm:
                    ok(f"{label} has frontmatter with description")
                    return content
                else:
                    error(f"{label} frontmatter missing 'description' key")
            else:
                error(f"{label} frontmatter not properly closed")
        else:
            error(f"{label} missing YAML frontmatter (must start with ---)")
        return content
    except FileNotFoundError:
        error(f"{label} file not found at {path}")
        return None


def check_skill_content(content, skill_name, required_phrases):
    """Check that a skill's content covers expected topics."""
    missing = []
    for phrase in required_phrases:
        if isinstance(phrase, str):
            if phrase.lower() not in content.lower():
                missing.append(phrase)
        elif isinstance(phrase, (list, tuple)):
            # Any of these must match
            if not any(p.lower() in content.lower() for p in phrase):
                missing.append(f"one of: {phrase}")
    if missing:
        for m in missing:
            error(f"Skill '{skill_name}' missing coverage of: {m}")
    else:
        ok(f"Skill '{skill_name}' covers all required topics")


# =============================================================
print("\n=== 1. Marketplace Structure ===")

mp = check_json(".claude-plugin/marketplace.json", "marketplace.json")
if mp:
    for field in ["name", "owner", "plugins"]:
        if field in mp:
            ok(f"marketplace.json has '{field}'")
        else:
            error(f"marketplace.json missing '{field}'")

    if "name" in mp.get("owner", {}):
        ok("marketplace owner has name")
    else:
        error("marketplace owner missing name")

    if mp.get("metadata", {}).get("description"):
        ok("marketplace has metadata.description")
    else:
        warn("marketplace missing metadata.description")

# =============================================================
print("\n=== 2. Plugin Manifests ===")

if mp and "plugins" in mp:
    for plugin in mp["plugins"]:
        pname = plugin.get("name", "UNKNOWN")
        source = plugin.get("source", "")

        if isinstance(source, str) and source.startswith("./"):
            resolved = os.path.join(".claude-plugin", source[2:])
            pj_path = os.path.join(resolved, ".claude-plugin", "plugin.json")
            pj = check_json(pj_path, f"{pname}/plugin.json")
            if pj:
                for field in ["name", "description", "version"]:
                    if field in pj:
                        ok(f"{pname}/plugin.json has '{field}'")
                    else:
                        error(f"{pname}/plugin.json missing '{field}'")

                # Verify name consistency
                if pj.get("name") != pname:
                    error(
                        f"{pname}: plugin.json name '{pj.get('name')}' "
                        f"doesn't match marketplace entry '{pname}'"
                    )

# =============================================================
print("\n=== 3. Skills ===")

if mp and "plugins" in mp:
    for plugin in mp["plugins"]:
        pname = plugin.get("name", "UNKNOWN")
        source = plugin.get("source", "")
        if isinstance(source, str) and source.startswith("./"):
            resolved = os.path.join(".claude-plugin", source[2:])
            skills_dir = os.path.join(ROOT, resolved, "skills")
            if os.path.isdir(skills_dir):
                for skill in os.listdir(skills_dir):
                    skill_path = os.path.join(resolved, "skills", skill, "SKILL.md")
                    content = check_frontmatter(skill_path, f"{pname}/skills/{skill}")
            else:
                warn(f"{pname} has no skills/ directory")

# =============================================================
print("\n=== 4. Agents ===")

if mp and "plugins" in mp:
    for plugin in mp["plugins"]:
        pname = plugin.get("name", "UNKNOWN")
        source = plugin.get("source", "")
        if isinstance(source, str) and source.startswith("./"):
            resolved = os.path.join(".claude-plugin", source[2:])
            agents_dir = os.path.join(ROOT, resolved, "agents")
            if os.path.isdir(agents_dir):
                for agent_file in os.listdir(agents_dir):
                    if agent_file.endswith(".md"):
                        agent_path = os.path.join(
                            resolved, "agents", agent_file
                        )
                        check_frontmatter(agent_path, f"{pname}/agents/{agent_file}")
            else:
                warn(f"{pname} has no agents/ directory")

# =============================================================
print("\n=== 5. Settings Integration ===")

settings = check_json(".claude/settings.json", "settings.json")
if settings and mp:
    mp_name = mp.get("name", "")
    if mp_name in settings.get("extraKnownMarketplaces", {}):
        ok(f"settings.json references marketplace '{mp_name}'")
    else:
        error(f"settings.json missing marketplace '{mp_name}' in extraKnownMarketplaces")

    for pname in [p["name"] for p in mp.get("plugins", [])]:
        key = f"{pname}@{mp_name}"
        if key in settings.get("enabledPlugins", {}):
            ok(f"settings.json enables '{key}'")
        else:
            error(f"settings.json missing '{key}' in enabledPlugins")

# =============================================================
print("\n=== 6. Contributing Skill — Jaime Workflow Coverage ===")

contrib_path = os.path.join(
    ROOT,
    ".claude-plugin/plugins/iowarp-contributing/skills/contributing/SKILL.md",
)
if os.path.isfile(contrib_path):
    with open(contrib_path) as f:
        contrib = f.read()

    # Jaime's workflow steps:
    # 1. Explore error → create issue on GitHub
    # 2. Branch from issue
    # 3. Write failing test, run it, verify failure
    # 4. Implement fix, run all tests, verify pass
    # 5. Push to branch, create PR

    jaime_coverage = [
        # Step 1: Issue creation
        ("gh issue create", ["gh issue create", "issue create", "create issue"]),
        # Step 2: Branch from issue
        ("issue-based branch naming", ["<issue-number>"]),
        # Step 3: Test-driven: write test first
        (
            "test-first / TDD workflow",
            ["failing test", "test that fails", "write a test", "test-driven"],
        ),
        # Step 4: Run all tests after fix
        ("run all tests after fix", ["ctest", "run all test", "regression"]),
        # Step 5: Push and create PR
        ("push and create PR", ["git push", "gh pr create"]),
        # PR body with link to issue
        ("PR links to issue", ["link to issue", "fixes #", "closes #", "motivation"]),
    ]

    print(f"  Checking Jaime's 5-step workflow coverage...")
    for label, patterns in jaime_coverage:
        found = any(p.lower() in contrib.lower() for p in patterns)
        if found:
            ok(f"Covers: {label}")
        else:
            error(f"Missing: {label}")

# =============================================================
print("\n=== 7. Dev-Setup Skill — Docker Path Coverage ===")

devsetup_path = os.path.join(
    ROOT,
    ".claude-plugin/plugins/iowarp-dev-setup/skills/dev-setup/SKILL.md",
)
if os.path.isfile(devsetup_path):
    with open(devsetup_path) as f:
        devsetup = f.read()

    docker_coverage = [
        ("git clone --recurse-submodules", ["recurse-submodules"]),
        ("devcontainer.json reference", ["devcontainer.json", "devcontainer"]),
        ("cmake --preset=debug", ["cmake --preset=debug"]),
        ("cmake --build build", ["cmake --build build"]),
        ("ctest", ["ctest"]),
        ("Docker build command", ["docker build"]),
        ("UID/GID mapping", ["HOST_UID", "uid"]),
        ("troubleshooting section", ["troubleshoot"]),
        ("GPU container path", ["nvidia-gpu", "cuda"]),
        ("IPC transport modes", ["CHI_IPC_MODE"]),
    ]

    for label, patterns in docker_coverage:
        found = any(p.lower() in devsetup.lower() for p in patterns)
        if found:
            ok(f"Covers: {label}")
        else:
            error(f"Missing: {label}")


# =============================================================
print("\n" + "=" * 60)
if ERRORS:
    print(f"RESULT: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
    for e in ERRORS:
        print(f"  ERROR: {e}")
    sys.exit(1)
else:
    print(f"RESULT: ALL CHECKS PASSED ({len(WARNINGS)} warning(s))")
    sys.exit(0)
