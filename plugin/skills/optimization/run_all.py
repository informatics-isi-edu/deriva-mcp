#!/usr/bin/env python3
"""Run skill description optimization for all DerivaML MCP skills."""

import json
import os
import sys
from pathlib import Path

# Add skill-creator to path
SKILL_CREATOR = Path(os.path.expanduser(
    "~/.claude/plugins/cache/claude-plugins-official/skill-creator/205b6e0b3036/skills/skill-creator"
))
sys.path.insert(0, str(SKILL_CREATOR))
os.chdir(SKILL_CREATOR)

from scripts.run_loop import run_loop
from scripts.utils import parse_skill_md

DERIVA_MCP = Path("/Users/carl/GitHub/deriva-mcp")
OPT_DIR = DERIVA_MCP / "plugin" / "skills" / "optimization"
RESULTS_DIR = OPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Map skill names to paths
SKILL_MAP = {}
for skill_md in (DERIVA_MCP / "plugin" / "skills").rglob("SKILL.md"):
    skill_dir = skill_md.parent
    name, description, content = parse_skill_md(skill_dir)
    if name:
        SKILL_MAP[name] = str(skill_dir)

MODEL = "claude-opus-4-6"
MAX_ITERATIONS = 3

def run_one(name: str):
    eval_file = OPT_DIR / f"{name}-evals.json"
    skill_path = SKILL_MAP.get(name)
    result_file = RESULTS_DIR / f"{name}-result.json"

    if not eval_file.exists():
        print(f"SKIP {name}: no eval file")
        return
    if not skill_path:
        print(f"SKIP {name}: skill not found")
        return
    if result_file.exists():
        print(f"SKIP {name}: already optimized")
        return

    print(f"\n{'='*60}")
    print(f"OPTIMIZING: {name}")
    print(f"  Skill: {skill_path}")
    print(f"  Evals: {eval_file}")
    print(f"{'='*60}\n")

    with open(eval_file) as f:
        eval_set = json.load(f)

    try:
        result = run_loop(
            eval_set=eval_set,
            skill_path=Path(skill_path),
            description_override=None,
            num_workers=3,
            timeout=30,
            max_iterations=MAX_ITERATIONS,
            runs_per_query=3,
            trigger_threshold=0.5,
            holdout=0.4,
            model=MODEL,
            verbose=True,
            log_dir=RESULTS_DIR / name,
        )

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nRESULT for {name}:")
        if isinstance(result, dict):
            best = result.get("best_description", "N/A")
            print(f"  Best description: {best[:100]}...")
        print(f"  Saved to: {result_file}")

    except Exception as e:
        print(f"ERROR optimizing {name}: {e}")
        with open(result_file, "w") as f:
            json.dump({"error": str(e), "skill": name}, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", help="Optimize a single skill by name")
    parser.add_argument("--list", action="store_true", help="List all skills")
    args = parser.parse_args()

    if args.list:
        for name in sorted(SKILL_MAP.keys()):
            eval_exists = (OPT_DIR / f"{name}-evals.json").exists()
            result_exists = (RESULTS_DIR / f"{name}-result.json").exists()
            status = "done" if result_exists else ("ready" if eval_exists else "no-evals")
            print(f"  [{status}] {name}")
        sys.exit(0)

    if args.skill:
        run_one(args.skill)
    else:
        # Run all skills that have eval sets
        names = sorted(n for n in SKILL_MAP if (OPT_DIR / f"{n}-evals.json").exists())
        print(f"Optimizing {len(names)} skills...")
        for name in names:
            run_one(name)
        print(f"\nDone! Results in {RESULTS_DIR}")
