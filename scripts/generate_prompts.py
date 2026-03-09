#!/usr/bin/env python3
"""Generate MCP prompts from skill SKILL.md files.

Reads each skill's SKILL.md (and references/workflow.md if it exists) and
generates the prompts.py module that registers them as MCP prompts.

Usage:
    python scripts/generate_prompts.py > src/deriva_mcp/prompts.py
    python scripts/generate_prompts.py --dry-run   # Print to stdout without writing
    python scripts/generate_prompts.py --list       # List skills and their prompts
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Skills that should NOT get prompts (environment/utility skills)
SKIP_SKILLS = {"check-versions"}

# Mapping from skill directory name to prompt name (where they differ)
# Most skills use their directory name as the prompt name
PROMPT_NAME_OVERRIDES = {
    "coding-guidelines": "derivaml-coding-guidelines",
}


def parse_skill_frontmatter(skill_md: Path) -> dict:
    """Parse YAML frontmatter from SKILL.md."""
    text = skill_md.read_text()
    if not text.startswith("---"):
        return {}

    end = text.index("---", 3)
    frontmatter = text[3:end].strip()
    result = {}
    for line in frontmatter.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            result[key.strip()] = value
    return result


def get_skill_body(skill_md: Path) -> str:
    """Get the body of SKILL.md (after frontmatter)."""
    text = skill_md.read_text()
    if text.startswith("---"):
        end = text.index("---", 3) + 3
        return text[end:].strip()
    return text.strip()


def get_workflow_content(skill_dir: Path) -> str | None:
    """Get references/workflow.md content if it exists."""
    workflow = skill_dir / "references" / "workflow.md"
    if workflow.exists():
        return workflow.read_text().strip()
    return None


def make_function_name(name: str) -> str:
    """Convert a skill name to a Python function name."""
    return name.replace("-", "_") + "_prompt"


def escape_for_triple_quotes(text: str) -> str:
    """Escape text for inclusion in triple-quoted strings."""
    # Escape backslashes first, then triple quotes
    text = text.replace("\\", "\\\\")
    text = text.replace('"""', '\\"\\"\\"')
    return text


def discover_skills(plugin_dir: Path) -> list[tuple[str, Path]]:
    """Discover all skills and return (name, directory) pairs."""
    skills = []
    for skill_md in sorted(plugin_dir.rglob("SKILL.md")):
        skill_dir = skill_md.parent
        frontmatter = parse_skill_frontmatter(skill_md)
        name = frontmatter.get("name", skill_dir.name)
        if name in SKIP_SKILLS:
            continue
        skills.append((name, skill_dir))
    return skills


def generate_prompt_function(
    name: str, skill_dir: Path, frontmatter: dict
) -> str:
    """Generate a single prompt function."""
    prompt_name = PROMPT_NAME_OVERRIDES.get(name, name)
    func_name = make_function_name(prompt_name)
    description = frontmatter.get("description", f"Guide for {name}")

    # Strip "ALWAYS use this skill when" prefix for prompt description
    # since prompts are explicitly selected, not auto-triggered
    desc = description
    if desc.startswith("ALWAYS use this skill when"):
        desc = "Guide for" + desc[len("ALWAYS use this skill when"):]
    # Remove "Triggers on:" suffix
    if "Triggers on:" in desc:
        desc = desc[:desc.index("Triggers on:")].rstrip(". ")

    body = get_skill_body(skill_dir / "SKILL.md")
    workflow = get_workflow_content(skill_dir)

    # Build the full prompt content
    if workflow:
        full_content = f"{body}\n\n---\n\n# Detailed Guide\n\n{workflow}"
    else:
        full_content = body

    escaped = escape_for_triple_quotes(full_content)

    return f'''    @mcp.prompt(
        name="{prompt_name}",
        description="{escape_for_triple_quotes(desc)}",
    )
    def {func_name}() -> str:
        """{frontmatter.get("name", name)} workflow guide."""
        return """{escaped}"""
'''


def generate_prompts_module(plugin_dir: Path) -> str:
    """Generate the complete prompts.py module."""
    skills = discover_skills(plugin_dir)

    header = '''"""MCP Prompts for DerivaML.

Auto-generated from skill SKILL.md files by scripts/generate_prompts.py.
Do not edit manually — regenerate with:
    python scripts/generate_prompts.py

Prompts provide interactive, step-by-step guidance for common tasks:
- ML execution lifecycle (training, inference workflows)
- Dataset preparation and management
- Catalog operations (tables, features, annotations)
- Experiment configuration and execution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from deriva_mcp.connection import ConnectionManager


def register_prompts(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML prompts with the MCP server."""
'''

    functions = []
    for name, skill_dir in skills:
        frontmatter = parse_skill_frontmatter(skill_dir / "SKILL.md")
        func = generate_prompt_function(name, skill_dir, frontmatter)
        functions.append(func)

    return header + "\n".join(functions)


def main():
    parser = argparse.ArgumentParser(description="Generate prompts.py from skills")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout without writing")
    parser.add_argument("--list", action="store_true",
                        help="List skills and their prompt names")
    parser.add_argument("--plugin-dir", type=Path,
                        default=Path(__file__).parent.parent / "plugin" / "skills",
                        help="Path to the plugin/skills directory")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent.parent / "src" / "deriva_mcp" / "prompts.py",
                        help="Output path for prompts.py")
    args = parser.parse_args()

    if args.list:
        skills = discover_skills(args.plugin_dir)
        for name, skill_dir in skills:
            prompt_name = PROMPT_NAME_OVERRIDES.get(name, name)
            has_workflow = (skill_dir / "references" / "workflow.md").exists()
            marker = " (+workflow)" if has_workflow else ""
            print(f"  {prompt_name:<40s} <- {skill_dir.relative_to(args.plugin_dir)}{marker}")
        print(f"\n  Total: {len(skills)} prompts")
        return

    module = generate_prompts_module(args.plugin_dir)

    if args.dry_run:
        print(module)
    else:
        args.output.write_text(module)
        print(f"Generated {args.output} ({len(module)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
