"""
Skill loader for pipeline stage strategy prompts.

Thin markdown files with YAML frontmatter. Skills describe strategy (what
trade-offs to make, what metrics matter) -- not mechanics (how to build
configs). Business logic stays in Python; prompts guide LLM reasoning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path(__file__).parent.parent.parent / "configs" / "agent" / "skills"


class Skill:
    def __init__(self, name: str, description: str, tools: list, meta: Dict[str, Any], prompt: str):
        self.name = name
        self.description = description
        self.tools = tools          # list of allowed tool names for this stage
        self.meta = meta
        self.prompt = prompt        # the strategy prompt (markdown body)

    def to_system_prompt(self) -> str:
        return f"# Stage: {self.name}\n\n{self.description}\n\n{self.prompt}"


class SkillLoader:
    """Loads skill files from configs/agent/skills/."""

    def __init__(self, skills_dir: Optional[Path] = None):
        self._dir = skills_dir or _DEFAULT_SKILLS_DIR
        self._cache: Dict[str, Skill] = {}

    def load(self, stage: str) -> Skill:
        """Load skill for a stage. Caches after first load."""
        if stage in self._cache:
            return self._cache[stage]

        path = self._dir / f"{stage}.md"
        if not path.exists():
            raise FileNotFoundError(f"No skill file for stage {stage!r} at {path}")

        raw = path.read_text(encoding="utf-8")
        skill = _parse_skill_file(stage, raw)
        self._cache[stage] = skill
        logger.debug("Loaded skill: %s", stage)
        return skill

    def available(self) -> List[str]:
        return [p.stem for p in self._dir.glob("*.md")]


def _parse_skill_file(name: str, raw: str) -> Skill:
    """Parse ---frontmatter---\n\nbody format."""
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
        else:
            meta = {}
            body = raw
    else:
        meta = {}
        body = raw

    return Skill(
        name=name,
        description=meta.get("description", ""),
        tools=meta.get("tools", []),
        meta=meta,
        prompt=body,
    )
