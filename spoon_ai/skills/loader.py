"""
Skill loader for parsing SKILL.md files.

Discovers and loads skills from multiple paths:
1. ~/.spoon/skills/ - Global user skills
2. ./skills/ - Project-local skills
3. Additional user-specified paths

Note: No built-in skills are included by default.
Use additional_paths to specify skill directories.
"""

import re
import yaml
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from spoon_ai.skills.models import Skill, SkillMetadata, ScriptConfig
from spoon_ai.tools.base import BaseTool

logger = logging.getLogger(__name__)


import os


class SkillLoader:
    """
    Loads skills from SKILL.md files with optional Python tool modules.

    SKILL.md Format:
    ---
    name: my-skill
    description: What this skill does
    version: 1.0.0
    triggers:
      - type: keyword
        keywords: [analyze, review]
    ---

    # Skill Instructions

    Markdown content here...
    """

    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n(.*)$',
        re.DOTALL
    )

    def __init__(self, additional_paths: Optional[List[Path]] = None, include_default_paths: bool = True):
        """
        Initialize loader with skill search paths.

        Args:
            additional_paths: Additional directories to search for skills
        """
        self._paths: List[Path] = []

        if include_default_paths:
            # Global user skills
            global_path = Path.home() / ".spoon" / "skills"
            if global_path.exists():
                self._paths.append(global_path)

            # Project-local skills
            local_path = Path.cwd() / "skills"
            if local_path.exists():
                self._paths.append(local_path)

        # Additional paths
        if additional_paths:
            for p in additional_paths:
                if isinstance(p, str):
                    p = Path(p)
                if p.exists():
                    self._paths.append(p)

        # Note: No built-in skills are included by default
        # Users should specify skill paths via additional_paths

        # Caches
        self._skill_cache: Dict[str, Skill] = {}
        self._tool_cache: Dict[str, List[BaseTool]] = {}

    @property
    def paths(self) -> List[Path]:
        """Get configured skill paths."""
        return self._paths.copy()

    def add_path(self, path: Path) -> None:
        """Add a path to search for skills."""
        if isinstance(path, str):
            path = Path(path)
        if path.exists() and path not in self._paths:
            # Insert before builtin
            if self._paths and "builtin" in str(self._paths[-1]):
                self._paths.insert(-1, path)
            else:
                self._paths.append(path)

    def discover(self) -> List[Path]:
        """
        Discover all SKILL.md files in configured paths.

        Returns:
            List of paths to SKILL.md files
        """
        skill_files: List[Path] = []
        seen_resolved: set = set()

        for base_path in self._paths:
            if not base_path.exists():
                continue

            # Use a single glob and case-insensitive name check to avoid
            # duplicates on Windows (where rglob("SKILL.md") and
            # rglob("skill.md") may return the same files).
            for md_file in base_path.rglob("*.md"):
                if md_file.name.lower() != "skill.md":
                    continue
                resolved = md_file.resolve()
                if resolved in seen_resolved:
                    continue
                seen_resolved.add(resolved)
                skill_files.append(md_file)
                logger.debug(f"Discovered skill: {md_file}")

        return skill_files

    def parse(self, file_path: Path) -> Tuple[SkillMetadata, str]:
        """
        Parse a SKILL.md file into metadata and instructions.

        Args:
            file_path: Path to SKILL.md file

        Returns:
            Tuple of (SkillMetadata, instructions_markdown)

        Raises:
            ValueError: If file format is invalid
        """
        # Use utf-8-sig to strip BOM if present (common on Windows editors)
        content = file_path.read_text(encoding='utf-8-sig')
        # Normalize line endings for regex reliability (\r\n → \n)
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError(
                f"Invalid SKILL.md format in {file_path}: "
                "missing YAML frontmatter (must start with ---)"
            )

        yaml_content = match.group(1)
        instructions = match.group(2).strip()

        try:
            metadata_dict = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")

        if not isinstance(metadata_dict, dict):
            raise ValueError(f"YAML frontmatter must be a dictionary in {file_path}")

        # Validate required fields
        if 'name' not in metadata_dict:
            raise ValueError(f"Missing required 'name' field in {file_path}")
        if 'description' not in metadata_dict:
            raise ValueError(f"Missing required 'description' field in {file_path}")

        try:
            metadata = SkillMetadata(**metadata_dict)
        except Exception as e:
            raise ValueError(f"Invalid metadata in {file_path}: {e}")

        return metadata, instructions

    def _resolve_scripts(self, metadata: SkillMetadata, skill_dir: Path) -> None:
        """
        Resolve script file paths relative to skill directory.

        Args:
            metadata: Skill metadata with script config
            skill_dir: Directory containing SKILL.md
        """
        if not metadata.scripts or not metadata.scripts.definitions:
            return

        # Determine base directory for scripts
        if metadata.scripts.working_directory:
            if os.path.isabs(metadata.scripts.working_directory):
                base_dir = Path(metadata.scripts.working_directory)
            else:
                base_dir = skill_dir / metadata.scripts.working_directory
        else:
            base_dir = skill_dir

        # Resolve each script's file path
        for script in metadata.scripts.definitions:
            if script.file:
                # Per-script working directory override
                if script.working_directory:
                    if os.path.isabs(script.working_directory):
                        script_base = Path(script.working_directory)
                    else:
                        script_base = base_dir / script.working_directory
                else:
                    script_base = base_dir

                # Resolve and store the full path
                resolved = (script_base / script.file).resolve()
                script._resolved_path = str(resolved)

                if not resolved.exists():
                    logger.warning(
                        f"Script file not found: {resolved} "
                        f"(skill: {metadata.name}, script: {script.name})"
                    )

        # Update working directory to absolute path
        metadata.scripts.working_directory = str(base_dir.resolve())

    def load_tools(self, skill_dir: Path) -> List[BaseTool]:
        """
        Load Python tools from a skill directory.

        Looks for tools.py containing BaseTool subclasses.

        Args:
            skill_dir: Directory containing the skill

        Returns:
            List of loaded tool instances
        """
        tools = []
        tools_file = skill_dir / "tools.py"

        if not tools_file.exists():
            return tools

        try:
            # Create unique module name to avoid conflicts
            module_name = f"skill_tools_{skill_dir.name}_{id(skill_dir)}"

            spec = importlib.util.spec_from_file_location(module_name, tools_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all BaseTool subclasses
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type) and
                        issubclass(obj, BaseTool) and
                        obj is not BaseTool
                    ):
                        try:
                            tool_instance = obj()
                            tools.append(tool_instance)
                            logger.info(f"Loaded skill tool: {tool_instance.name}")
                        except Exception as e:
                            logger.error(f"Failed to instantiate tool {name}: {e}")

        except Exception as e:
            logger.error(f"Failed to load tools from {tools_file}: {e}")

        return tools

    def load(self, file_path: Path) -> Skill:
        """
        Load a complete skill from SKILL.md and optional modules.

        Args:
            file_path: Path to SKILL.md file

        Returns:
            Loaded Skill instance
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        metadata, instructions = self.parse(file_path)
        skill_dir = file_path.parent

        # Load associated tools
        tools = self.load_tools(skill_dir)
        tool_names = [t.name for t in tools]

        # Resolve script paths
        if metadata.has_scripts():
            self._resolve_scripts(metadata, skill_dir)

        # Get script names
        script_names = []
        if metadata.has_scripts() and metadata.scripts_enabled():
            script_names = [s.name for s in metadata.scripts.definitions]

        skill = Skill(
            metadata=metadata,
            instructions=instructions,
            source_path=str(file_path),
            tool_names=tool_names,
            script_names=script_names,
            loaded_at=datetime.now()
        )

        # Cache skill and tools
        self._skill_cache[metadata.name] = skill
        self._tool_cache[metadata.name] = tools

        # Log info
        info_parts = [f"Loaded skill: {metadata.name}"]
        if tool_names:
            info_parts.append(f"tools={tool_names}")
        if script_names:
            info_parts.append(f"scripts={script_names}")
        logger.info(" ".join(info_parts))

        return skill

    def load_all(self) -> Dict[str, Skill]:
        """
        Discover and load all skills from configured paths.

        Returns:
            Dictionary mapping skill names to Skill instances
        """
        skill_files = self.discover()

        for file_path in skill_files:
            try:
                self.load(file_path)
            except Exception as e:
                logger.error(f"Failed to load skill from {file_path}: {e}")

        logger.info(f"Loaded {len(self._skill_cache)} skills")
        return self._skill_cache.copy()

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a loaded skill by name."""
        return self._skill_cache.get(name)

    def get_tools(self, skill_name: str) -> List[BaseTool]:
        """Get loaded tools for a skill."""
        return self._tool_cache.get(skill_name, [])

    def clear_cache(self) -> None:
        """Clear all cached skills and tools."""
        self._skill_cache.clear()
        self._tool_cache.clear()

    def reload(self, name: str) -> Optional[Skill]:
        """
        Reload a specific skill from disk.

        Args:
            name: Skill name to reload

        Returns:
            Reloaded Skill or None if not found
        """
        skill = self._skill_cache.get(name)
        if skill and skill.source_path:
            # Remove from cache
            del self._skill_cache[name]
            if name in self._tool_cache:
                del self._tool_cache[name]

            # Reload
            return self.load(Path(skill.source_path))

        return None
