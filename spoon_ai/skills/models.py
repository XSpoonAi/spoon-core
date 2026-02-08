"""
Skill system data models.

Pydantic schemas for skill definition, following Anthropic Skills specification
with XSpoonAi extensions.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from datetime import datetime


class SkillState(str, Enum):
    """Skill activation state."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    LOADING = "loading"
    ERROR = "error"


class ScriptType(str, Enum):
    """Supported script execution types."""
    PYTHON = "python"
    SHELL = "shell"
    BASH = "bash"


class SkillScript(BaseModel):
    """
    Script definition within a skill.

    Scripts are executed by the agent when needed. The AI decides
    how to call scripts - users only control whether scripts are allowed.

    Example in SKILL.md:
        scripts:
          - name: fetch_data
            description: Fetch latest market data
            type: python
            file: scripts/fetch_data.py
    """
    name: str = Field(..., description="Script identifier")
    description: str = Field(default="", description="What this script does")
    type: ScriptType = Field(default=ScriptType.PYTHON, description="Script type")

    # Script source (one of file or inline)
    file: Optional[str] = Field(default=None, description="Script file path (relative to skill dir)")
    inline: Optional[str] = Field(default=None, description="Inline script content")

    # Input schema for LLM tool integration
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema describing the expected stdin input format"
    )

    # Execution settings
    timeout: int = Field(default=30, ge=1, le=600, description="Execution timeout in seconds")
    working_directory: Optional[str] = Field(default=None, description="Working directory override")

    # Lifecycle hooks
    run_on_activation: bool = Field(default=False, description="Run when skill activates")
    run_on_deactivation: bool = Field(default=False, description="Run when skill deactivates")

    # Runtime state (not from YAML)
    _resolved_path: Optional[str] = None

    @model_validator(mode='after')
    def validate_source(self):
        """Ensure either file or inline is provided."""
        if not self.file and not self.inline:
            raise ValueError(f"Script '{self.name}': Either 'file' or 'inline' must be provided")
        if self.file and self.inline:
            raise ValueError(f"Script '{self.name}': Cannot specify both 'file' and 'inline'")
        return self


class ScriptConfig(BaseModel):
    """
    Script configuration section in skill metadata.

    Example in SKILL.md:
        scripts:
          enabled: true
          working_directory: ./scripts
          definitions:
            - name: analyze
              type: python
              file: analyze.py
    """
    enabled: bool = Field(default=True, description="Enable scripts for this skill")
    working_directory: Optional[str] = Field(default=None, description="Base directory for scripts")
    definitions: List[SkillScript] = Field(default_factory=list, alias="definitions")

    def get_script(self, name: str) -> Optional[SkillScript]:
        """Get script by name."""
        for script in self.definitions:
            if script.name == name:
                return script
        return None

    def get_activation_scripts(self) -> List[SkillScript]:
        """Get scripts to run on activation."""
        return [s for s in self.definitions if s.run_on_activation]

    def get_deactivation_scripts(self) -> List[SkillScript]:
        """Get scripts to run on deactivation."""
        return [s for s in self.definitions if s.run_on_deactivation]


class ScriptResult(BaseModel):
    """Result of script execution."""
    script_name: str = Field(..., description="Name of executed script")
    success: bool = Field(..., description="Whether execution succeeded")
    exit_code: int = Field(default=-1, description="Process exit code")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def to_string(self) -> str:
        """Convert to string for agent context."""
        if self.success:
            return self.stdout.strip() if self.stdout else "(no output)"
        return f"Error: {self.error or self.stderr}"


class SkillTrigger(BaseModel):
    """Trigger configuration for skill activation."""
    type: Literal["keyword", "pattern", "intent"] = Field(
        default="keyword",
        description="Trigger type: keyword (exact match), pattern (regex), intent (LLM-powered)"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that trigger this skill"
    )
    patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns that trigger this skill"
    )
    intent_category: Optional[str] = Field(
        default=None,
        description="Intent category for LLM-powered matching"
    )
    priority: int = Field(
        default=0,
        description="Higher priority triggers are checked first"
    )


class SkillParameter(BaseModel):
    """Parameter definition for skills."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="Parameter type")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values")


class SkillPrerequisite(BaseModel):
    """Prerequisites for skill execution."""
    skills: List[str] = Field(
        default_factory=list,
        description="Required skills that must be available"
    )
    tools: List[str] = Field(
        default_factory=list,
        description="Required tools that must be available"
    )
    env_vars: List[str] = Field(
        default_factory=list,
        description="Required environment variables"
    )


class SkillMetadata(BaseModel):
    """
    Skill metadata from YAML frontmatter.

    Required fields (Anthropic-compatible):
    - name: Unique skill identifier
    - description: Human-readable description

    Optional fields (XSpoonAi extensions):
    - triggers, parameters, prerequisites, composes, etc.
    """
    # Required fields (Anthropic-compatible)
    name: str = Field(..., description="Unique skill identifier")
    description: str = Field(..., description="Human-readable description")

    # Optional metadata
    version: str = Field(default="1.0.0", description="Skill version")
    author: Optional[str] = Field(default=None, description="Skill author")
    tags: List[str] = Field(default_factory=list, description="Skill tags for categorization")

    # Trigger configuration
    triggers: List[SkillTrigger] = Field(
        default_factory=list,
        description="Trigger configurations for skill activation"
    )

    # Parameters
    parameters: List[SkillParameter] = Field(
        default_factory=list,
        description="Parameters the skill accepts"
    )

    # Prerequisites
    prerequisites: SkillPrerequisite = Field(
        default_factory=SkillPrerequisite,
        description="Prerequisites for skill execution"
    )

    # Composition
    composes: List[str] = Field(
        default_factory=list,
        description="Skills this skill can invoke"
    )
    composable: bool = Field(
        default=True,
        description="Whether this skill can be invoked by other skills"
    )

    # State management
    persist_state: bool = Field(
        default=False,
        description="Whether to persist skill state between calls"
    )
    context_window: int = Field(
        default=4096,
        description="Token budget for skill context"
    )

    # Script configuration
    scripts: Optional[ScriptConfig] = Field(
        default=None,
        description="Script execution configuration"
    )

    def has_scripts(self) -> bool:
        """Check if skill has scripts defined."""
        return self.scripts is not None and len(self.scripts.definitions) > 0

    def scripts_enabled(self) -> bool:
        """Check if scripts are enabled for this skill."""
        return self.scripts is not None and self.scripts.enabled


class Skill(BaseModel):
    """
    Complete skill definition.

    Combines metadata from YAML frontmatter with markdown instructions.
    """
    metadata: SkillMetadata = Field(..., description="Skill metadata from YAML frontmatter")
    instructions: str = Field(..., description="Markdown instructions from SKILL.md")

    # Runtime state
    state: SkillState = Field(
        default=SkillState.INACTIVE,
        description="Current activation state"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Path to SKILL.md file"
    )
    loaded_at: Optional[datetime] = Field(
        default=None,
        description="When the skill was loaded"
    )

    # Associated tools (from optional tools.py)
    tool_names: List[str] = Field(
        default_factory=list,
        description="Names of tools loaded from skill directory"
    )

    # Associated scripts
    script_names: List[str] = Field(
        default_factory=list,
        description="Names of scripts available in this skill"
    )

    # Execution context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime context for skill execution"
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def name(self) -> str:
        """Convenience property for skill name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Convenience property for skill description."""
        return self.metadata.description

    def get_prompt_injection(self) -> str:
        """
        Generate prompt content to inject into agent's system prompt.

        Returns:
            Formatted skill instructions with metadata
        """
        lines = [
            f"## Skill: {self.metadata.name}",
            f"*{self.metadata.description}*",
            "",
            self.instructions,
        ]

        if self.metadata.parameters:
            lines.append("")
            lines.append("### Parameters")
            for param in self.metadata.parameters:
                req = "(required)" if param.required else "(optional)"
                default = f" [default: {param.default}]" if param.default is not None else ""
                lines.append(f"- **{param.name}** ({param.type}) {req}{default}: {param.description}")

        # Add script information
        if self.metadata.has_scripts() and self.metadata.scripts_enabled():
            lines.append("")
            lines.append("### Available Scripts")
            lines.append("You can execute these scripts when needed:")
            for script in self.metadata.scripts.definitions:
                desc = f" - {script.description}" if script.description else ""
                lines.append(f"- **{script.name}** ({script.type.value}){desc}")

        return "\n".join(lines)
