"""
Script-based tool for agent integration.

Wraps SkillScript as a BaseTool that agents can call.
AI decides how to use scripts - users only control whether scripts are allowed.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from spoon_ai.tools.base import BaseTool
from spoon_ai.skills.models import SkillScript, ScriptResult
from spoon_ai.skills.executor import get_executor

logger = logging.getLogger(__name__)


class ScriptTool(BaseTool):
    """
    Tool wrapper for skill scripts.

    Exposes a SkillScript as a callable tool that agents can invoke.
    When the script defines an ``input_schema``, the tool parameters are
    derived from that schema so the LLM receives a structured contract.
    Otherwise a generic ``input`` string parameter is used for backward
    compatibility.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict = Field(default_factory=dict, description="Tool parameters schema")

    # Script configuration
    script: SkillScript = Field(..., exclude=True)
    skill_name: str = Field(..., exclude=True)
    working_directory: Optional[str] = Field(default=None, exclude=True)
    _uses_structured_schema: bool = False

    def __init__(
        self,
        script: SkillScript,
        skill_name: str,
        working_directory: Optional[str] = None
    ):
        """
        Create a tool from a script definition.

        Args:
            script: SkillScript to wrap
            skill_name: Parent skill name
            working_directory: Base working directory
        """
        # Generate tool name
        tool_name = f"run_script_{skill_name}_{script.name}"

        # Build description
        desc = script.description or f"Execute the '{script.name}' script"
        description = f"{desc} (Type: {script.type.value})"

        # Derive parameter schema from script.input_schema when available (#8)
        uses_structured = False
        if script.input_schema and isinstance(script.input_schema, dict):
            schema_type = script.input_schema.get("type", "object")
            # Tool/function calling interfaces expect top-level object schema.
            # If skill metadata declares non-object type, degrade gracefully.
            if schema_type != "object":
                logger.warning(
                    "Script '%s' in skill '%s' has non-object input_schema.type=%s; "
                    "falling back to generic object schema",
                    script.name,
                    skill_name,
                    schema_type,
                )
                parameters = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Optional input text to pass to the script via stdin"
                        }
                    },
                    "required": []
                }
            else:
                parameters = {
                    "type": "object",
                    "properties": script.input_schema.get("properties", {}),
                    "required": script.input_schema.get("required", []),
                }
                uses_structured = True
        else:
            # Fallback: generic optional input string (backward compat)
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Optional input text to pass to the script via stdin"
                    }
                },
                "required": []
            }

        super().__init__(
            name=tool_name,
            description=description,
            parameters=parameters,
            script=script,
            skill_name=skill_name,
            working_directory=working_directory
        )
        object.__setattr__(self, "_uses_structured_schema", uses_structured)

    async def execute(self, input: Optional[str] = None, **kwargs) -> str:
        """
        Execute the script.

        When the script declares an ``input_schema``, the LLM's structured
        kwargs are serialized to JSON and piped to stdin.  For legacy scripts
        that only declare a generic ``input`` string, the raw value is passed
        through as-is.

        Args:
            input: Optional input text (legacy path)
            **kwargs: Structured arguments matching input_schema

        Returns:
            Script output as string
        """
        executor = get_executor()

        logger.debug(f"ScriptTool '{self.name}' executing")

        # Decide what to send to the script on stdin
        if self._uses_structured_schema:
            # Build a JSON payload from all kwargs (including 'input' if present)
            payload: Dict[str, Any] = {}
            if input is not None:
                payload["input"] = input
            payload.update(kwargs)
            input_text = json.dumps(payload, ensure_ascii=False)
        else:
            # Legacy path: plain string or try JSON passthrough
            input_text = input
            if input_text is None and kwargs:
                # Model may have sent structured args despite generic schema
                input_text = json.dumps(kwargs, ensure_ascii=False)

        result: ScriptResult = await executor.execute(
            script=self.script,
            input_text=input_text,
            working_directory=self.working_directory
        )

        if result.success:
            return result.stdout if result.stdout else "(script completed with no output)"
        else:
            # On failure, provide as much context as possible
            error_msg = result.error or result.stderr
            if not error_msg and result.stdout:
                # Some scripts (like tavily_search.py) print error JSON to stdout
                error_msg = result.stdout
            
            return f"Script failed: {error_msg or 'Unknown error (no output captured)'}"

    def to_param(self) -> dict:
        """Generate OpenAI-compatible function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def create_script_tools(
    skill_name: str,
    scripts: List[SkillScript],
    working_directory: Optional[str] = None
) -> List[ScriptTool]:
    """
    Create ScriptTool instances from script definitions.

    Args:
        skill_name: Parent skill name
        scripts: List of script definitions
        working_directory: Base working directory (fallback if script has none)

    Returns:
        List of ScriptTool instances
    """
    tools = []

    for script in scripts:
        try:
            # Per-script working_directory takes precedence over skill-level
            script_working_dir = script.working_directory or working_directory
            tool = ScriptTool(
                script=script,
                skill_name=skill_name,
                working_directory=script_working_dir
            )
            tools.append(tool)
            logger.debug(f"Created script tool: {tool.name}")
        except Exception as e:
            logger.error(f"Failed to create tool for script '{script.name}': {e}")

    return tools
