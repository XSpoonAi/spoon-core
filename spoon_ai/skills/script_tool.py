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
    The AI decides what input to provide - there's no fixed parameter schema.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict = Field(default_factory=dict, description="Tool parameters schema")

    # Script configuration
    script: SkillScript = Field(..., exclude=True)
    skill_name: str = Field(..., exclude=True)
    working_directory: Optional[str] = Field(default=None, exclude=True)

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

        # Build description and parameters based on input_schema
        desc = script.description or f"Execute the '{script.name}' script"

        if script.input_schema:
            # Script declares its own input schema
            description = f"{desc}. Pass a JSON string as 'input' with this structure: {json.dumps(script.input_schema)}. The JSON is sent to the script via stdin."
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": f"JSON string to pass to script via stdin. Must match schema: {json.dumps(script.input_schema)}"
                    }
                },
                "required": ["input"]
            }
        else:
            # Fallback: generic description
            description = f"{desc} (Type: {script.type.value}). If the script expects input, pass a JSON string as 'input'."
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input text (usually JSON) to pass to the script via stdin"
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

    async def execute(self, input: Optional[str] = None, **kwargs) -> str:
        """
        Execute the script.

        Args:
            input: Optional input text to pass to script via stdin
            **kwargs: Additional arguments (ignored)

        Returns:
            Script output as string
        """
        executor = get_executor()

        logger.debug(f"ScriptTool '{self.name}' executing")

        result: ScriptResult = await executor.execute(
            script=self.script,
            input_text=input,
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
