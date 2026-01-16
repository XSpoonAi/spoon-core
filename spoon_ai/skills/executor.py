"""
Script execution engine for skills.

Provides async subprocess management for executing skill scripts.
AI decides how to call scripts - users only control whether scripts are allowed.
"""

import os
import sys
import asyncio
import shutil
import time
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from spoon_ai.skills.models import SkillScript, ScriptType, ScriptResult

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 600
MAX_OUTPUT_SIZE = 5 * 1024 * 1024  # 5MB


class ScriptExecutionError(Exception):
    """Raised when script execution fails."""
    pass


class ScriptExecutor:
    """
    Async script executor for skill scripts.

    Features:
    - Async subprocess execution with timeout
    - Support for Python, shell, bash scripts
    - Environment variable passthrough
    - Output capture and size limiting
    - Global enable/disable control
    """

    def __init__(
        self,
        enabled: bool = True,
        default_timeout: int = DEFAULT_TIMEOUT,
        max_output_size: int = MAX_OUTPUT_SIZE,
        env_passthrough: Optional[List[str]] = None
    ):
        """
        Initialize script executor.

        Args:
            enabled: Whether script execution is allowed
            default_timeout: Default timeout in seconds
            max_output_size: Max output capture size in bytes
            env_passthrough: Environment variables to pass through
        """
        self.enabled = enabled
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.env_passthrough = env_passthrough or [
            'PATH', 'HOME', 'USER', 'LANG', 'TERM',
            'PYTHONPATH', 'NODE_PATH', 'VIRTUAL_ENV',
            # Common API keys
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'TAVILY_API_KEY',
        ]

        # Detect available interpreters
        self._interpreters = self._detect_interpreters()

        # Execution stats
        self._stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "timed_out": 0,
            "blocked": 0
        }

    def _detect_interpreters(self) -> Dict[ScriptType, str]:
        """Detect available script interpreters."""
        interpreters = {}

        # Python - use current interpreter
        python_path = sys.executable
        if python_path:
            interpreters[ScriptType.PYTHON] = python_path

        # Bash
        bash_path = shutil.which('bash')
        if bash_path:
            interpreters[ScriptType.BASH] = bash_path

        # Shell (sh on Unix, bash or cmd on Windows)
        if os.name == 'nt':
            # Windows - prefer bash if available, else cmd
            if bash_path:
                interpreters[ScriptType.SHELL] = bash_path
            else:
                cmd_path = shutil.which('cmd')
                if cmd_path:
                    interpreters[ScriptType.SHELL] = cmd_path
        else:
            # Unix - use sh
            sh_path = shutil.which('sh') or '/bin/sh'
            if os.path.exists(sh_path):
                interpreters[ScriptType.SHELL] = sh_path

        logger.debug(f"Available interpreters: {interpreters}")
        return interpreters

    def _build_environment(self, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build environment for script execution."""
        env = {}

        # Pass through allowed variables
        for key in self.env_passthrough:
            if key in os.environ:
                env[key] = os.environ[key]

        # Add extra environment variables
        if extra_env:
            env.update(extra_env)

        return env

    def is_available(self, script_type: ScriptType) -> bool:
        """Check if a script type can be executed."""
        return script_type in self._interpreters

    def get_interpreter(self, script_type: ScriptType) -> Optional[str]:
        """Get interpreter path for a script type."""
        return self._interpreters.get(script_type)

    async def execute(
        self,
        script: SkillScript,
        input_text: Optional[str] = None,
        working_directory: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> ScriptResult:
        """
        Execute a script asynchronously.

        Args:
            script: Script to execute
            input_text: Optional text to pass to script via stdin
            working_directory: Working directory for execution
            extra_env: Additional environment variables
            timeout: Timeout override (uses script.timeout or default)

        Returns:
            ScriptResult with output and status
        """
        start_time = time.time()
        self._stats["total"] += 1

        # Check if execution is enabled
        if not self.enabled:
            self._stats["blocked"] += 1
            return ScriptResult(
                script_name=script.name,
                success=False,
                exit_code=-1,
                error="Script execution is disabled"
            )

        # Check interpreter availability
        if script.type not in self._interpreters:
            self._stats["failed"] += 1
            return ScriptResult(
                script_name=script.name,
                success=False,
                exit_code=-1,
                error=f"No interpreter available for {script.type.value}"
            )

        interpreter = self._interpreters[script.type]

        # Determine working directory
        cwd = working_directory or script.working_directory
        if cwd:
            cwd = str(Path(cwd).resolve())
            if not os.path.isdir(cwd):
                self._stats["failed"] += 1
                return ScriptResult(
                    script_name=script.name,
                    success=False,
                    exit_code=-1,
                    error=f"Working directory does not exist: {cwd}"
                )

        # Determine timeout
        exec_timeout = timeout or script.timeout or self.default_timeout
        exec_timeout = min(exec_timeout, MAX_TIMEOUT)

        # Build environment
        env = self._build_environment(extra_env)

        # Build command
        try:
            cmd, temp_file = await self._build_command(script, interpreter)
        except Exception as e:
            self._stats["failed"] += 1
            return ScriptResult(
                script_name=script.name,
                success=False,
                exit_code=-1,
                error=f"Failed to build command: {e}"
            )

        logger.info(f"Executing script '{script.name}' ({script.type.value})")
        logger.debug(f"Command: {cmd[:3]}... cwd={cwd}, timeout={exec_timeout}s")

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_text else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env if env else None
            )

            # Execute with timeout
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(
                        input=input_text.encode('utf-8') if input_text else None
                    ),
                    timeout=exec_timeout
                )
            except asyncio.TimeoutError:
                # Kill process on timeout
                process.kill()
                await process.wait()
                self._stats["timed_out"] += 1

                return ScriptResult(
                    script_name=script.name,
                    success=False,
                    exit_code=-1,
                    execution_time=exec_timeout,
                    error=f"Script timed out after {exec_timeout}s"
                )

            # Decode and truncate output
            stdout = stdout_data.decode('utf-8', errors='replace')
            stderr = stderr_data.decode('utf-8', errors='replace')

            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... [output truncated]"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... [output truncated]"

            success = process.returncode == 0
            execution_time = time.time() - start_time

            if success:
                self._stats["successful"] += 1
                logger.info(f"Script '{script.name}' completed successfully in {execution_time:.2f}s")
            else:
                self._stats["failed"] += 1
                logger.warning(f"Script '{script.name}' failed with exit code {process.returncode}")

            return ScriptResult(
                script_name=script.name,
                success=success,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                error=stderr if not success and stderr else None
            )

        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Script '{script.name}' execution error: {e}")

            return ScriptResult(
                script_name=script.name,
                success=False,
                exit_code=-1,
                execution_time=time.time() - start_time,
                error=str(e)
            )

        finally:
            # Clean up temp file if created
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

    async def _build_command(
        self,
        script: SkillScript,
        interpreter: str
    ) -> tuple[List[str], Optional[str]]:
        """
        Build command list for subprocess execution.

        Returns:
            Tuple of (command_list, temp_file_path_if_created)
        """
        temp_file = None

        if script.file:
            # File-based script
            script_path = script.file
            if hasattr(script, '_resolved_path') and script._resolved_path:
                script_path = script._resolved_path

            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script file not found: {script_path}")

            cmd = [interpreter, script_path]

        else:
            # Inline script - write to temp file
            suffix = '.py' if script.type == ScriptType.PYTHON else '.sh'
            fd, temp_file = tempfile.mkstemp(suffix=suffix, prefix=f'skill_script_{script.name}_')

            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(script.inline)

                cmd = [interpreter, temp_file]

            except Exception:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
                raise

        return cmd, temp_file

    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return self._stats.copy()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable script execution."""
        self.enabled = enabled
        logger.info(f"Script execution {'enabled' if enabled else 'disabled'}")


# Global executor instance
_executor: Optional[ScriptExecutor] = None


def get_executor() -> ScriptExecutor:
    """Get or create the global script executor."""
    global _executor
    if _executor is None:
        _executor = ScriptExecutor()
    return _executor


def configure_executor(**kwargs) -> ScriptExecutor:
    """Configure and return the global executor."""
    global _executor
    _executor = ScriptExecutor(**kwargs)
    return _executor


def set_scripts_enabled(enabled: bool) -> None:
    """Enable or disable script execution globally."""
    get_executor().set_enabled(enabled)
