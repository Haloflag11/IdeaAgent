"""Bash command executor for IdeaAgent.

This module provides functionality to execute shell commands in the .venv
virtual environment with real-time output streaming.
"""

import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple


class BashExecutor:
    """Execute bash commands in the .venv virtual environment.
    
    Features:
    - Runs commands in .venv environment
    - Real-time output streaming
    - Timeout control
    - Returns (returncode, stdout, stderr)
    """
    
    def __init__(
        self,
        venv_path: Optional[Path] = None,
        timeout: int = 300,
    ):
        """Initialize the bash executor.
        
        Args:
            venv_path: Path to the .venv directory. If None, will be auto-detected.
            timeout: Default timeout for command execution in seconds.
        """
        self.venv_path = venv_path or self._find_venv()
        self.timeout = timeout
        self._python_path = self._get_python_path()
        self._pip_path = self._get_pip_path()
    
    def _find_venv(self) -> Path:
        """Find .venv directory by searching from current directory upwards."""
        current = Path.cwd()
        while current != current.parent:
            venv_path = current / ".venv"
            if venv_path.exists():
                return venv_path
            current = current.parent
        
        # Try common locations
        possible_paths = [
            Path.cwd() / ".venv",
            Path.home() / ".venv",
        ]
        
        for venv_path in possible_paths:
            if venv_path.exists():
                return venv_path
        
        raise FileNotFoundError(
            ".venv not found. Please create virtual environment with: python -m venv .venv"
        )
    
    def _get_python_path(self) -> Path:
        """Get python executable path in venv."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def _get_pip_path(self) -> Path:
        """Get pip executable path in venv."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def _build_env(self) -> dict:
        """Build environment variables for subprocess execution.
        
        Returns:
            dict suitable for passing as env= to subprocess.run/Popen.
        """
        import os
        
        env = os.environ.copy()
        
        # Prepend venv Scripts/bin to PATH
        if sys.platform == "win32":
            venv_bin = str(self.venv_path / "Scripts")
        else:
            venv_bin = str(self.venv_path / "bin")
        
        existing_path = env.get("PATH", "")
        env["PATH"] = venv_bin + os.pathsep + existing_path
        
        # Set VIRTUAL_ENV to advertise the active venv
        env["VIRTUAL_ENV"] = str(self.venv_path)
        
        # Force UTF-8 encoding for stdout/stderr
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        
        # Disable output buffering for real-time streaming
        env["PYTHONUNBUFFERED"] = "1"
        
        # Remove conda-related variables to prevent environment leakage
        for var in (
            "PYTHONPATH", "PYTHONSTARTUP",
            "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
            "CONDA_EXE", "CONDA_PYTHON_EXE", "_CONDA_SET_OLDPATH",
        ):
            env.pop(var, None)
        
        return env
    
    def run(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        realtime_output: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute a bash command in the .venv environment.
        
        Args:
            command: Shell command to execute.
            cwd: Working directory for the command.
            timeout: Execution timeout in seconds (overrides default).
            realtime_output: Whether to print output in real-time.
        
        Returns:
            Tuple of (returncode, stdout, stderr).
        
        Raises:
            TimeoutError: If command execution times out.
            RuntimeError: If command execution fails.
        """
        exec_timeout = timeout or self.timeout
        
        if realtime_output:
            return self._run_with_realtime_output(command, cwd, exec_timeout)
        else:
            return self._run_captured_output(command, cwd, exec_timeout)
    
    def _run_captured_output(
        self,
        command: str,
        cwd: Path,
        timeout: int,
    ) -> Tuple[int, str, str]:
        """Run command with captured output (no real-time display)."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                encoding='utf-8',
                errors='replace',
                env=self._build_env(),
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout}s: {command}")
        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {e}")
    
    def _run_with_realtime_output(
        self,
        command: str,
        cwd: Path,
        timeout: int,
    ) -> Tuple[int, str, str]:
        """Run command with real-time output streaming."""
        import time
        
        collected_stdout = []
        collected_stderr = []
        
        def read_and_print_stream(stream, collected_list, is_stderr):
            """Read stream character by character, print immediately, and collect."""
            try:
                while True:
                    chunk = stream.read(1)
                    if not chunk:
                        break
                    
                    # Write directly to terminal for immediate display
                    if is_stderr:
                        sys.stderr.write(chunk)
                        sys.stderr.flush()
                    else:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                    
                    # Collect for return
                    collected_list.append(chunk)
            except Exception:
                pass
            finally:
                stream.close()
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=self._build_env(),
            )
            
            # Start threads to read, print, and collect output
            stdout_thread = threading.Thread(
                target=read_and_print_stream,
                args=(process.stdout, collected_stdout, False),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=read_and_print_stream,
                args=(process.stderr, collected_stderr, True),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for completion with timeout
            start_time = time.time()
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    process.kill()
                    raise TimeoutError(f"Command timed out after {timeout}s: {command}")
                time.sleep(0.05)
            
            # Wait for threads to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            returncode = process.returncode
            stdout_result = ''.join(collected_stdout)
            stderr_result = ''.join(collected_stderr)
            
            return returncode, stdout_result, stderr_result
            
        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {e}")
    
    def run_python(
        self,
        script: str,
        args: Optional[list] = None,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        realtime_output: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute a Python script in the .venv environment.

        Args:
            script: Path to the Python script OR an inline Python code string.
            args: Optional list of command-line arguments.
            cwd: Working directory for the command.
            timeout: Execution timeout in seconds.
            realtime_output: Whether to print output in real-time.

        Returns:
            Tuple of (returncode, stdout, stderr).

        Notes:
            When *script* is an inline code string (not an existing .py file),
            the code is written to a **temporary file** and executed via
            ``python <tempfile>``.  This avoids all shell-quoting / multi-line
            breakage that occurs when code is passed via ``python -c "..."``.
            The temporary file is always removed after execution.
        """
        import tempfile
        import os

        exec_cwd = cwd or Path.cwd()

        # ── Case 1: existing .py file path ───────────────────────────────────
        if script.endswith('.py') and Path(script).exists():
            cmd_parts = [str(self._python_path), script]
            if args:
                cmd_parts.extend(args)
            return self.run(
                command=' '.join(cmd_parts),
                cwd=exec_cwd,
                timeout=timeout,
                realtime_output=realtime_output,
            )

        # ── Case 2: inline code string ───────────────────────────────────────
        # Write to a temp file so that:
        #   * Multi-line code is never corrupted by shell quoting.
        #   * Single/double quotes inside the code don't break the command.
        #   * The .venv Python that already has all packages is used.
        tmp_file = None
        try:
            # Use a named temp file in the working directory so relative
            # imports / file paths inside the code resolve correctly.
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                dir=exec_cwd,
                delete=False,
                encoding='utf-8',
            ) as f:
                tmp_file = Path(f.name)
                f.write(script)

            cmd_parts = [str(self._python_path), str(tmp_file)]
            if args:
                cmd_parts.extend(args)

            return self.run(
                command=' '.join(f'"{p}"' if ' ' in p else p for p in cmd_parts),
                cwd=exec_cwd,
                timeout=timeout,
                realtime_output=realtime_output,
            )
        finally:
            # Always clean up the temp file
            if tmp_file is not None:
                try:
                    tmp_file.unlink(missing_ok=True)
                except Exception:
                    pass
    
    def run_pip(
        self,
        packages: list[str],
        upgrade: bool = False,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        realtime_output: bool = True,
    ) -> Tuple[int, str, str]:
        """Install packages using pip in the .venv environment.
        
        Args:
            packages: List of package names to install.
            upgrade: Whether to upgrade existing packages.
            cwd: Working directory for the command.
            timeout: Execution timeout in seconds.
            realtime_output: Whether to print output in real-time.
        
        Returns:
            Tuple of (returncode, stdout, stderr).
        """
        cmd = [str(self._python_path), "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(packages)
        
        return self.run(
            command=' '.join(cmd),
            cwd=cwd,
            timeout=timeout,
            realtime_output=realtime_output,
        )