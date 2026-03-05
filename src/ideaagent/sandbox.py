"""Sandboxed execution using .venv virtual environment for IdeaAgent."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import json

from dotenv import load_dotenv

load_dotenv()


class SandboxExecutionError(Exception):
    """Exception raised when sandbox execution fails."""
    pass


class VenvSandbox:
    """Manages .venv-based sandboxed execution environments."""

    def __init__(
        self,
        timeout: int = 300,
        workspace: Optional[Path] = None,
    ):
        """Initialize venv sandbox.
        
        Args:
            timeout: Execution timeout in seconds
            workspace: Base workspace directory for execution
        """
        self.timeout = timeout
        self.workspace = workspace or Path.cwd() / "sandbox_workspaces"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self._venv_path = self._find_venv()
        self._python_path = self._get_python_path()
        
    def _find_venv(self) -> Path:
        """Find .venv directory."""
        # Try current directory and parents
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
        
        raise SandboxExecutionError(
            ".venv not found. Please create virtual environment with: python -m venv .venv"
        )
    
    def _get_python_path(self) -> Path:
        """Get python executable path in venv."""
        if sys.platform == "win32":
            return self._venv_path / "Scripts" / "python.exe"
        else:
            return self._venv_path / "bin" / "python"
    
    def _get_pip_path(self) -> Path:
        """Get pip executable path in venv."""
        if sys.platform == "win32":
            return self._venv_path / "Scripts" / "pip.exe"
        else:
            return self._venv_path / "bin" / "pip"

    def _build_subprocess_env(self) -> dict:
        """Build a clean, isolated environment for subprocess execution.

        Starts from the current process environment and then:

        * Puts the venv ``Scripts`` / ``bin`` directory **first** in ``PATH``
          so that the venv's Python and scripts shadow anything on the system
          path (including a parent conda environment).
        * Sets ``VIRTUAL_ENV`` to the venv root so tools that inspect it
          behave correctly.
        * Removes ``PYTHONPATH`` – an inherited ``PYTHONPATH`` can inject
          site-packages from the conda base (or any other Python environment)
          into the subprocess, defeating the venv isolation completely.
        * Removes conda activation variables (``CONDA_PREFIX``,
          ``CONDA_DEFAULT_ENV``, ``CONDA_PROMPT_MODIFIER``,
          ``CONDA_EXE``, ``CONDA_PYTHON_EXE``) so that conda-aware code
          inside the script does not accidentally use the base environment.

        Returns:
            dict suitable for passing as ``env=`` to ``subprocess.run`` /
            ``subprocess.Popen``.
        """
        env = os.environ.copy()

        # Prepend the venv's bin directory to PATH
        if sys.platform == "win32":
            venv_bin = str(self._venv_path / "Scripts")
        else:
            venv_bin = str(self._venv_path / "bin")

        existing_path = env.get("PATH", "")
        env["PATH"] = venv_bin + os.pathsep + existing_path

        # Advertise the active venv (some tools / scripts inspect this)
        env["VIRTUAL_ENV"] = str(self._venv_path)

        # Force UTF-8 for stdout/stderr in all subprocesses.
        # This is the CRITICAL fix for UnicodeEncodeError on Windows GBK terminals.
        # Without this, any Unicode character in print() statements (e.g. ✓ \u2713,
        # ² \u00b2) causes: UnicodeEncodeError: 'gbk' codec can't encode character
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"  # PEP 540: Python UTF-8 Mode (Python 3.7+)
        
        # Disable output buffering for realtime output streaming.
        # When Python detects output is redirected to a pipe, it uses full
        # buffering instead of line buffering. Setting PYTHONUNBUFFERED=1 forces
        # unbuffered mode so print() output appears immediately.
        env["PYTHONUNBUFFERED"] = "1"

        # Strip variables that can leak packages from other environments
        for var in (
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_PROMPT_MODIFIER",
            "CONDA_EXE",
            "CONDA_PYTHON_EXE",
            "_CONDA_SET_OLDPATH",
        ):
            env.pop(var, None)

        return env
    
    def environment_exists(self) -> bool:
        """Check if virtual environment exists."""
        return self._venv_path.exists() and self._python_path.exists()
    
    def install_packages(self, packages: list[str], upgrade: bool = False) -> bool:
        """Install packages in the virtual environment with realtime output.

        Uses ``python -m pip install`` (not ``pip.exe``) so packages
        are written to exactly the same ``sys.path`` location as the Python
        interpreter that will later execute the generated scripts.

        Args:
            packages: List of packages to install
            upgrade: Whether to upgrade packages

        Returns:
            True if successful

        Raises:
            SandboxExecutionError: If installation fails
        """
        import threading
        import queue
        
        try:
            # Always drive pip through the *same* Python that runs scripts so
            # that packages land in the right site-packages directory.
            cmd = [str(self._python_path), "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.extend(packages)

            # Use Popen with realtime output streaming
            stdout_queue = queue.Queue()
            stderr_queue = queue.Queue()
            
            def read_stream(stream, target_queue, is_stdout):
                """Read stream, print in realtime, and collect."""
                try:
                    for line in iter(stream.readline, ''):
                        if line:
                            # Print in realtime
                            if is_stdout:
                                print(line, end='', flush=True)
                            else:
                                print(line, end='', file=sys.stderr, flush=True)
                            # Also collect
                            target_queue.put(line)
                except Exception:
                    pass
                finally:
                    stream.close()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.workspace,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=self._build_subprocess_env(),
            )
            
            # Start threads
            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_queue, True))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_queue, False))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Collect output
            stdout_lines = []
            stderr_lines = []
            
            def drain_queue(q, lines_list):
                while not q.empty():
                    try:
                        lines_list.append(q.get_nowait())
                    except queue.Empty:
                        break
            
            # Wait with timeout
            start_time = datetime.now()
            import time
            while process.poll() is None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > self.timeout * 2:
                    process.kill()
                    raise SandboxExecutionError(f"Package installation timed out after {self.timeout * 2}s")
                
                drain_queue(stdout_queue, stdout_lines)
                drain_queue(stderr_queue, stderr_lines)
                time.sleep(0.05)
            
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            drain_queue(stdout_queue, stdout_lines)
            drain_queue(stderr_queue, stderr_lines)
            
            if process.returncode != 0:
                raise SandboxExecutionError(
                    f"Package installation failed:\n{''.join(stderr_lines)}\n{''.join(stdout_lines)}"
                )

            return True

        except subprocess.TimeoutExpired:
            raise SandboxExecutionError(f"Package installation timed out after {self.timeout * 2}s")
        except SandboxExecutionError:
            raise
        except Exception as e:
            raise SandboxExecutionError(f"Failed to install packages: {e}")
    
    def run_script(
        self,
        script_path: Path,
        cwd: Optional[Path] = None,
        realtime_output: bool = True,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Run a Python script in the virtual environment with optional realtime output.
        
        Args:
            script_path: Path to the Python script
            cwd: Working directory
            realtime_output: Whether to print output in realtime
            timeout: Execution timeout (overrides default)
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            if not self.environment_exists():
                raise SandboxExecutionError("Virtual environment does not exist")
            
            cmd = [str(self._python_path), str(script_path)]
            
            exec_timeout = timeout or self.timeout
            
            if realtime_output:
                # Run with realtime output streaming
                return self._run_with_realtime_output(cmd, cwd or self.workspace, exec_timeout)
            else:
                # Run with captured output (original behavior)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=exec_timeout,
                    cwd=cwd or self.workspace,
                    encoding='utf-8',
                    errors='replace',
                    env=self._build_subprocess_env(),
                )
                return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            raise SandboxExecutionError(f"Script execution timed out after {exec_timeout}s")
        except Exception as e:
            raise SandboxExecutionError(f"Failed to run script: {e}")
    
    def _run_with_realtime_output(
        self,
        cmd: list,
        cwd: Path,
        timeout: int,
    ) -> Tuple[int, str, str]:
        """Run command with realtime output streaming.
        
        This method streams output directly to the terminal in real-time,
        AND collects it for return to the caller. This ensures that all
        print() and log() statements from the subprocess are visible immediately.
        
        Args:
            cmd: Command to run
            cwd: Working directory
            timeout: Execution timeout
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        import threading
        import time
        
        collected_stdout = []
        collected_stderr = []
        
        def read_and_print_stream(stream, collected_list, is_stderr):
            """Read stream char by char, print immediately, and collect."""
            try:
                while True:
                    chunk = stream.read(1)  # Read one character at a time
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
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=self._build_subprocess_env(),
            )
            
            # Start threads to read, print, and collect output
            stdout_thread = threading.Thread(
                target=read_and_print_stream, 
                args=(process.stdout, collected_stdout, False)
            )
            stderr_thread = threading.Thread(
                target=read_and_print_stream, 
                args=(process.stderr, collected_stderr, True)
            )
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for completion with timeout
            start_time = datetime.now()
            while process.poll() is None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    process.kill()
                    raise SandboxExecutionError(f"Script execution timed out after {timeout}s")
                time.sleep(0.05)
            
            # Wait for threads to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            returncode = process.returncode
            stdout_result = ''.join(collected_stdout)
            stderr_result = ''.join(collected_stderr)
            
            return returncode, stdout_result, stderr_result
            
        except Exception as e:
            raise SandboxExecutionError(f"Failed to run script with realtime output: {e}")
    
    # Standard subdirectories created in every task workspace
    WORKSPACE_SUBDIRS: tuple[str, ...] = ("data", "models", "plots", "results", "logs")

    def create_task_workspace(self, task_name: str) -> Path:
        """Create a workspace directory for a task, including standard subdirs.

        The following subdirectories are always pre-created so that LLM-generated
        code can write to them without extra ``mkdir`` calls:

        * ``data/``    – raw and processed datasets (.csv, .json, .npy …)
        * ``models/``  – saved model artefacts (.pkl, .pth, .h5 …)
        * ``plots/``   – visualisation outputs (.png, .svg …)
        * ``results/`` – metrics, evaluation reports (.json, .csv …)
        * ``logs/``    – training / run logs (.txt, .csv …)

        Args:
            task_name: Name of the task (used for directory naming)

        Returns:
            Path to the task workspace directory
        """
        # Sanitize task name for filesystem
        safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in task_name)
        safe_name = safe_name[:50]  # Limit length

        workspace_dir = self.workspace / safe_name
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create standard subdirectories
        for sub in self.WORKSPACE_SUBDIRS:
            (workspace_dir / sub).mkdir(exist_ok=True)

        return workspace_dir
    
    def execute_in_sandbox(
        self,
        workspace_dir: Path,
        code: str,
        packages: Optional[list[str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str, str]:
        """Execute Python code in sandbox with automatic package installation.
        
        Args:
            workspace_dir: Task workspace directory (all files saved here)
            code: Python code to execute
            packages: Optional list of packages to install
            timeout: Execution timeout
            
        Returns:
            Tuple of (success, output, error)
        """
        try:
            # Install packages if specified
            if packages:
                self.install_packages(packages)
            
            # Write code to script in the workspace.
            # Prepend a UTF-8 stdout/stderr reconfigure header so that any
            # Unicode characters that slip through sanitize_unicode() don't
            # trigger UnicodeEncodeError on Windows GBK/CP936 consoles.
            # This is the most reliable fix: PYTHONIOENCODING env var alone
            # is sometimes ignored when the stream is already open.
            encoding_header = (
                "import sys as _sys\n"
                "try:\n"
                "    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
                "    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')\n"
                "except AttributeError:\n"
                "    pass  # Python < 3.7 fallback\n"
                "\n"
            )
            # Also run sanitize_unicode as a final safety net before writing
            from .utils.code_parser import sanitize_unicode
            safe_code = sanitize_unicode(code)
            script_path = workspace_dir / "script.py"
            script_path.write_text(encoding_header + safe_code, encoding='utf-8')
            
            # Execute script in the workspace directory
            returncode, stdout, stderr = self.run_script(
                script_path=script_path,
                cwd=workspace_dir,
                realtime_output=True,
                timeout=timeout or self.timeout,
            )
            
            success = returncode == 0
            return success, stdout, stderr
            
        except SandboxExecutionError as e:
            return False, "", str(e)
        except Exception as e:
            return False, "", f"Unexpected error: {e}"
