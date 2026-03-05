"""File manager for IdeaAgent workspace operations.

This module provides functionality to manage files and directories
in the workspace, including mkdir, write_file, and read_file operations.
"""

from pathlib import Path
from typing import Optional


class FileManager:
    """Manage files and directories in the workspace.
    
    Features:
    - Create directories (mkdir)
    - Write files (write_file)
    - Read files (read_file)
    - All operations are relative to workspace directory
    """
    
    def __init__(self, workspace_dir: Path):
        """Initialize the file manager.
        
        Args:
            workspace_dir: Root directory of the workspace.
        """
        self.workspace_dir = workspace_dir
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a relative path to absolute path within workspace.
        
        Args:
            path: Relative path string.
        
        Returns:
            Absolute Path within workspace.
        
        Raises:
            ValueError: If path is absolute or tries to escape workspace.
        """
        # Convert to Path object
        rel_path = Path(path)
        
        # Check for absolute paths
        if rel_path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed: {path}")
        
        # Check for path traversal attempts
        try:
            resolved = (self.workspace_dir / rel_path).resolve()
            if not str(resolved).startswith(str(self.workspace_dir.resolve())):
                raise ValueError(f"Path escapes workspace: {path}")
            return resolved
        except Exception as e:
            raise ValueError(f"Invalid path: {path} - {e}")
    
    def mkdir(
        self,
        path: str,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> dict:
        """Create a directory in the workspace.
        
        Args:
            path: Relative path to the directory.
            parents: If True, create parent directories as needed (like mkdir -p).
            exist_ok: If True, don't raise error if directory exists.
        
        Returns:
            dict with status and message.
        """
        try:
            dir_path = self._resolve_path(path)
            dir_path.mkdir(parents=parents, exist_ok=exist_ok)
            return {
                "success": True,
                "path": str(dir_path),
                "message": f"Created directory: {path}",
            }
        except Exception as e:
            return {
                "success": False,
                "path": path,
                "error": str(e),
            }
    
    def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        parents: bool = True,
    ) -> dict:
        """Write content to a file in the workspace.
        
        Args:
            path: Relative path to the file.
            content: Content to write to the file.
            encoding: File encoding (default: utf-8).
            parents: If True, create parent directories as needed.
        
        Returns:
            dict with status and message.
        """
        try:
            file_path = self._resolve_path(path)
            
            # Create parent directories if needed
            if parents:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            file_path.write_text(content, encoding=encoding)
            
            return {
                "success": True,
                "path": str(file_path),
                "message": f"Written to file: {path}",
                "bytes_written": len(content.encode(encoding)),
            }
        except Exception as e:
            return {
                "success": False,
                "path": path,
                "error": str(e),
            }
    
    def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
    ) -> dict:
        """Read content from a file in the workspace.
        
        Args:
            path: Relative path to the file.
            encoding: File encoding (default: utf-8).
        
        Returns:
            dict with status, content (if successful), or error message.
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    "success": False,
                    "path": path,
                    "error": f"File not found: {path}",
                }
            
            if not file_path.is_file():
                return {
                    "success": False,
                    "path": path,
                    "error": f"Not a file: {path}",
                }
            
            content = file_path.read_text(encoding=encoding)
            
            return {
                "success": True,
                "path": str(file_path),
                "content": content,
                "bytes_read": len(content.encode(encoding)),
            }
        except Exception as e:
            return {
                "success": False,
                "path": path,
                "error": str(e),
            }
    
    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the workspace.
        
        Args:
            path: Relative path to the file.
        
        Returns:
            True if file exists, False otherwise.
        """
        try:
            file_path = self._resolve_path(path)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False
    
    def dir_exists(self, path: str) -> bool:
        """Check if a directory exists in the workspace.
        
        Args:
            path: Relative path to the directory.
        
        Returns:
            True if directory exists, False otherwise.
        """
        try:
            dir_path = self._resolve_path(path)
            return dir_path.exists() and dir_path.is_dir()
        except Exception:
            return False
    
    def list_dir(self, path: str = ".") -> dict:
        """List contents of a directory in the workspace.
        
        Args:
            path: Relative path to the directory (default: workspace root).
        
        Returns:
            dict with status, files, and directories lists.
        """
        try:
            dir_path = self._resolve_path(path)
            
            if not dir_path.exists():
                return {
                    "success": False,
                    "path": path,
                    "error": f"Directory not found: {path}",
                }
            
            if not dir_path.is_dir():
                return {
                    "success": False,
                    "path": path,
                    "error": f"Not a directory: {path}",
                }
            
            files = []
            directories = []
            
            for item in dir_path.iterdir():
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    directories.append(item.name)
            
            return {
                "success": True,
                "path": str(dir_path),
                "files": sorted(files),
                "directories": sorted(directories),
            }
        except Exception as e:
            return {
                "success": False,
                "path": path,
                "error": str(e),
            }
    
    def delete_file(self, path: str) -> dict:
        """Delete a file from the workspace.
        
        Args:
            path: Relative path to the file.
        
        Returns:
            dict with status and message.
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    "success": False,
                    "path": path,
                    "error": f"File not found: {path}",
                }
            
            if not file_path.is_file():
                return {
                    "success": False,
                    "path": path,
                    "error": f"Not a file: {path}",
                }
            
            file_path.unlink()
            
            return {
                "success": True,
                "path": str(file_path),
                "message": f"Deleted file: {path}",
            }
        except Exception as e:
            return {
                "success": False,
                "path": path,
                "error": str(e),
            }