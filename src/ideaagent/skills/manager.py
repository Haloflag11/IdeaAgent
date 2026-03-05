"""AgentSkills integration for IdeaAgent."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import yaml

from .errors import SkillError, ParseError, ValidationError


@dataclass
class SkillProperties:
    """Properties parsed from a skill's SKILL.md frontmatter."""
    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        result = {"name": self.name, "description": self.description}
        if self.license:
            result["license"] = self.license
        if self.compatibility:
            result["compatibility"] = self.compatibility
        if self.allowed_tools:
            result["allowed-tools"] = self.allowed_tools
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class SkillManager:
    """Manages discovery and loading of AgentSkills."""

    def __init__(self, skills_root: Path):
        """Initialize skill manager.
        
        Args:
            skills_root: Root directory containing skills
        """
        self.skills_root = Path(skills_root)
        self._skills_cache: dict[str, SkillProperties] = {}

    def find_skill_md(self, skill_dir: Path) -> Optional[Path]:
        """Find the SKILL.md file in a skill directory."""
        for name in ("SKILL.md", "skill.md"):
            path = skill_dir / name
            if path.exists():
                return path
        return None

    def parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from SKILL.md content."""
        if not content.startswith("---"):
            raise ParseError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ParseError("SKILL.md frontmatter not properly closed with ---")

        frontmatter_str = parts[1]
        body = parts[2].strip()

        try:
            metadata = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            raise ParseError(f"Invalid YAML in frontmatter: {e}")

        if not isinstance(metadata, dict):
            raise ParseError("SKILL.md frontmatter must be a YAML mapping")

        return metadata, body

    def read_properties(self, skill_dir: Path) -> SkillProperties:
        """Read skill properties from SKILL.md frontmatter."""
        skill_dir = Path(skill_dir)
        skill_md = self.find_skill_md(skill_dir)

        if skill_md is None:
            raise ParseError(f"SKILL.md not found in {skill_dir}")

        content = skill_md.read_text(encoding="utf-8")
        metadata, _ = self.parse_frontmatter(content)

        if "name" not in metadata:
            raise ValidationError("Missing required field in frontmatter: name")
        if "description" not in metadata:
            raise ValidationError("Missing required field in frontmatter: description")

        name = metadata["name"]
        description = metadata["description"]

        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Field 'name' must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValidationError("Field 'description' must be a non-empty string")

        return SkillProperties(
            name=name.strip(),
            description=description.strip(),
            license=metadata.get("license"),
            compatibility=metadata.get("compatibility"),
            allowed_tools=metadata.get("allowed-tools"),
            metadata=metadata.get("metadata", {}),
        )

    def validate(self, skill_dir: Path) -> list[str]:
        """Validate a skill directory.
        
        Args:
            skill_dir: Path to the skill directory
            
        Returns:
            List of validation error messages. Empty list means valid.
        """
        errors = []
        skill_dir = Path(skill_dir)

        if not skill_dir.exists():
            return [f"Path does not exist: {skill_dir}"]

        if not skill_dir.is_dir():
            return [f"Not a directory: {skill_dir}"]

        skill_md = self.find_skill_md(skill_dir)
        if skill_md is None:
            return ["Missing required file: SKILL.md"]

        try:
            content = skill_md.read_text(encoding="utf-8")
            metadata, _ = self.parse_frontmatter(content)
        except (ParseError, Exception) as e:
            return [str(e)]

        # Validate name
        if "name" not in metadata:
            errors.append("Missing required field: name")
        else:
            name = metadata["name"]
            if not isinstance(name, str) or not name.strip():
                errors.append("Field 'name' must be a non-empty string")
            elif len(name) > 64:
                errors.append(f"Skill name exceeds 64 characters")
            elif name != name.lower():
                errors.append(f"Skill name must be lowercase")
            elif name.startswith("-") or name.endswith("-"):
                errors.append("Skill name cannot start or end with a hyphen")
            elif "--" in name:
                errors.append("Skill name cannot contain consecutive hyphens")
            elif skill_dir.name != name:
                errors.append(f"Directory name must match skill name")

        # Validate description
        if "description" not in metadata:
            errors.append("Missing required field: description")
        else:
            desc = metadata["description"]
            if not isinstance(desc, str) or not desc.strip():
                errors.append("Field 'description' must be a non-empty string")
            elif len(desc) > 1024:
                errors.append(f"Description exceeds 1024 characters")

        return errors

    def discover_skills(self) -> list[SkillProperties]:
        """Discover all valid skills in the skills root directory.
        
        Returns:
            List of valid SkillProperties
        """
        skills = []
        
        if not self.skills_root.exists():
            return skills

        for item in self.skills_root.iterdir():
            if item.is_dir():
                errors = self.validate(item)
                if not errors:
                    try:
                        props = self.read_properties(item)
                        skills.append(props)
                        self._skills_cache[props.name] = props
                    except Exception:
                        continue

        return skills

    def get_skill(self, skill_name: str) -> Optional[SkillProperties]:
        """Get a skill by name.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            SkillProperties if found, None otherwise
        """
        if skill_name in self._skills_cache:
            return self._skills_cache[skill_name]

        # Try to find in skills root
        skill_dir = self.skills_root / skill_name
        if skill_dir.exists():
            try:
                props = self.read_properties(skill_dir)
                self._skills_cache[skill_name] = props
                return props
            except Exception:
                return None

        return None

    def get_skill_path(self, skill_name: str) -> Optional[Path]:
        """Get the path to a skill directory.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Path to skill directory if found, None otherwise
        """
        skill_dir = self.skills_root / skill_name
        if skill_dir.exists() and self.find_skill_md(skill_dir):
            return skill_dir
        return None

    def get_skill_instructions(self, skill_name: str) -> Optional[str]:
        """Get the full instructions from a skill's SKILL.md.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Full content of SKILL.md if found, None otherwise
        """
        skill_dir = self.get_skill_path(skill_name)
        if skill_dir is None:
            return None

        skill_md = self.find_skill_md(skill_dir)
        if skill_md:
            return skill_md.read_text(encoding="utf-8")

        return None

    def to_prompt_xml(self, skill_names: Optional[list[str]] = None) -> str:
        """Generate <available_skills> XML for agent prompts.
        
        Args:
            skill_names: Optional list of specific skill names to include
            
        Returns:
            XML string with <available_skills> block
        """
        if skill_names:
            skills = [self.get_skill(name) for name in skill_names]
            skills = [s for s in skills if s is not None]
        else:
            skills = self.discover_skills()

        if not skills:
            return "<available_skills>\n</available_skills>"

        lines = ["<available_skills>"]

        for skill in skills:
            skill_path = self.get_skill_path(skill.name)
            skill_md_path = self.find_skill_md(skill_path) if skill_path else None

            lines.append("<skill>")
            lines.append("<name>")
            lines.append(self._escape_xml(skill.name))
            lines.append("</name>")
            lines.append("<description>")
            lines.append(self._escape_xml(skill.description))
            lines.append("</description>")

            if skill_md_path:
                lines.append("<location>")
                lines.append(str(skill_md_path.absolute()))
                lines.append("</location>")

            lines.append("</skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_requirements_file(self, skill_name: str) -> Optional[Path]:
        """Return the path to a skill's requirements.txt, or None if absent.

        Args:
            skill_name: Name of the skill.

        Returns:
            Path to ``requirements.txt`` if it exists, otherwise ``None``.
        """
        skill_path = self.get_skill_path(skill_name)
        if skill_path is None:
            return None
        req = skill_path / "requirements.txt"
        return req if req.exists() else None

    def read_requirements(self, skill_name: str) -> list[str]:
        """Parse the package specs from a skill's requirements.txt.

        Lines starting with ``#`` and blank lines are ignored.  Version
        specifiers (e.g. ``scikit-learn>=1.0``) are kept as-is so pip can
        honour them.

        Args:
            skill_name: Name of the skill.

        Returns:
            List of package spec strings.  Empty list when no file found.
        """
        req_file = self.get_requirements_file(skill_name)
        if req_file is None:
            return []
        lines = req_file.read_text(encoding="utf-8").splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]

    def install_skill_requirements(self, skill_name: str, sandbox) -> list[str]:
        """Install the packages declared in a skill's requirements.txt.

        Silently succeeds when the skill has no requirements file.

        Args:
            skill_name: Name of the skill whose requirements should be installed.
            sandbox: A :class:`~ideaagent.sandbox.VenvSandbox` (or compatible)
                instance whose ``install_packages`` method will be called.

        Returns:
            The list of package specs that were passed to pip.  Empty list
            when nothing was installed.

        Raises:
            :class:`~ideaagent.skills.errors.SkillError`: Re-raised from the
                sandbox if installation fails, so the caller can decide how to
                handle it.
        """
        packages = self.read_requirements(skill_name)
        if not packages:
            return []
        try:
            sandbox.install_packages(packages)
        except Exception as exc:
            raise SkillError(
                f"Failed to install requirements for skill '{skill_name}': {exc}"
            ) from exc
        return packages

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))
