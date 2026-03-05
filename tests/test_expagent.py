"""Test suite for IdeaAgent."""

import pytest
from pathlib import Path
from IdeaAgent.models import (
    ResearchType,
    Task,
    TaskStatus,
    ExperimentPlan,
    ExperimentStep,
    ExecutionResult,
)


class TestModels:
    """Test data models."""

    def test_research_type_values(self):
        """Test ResearchType enum values."""
        assert ResearchType.DEEP_LEARNING.value == "deep-learning"
        assert ResearchType.MACHINE_LEARNING.value == "machine-learning"
        assert ResearchType.AGENT.value == "agent"

    def test_task_creation(self):
        """Test Task creation."""
        task = Task(
            research_type=ResearchType.MACHINE_LEARNING,
            idea_description="Test ML experiment"
        )
        
        assert task.research_type == ResearchType.MACHINE_LEARNING
        assert task.idea_description == "Test ML experiment"
        assert task.status == TaskStatus.PENDING
        assert task.id is not None
        assert task.loop_count == 0

    def test_task_to_dict(self):
        """Test Task serialization."""
        task = Task(
            research_type=ResearchType.DEEP_LEARNING,
            idea_description="Test DL experiment"
        )
        
        task_dict = task.to_dict()
        
        assert task_dict["id"] == task.id
        assert task_dict["research_type"] == "deep-learning"
        assert task_dict["idea_description"] == "Test DL experiment"
        assert task_dict["status"] == "pending"

    def test_task_from_dict(self):
        """Test Task deserialization."""
        task_dict = {
            "id": "test-id",
            "research_type": "machine-learning",
            "idea_description": "Test experiment",
            "status": "running",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "loop_count": 0,
            "metadata": {}
        }
        
        task = Task.from_dict(task_dict)
        
        assert task.id == "test-id"
        assert task.research_type == ResearchType.MACHINE_LEARNING
        assert task.status == TaskStatus.RUNNING

    def test_experiment_plan(self):
        """Test ExperimentPlan creation."""
        plan = ExperimentPlan(
            title="Test Plan",
            description="Test experiment plan",
            steps=[
                ExperimentStep(
                    step_number=1,
                    description="Step 1",
                    estimated_duration=10
                ),
                ExperimentStep(
                    step_number=2,
                    description="Step 2",
                    skill_required="data-preprocessing"
                )
            ]
        )
        
        assert plan.title == "Test Plan"
        assert len(plan.steps) == 2
        assert plan.steps[0].estimated_duration == 10
        assert plan.steps[1].skill_required == "data-preprocessing"

    def test_execution_result(self):
        """Test ExecutionResult creation."""
        result = ExecutionResult(
            step_number=1,
            success=True,
            output="Success output",
            execution_time=1.5
        )
        
        assert result.step_number == 1
        assert result.success is True
        assert result.output == "Success output"
        assert result.execution_time == 1.5
        assert result.error is None


class TestSkillProperties:
    """Test skill properties parsing."""

    def test_skill_properties_creation(self):
        """Test SkillProperties creation."""
        from IdeaAgent.skills import SkillProperties
        
        props = SkillProperties(
            name="test-skill",
            description="Test skill description"
        )
        
        assert props.name == "test-skill"
        assert props.description == "Test skill description"
        assert props.license is None
        assert props.metadata == {}

    def test_skill_properties_to_dict(self):
        """Test SkillProperties serialization."""
        from IdeaAgent.skills import SkillProperties
        
        props = SkillProperties(
            name="test-skill",
            description="Test skill",
            license="MIT",
            metadata={"author": "Test"}
        )
        
        props_dict = props.to_dict()
        
        assert props_dict["name"] == "test-skill"
        assert props_dict["description"] == "Test skill"
        assert props_dict["license"] == "MIT"
        assert props_dict["metadata"]["author"] == "Test"


class TestValidator:
    """Test skill validation."""

    def test_validate_valid_skill(self, tmp_path):
        """Test validation of a valid skill."""
        from IdeaAgent.skills import SkillManager
        
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: A test skill
---
# Test Skill

Test content.
""")
        
        manager = SkillManager(tmp_path)
        errors = manager.validate(skill_dir)
        
        assert len(errors) == 0

    def test_validate_missing_skill_md(self, tmp_path):
        """Test validation with missing SKILL.md."""
        from IdeaAgent.skills import SkillManager
        
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        
        manager = SkillManager(tmp_path)
        errors = manager.validate(skill_dir)
        
        assert len(errors) == 1
        assert "Missing required file" in errors[0]

    def test_validate_invalid_name_uppercase(self, tmp_path):
        """Test validation with uppercase name."""
        from IdeaAgent.skills import SkillManager
        
        skill_dir = tmp_path / "Test-Skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: Test-Skill
description: A test skill
---
Content
""")
        
        manager = SkillManager(tmp_path)
        errors = manager.validate(skill_dir)
        
        assert any("lowercase" in e for e in errors)

    def test_validate_missing_description(self, tmp_path):
        """Test validation with missing description."""
        from IdeaAgent.skills import SkillManager
        
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
---
Content
""")
        
        manager = SkillManager(tmp_path)
        errors = manager.validate(skill_dir)
        
        assert any("Missing required field" in e for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
