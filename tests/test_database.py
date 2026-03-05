"""Unit tests for ideaagent.database (SQLite backend)."""

import tempfile
from pathlib import Path

import pytest

from ideaagent.database import Database
from ideaagent.models import Task, TaskStatus, ResearchType


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database for each test."""
    db = Database(db_path=tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture
def sample_task():
    """Return a sample Task object."""
    return Task(
        research_type=ResearchType.MACHINE_LEARNING,
        idea_description="Test a random forest on iris dataset",
    )


class TestDatabaseConnection:
    def test_is_connected(self, tmp_db):
        assert tmp_db.is_connected() is True

    def test_file_created(self, tmp_path):
        db = Database(db_path=tmp_path / "new.db")
        assert (tmp_path / "new.db").exists()
        db.close()


class TestSaveAndGetTask:
    def test_save_and_retrieve(self, tmp_db, sample_task):
        assert tmp_db.save_task(sample_task) is True
        retrieved = tmp_db.get_task(sample_task.id)
        assert retrieved is not None
        assert retrieved.id == sample_task.id
        assert retrieved.research_type == sample_task.research_type
        assert retrieved.idea_description == sample_task.idea_description

    def test_upsert_updates_existing(self, tmp_db, sample_task):
        tmp_db.save_task(sample_task)
        sample_task.idea_description = "Updated description"
        tmp_db.save_task(sample_task)
        retrieved = tmp_db.get_task(sample_task.id)
        assert retrieved.idea_description == "Updated description"

    def test_get_nonexistent(self, tmp_db):
        assert tmp_db.get_task("does-not-exist") is None


class TestGetAllTasks:
    def test_returns_empty_list(self, tmp_db):
        assert tmp_db.get_all_tasks() == []

    def test_returns_all_saved(self, tmp_db):
        for _ in range(3):
            tmp_db.save_task(
                Task(
                    research_type=ResearchType.DEEP_LEARNING,
                    idea_description="Test",
                )
            )
        tasks = tmp_db.get_all_tasks()
        assert len(tasks) == 3

    def test_limit_respected(self, tmp_db):
        for _ in range(5):
            tmp_db.save_task(
                Task(
                    research_type=ResearchType.AGENT,
                    idea_description="Test",
                )
            )
        assert len(tmp_db.get_all_tasks(limit=2)) == 2


class TestGetTasksByStatus:
    def test_filter_by_status(self, tmp_db):
        t1 = Task(research_type=ResearchType.MACHINE_LEARNING, idea_description="A")
        t2 = Task(research_type=ResearchType.MACHINE_LEARNING, idea_description="B")
        t2.status = TaskStatus.COMPLETED
        tmp_db.save_task(t1)
        tmp_db.save_task(t2)

        pending = tmp_db.get_tasks_by_status(TaskStatus.PENDING)
        completed = tmp_db.get_tasks_by_status(TaskStatus.COMPLETED)
        assert len(pending) == 1
        assert len(completed) == 1


class TestDeleteTask:
    def test_delete_existing(self, tmp_db, sample_task):
        tmp_db.save_task(sample_task)
        assert tmp_db.delete_task(sample_task.id) is True
        assert tmp_db.get_task(sample_task.id) is None

    def test_delete_nonexistent(self, tmp_db):
        assert tmp_db.delete_task("ghost-id") is False


class TestGetStatistics:
    def test_empty_db(self, tmp_db):
        stats = tmp_db.get_task_statistics()
        assert stats["total"] == 0
        assert stats["by_status"] == {}
        assert stats["by_research_type"] == {}

    def test_counts(self, tmp_db):
        tmp_db.save_task(
            Task(
                research_type=ResearchType.MACHINE_LEARNING,
                idea_description="ML task",
            )
        )
        stats = tmp_db.get_task_statistics()
        assert stats["total"] == 1
        assert stats["by_research_type"].get("machine-learning") == 1
