"""LLM integration using OpenAI library for IdeaAgent."""

import os
import json
import logging
from typing import Optional, Generator

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from .models import ExperimentPlan, ExperimentStep, ResearchType
from . import prompts

# Force load from .env file, overriding system environment variables
env_path = find_dotenv()
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger("IdeaAgent.llm")


class LLMClient:
    """OpenAI LLM client for IdeaAgent."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_tokens = int(os.getenv("MAX_TOKENS", str(max_tokens)))

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in .env file.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        research_type: ResearchType,
        idea_description: str,
        available_skills: str = "",
    ) -> ExperimentPlan:
        """Generate an experiment plan based on the research idea."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.get_plan_system_prompt(research_type, available_skills)},
                {"role": "user",   "content": prompts.get_plan_user_prompt(idea_description)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response")
        try:
            return self._parse_plan(json.loads(content))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    def regenerate_plan(
        self,
        research_type: ResearchType,
        idea_description: str,
        previous_plan: str,
        feedback: str,
        available_skills: str = "",
    ) -> ExperimentPlan:
        """Regenerate a plan based on user feedback."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.get_plan_system_prompt(research_type, available_skills)},
                {"role": "user",   "content": prompts.get_plan_regeneration_prompt(idea_description, previous_plan, feedback)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response")
        try:
            return self._parse_plan(json.loads(content))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def execute_step(
        self,
        research_type: ResearchType,
        step_description: str,
        context: str,
        skill_instructions: Optional[str] = None,
    ) -> str:
        """Generate executable code for a single step (non-streaming)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.get_execution_system_prompt(research_type)},
                {"role": "user",   "content": prompts.get_execution_user_prompt(step_description, context, skill_instructions)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def execute_step_with_tools(
        self,
        research_type: ResearchType,
        step_description: str,
        context: str,
        skill_instructions: Optional[str],
        available_tools: list,
    ) -> str:
        """Generate executable code with tool-usage awareness."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts.get_execution_system_prompt_with_tools(research_type, available_tools)},
                    {"role": "user",   "content": prompts.get_execution_user_prompt(step_description, context, skill_instructions)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"# Error generating code: {e}"

    def stream_execute_step_with_thinking(
        self,
        research_type: ResearchType,
        step_description: str,
        context: str,
        skill_instructions: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> tuple[str, str]:
        """Stream step execution with real-time thinking output.

        Streams the LLM response and forwards reasoning/thinking chunks via
        *callback* as they arrive.  For OpenAI-compatible APIs that expose a
        ``reasoning_content`` (or ``thinking`` / ``reasoning``) field on each
        delta, those chunks are emitted with ``chunk_type='thinking'``.

        Args:
            research_type: Type of research task.
            step_description: Description of the current step.
            context: Execution context from previous steps.
            skill_instructions: Optional skill documentation.
            callback: Called for each incoming chunk.
                Signature: ``callback(chunk_type: str, content: str)``
                ``chunk_type`` is ``'thinking'`` for reasoning content.

        Returns:
            ``(full_response_text, extracted_code)``
        """
        from .utils.stream_parser import extract_python_code

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.get_execution_system_prompt(research_type)},
                {"role": "user",   "content": prompts.get_execution_user_prompt(step_description, context, skill_instructions)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        full_response: list[str] = []

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Regular content
            content = getattr(delta, "content", None) or ""
            if content:
                full_response.append(content)

            # Reasoning / thinking fields (OpenAI-compatible extensions)
            delta_dict = delta.model_dump() if hasattr(delta, "model_dump") else (
                delta.dict() if hasattr(delta, "dict") else {}
            )
            for field in ("reasoning_content", "thinking", "reasoning"):
                thinking = delta_dict.get(field)
                if thinking and callback:
                    callback("thinking", thinking)
                    break

        full_text = "".join(full_response)
        return full_text, extract_python_code(full_text)

    # ------------------------------------------------------------------
    # Error analysis & fix
    # ------------------------------------------------------------------

    def analyze_and_fix(
        self,
        research_type: ResearchType,
        step_description: str,
        failed_code: str,
        error_message: str,
        context: str,
        skill_instructions: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> str:
        """Diagnose a failed execution and return corrected code."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts.get_execution_system_prompt(research_type)},
                    {"role": "user",   "content": prompts.get_fix_user_prompt(
                        step_description, failed_code, error_message,
                        context, attempt, max_attempts, skill_instructions,
                    )},
                ],
                temperature=min(self.temperature + 0.1 * attempt, 1.0),
                max_tokens=self.max_tokens,
            )
            result = response.choices[0].message.content or ""
            logger.info("analyze_and_fix attempt %d/%d returned %d chars", attempt, max_attempts, len(result))
            return result
        except Exception as exc:
            logger.error("analyze_and_fix LLM call failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Judge execution output and fix if needed (true Agentic Loop)
    # ------------------------------------------------------------------

    def judge_and_fix(
        self,
        research_type: ResearchType,
        step_description: str,
        code: str,
        stdout: str,
        stderr: str,
        returncode: int,
        context: str,
        skill_instructions: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 5,
    ) -> dict:
        """Let the LLM judge whether execution succeeded; return fix if needed.

        Returns:
            ``{"status": "success"}``
            or
            ``{"status": "fix", "reason": "...", "code": "..."}``
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts.get_execution_system_prompt(research_type)},
                    {"role": "user",   "content": prompts.get_judge_user_prompt(
                        step_description, code, stdout, stderr,
                        returncode, context, attempt, max_attempts, skill_instructions,
                    )},
                ],
                temperature=0.2,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            logger.info("judge_and_fix attempt %d/%d returned %d chars", attempt, max_attempts, len(content))
            result = json.loads(content)
            if result.get("status") not in ("success", "fix"):
                logger.warning("Unexpected judge_and_fix status: %s", result)
                return {"status": "success"}
            return result
        except Exception as exc:
            logger.error("judge_and_fix LLM call failed: %s", exc)
            return {
                "status": "success" if returncode == 0 else "fix",
                "reason": str(exc),
                "code": code,
            }

    # ------------------------------------------------------------------
    # Plan parsing (internal)
    # ------------------------------------------------------------------

    def _parse_plan(self, plan_data: dict) -> ExperimentPlan:
        """Parse plan dict from LLM JSON response into an ExperimentPlan."""
        for required in ("title", "description"):
            if required not in plan_data:
                raise ValueError(f"Missing '{required}' in plan response")
        if not isinstance(plan_data.get("steps"), list):
            raise ValueError("Missing or invalid 'steps' in plan response")

        steps = []
        for step_data in plan_data["steps"]:
            if not isinstance(step_data, dict):
                continue
            steps.append(
                ExperimentStep(
                    step_number=step_data.get("step_number", len(steps) + 1),
                    description=step_data.get("description", ""),
                    skill_required=step_data.get("skill_required"),
                    estimated_duration=step_data.get("estimated_duration"),
                )
            )
        if not steps:
            raise ValueError("No valid steps in plan")

        return ExperimentPlan(
            title=plan_data["title"],
            description=plan_data["description"],
            steps=steps,
            estimated_total_time=plan_data.get("estimated_total_time"),
            skills_needed=plan_data.get("skills_needed", []),
        )
