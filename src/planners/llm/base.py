"""
Base LLM client interfaces and data models for intelligent planning.

This module defines the core abstractions that all LLM clients must implement
for consistent integration with the planning engine.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""

    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit_requests_per_minute: int = 60


class PlanningPrompt(BaseModel):
    """Structured prompt for planning operations."""

    system_prompt: str
    user_prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class LLMResponse(BaseModel):
    """Response from LLM with metadata."""

    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    model: str
    provider: str
    response_time: float
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class RequirementsAnalysis(BaseModel):
    """Analysis of user requirements."""

    technologies: List[str] = field(default_factory=list)
    architecture_pattern: Optional[str] = None
    complexity_score: float = 0.0
    estimated_duration: Optional[float] = None
    identified_constraints: List[str] = field(default_factory=list)
    security_requirements: List[str] = field(default_factory=list)
    performance_requirements: List[str] = field(default_factory=list)
    scalability_requirements: List[str] = field(default_factory=list)
    budget_constraints: Optional[Dict[str, float]] = None
    confidence_score: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class PlanGenerationRequest(BaseModel):
    """Request for plan generation."""

    requirements_description: str
    requirements_analysis: Optional[RequirementsAnalysis] = None
    available_tools: List[str] = field(default_factory=list)
    budget_constraints: Optional[Dict[str, float]] = None
    timeline_constraints: Optional[Dict[str, Any]] = None
    preferences: Dict[str, Any] = field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class PlanOptimizationRequest(BaseModel):
    """Request for plan optimization."""

    original_plan: Dict[str, Any]
    optimization_goals: List[str] = field(
        default_factory=list
    )  # cost, performance, security, speed
    constraints: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None
        self._request_count = 0
        self._last_request_time = 0.0

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM client."""

    @abstractmethod
    async def generate_text(self, prompt: PlanningPrompt) -> LLMResponse:
        """Generate text response from prompt."""

    async def analyze_requirements(self, requirements: str) -> RequirementsAnalysis:
        """Analyze user requirements and extract structured information."""
        prompt = self._create_requirements_analysis_prompt(requirements)
        response = await self.generate_text(prompt)
        return self._parse_requirements_analysis(response.content)

    async def generate_plan(self, request: PlanGenerationRequest) -> Dict[str, Any]:
        """Generate an execution plan from requirements."""
        prompt = self._create_plan_generation_prompt(request)
        response = await self.generate_text(prompt)
        return self._parse_plan_response(response.content)

    async def optimize_plan(self, request: PlanOptimizationRequest) -> Dict[str, Any]:
        """Optimize an existing plan."""
        prompt = self._create_plan_optimization_prompt(request)
        response = await self.generate_text(prompt)
        return self._parse_plan_response(response.content)

    async def explain_plan(self, plan: Dict[str, Any]) -> str:
        """Generate human-readable explanation of a plan."""
        prompt = self._create_plan_explanation_prompt(plan)
        response = await self.generate_text(prompt)
        return response.content

    async def validate_plan(
        self, plan: Dict[str, Any], requirements: str
    ) -> Dict[str, Any]:
        """Validate a plan against requirements."""
        prompt = self._create_plan_validation_prompt(plan, requirements)
        response = await self.generate_text(prompt)
        return self._parse_validation_response(response.content)

    def _create_requirements_analysis_prompt(self, requirements: str) -> PlanningPrompt:
        """Create prompt for requirements analysis."""
        system_prompt = """You are an expert DevOps architect analyzing user requirements for development environment setup.

Analyze the following requirements and extract:
1. Technologies mentioned (frameworks, databases, cloud providers, etc.)
2. Architecture pattern (microservices, monolithic, serverless, etc.)
3. Complexity score (1-10, where 10 is most complex)
4. Estimated duration in hours
5. Constraints (budget, security, performance, timeline)
6. Security requirements
7. Performance requirements
8. Scalability requirements

Return your analysis in JSON format with the specified structure."""

        user_prompt = f"""Requirements to analyze:
{requirements}

Please provide a detailed analysis in JSON format with the following structure:
{{
    "technologies": ["list", "of", "technologies"],
    "architecture_pattern": "pattern_name",
    "complexity_score": 5.5,
    "estimated_duration": 40.0,
    "identified_constraints": ["constraint1", "constraint2"],
    "security_requirements": ["requirement1", "requirement2"],
    "performance_requirements": ["requirement1", "requirement2"],
    "scalability_requirements": ["requirement1", "requirement2"],
    "budget_constraints": {{"max_monthly_cost": 500.0}},
    "confidence_score": 0.85
}}"""

        return PlanningPrompt(system_prompt=system_prompt, user_prompt=user_prompt)

    def _create_plan_generation_prompt(
        self, request: PlanGenerationRequest
    ) -> PlanningPrompt:
        """Create prompt for plan generation."""
        available_tools_str = ", ".join(request.available_tools)

        system_prompt = f"""You are an expert DevOps architect creating detailed execution plans for development environment setup.

Available tools: {available_tools_str}

Create a step-by-step execution plan that:
1. Uses only available tools
2. Follows best practices for security and performance
3. Optimizes for cost efficiency
4. Handles dependencies correctly
5. Includes proper error handling and rollback strategies

Return the plan in JSON format with clear step definitions, dependencies, and estimated costs/durations."""

        requirements_str = request.requirements_description
        if request.requirements_analysis:
            analysis = request.requirements_analysis
            requirements_str += f"\n\nAnalysis: Technologies: {analysis.technologies}, Pattern: {analysis.architecture_pattern}, Complexity: {analysis.complexity_score}"

        constraints_str = ""
        if request.budget_constraints:
            constraints_str += f"Budget constraints: {request.budget_constraints}\n"
        if request.timeline_constraints:
            constraints_str += f"Timeline constraints: {request.timeline_constraints}\n"

        user_prompt = f"""Requirements:
{requirements_str}

{constraints_str}

Please create a detailed execution plan in JSON format with this structure:
{{
    "steps": [
        {{
            "id": "step_1",
            "name": "Step Name",
            "description": "Detailed description",
            "tool": "tool_name",
            "action": "action_name",
            "parameters": {{}},
            "dependencies": [],
            "estimated_duration": 300.0,
            "estimated_cost": 50.0
        }}
    ],
    "dependencies": {{"step_2": ["step_1"]}},
    "estimated_cost": 225.0,
    "estimated_duration": 1300.0,
    "risk_assessment": {{
        "overall_risk": 0.3,
        "risk_factors": ["Network configuration"]
    }}
}}"""

        return PlanningPrompt(system_prompt=system_prompt, user_prompt=user_prompt)

    def _create_plan_optimization_prompt(
        self, request: PlanOptimizationRequest
    ) -> PlanningPrompt:
        """Create prompt for plan optimization."""
        goals_str = ", ".join(request.optimization_goals)
        constraints_str = (
            ", ".join(request.constraints) if request.constraints else "None"
        )

        system_prompt = f"""You are an expert DevOps architect optimizing execution plans for development environments.

Optimization goals: {goals_str}
Constraints: {constraints_str}

Optimize the given plan by:
1. Reducing costs where possible
2. Improving performance and reliability
3. Enhancing security
4. Minimizing execution time
5. Following best practices

Return the optimized plan in the same JSON format, highlighting the changes made."""

        user_prompt = f"""Original plan to optimize:
{request.original_plan}

Please provide an optimized version of this plan that addresses the specified goals while respecting constraints. Include a summary of changes made."""

        return PlanningPrompt(system_prompt=system_prompt, user_prompt=user_prompt)

    def _create_plan_explanation_prompt(self, plan: Dict[str, Any]) -> PlanningPrompt:
        """Create prompt for plan explanation."""
        system_prompt = """You are an expert DevOps architect explaining technical implementation plans to users.

Provide a clear, non-technical explanation of the execution plan that:
1. Explains what will be created/configured
2. Describes the purpose of each major component
3. Highlights important considerations
4. Estimates timeline and costs
5. Mentions any risks or considerations

Write in a friendly, accessible tone suitable for both technical and non-technical audiences."""

        user_prompt = f"""Plan to explain:
{plan}

Please provide a clear, comprehensive explanation of this execution plan."""

        return PlanningPrompt(system_prompt=system_prompt, user_prompt=user_prompt)

    def _create_plan_validation_prompt(
        self, plan: Dict[str, Any], requirements: str
    ) -> PlanningPrompt:
        """Create prompt for plan validation."""
        system_prompt = """You are an expert DevOps architect validating execution plans against user requirements.

Validate the plan by checking:
1. Does it meet all stated requirements?
2. Are the technologies appropriate?
3. Is the architecture sound?
4. Are security best practices followed?
5. Is the cost estimation reasonable?
6. Are there any missing components?
7. Are dependencies correctly specified?

Return validation results in JSON format with specific feedback."""

        user_prompt = f"""Original requirements:
{requirements}

Plan to validate:
{plan}

Please validate this plan and provide feedback in JSON format:
{{
    "is_valid": true,
    "completeness_score": 0.95,
    "issues": [],
    "suggestions": [],
    "missing_components": [],
    "estimated_success_probability": 0.90
}}"""

        return PlanningPrompt(system_prompt=system_prompt, user_prompt=user_prompt)

    def _parse_requirements_analysis(self, response: str) -> RequirementsAnalysis:
        """Parse requirements analysis response."""
        try:
            import json

            # Extract JSON from response if it contains other text
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                return RequirementsAnalysis(**data)
            else:
                # Fallback to basic parsing
                return RequirementsAnalysis(confidence_score=0.0)
        except Exception:
            return RequirementsAnalysis(confidence_score=0.0)

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse plan generation/optimization response."""
        try:
            import json

            # Extract JSON from response if it contains other text
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                parsed_result: Dict[str, Any] = json.loads(json_str)
                return parsed_result
            else:
                return {"error": "Could not parse plan response"}
        except Exception as e:
            return {"error": f"Plan parsing failed: {str(e)}"}

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse plan validation response."""
        try:
            import json

            # Extract JSON from response if it contains other text
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                parsed_result: Dict[str, Any] = json.loads(json_str)
                return parsed_result
            else:
                return {
                    "is_valid": False,
                    "completeness_score": 0.0,
                    "issues": ["Could not parse validation response"],
                    "estimated_success_probability": 0.0,
                }
        except Exception as e:
            return {
                "is_valid": False,
                "completeness_score": 0.0,
                "issues": [f"Validation parsing failed: {str(e)}"],
                "estimated_success_probability": 0.0,
            }

    async def _rate_limit_check(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time

        # Basic rate limiting - 1 request per second minimum
        if time_since_last_request < 1.0:
            import asyncio

            await asyncio.sleep(1.0 - time_since_last_request)

        self._last_request_time = time.time()
        self._request_count += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the client."""
        return {
            "total_requests": self._request_count,
            "provider": self.config.provider.value,
            "model": self.config.model,
        }
