"""
Intelligent planning engine for Orcastrate.

This module provides AI-powered planning capabilities that understand natural language
requirements and generate sophisticated, optimized execution plans.
"""

from typing import Any, Dict, List, Optional, Set

from ..agent.base import Plan as AgentPlan
from ..agent.base import Requirements
from .analysis.requirements_analyzer import AnalysisResult, RequirementsAnalyzer
from .base import Planner, PlannerConfig, PlanStep, PlanStructure, ValidationResult
from .llm.anthropic_client import AnthropicClient
from .llm.base import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    PlanGenerationRequest,
    PlanOptimizationRequest,
)
from .llm.openai_client import OpenAIClient
from .llm.prompt_templates import PromptTemplateManager


class IntelligentPlanningStrategy:
    """AI-powered planning strategy constants."""

    LLM_POWERED = "llm_powered"
    HYBRID_ANALYSIS = "hybrid_analysis"
    REQUIREMENTS_DRIVEN = "requirements_driven"
    COST_OPTIMIZED = "cost_optimized"


class IntelligentPlanner(Planner):
    """
    Intelligent planner that uses LLMs and sophisticated analysis to create
    optimized execution plans from natural language requirements.
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        strategy: str = IntelligentPlanningStrategy.HYBRID_ANALYSIS,
        fallback_to_template: bool = True,
    ):
        if config is None:
            config = PlannerConfig()
        super().__init__(config)
        self.intelligent_strategy = strategy
        self.fallback_to_template = fallback_to_template

        # Initialize LLM client if config provided
        self.llm_client: Optional[LLMClient] = None
        if llm_config:
            self._initialize_llm_client(llm_config)

        # Initialize core components
        self.requirements_analyzer = RequirementsAnalyzer(self.llm_client)
        self.prompt_manager = PromptTemplateManager()

        # Planning state
        self._available_tools: Set[str] = set()
        self._tool_capabilities: Dict[str, Dict[str, Any]] = {}
        self._cost_estimates: Dict[str, float] = {}

        self.logger.info(f"IntelligentPlanner initialized with strategy: {strategy}")

    def _initialize_llm_client(self, config: LLMConfig) -> None:
        """Initialize the appropriate LLM client based on provider."""
        try:
            if config.provider == LLMProvider.OPENAI:
                self.llm_client = OpenAIClient(config)
            elif config.provider == LLMProvider.ANTHROPIC:
                self.llm_client = AnthropicClient(config)
            else:
                raise ValueError(f"Unsupported LLM provider: {config.provider}")

            self.logger.info(f"LLM client initialized: {config.provider.value}")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            if not self.fallback_to_template:
                raise
            self.llm_client = None

    async def initialize(self) -> None:
        """Initialize the intelligent planner."""
        try:
            # Initialize LLM client if available
            if self.llm_client:
                await self.llm_client.initialize()
                self.logger.info("LLM client initialized successfully")

            # Initialize requirements analyzer
            # (No async initialization needed for analyzer)

            self.logger.info("IntelligentPlanner initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize IntelligentPlanner: {e}")
            if not self.fallback_to_template:
                raise

    async def create_plan(self, requirements: Requirements) -> AgentPlan:
        """Create plan from Requirements object (base class interface)."""
        # Convert Requirements to string and call intelligent planning
        req_str = str(requirements)  # This will need proper conversion
        plan_steps = await self.create_intelligent_plan(req_str)

        # Convert List[PlanStep] to AgentPlan
        # Convert PlanStep objects to dictionaries as expected by AgentPlan
        steps_as_dicts = [
            {
                "id": step.id,
                "name": step.name,
                "description": step.description,
                "tool": step.tool,
                "action": step.action,
                "parameters": step.parameters,
                "dependencies": step.dependencies,
                "estimated_duration": step.estimated_duration,
                "estimated_cost": step.estimated_cost,
            }
            for step in plan_steps
        ]

        plan = AgentPlan(
            steps=steps_as_dicts,
            dependencies={},
            estimated_cost=0.0,
            estimated_duration=0.0,
            risk_assessment={},
            requirements=requirements,
        )
        return plan

    async def create_intelligent_plan(
        self,
        requirements: str,
        context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[PlanStep]:
        """
        Create an intelligent execution plan from natural language requirements.

        Args:
            requirements: Natural language description of what needs to be built
            context: Additional context for planning (team size, preferences, etc.)
            available_tools: List of available tools/integrations
            constraints: Budget, timeline, and other constraints

        Returns:
            List of optimized plan steps
        """
        if context is None:
            context = {}
        if available_tools is None:
            available_tools = list(self._available_tools)
        if constraints is None:
            constraints = {}

        self.logger.info(
            f"Creating intelligent plan for requirements: {requirements[:100]}..."
        )

        try:
            # Phase 1: Analyze requirements
            analysis_result = await self._analyze_requirements(requirements, context)

            # Phase 2: Generate plan using LLM if available
            if (
                self.llm_client
                and self.intelligent_strategy
                != IntelligentPlanningStrategy.REQUIREMENTS_DRIVEN
            ):
                plan_steps = await self._generate_llm_plan(
                    requirements, analysis_result, available_tools, constraints
                )
            else:
                # Fallback to rule-based planning
                plan_steps = await self._generate_rule_based_plan(
                    requirements, analysis_result, available_tools, constraints
                )

            # Phase 3: Optimize plan
            optimized_steps = await self._optimize_intelligent_plan(
                plan_steps, analysis_result, constraints
            )

            # Phase 4: Validate plan
            validation_result = await self._validate_intelligent_plan(
                optimized_steps, requirements, analysis_result
            )

            if not validation_result.valid:
                self.logger.warning(
                    f"Plan validation failed: {validation_result.errors}"
                )
                if self.fallback_to_template:
                    # Attempt template-based fallback
                    return await self._create_template_fallback_plan(
                        requirements, available_tools
                    )
                else:
                    raise ValueError(
                        f"Plan validation failed: {validation_result.errors}"
                    )

            self.logger.info(
                f"Intelligent plan created with {len(optimized_steps)} steps"
            )
            return optimized_steps

        except Exception as e:
            self.logger.error(f"Failed to create intelligent plan: {e}")

            if self.fallback_to_template:
                self.logger.info("Falling back to template-based planning")
                return await self._create_template_fallback_plan(
                    requirements, available_tools
                )
            else:
                raise

    async def _analyze_requirements(
        self, requirements: str, context: Dict[str, Any]
    ) -> AnalysisResult:
        """Analyze requirements using the requirements analyzer."""
        self.logger.debug("Starting requirements analysis")

        # Use LLM for analysis if available, otherwise rule-based only
        use_llm = self.llm_client is not None and self.intelligent_strategy in [
            IntelligentPlanningStrategy.LLM_POWERED,
            IntelligentPlanningStrategy.HYBRID_ANALYSIS,
        ]

        analysis_result = await self.requirements_analyzer.analyze(
            requirements, context, use_llm=use_llm
        )

        self.logger.debug(
            f"Requirements analysis completed. "
            f"Confidence: {analysis_result.analysis_confidence:.2f}, "
            f"Technologies: {len(analysis_result.detected_technologies)}, "
            f"Constraints: {len(analysis_result.constraints)}"
        )

        return analysis_result

    async def _generate_llm_plan(
        self,
        requirements: str,
        analysis: AnalysisResult,
        available_tools: List[str],
        constraints: Dict[str, Any],
    ) -> List[PlanStep]:
        """Generate plan using LLM capabilities."""
        self.logger.debug("Generating LLM-powered plan")

        # Create plan generation request
        request = PlanGenerationRequest(
            requirements_description=requirements,
            requirements_analysis=analysis.llm_analysis,
            available_tools=available_tools,
            budget_constraints=constraints.get("budget"),
            timeline_constraints=constraints.get("timeline"),
            preferences=constraints.get("preferences", {}),
        )

        # Generate plan using LLM
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")
        plan_data = await self.llm_client.generate_plan(request)

        # Convert LLM response to PlanStep objects
        plan_steps = self._convert_llm_response_to_steps(plan_data, analysis)

        self.logger.debug(f"LLM generated {len(plan_steps)} plan steps")
        return plan_steps

    async def _generate_rule_based_plan(
        self,
        requirements: str,
        analysis: AnalysisResult,
        available_tools: List[str],
        constraints: Dict[str, Any],
    ) -> List[PlanStep]:
        """Generate plan using rule-based approach with analysis insights."""
        self.logger.debug("Generating rule-based plan")

        plan_steps = []
        step_counter = 1

        # Use technology stack from analysis to drive plan generation
        tech_stack = analysis.technology_stack

        # Infrastructure setup steps
        if "infrastructure" in tech_stack and tech_stack["infrastructure"]:
            infra_steps = self._create_infrastructure_steps(
                tech_stack["infrastructure"], available_tools, step_counter
            )
            plan_steps.extend(infra_steps)
            step_counter += len(infra_steps)

        # Database setup steps
        if "database" in tech_stack and tech_stack["database"]:
            db_steps = self._create_database_steps(
                tech_stack["database"], available_tools, step_counter
            )
            plan_steps.extend(db_steps)
            step_counter += len(db_steps)

        # Backend setup steps
        if "backend" in tech_stack and tech_stack["backend"]:
            backend_steps = self._create_backend_steps(
                tech_stack["backend"], available_tools, step_counter
            )
            plan_steps.extend(backend_steps)
            step_counter += len(backend_steps)

        # Frontend setup steps
        if "frontend" in tech_stack and tech_stack["frontend"]:
            frontend_steps = self._create_frontend_steps(
                tech_stack["frontend"], available_tools, step_counter
            )
            plan_steps.extend(frontend_steps)
            step_counter += len(frontend_steps)

        # Security and monitoring steps
        security_steps = self._create_security_steps(
            analysis, available_tools, step_counter
        )
        plan_steps.extend(security_steps)

        self.logger.debug(f"Rule-based plan generated {len(plan_steps)} steps")
        return plan_steps

    def _convert_llm_response_to_steps(
        self, plan_data: Dict[str, Any], analysis: AnalysisResult
    ) -> List[PlanStep]:
        """Convert LLM plan response to PlanStep objects."""
        plan_steps: List[PlanStep] = []

        if "steps" not in plan_data:
            self.logger.warning("LLM response missing 'steps' field")
            return plan_steps

        for step_data in plan_data["steps"]:
            try:
                # Extract dependencies
                dependencies = step_data.get("dependencies", [])
                if isinstance(dependencies, str):
                    dependencies = [dependencies]

                step = PlanStep(
                    id=step_data.get("id", f"step_{len(plan_steps) + 1}"),
                    name=step_data.get("name", "Unnamed Step"),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", ""),
                    action=step_data.get("action", ""),
                    parameters=step_data.get("parameters", {}),
                    dependencies=dependencies,
                    estimated_duration=step_data.get("estimated_duration", 300.0),
                    estimated_cost=step_data.get("estimated_cost", 0.0),
                )

                plan_steps.append(step)

            except Exception as e:
                self.logger.warning(f"Failed to convert LLM step: {e}")
                continue

        return plan_steps

    def _create_infrastructure_steps(
        self,
        infrastructure_techs: List[str],
        available_tools: List[str],
        start_counter: int,
    ) -> List[PlanStep]:
        """Create infrastructure setup steps."""
        steps = []

        # Check for cloud providers
        if any("aws" in tech.lower() for tech in infrastructure_techs):
            if "aws" in available_tools:
                steps.append(
                    PlanStep(
                        id=f"step_{start_counter}",
                        name="Setup AWS Infrastructure",
                        description="Create VPC, subnets, and security groups",
                        tool="aws",
                        action="create_vpc",
                        parameters={"cidr": "10.0.0.0/16"},
                        dependencies=[],
                        estimated_duration=600.0,
                        estimated_cost=50.0,
                    )
                )

        # Check for containerization
        if any("docker" in tech.lower() for tech in infrastructure_techs):
            if "docker" in available_tools:
                steps.append(
                    PlanStep(
                        id=f"step_{start_counter + len(steps)}",
                        name="Setup Container Environment",
                        description="Configure Docker environment and registries",
                        tool="docker",
                        action="setup_environment",
                        parameters={},
                        dependencies=[],
                        estimated_duration=300.0,
                        estimated_cost=0.0,
                    )
                )

        return steps

    def _create_database_steps(
        self, database_techs: List[str], available_tools: List[str], start_counter: int
    ) -> List[PlanStep]:
        """Create database setup steps."""
        steps = []

        for i, db_tech in enumerate(database_techs):
            db_lower = db_tech.lower()

            if "postgresql" in db_lower and "postgresql" in available_tools:
                steps.append(
                    PlanStep(
                        id=f"step_{start_counter + i}",
                        name="Setup PostgreSQL Database",
                        description="Create and configure PostgreSQL instance",
                        tool="postgresql",
                        action="create_database",
                        parameters={"name": "main_db", "user": "app_user"},
                        dependencies=[],
                        estimated_duration=400.0,
                        estimated_cost=25.0,
                    )
                )

            elif "mysql" in db_lower and "mysql" in available_tools:
                steps.append(
                    PlanStep(
                        id=f"step_{start_counter + i}",
                        name="Setup MySQL Database",
                        description="Create and configure MySQL instance",
                        tool="mysql",
                        action="create_database",
                        parameters={"name": "main_db", "user": "app_user"},
                        dependencies=[],
                        estimated_duration=400.0,
                        estimated_cost=25.0,
                    )
                )

            elif "mongodb" in db_lower and "mongodb" in available_tools:
                steps.append(
                    PlanStep(
                        id=f"step_{start_counter + i}",
                        name="Setup MongoDB Database",
                        description="Create and configure MongoDB instance",
                        tool="mongodb",
                        action="create_database",
                        parameters={"name": "main_db"},
                        dependencies=[],
                        estimated_duration=400.0,
                        estimated_cost=30.0,
                    )
                )

        return steps

    def _create_backend_steps(
        self, backend_techs: List[str], available_tools: List[str], start_counter: int
    ) -> List[PlanStep]:
        """Create backend application steps."""
        steps = []

        # For now, create generic application deployment steps
        # This could be enhanced with technology-specific templates

        steps.append(
            PlanStep(
                id=f"step_{start_counter}",
                name="Deploy Backend Application",
                description=f"Deploy backend using {', '.join(backend_techs)}",
                tool="docker",  # Default to containerized deployment
                action="deploy_application",
                parameters={
                    "image": "backend-app",
                    "port": 8000,
                    "environment": "production",
                },
                dependencies=[],
                estimated_duration=600.0,
                estimated_cost=40.0,
            )
        )

        return steps

    def _create_frontend_steps(
        self, frontend_techs: List[str], available_tools: List[str], start_counter: int
    ) -> List[PlanStep]:
        """Create frontend application steps."""
        steps = []

        steps.append(
            PlanStep(
                id=f"step_{start_counter}",
                name="Deploy Frontend Application",
                description=f"Deploy frontend using {', '.join(frontend_techs)}",
                tool="aws",  # Default to S3 + CloudFront
                action="deploy_static_site",
                parameters={"bucket_name": "frontend-app", "enable_cloudfront": True},
                dependencies=[],
                estimated_duration=400.0,
                estimated_cost=20.0,
            )
        )

        return steps

    def _create_security_steps(
        self, analysis: AnalysisResult, available_tools: List[str], start_counter: int
    ) -> List[PlanStep]:
        """Create security-related steps based on analysis."""
        steps = []

        # Check for security requirements in analysis
        security_reqs = [
            req
            for req in analysis.extracted_requirements
            if req.type.value == "security"
        ]

        if security_reqs:
            steps.append(
                PlanStep(
                    id=f"step_{start_counter}",
                    name="Configure Security",
                    description="Setup SSL certificates and security policies",
                    tool="aws",
                    action="setup_security",
                    parameters={"enable_ssl": True, "enable_waf": True},
                    dependencies=[],
                    estimated_duration=300.0,
                    estimated_cost=15.0,
                )
            )

        return steps

    async def _optimize_plan(self, plan: PlanStructure) -> PlanStructure:
        """Optimize plan - base class interface."""
        # Convert PlanStructure to the format expected by intelligent optimization
        plan_steps = plan.steps
        # For now, use empty analysis and constraints
        analysis = AnalysisResult(original_requirements="")
        constraints: Dict[str, Any] = {}

        optimized_steps = await self._optimize_intelligent_plan(
            plan_steps, analysis, constraints
        )

        # Convert back to PlanStructure
        optimized_plan = PlanStructure(
            steps=optimized_steps,
            metadata=plan.metadata,
        )
        return optimized_plan

    async def _optimize_intelligent_plan(
        self,
        plan_steps: List[PlanStep],
        analysis: AnalysisResult,
        constraints: Dict[str, Any],
    ) -> List[PlanStep]:
        """Optimize the plan for cost, performance, and other factors."""
        self.logger.debug("Optimizing plan")

        # If LLM is available and we're using LLM-powered strategy, use it for optimization
        if self.llm_client and self.intelligent_strategy in [
            IntelligentPlanningStrategy.LLM_POWERED,
            IntelligentPlanningStrategy.COST_OPTIMIZED,
        ]:
            return await self._llm_optimize_plan(plan_steps, analysis, constraints)
        else:
            return await self._rule_based_optimize_plan(
                plan_steps, analysis, constraints
            )

    async def _llm_optimize_plan(
        self,
        plan_steps: List[PlanStep],
        analysis: AnalysisResult,
        constraints: Dict[str, Any],
    ) -> List[PlanStep]:
        """Use LLM to optimize the plan."""
        try:
            # Convert plan steps to dictionary format for LLM
            plan_dict = {
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "description": step.description,
                        "tool": step.tool,
                        "action": step.action,
                        "parameters": step.parameters,
                        "dependencies": step.dependencies,
                        "estimated_duration": step.estimated_duration,
                        "estimated_cost": step.estimated_cost,
                    }
                    for step in plan_steps
                ]
            }

            # Determine optimization goals based on constraints
            optimization_goals = []
            if constraints.get("budget"):
                optimization_goals.append("cost")
            if constraints.get("timeline"):
                optimization_goals.append("speed")
            optimization_goals.extend(
                ["security", "performance"]
            )  # Always optimize these

            request = PlanOptimizationRequest(
                original_plan=plan_dict,
                optimization_goals=optimization_goals,
                constraints=list(constraints.keys()),
                available_tools=list(self._available_tools),
            )

            if self.llm_client is None:
                raise RuntimeError("LLM client not initialized")
            optimized_plan_data = await self.llm_client.optimize_plan(request)

            # Convert back to PlanStep objects
            return self._convert_llm_response_to_steps(optimized_plan_data, analysis)

        except Exception as e:
            self.logger.warning(
                f"LLM optimization failed: {e}, falling back to rule-based"
            )
            return await self._rule_based_optimize_plan(
                plan_steps, analysis, constraints
            )

    async def _rule_based_optimize_plan(
        self,
        plan_steps: List[PlanStep],
        analysis: AnalysisResult,
        constraints: Dict[str, Any],
    ) -> List[PlanStep]:
        """Apply rule-based optimizations to the plan."""
        optimized_steps = plan_steps.copy()

        # Cost optimization
        if constraints.get("budget"):
            optimized_steps = self._optimize_for_cost(optimized_steps)

        # Timeline optimization
        if constraints.get("timeline"):
            optimized_steps = self._optimize_for_timeline(optimized_steps)

        # Remove redundant steps
        optimized_steps = self._remove_redundant_steps(optimized_steps)

        # Optimize dependencies for parallel execution
        optimized_steps = self._optimize_dependencies(optimized_steps)

        return optimized_steps

    def _optimize_for_cost(self, plan_steps: List[PlanStep]) -> List[PlanStep]:
        """Apply cost optimization rules."""
        # Simple cost optimization: prefer cheaper alternatives
        for step in plan_steps:
            if step.tool == "aws" and step.action == "create_instance":
                # Use smaller instance sizes
                if "instance_type" in step.parameters:
                    current_type = step.parameters["instance_type"]
                    if "large" in current_type:
                        step.parameters["instance_type"] = current_type.replace(
                            "large", "medium"
                        )
                        step.estimated_cost *= 0.7

        return plan_steps

    def _optimize_for_timeline(self, plan_steps: List[PlanStep]) -> List[PlanStep]:
        """Apply timeline optimization rules."""
        # Identify steps that can be parallelized
        for step in plan_steps:
            if not step.dependencies:
                # Independent steps can be marked for parallel execution
                step.parameters["parallel_execution"] = True

        return plan_steps

    def _remove_redundant_steps(self, plan_steps: List[PlanStep]) -> List[PlanStep]:
        """Remove redundant or duplicate steps."""
        seen_actions = set()
        unique_steps = []

        for step in plan_steps:
            action_key = f"{step.tool}:{step.action}"
            if action_key not in seen_actions:
                unique_steps.append(step)
                seen_actions.add(action_key)

        return unique_steps

    def _optimize_dependencies(self, plan_steps: List[PlanStep]) -> List[PlanStep]:
        """Optimize step dependencies for better execution flow."""
        # This is a simplified dependency optimization
        # In a production system, this would use graph algorithms

        step_dict = {step.id: step for step in plan_steps}

        for step in plan_steps:
            # Remove self-dependencies
            step.dependencies = [dep for dep in step.dependencies if dep != step.id]

            # Remove non-existent dependencies
            step.dependencies = [dep for dep in step.dependencies if dep in step_dict]

        return plan_steps

    async def _validate_plan(self, plan: PlanStructure) -> ValidationResult:
        """Validate plan - base class interface."""
        # Convert PlanStructure to the format expected by intelligent validation
        plan_steps = plan.steps
        # For now, use empty requirements and analysis
        requirements = ""
        analysis = AnalysisResult(original_requirements="")

        return await self._validate_intelligent_plan(plan_steps, requirements, analysis)

    async def _validate_intelligent_plan(
        self, plan_steps: List[PlanStep], requirements: str, analysis: AnalysisResult
    ) -> ValidationResult:
        """Validate the generated plan."""
        self.logger.debug("Validating plan")

        issues = []

        # Check for tool availability
        for step in plan_steps:
            if step.tool not in self._available_tools:
                issues.append(
                    f"Tool '{step.tool}' not available for step '{step.name}'"
                )

        # Check for circular dependencies
        if self._has_circular_dependencies(plan_steps):
            issues.append("Plan contains circular dependencies")

        # Check for missing dependencies
        step_ids = {step.id for step in plan_steps}
        for step in plan_steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    issues.append(
                        f"Step '{step.name}' depends on non-existent step '{dep}'"
                    )

        # Use LLM for additional validation if available
        if self.llm_client:
            try:
                llm_validation = await self._llm_validate_plan(plan_steps, requirements)
                if not llm_validation.get("is_valid", True):
                    issues.extend(llm_validation.get("issues", []))
            except Exception as e:
                self.logger.warning(f"LLM validation failed: {e}")

        return ValidationResult(
            valid=len(issues) == 0,
            errors=issues,
            confidence=0.9 if len(issues) == 0 else max(0.1, 0.9 - len(issues) * 0.1),
        )

    def _has_circular_dependencies(self, plan_steps: List[PlanStep]) -> bool:
        """Check for circular dependencies in the plan."""
        # Simple cycle detection using DFS
        step_dict = {step.id: step for step in plan_steps}
        visited = set()
        rec_stack = set()

        def has_cycle(step_id):
            if step_id not in step_dict:
                return False

            visited.add(step_id)
            rec_stack.add(step_id)

            for dep in step_dict[step_id].dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(step_id)
            return False

        for step in plan_steps:
            if step.id not in visited:
                if has_cycle(step.id):
                    return True

        return False

    async def _llm_validate_plan(
        self, plan_steps: List[PlanStep], requirements: str
    ) -> Dict[str, Any]:
        """Use LLM to validate the plan against requirements."""
        plan_dict = {
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "description": step.description,
                    "tool": step.tool,
                    "action": step.action,
                    "dependencies": step.dependencies,
                }
                for step in plan_steps
            ]
        }

        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")
        return await self.llm_client.validate_plan(plan_dict, requirements)

    async def _create_template_fallback_plan(
        self, requirements: str, available_tools: List[str]
    ) -> List[PlanStep]:
        """Create a basic plan using template-based approach as fallback."""
        self.logger.info("Creating template-based fallback plan")

        # This is a simplified fallback - in production this would use
        # the existing template planner
        fallback_steps = [
            PlanStep(
                id="fallback_1",
                name="Basic Infrastructure Setup",
                description="Create basic infrastructure components",
                tool="aws" if "aws" in available_tools else "docker",
                action="create_basic_environment",
                parameters={},
                dependencies=[],
                estimated_duration=1200.0,
                estimated_cost=100.0,
            )
        ]

        return fallback_steps

    def register_tool_capabilities(
        self, tool_name: str, capabilities: Dict[str, Any]
    ) -> None:
        """Register tool capabilities for planning."""
        self._available_tools.add(tool_name)
        self._tool_capabilities[tool_name] = capabilities
        self.logger.debug(f"Registered tool capabilities: {tool_name}")

    def update_cost_estimates(self, cost_data: Dict[str, float]) -> None:
        """Update cost estimates for tools and actions."""
        self._cost_estimates.update(cost_data)
        self.logger.debug(f"Updated cost estimates for {len(cost_data)} items")

    async def explain_plan(self, plan_steps: List[PlanStep]) -> str:
        """Generate human-readable explanation of the plan."""
        if self.llm_client:
            try:
                plan_dict = {
                    "steps": [
                        {
                            "name": step.name,
                            "description": step.description,
                            "estimated_duration": step.estimated_duration,
                            "estimated_cost": step.estimated_cost,
                        }
                        for step in plan_steps
                    ],
                    "total_cost": sum(step.estimated_cost for step in plan_steps),
                    "total_duration": sum(
                        step.estimated_duration for step in plan_steps
                    ),
                }

                return await self.llm_client.explain_plan(plan_dict)
            except Exception as e:
                self.logger.warning(f"LLM plan explanation failed: {e}")

        # Fallback to basic explanation
        total_cost = sum(step.estimated_cost for step in plan_steps)
        total_duration = (
            sum(step.estimated_duration for step in plan_steps) / 3600
        )  # Convert to hours

        explanation = f"""
        This plan consists of {len(plan_steps)} steps that will:

        """

        for i, step in enumerate(plan_steps, 1):
            explanation += f"{i}. {step.name}: {step.description}\n"

        explanation += f"""

        Estimated total cost: ${total_cost:.2f}
        Estimated total duration: {total_duration:.1f} hours
        """

        return explanation.strip()

    async def get_planning_recommendations(self, requirements: str) -> Dict[str, Any]:
        """Get recommendations for improving requirements or planning."""
        analysis = await self._analyze_requirements(requirements, {})

        recommendations_list: List[str] = []
        recommendations = {
            "analysis_confidence": analysis.analysis_confidence,
            "completeness_score": analysis.completeness_score,
            "ambiguity_score": analysis.ambiguity_score,
            "recommendations": recommendations_list,
        }

        if analysis.ambiguity_score > 0.5:
            recommendations_list.append(
                "Consider providing more specific details about your requirements"
            )

        if analysis.completeness_score < 0.7:
            recommendations_list.append(
                "Consider specifying performance, security, or scalability requirements"
            )

        if not analysis.detected_technologies:
            recommendations_list.append(
                "Consider mentioning specific technologies or frameworks you prefer"
            )

        return recommendations

    async def _gather_context(self, requirements: Requirements) -> Dict[str, Any]:
        """Gather context for planning from requirements."""
        # Convert the Requirements object to a context dictionary
        # This bridges the base Planner interface with our intelligent planner
        return {
            "requirements_text": requirements.description
            if hasattr(requirements, "description")
            else str(requirements),
            "technologies": getattr(requirements, "technologies", []),
            "constraints": getattr(requirements, "constraints", {}),
            "preferences": getattr(requirements, "preferences", {}),
        }

    async def _generate_initial_plan(self, context: Dict[str, Any]) -> Any:
        """Generate initial plan from context."""
        # Extract requirements from context
        requirements_text = context.get("requirements_text", "")
        context.get("technologies", [])
        constraints = context.get("constraints", {})

        # Use our intelligent create_plan method
        plan_steps = await self.create_intelligent_plan(
            requirements=requirements_text,
            context=context,
            available_tools=list(self._available_tools),
            constraints=constraints,
        )

        # Return a simple object with steps for compatibility with base class
        class InitialPlan:
            def __init__(self, steps):
                self.steps = steps

        return InitialPlan(plan_steps)
