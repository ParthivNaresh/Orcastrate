"""
Pytest configuration and shared fixtures.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from src.agent.base import Agent, AgentRegistry, Plan, Requirements
from src.executors.base import Executor, ExecutorConfig
from src.planners.base import Planner, PlannerConfig, PlanStep
from src.security.manager import SecurityManager
from src.tools.base import CostEstimate, Tool, ToolConfig, ToolSchema


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_requirements() -> Requirements:
    """Create sample requirements for testing."""
    return Requirements(
        description="Create a FastAPI web application with PostgreSQL database",
        framework="fastapi",
        database="postgresql",
        cloud_provider="aws",
        scaling_requirements={"min_instances": 1, "max_instances": 10},
        security_requirements={"encryption": True, "authentication": "oauth2"},
        budget_constraints={"max_monthly_cost": 500.0},
    )


@pytest.fixture
def sample_plan_steps() -> List[PlanStep]:
    """Create sample plan steps for testing."""
    return [
        PlanStep(
            id="step1",
            name="Create VPC",
            description="Create Virtual Private Cloud",
            tool="aws_tool",
            action="create_vpc",
            parameters={"cidr": "10.0.0.0/16"},
            estimated_duration=300.0,
            estimated_cost=50.0,
        ),
        PlanStep(
            id="step2",
            name="Create Database",
            description="Create PostgreSQL database",
            tool="aws_tool",
            action="create_rds",
            parameters={"engine": "postgresql", "instance_class": "t3.micro"},
            dependencies=["step1"],
            estimated_duration=600.0,
            estimated_cost=100.0,
        ),
        PlanStep(
            id="step3",
            name="Deploy Application",
            description="Deploy FastAPI application",
            tool="aws_tool",
            action="create_ecs_service",
            parameters={"image": "myapp:latest", "port": 8000},
            dependencies=["step1", "step2"],
            estimated_duration=400.0,
            estimated_cost=75.0,
        ),
    ]


@pytest.fixture
def sample_plan(
    sample_requirements: Requirements, sample_plan_steps: List[PlanStep]
) -> Plan:
    """Create a sample plan for testing."""
    return Plan(
        steps=[step.model_dump() for step in sample_plan_steps],
        dependencies={"step2": ["step1"], "step3": ["step1", "step2"]},
        estimated_cost=225.0,
        estimated_duration=1300.0,
        risk_assessment={
            "overall_risk": 0.3,
            "risk_factors": ["Network configuration"],
        },
        requirements=sample_requirements,
    )


@pytest.fixture
def tool_config() -> ToolConfig:
    """Create tool configuration for testing."""
    return ToolConfig(
        name="test_tool",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        retry_delay=5,
        environment={"TEST_MODE": "true"},
        credentials={"api_key": "test_key"},
    )


@pytest.fixture
def planner_config() -> PlannerConfig:
    """Create planner configuration for testing."""
    return PlannerConfig(
        max_plan_steps=50,
        max_planning_time=300,
        cost_optimization=True,
        risk_threshold=0.7,
        parallel_execution=True,
    )


@pytest.fixture
def executor_config() -> ExecutorConfig:
    """Create executor configuration for testing."""
    return ExecutorConfig(
        max_concurrent_steps=5,
        step_timeout=3600,
        retry_policy={"max_retries": 3, "backoff_factor": 2.0, "max_delay": 300},
        enable_rollback=True,
    )


@pytest.fixture
def security_config() -> Dict[str, Any]:
    """Create security manager configuration for testing."""
    return {
        "enabled": True,
        "strict_mode": False,
        "audit_enabled": True,
        "blacklisted_tools": ["dangerous_tool"],
        "blacklisted_actions": ["format_disk"],
    }


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, config: ToolConfig, fail_actions: List[str] = None):
        super().__init__(config)
        self.fail_actions = fail_actions or []
        self.execution_count = 0
        self.last_action = None
        self.last_params = None

    async def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.config.name,
            description="Mock tool for testing",
            version=self.config.version,
            actions={
                "create_vpc": {"description": "Create VPC"},
                "create_rds": {"description": "Create RDS instance"},
                "create_ecs_service": {"description": "Create ECS service"},
            },
        )

    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        return CostEstimate(
            estimated_cost=50.0, cost_breakdown={"compute": 30.0, "storage": 20.0}
        )

    async def _create_client(self) -> Any:
        return Mock()

    async def _create_validator(self) -> Any:
        return Mock()

    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.execution_count += 1
        self.last_action = action
        self.last_params = params

        if action in self.fail_actions:
            raise Exception(f"Mock failure for action: {action}")

        return {
            "action": action,
            "params": params,
            "result": "success",
            "resource_id": f"{action}_resource_123",
        }

    async def _execute_rollback(self, execution_id: str) -> Dict[str, Any]:
        return {"rollback": "success", "execution_id": execution_id}

    async def _get_supported_actions(self) -> List[str]:
        return ["create_vpc", "create_rds", "create_ecs_service"]


class MockPlanner(Planner):
    """Mock planner for testing."""

    def __init__(self, config: PlannerConfig, should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.plan_count = 0

    async def _generate_initial_plan(self, context: Dict[str, Any]) -> Any:
        self.plan_count += 1

        if self.should_fail:
            raise Exception("Mock planner failure")

        from src.planners.base import PlanStep, PlanStructure

        steps = [
            PlanStep(
                id="mock_step_1",
                name="Mock Step 1",
                description="First mock step",
                tool="mock_tool",
                action="create_vpc",
                parameters={"cidr": "10.0.0.0/16"},
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
            PlanStep(
                id="mock_step_2",
                name="Mock Step 2",
                description="Second mock step",
                tool="mock_tool",
                action="create_rds",
                parameters={"engine": "postgresql"},
                dependencies=["mock_step_1"],
                estimated_duration=600.0,
                estimated_cost=100.0,
            ),
        ]

        return PlanStructure(steps=steps)

    async def _gather_context(self, requirements: Requirements) -> Dict[str, Any]:
        return {
            "requirements": requirements.model_dump(),
            "templates": [],
            "constraints": {},
        }


class MockExecutor(Executor):
    """Mock executor for testing."""

    def __init__(self, config: ExecutorConfig, should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.execution_count = 0
        self.last_plan = None

    async def _execute_plan_with_strategy(self, plan: Plan, context: Any) -> Any:
        from src.agent.base import ExecutionResult

        self.execution_count += 1
        self.last_plan = plan

        if self.should_fail:
            return ExecutionResult(
                success=False,
                execution_id=context.execution_id,
                error="Mock executor failure",
            )

        return ExecutionResult(
            success=True,
            execution_id=context.execution_id,
            artifacts={"deployed_resources": ["vpc-123", "rds-456"]},
            logs=["Step 1 completed", "Step 2 completed"],
            metrics={"execution_time": 900.0, "cost": 150.0},
        )


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(
        self, agent_id: str, config: Dict[str, Any], should_fail: bool = False
    ):
        super().__init__(agent_id, config)
        self.should_fail = should_fail
        self.plan_count = 0
        self.execution_count = 0

    async def plan(self, requirements: Requirements) -> Plan:
        self.plan_count += 1

        if self.should_fail:
            raise Exception("Mock agent planning failure")

        return Plan(
            steps=[
                {
                    "id": "mock_step",
                    "name": "Mock Step",
                    "tool": "mock_tool",
                    "action": "mock_action",
                    "parameters": {},
                }
            ],
            estimated_cost=100.0,
            estimated_duration=600.0,
            requirements=requirements,
        )

    async def execute(self, plan: Plan) -> Any:
        from src.agent.base import ExecutionResult

        self.execution_count += 1

        if self.should_fail:
            return ExecutionResult(
                success=False,
                execution_id="mock_exec",
                error="Mock agent execution failure",
            )

        return ExecutionResult(
            success=True, execution_id="mock_exec", artifacts={"result": "success"}
        )

    async def monitor(self, execution_id: str) -> Dict[str, Any]:
        return {"execution_id": execution_id, "status": "completed", "progress": 1.0}


@pytest.fixture
def mock_tool(tool_config: ToolConfig) -> MockTool:
    """Create a mock tool for testing."""
    return MockTool(tool_config)


@pytest.fixture
def mock_planner(planner_config: PlannerConfig) -> MockPlanner:
    """Create a mock planner for testing."""
    return MockPlanner(planner_config)


@pytest.fixture
def mock_executor(executor_config: ExecutorConfig) -> MockExecutor:
    """Create a mock executor for testing."""
    return MockExecutor(executor_config)


@pytest.fixture
def mock_agent() -> MockAgent:
    """Create a mock agent for testing."""
    return MockAgent("test_agent", {"test": True})


@pytest.fixture
def security_manager(security_config: Dict[str, Any]) -> SecurityManager:
    """Create a security manager for testing."""
    return SecurityManager(security_config)


@pytest.fixture(autouse=True)
def reset_agent_registry():
    """Reset the agent registry before each test."""
    # Clear the singleton instance
    AgentRegistry._instance = None
    yield
    # Clear it again after the test
    AgentRegistry._instance = None
