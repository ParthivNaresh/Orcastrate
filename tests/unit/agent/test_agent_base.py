"""
Tests for agent base classes and core functionality.
"""

from datetime import datetime

import pytest

from src.agent.base import (
    Agent,
    AgentError,
    AgentRegistry,
    AgentStatus,
    ExecutionError,
    ExecutionResult,
    MonitoringError,
    Plan,
    PlanningError,
    Requirements,
)


class TestRequirements:
    """Test Requirements model."""

    def test_requirements_creation(self):
        """Test basic requirements creation."""
        req = Requirements(
            description="Test environment", framework="fastapi", database="postgresql"
        )

        assert req.description == "Test environment"
        assert req.framework == "fastapi"
        assert req.database == "postgresql"
        assert req.cloud_provider is None
        assert req.metadata == {}

    def test_requirements_with_optional_fields(self):
        """Test requirements with all optional fields."""
        req = Requirements(
            description="Complex environment",
            framework="django",
            database="mysql",
            cloud_provider="aws",
            scaling_requirements={"min": 1, "max": 10},
            security_requirements={"tls": True},
            budget_constraints={"monthly": 500},
            metadata={"project": "test"},
        )

        assert req.scaling_requirements == {"min": 1, "max": 10}
        assert req.security_requirements == {"tls": True}
        assert req.budget_constraints == {"monthly": 500}
        assert req.metadata == {"project": "test"}


class TestPlan:
    """Test Plan model."""

    def test_plan_creation(self, sample_requirements):
        """Test basic plan creation."""
        plan = Plan(
            steps=[{"id": "step1", "action": "create_vpc"}],
            dependencies={"step1": []},
            estimated_cost=100.0,
            estimated_duration=600.0,
            requirements=sample_requirements,
        )

        assert len(plan.steps) == 1
        assert plan.estimated_cost == 100.0
        assert plan.estimated_duration == 600.0
        assert plan.requirements == sample_requirements
        assert plan.id is not None  # UUID should be generated

    def test_plan_with_complex_dependencies(self, sample_requirements):
        """Test plan with complex dependency structure."""
        plan = Plan(
            steps=[
                {"id": "step1", "action": "create_vpc"},
                {"id": "step2", "action": "create_db"},
                {"id": "step3", "action": "deploy_app"},
            ],
            dependencies={"step2": ["step1"], "step3": ["step1", "step2"]},
            estimated_cost=250.0,
            estimated_duration=1200.0,
            requirements=sample_requirements,
        )

        assert plan.dependencies["step2"] == ["step1"]
        assert plan.dependencies["step3"] == ["step1", "step2"]


class TestExecutionResult:
    """Test ExecutionResult model."""

    def test_successful_execution_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            execution_id="exec_123",
            artifacts={"vpc_id": "vpc-123"},
            logs=["Step 1 completed"],
            metrics={"duration": 300.0},
        )

        assert result.success is True
        assert result.execution_id == "exec_123"
        assert result.artifacts == {"vpc_id": "vpc-123"}
        assert result.logs == ["Step 1 completed"]
        assert result.metrics == {"duration": 300.0}
        assert result.error is None
        assert isinstance(result.created_at, datetime)

    def test_failed_execution_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            execution_id="exec_456",
            error="Network timeout",
            duration=30.0,
        )

        assert result.success is False
        assert result.error == "Network timeout"
        assert result.duration == 30.0


class TestAgentRegistry:
    """Test AgentRegistry singleton."""

    def test_singleton_behavior(self):
        """Test that AgentRegistry is a singleton."""
        registry1 = AgentRegistry()
        registry2 = AgentRegistry()

        assert registry1 is registry2

    def test_register_agent_type(self):
        """Test registering agent types."""
        registry = AgentRegistry()

        class TestAgent(Agent):
            async def plan(self, requirements):
                pass

            async def execute(self, plan):
                pass

            async def monitor(self, execution_id):
                pass

        registry.register_agent_type("test_agent", TestAgent)

        assert "test_agent" in registry._agent_types
        assert registry._agent_types["test_agent"] == TestAgent

    def test_create_agent(self):
        """Test creating agent instances."""
        registry = AgentRegistry()

        class TestAgent(Agent):
            async def plan(self, requirements):
                pass

            async def execute(self, plan):
                pass

            async def monitor(self, execution_id):
                pass

        registry.register_agent_type("test_agent", TestAgent)
        agent = registry.create_agent("test_agent", "agent_1", {"config": "test"})

        assert agent.agent_id == "agent_1"
        assert agent.config == {"config": "test"}
        assert registry.get_agent("agent_1") is agent

    def test_create_unknown_agent_type(self):
        """Test creating agent with unknown type raises error."""
        registry = AgentRegistry()

        with pytest.raises(ValueError, match="Unknown agent type"):
            registry.create_agent("unknown_type", "agent_1", {})

    def test_agent_management(self):
        """Test agent management operations."""
        registry = AgentRegistry()

        class TestAgent(Agent):
            async def plan(self, requirements):
                pass

            async def execute(self, plan):
                pass

            async def monitor(self, execution_id):
                pass

        registry.register_agent_type("test_agent", TestAgent)
        agent = registry.create_agent("test_agent", "agent_1", {})

        # Test listing agents
        assert "agent_1" in registry.list_agents()

        # Test getting agent
        assert registry.get_agent("agent_1") is agent
        assert registry.get_agent("nonexistent") is None

        # Test removing agent
        assert registry.remove_agent("agent_1") is True
        assert registry.remove_agent("agent_1") is False  # Already removed
        assert registry.get_agent("agent_1") is None


class TestAgent:
    """Test Agent base class."""

    @pytest.fixture
    def concrete_agent(self):
        """Create a concrete agent for testing."""

        class ConcreteAgent(Agent):
            def __init__(self, agent_id, config):
                super().__init__(agent_id, config)
                self.plan_called = False
                self.execute_called = False
                self.monitor_called = False

            async def plan(self, requirements):
                self.plan_called = True
                return Plan(
                    steps=[{"id": "step1", "action": "test"}], requirements=requirements
                )

            async def execute(self, plan):
                self.execute_called = True
                return ExecutionResult(success=True, execution_id="test_exec")

            async def monitor(self, execution_id):
                self.monitor_called = True
                return {"status": "running"}

        return ConcreteAgent("test_agent", {"test": True})

    def test_agent_initialization(self, concrete_agent):
        """Test agent initialization."""
        assert concrete_agent.agent_id == "test_agent"
        assert concrete_agent.config == {"test": True}
        assert concrete_agent.status == AgentStatus.IDLE
        assert len(concrete_agent._execution_history) == 0

    @pytest.mark.asyncio
    async def test_validate_requirements(self, concrete_agent, sample_requirements):
        """Test requirements validation."""
        # Valid requirements
        valid = await concrete_agent.validate_requirements(sample_requirements)
        assert valid is True

        # Invalid requirements (empty description)
        invalid_req = Requirements(description="")
        invalid = await concrete_agent.validate_requirements(invalid_req)
        assert invalid is False

    def test_execution_history_management(self, concrete_agent):
        """Test execution history management."""
        # Add execution results
        for i in range(5):
            result = ExecutionResult(success=True, execution_id=f"exec_{i}")
            concrete_agent.add_execution_result(result)

        history = concrete_agent.get_execution_history()
        assert len(history) == 5
        assert history[0].execution_id == "exec_0"
        assert history[-1].execution_id == "exec_4"

    def test_execution_history_limit(self, concrete_agent):
        """Test execution history is limited to prevent memory issues."""
        # Add more than 100 results
        for i in range(105):
            result = ExecutionResult(success=True, execution_id=f"exec_{i}")
            concrete_agent.add_execution_result(result)

        history = concrete_agent.get_execution_history()
        assert len(history) == 100
        # Should keep the last 100
        assert history[0].execution_id == "exec_5"
        assert history[-1].execution_id == "exec_104"


class TestAgentExceptions:
    """Test agent-related exceptions."""

    def test_agent_error(self):
        """Test AgentError exception."""
        error = AgentError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_planning_error(self):
        """Test PlanningError exception."""
        error = PlanningError("Planning failed")
        assert str(error) == "Planning failed"
        assert isinstance(error, AgentError)

    def test_execution_error(self):
        """Test ExecutionError exception."""
        error = ExecutionError("Execution failed")
        assert str(error) == "Execution failed"
        assert isinstance(error, AgentError)

    def test_monitoring_error(self):
        """Test MonitoringError exception."""
        error = MonitoringError("Monitoring failed")
        assert str(error) == "Monitoring failed"
        assert isinstance(error, AgentError)
