# Testing Guide

## Overview

Orcastrate uses a comprehensive testing strategy with unit tests, integration tests, and end-to-end tests to ensure reliability and maintainability.

## Test Structure

```
tests/
├── unit/                     # Unit tests (isolated components)
│   ├── agent/               # Agent layer tests
│   ├── planners/            # Planner tests
│   ├── executors/           # Executor tests
│   ├── tools/               # Tool tests
│   └── security/            # Security tests
├── integration/             # Integration tests (component interaction)
└── conftest.py             # Shared test configuration
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test directory
pytest tests/unit/planners/

# Run specific test file
pytest tests/unit/planners/test_template_planner.py

# Run specific test method
pytest tests/unit/planners/test_template_planner.py::TestTemplatePlanner::test_create_plan
```

### Test Coverage

```bash
# Run tests with coverage
pytest --cov=src

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Parallel Test Execution

```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## Test Categories

### Unit Tests

Test individual components in isolation using mocks for dependencies.

```python
# Example: Testing template planner
class TestTemplatePlanner:
    @pytest.fixture
    def planner_config(self):
        return PlannerConfig(
            strategy=PlanningStrategy.TEMPLATE_MATCHING,
            max_plan_steps=20
        )
    
    @pytest.fixture
    async def planner(self, planner_config):
        planner = TemplatePlanner(planner_config)
        await planner.initialize()
        return planner
    
    @pytest.mark.asyncio
    async def test_template_selection(self, planner):
        requirements = Requirements(
            description="Node.js web application",
            framework="nodejs"
        )
        
        template = await planner._select_template(requirements)
        assert template is not None
        assert template["framework"] == "nodejs"
```

### Integration Tests

Test component interactions and workflows.

```python
# Example: Testing planner-executor integration
class TestPlannerExecutorIntegration:
    @pytest.mark.asyncio
    async def test_plan_generation_and_validation(self):
        # Create planner
        planner = TemplatePlanner(planner_config)
        await planner.initialize()
        
        # Create executor
        executor = ConcreteExecutor(executor_config)
        await executor.initialize()
        
        # Generate plan
        requirements = Requirements(description="Node.js app")
        plan = await planner.create_plan(requirements)
        
        # Validate plan
        validation = await executor.validate_plan_requirements(plan)
        assert validation["valid"] is True
```

### End-to-End Tests

Test complete workflows from requirements to execution.

```python
# Example: Complete workflow test
class TestEndToEndWorkflow:
    @pytest.mark.asyncio
    async def test_nodejs_workflow(self):
        requirements = Requirements(
            description="Create a Node.js web application",
            framework="nodejs"
        )
        
        # Generate plan
        plan = await planner.create_plan(requirements)
        assert len(plan.steps) > 0
        
        # Execute plan (with mocked tools)
        result = await executor.execute_plan(plan)
        assert result.success is True
```

## Test Configuration

### Pytest Configuration

```ini
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--disable-warnings",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "slow: Slow tests",
]
asyncio_mode = "auto"
```

### Fixtures

```python
# tests/conftest.py
import pytest
import asyncio
from src.planners.template_planner import TemplatePlanner
from src.executors.concrete_executor import ConcreteExecutor

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def template_planner():
    """Create initialized template planner."""
    config = PlannerConfig(strategy=PlanningStrategy.TEMPLATE_MATCHING)
    planner = TemplatePlanner(config)
    await planner.initialize()
    return planner

@pytest.fixture
async def mock_executor():
    """Create executor with mocked tools."""
    config = ExecutorConfig(strategy=ExecutionStrategy.SEQUENTIAL)
    executor = ConcreteExecutor(config)
    
    # Mock tool initialization
    with patch.object(executor, '_initialize_tools'):
        await executor.initialize()
    
    return executor
```

## Mocking Strategies

### External Dependencies

```python
# Mock Docker client
@patch('src.tools.docker.docker.from_env')
async def test_docker_tool(mock_docker):
    mock_client = AsyncMock()
    mock_docker.return_value = mock_client
    
    tool = DockerTool(config)
    await tool.initialize()
    
    # Test tool operations
    result = await tool.execute("list_containers", {})
    assert result.success is True
```

### Filesystem Operations

```python
# Mock filesystem operations
@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.write_text')
async def test_filesystem_tool(mock_write, mock_mkdir):
    tool = FileSystemTool(config)
    await tool.initialize()
    
    result = await tool.execute("create_directory", {"path": "/test"})
    mock_mkdir.assert_called_once()
```

### Network Operations

```python
# Mock HTTP requests
@patch('aiohttp.ClientSession.get')
async def test_api_call(mock_get):
    mock_response = AsyncMock()
    mock_response.json.return_value = {"status": "success"}
    mock_get.return_value.__aenter__.return_value = mock_response
    
    # Test API interaction
    result = await api_client.fetch_data()
    assert result["status"] == "success"
```

## Test Data Management

### Test Fixtures

```python
# Create test data directory
tests/
├── fixtures/
│   ├── plans/
│   │   ├── nodejs_plan.json
│   │   └── fastapi_plan.json
│   ├── templates/
│   │   └── custom_template.json
│   └── requirements/
│       └── sample_requirements.json
```

### Loading Test Data

```python
import json
from pathlib import Path

@pytest.fixture
def sample_plan():
    """Load sample plan from fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "plans" / "nodejs_plan.json"
    with fixture_path.open() as f:
        return json.load(f)

@pytest.fixture
def sample_requirements():
    """Create sample requirements for testing."""
    return Requirements(
        description="Test Node.js application",
        framework="nodejs",
        database="mongodb"
    )
```

## Performance Testing

### Load Testing

```python
import asyncio
import time
import pytest

@pytest.mark.slow
async def test_concurrent_plan_generation():
    """Test planner performance under load."""
    planner = TemplatePlanner(config)
    await planner.initialize()
    
    requirements = Requirements(description="Node.js app", framework="nodejs")
    
    # Test concurrent plan generation
    start_time = time.time()
    tasks = [planner.create_plan(requirements) for _ in range(10)]
    plans = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Verify all plans were generated
    assert len(plans) == 10
    assert all(len(plan.steps) > 0 for plan in plans)
    
    # Check performance
    total_time = end_time - start_time
    assert total_time < 5.0  # Should complete within 5 seconds
```

### Memory Testing

```python
import psutil
import os

def test_memory_usage():
    """Test memory usage during plan execution."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operations
    plans = []
    for i in range(100):
        plan = create_large_plan()
        plans.append(plan)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - quality

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - pytest --cov=src --cov-report=xml --cov-report=term
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

quality:
  stage: quality
  image: python:3.9
  script:
    - pip install -r requirements-dev.txt
    - flake8 src/ tests/
    - black --check src/ tests/
    - mypy src/
```

## Test Best Practices

### Test Organization

1. **One test per behavior**: Each test should verify one specific behavior
2. **Descriptive names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent tests**: Tests should not depend on each other

### Test Data

1. **Use fixtures**: Create reusable test data with pytest fixtures
2. **Minimal data**: Use the minimum data necessary to test the behavior
3. **Deterministic**: Test data should produce consistent results
4. **Realistic**: Use data that represents real-world scenarios

### Assertion Guidelines

```python
# Good: Specific assertions
assert result.success is True
assert len(plan.steps) == 5
assert plan.estimated_cost == 0.15

# Bad: Vague assertions
assert result
assert plan.steps
assert plan.estimated_cost
```

### Error Testing

```python
# Test error conditions explicitly
with pytest.raises(PlannerError, match="No suitable template"):
    await planner.create_plan(invalid_requirements)

# Test error messages
try:
    await tool.execute("invalid_action", {})
except ToolError as e:
    assert "Unknown action" in str(e)
```
