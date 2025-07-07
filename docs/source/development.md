# Development Guide

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Docker (optional, for full testing)
- Make (optional, for build automation)

### Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd orcastrate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Project Structure

```
orcastrate/
├── src/                    # Source code
│   ├── agent/             # Agent orchestration
│   ├── planners/          # Plan generation
│   ├── executors/         # Plan execution
│   ├── tools/             # Concrete tools
│   ├── security/          # Security components
│   ├── cli/               # Command-line interface
│   └── api/               # REST API (future)
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   │   ├── agent/
│   │   ├── planners/
│   │   ├── executors/
│   │   ├── tools/
│   │   └── security/
│   └── integration/       # Integration tests
├── docs/                  # Documentation
│   ├── source/            # Sphinx source files
│   └── build/             # Generated documentation
├── demo.py               # Demo script
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── pyproject.toml        # Project configuration
└── README.md            # Project overview
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-tool

# Make changes
# ... edit code ...

# Run tests
make test

# Check code quality
make lint
make format

# Commit changes
git add .
git commit -m "Add new tool implementation"

# Push and create PR
git push origin feature/new-tool
```

### 2. Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/unit/tools/

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/unit/planners/test_template_planner.py::TestTemplatePlanner::test_create_plan
```

### 3. Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
pylint src/

# Type checking
mypy src/
```

## Adding New Components

### Adding a New Tool

1. **Create the tool class:**

```python
# src/tools/my_tool.py
from typing import Any, Dict
from .base import Tool, ToolConfig, ToolSchema, ToolResult

class MyTool(Tool):
    """Description of what this tool does."""
    
    async def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="my_tool",
            description="Tool description",
            version=self.config.version,
            actions={
                "my_action": {
                    "description": "Action description",
                    "parameters": {
                        "param1": {"type": "string", "required": True},
                        "param2": {"type": "integer", "required": False}
                    }
                }
            },
            required_permissions=["my_permission"]
        )
    
    async def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if action == "my_action":
            return await self._my_action(parameters)
        else:
            raise ToolError(f"Unknown action: {action}")
    
    async def _my_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implement action logic
        return {"success": True, "result": "action completed"}
```

2. **Create tests:**

```python
# tests/unit/tools/test_my_tool.py
import pytest
from src.tools.my_tool import MyTool
from src.tools.base import ToolConfig

class TestMyTool:
    @pytest.fixture
    def tool_config(self):
        return ToolConfig(name="my_tool", version="1.0.0")
    
    @pytest.fixture
    def tool(self, tool_config):
        return MyTool(tool_config)
    
    @pytest.mark.asyncio
    async def test_my_action(self, tool):
        result = await tool.execute("my_action", {"param1": "value"})
        assert result.success is True
```

3. **Register with executor:**

```python
# src/executors/concrete_executor.py
from ..tools.my_tool import MyTool

class ConcreteExecutor(Executor):
    async def _initialize_tools(self) -> None:
        # Add to tool initialization
        my_tool_config = ToolConfig(name="my_tool", version="1.0.0")
        my_tool = MyTool(my_tool_config)
        await my_tool.initialize()
        self._tools["my_tool"] = my_tool
```

### Adding a New Planner

1. **Create the planner class:**

```python
# src/planners/my_planner.py
from typing import Dict, Any
from .base import Planner, Plan, Requirements, PlanStructure

class MyPlanner(Planner):
    """Custom planner implementation."""
    
    async def _generate_initial_plan(self, context: Dict[str, Any]) -> PlanStructure:
        # Implement planning logic
        requirements = context["requirements"]
        steps = self._create_steps(requirements)
        
        return PlanStructure(
            steps=steps,
            metadata={"planner": "my_planner"}
        )
    
    def _create_steps(self, requirements: Requirements) -> List[PlanStep]:
        # Generate plan steps based on requirements
        pass
```

2. **Add configuration:**

```python
# src/planners/base.py
class PlanningStrategy(str, Enum):
    TEMPLATE_MATCHING = "template_matching"
    MY_STRATEGY = "my_strategy"  # Add new strategy
```

### Adding a New Template

```python
# In template planner or configuration
custom_template = {
    "name": "My Custom Template",
    "description": "Description of the template",
    "framework": "my_framework",
    "steps": [
        {
            "id": "step1",
            "name": "Step Name",
            "description": "Step description",
            "tool": "filesystem",
            "action": "create_directory",
            "parameters": {"path": "{project_path}"},
            "estimated_duration": 30.0,
            "estimated_cost": 0.0
        }
    ],
    "variables": ["project_name", "project_path"],
    "estimated_total_duration": 300.0,
    "estimated_total_cost": 0.1
}

planner.add_custom_template("my_template", custom_template)
```

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names
- Add docstrings for all public methods

### Naming Conventions

```python
# Classes: PascalCase
class TemplatePlanner:
    pass

# Functions and variables: snake_case
def create_plan():
    plan_id = generate_id()

# Constants: UPPER_SNAKE_CASE
MAX_PLAN_STEPS = 20

# Private methods: _underscore_prefix
def _internal_method(self):
    pass
```

### Documentation Style

Use Google-style docstrings:

```python
def execute_plan(self, plan: Plan) -> ExecutionResult:
    """Execute a plan and return the result.
    
    Args:
        plan: The plan to execute containing steps and metadata.
        
    Returns:
        ExecutionResult containing success status, artifacts, and metrics.
        
    Raises:
        ExecutorError: If execution fails due to tool or system errors.
        ValidationError: If plan validation fails.
        
    Example:
        >>> executor = ConcreteExecutor(config)
        >>> result = await executor.execute_plan(plan)
        >>> if result.success:
        ...     print("Execution completed successfully")
    """
```

## Testing Guidelines

### Test Structure

```python
class TestMyComponent:
    """Test class for MyComponent."""
    
    @pytest.fixture
    def component_config(self):
        """Create component configuration for testing."""
        return ComponentConfig(param1="value1")
    
    @pytest.fixture
    async def component(self, component_config):
        """Create and initialize component instance."""
        comp = MyComponent(component_config)
        await comp.initialize()
        return comp
    
    @pytest.mark.asyncio
    async def test_basic_functionality(self, component):
        """Test basic component functionality."""
        result = await component.do_something()
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, component):
        """Test component error handling."""
        with pytest.raises(ComponentError):
            await component.do_invalid_operation()
```

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics

### Mocking Strategy

```python
# Mock external dependencies
@patch('src.tools.docker.docker.from_env')
async def test_docker_tool(self, mock_docker):
    mock_client = AsyncMock()
    mock_docker.return_value = mock_client
    
    tool = DockerTool(config)
    await tool.initialize()
    
    result = await tool.execute("list_containers", {})
    assert result.success is True
```

## Debugging

### Logging

```python
import logging

# Configure logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Use in components
logger = logging.getLogger(__name__)
logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### Debug Tools

```python
# Use pdb for debugging
import pdb; pdb.set_trace()

# Use rich for better output
from rich.console import Console
from rich.pretty import pprint

console = Console()
console.print("Debug output", style="bold red")
pprint(complex_object)
```

### Performance Profiling

```python
# Profile async code
import asyncio
import cProfile

async def profile_execution():
    # Your async code here
    pass

# Run with profiling
cProfile.run('asyncio.run(profile_execution())')
```

## Release Process

### Version Management

1. Update version in `pyproject.toml`
2. Update version in `docs/source/conf.py`
3. Update `CHANGELOG.md`
4. Create release tag

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version numbers are updated
- [ ] Changelog is updated
- [ ] Release notes are prepared
- [ ] Security review completed

### Automated Checks

```bash
# Pre-commit checks
make pre-commit

# Full test suite
make test-all

# Documentation build
make docs

# Security scan
make security-check
```
