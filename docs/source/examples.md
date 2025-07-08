# Examples

## Basic Usage Examples

### Creating a Node.js Web Application

```bash
# Simple Node.js app
python -m src.cli.main create "Node.js web application"

# Node.js with specific features
python -m src.cli.main create "Node.js web app with Express and MongoDB" \
    --framework nodejs \
    --database mongodb
```

### Creating a Python FastAPI Service

```bash
# Basic FastAPI service
python -m src.cli.main create "FastAPI REST API"

# FastAPI with PostgreSQL
python -m src.cli.main create "FastAPI service with PostgreSQL database" \
    --framework fastapi \
    --database postgresql
```

### Using the Programmatic API

```python
import asyncio
from src.agent.base import Requirements
from src.planners.template_planner import TemplatePlanner
from src.planners.base import PlannerConfig, PlanningStrategy
from src.executors.concrete_executor import ConcreteExecutor
from src.executors.base import ExecutorConfig, ExecutionStrategy

async def create_environment():
    # Define requirements
    requirements = Requirements(
        description="Create a Node.js web application with Express",
        framework="nodejs"
    )
    
    # Initialize planner
    planner_config = PlannerConfig(
        strategy=PlanningStrategy.TEMPLATE_MATCHING,
        max_plan_steps=20,
        cost_optimization=True
    )
    planner = TemplatePlanner(planner_config)
    await planner.initialize()
    
    # Generate plan
    plan = await planner.create_plan(requirements)
    print(f"Generated plan with {len(plan.steps)} steps")
    
    # Initialize executor
    executor_config = ExecutorConfig(
        strategy=ExecutionStrategy.SEQUENTIAL,
        max_concurrent_steps=5
    )
    executor = ConcreteExecutor(executor_config)
    await executor.initialize()
    
    # Execute plan
    result = await executor.execute_plan(plan)
    if result.success:
        print("Environment created successfully!")
        print(f"Artifacts: {result.artifacts}")
    else:
        print(f"Execution failed: {result.error}")

# Run the example
asyncio.run(create_environment())
```

## Advanced Examples

### Custom Template Creation

```python
from src.planners.template_planner import TemplatePlanner

async def add_custom_template():
    planner = TemplatePlanner(config)
    await planner.initialize()
    
    # Define custom React template
    react_template = {
        "name": "React Application",
        "description": "React frontend application with TypeScript",
        "framework": "react",
        "steps": [
            {
                "id": "setup_directory",
                "name": "Setup Project Directory",
                "description": "Create project directory",
                "tool": "filesystem",
                "action": "create_directory",
                "parameters": {
                    "path": "{project_path}",
                    "mode": "755"
                }
            },
            {
                "id": "create_package_json",
                "name": "Create package.json",
                "description": "Initialize React project",
                "tool": "filesystem",
                "action": "write_file",
                "parameters": {
                    "path": "{project_path}/package.json",
                    "content": {
                        "name": "{project_name}",
                        "version": "1.0.0",
                        "dependencies": {
                            "react": "^18.2.0",
                            "react-dom": "^18.2.0",
                            "typescript": "^4.9.0"
                        }
                    }
                }
            }
        ],
        "variables": ["project_name", "project_path"],
        "estimated_total_duration": 180.0,
        "estimated_total_cost": 0.10
    }
    
    # Add the template
    planner.add_custom_template("react_app", react_template)
    
    # Now it can be used
    requirements = Requirements(
        description="React frontend application",
        framework="react"
    )
    plan = await planner.create_plan(requirements)
```

### Tool Integration Example

```python
from src.tools.filesystem import FileSystemTool
from src.tools.git import GitTool
from src.tools.base import ToolConfig

async def tool_integration_example():
    # Initialize tools
    fs_config = ToolConfig(name="filesystem", version="1.0.0")
    fs_tool = FileSystemTool(fs_config)
    await fs_tool.initialize()
    
    git_config = ToolConfig(name="git", version="1.0.0")
    git_tool = GitTool(git_config)
    await git_tool.initialize()
    
    # Create project structure
    project_path = "/tmp/orcastrate/my-project"
    
    # Create directory
    result = await fs_tool.execute("create_directory", {
        "path": project_path,
        "mode": "755"
    })
    
    # Initialize Git repository
    result = await git_tool.execute("init", {
        "path": project_path,
        "initial_branch": "main"
    })
    
    # Create initial file
    result = await fs_tool.execute("write_file", {
        "path": f"{project_path}/README.md",
        "content": "# My Project\n\nCreated with Orcastrate!"
    })
    
    # Add and commit
    await git_tool.execute("add", {
        "path": project_path,
        "files": ["."]
    })
    
    await git_tool.execute("commit", {
        "path": project_path,
        "message": "Initial commit",
        "author_name": "Orcastrate",
        "author_email": "orcastrate@example.com"
    })
    
    print("Project created and initialized with Git!")
```

### Error Handling Example

```python
from src.executors.base import ExecutorError
from src.tools.base import ToolError

async def error_handling_example():
    try:
        # Attempt to create environment
        result = await executor.execute_plan(plan)
        
        if not result.success:
            print(f"Execution failed: {result.error}")
            
            # Check for specific error types
            if "Docker" in result.error:
                print("Docker is not available. Please install Docker.")
            elif "permission" in result.error.lower():
                print("Permission denied. Check file permissions.")
            else:
                print("Unknown error occurred.")
                
    except ExecutorError as e:
        print(f"Executor error: {e}")
        # Handle executor-specific errors
        
    except ToolError as e:
        print(f"Tool error: {e}")
        # Handle tool-specific errors
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle unexpected errors
```

### Configuration Example

```python
from src.planners.base import PlannerConfig, PlanningStrategy
from src.executors.base import ExecutorConfig, ExecutionStrategy

# Custom configuration
planner_config = PlannerConfig(
    strategy=PlanningStrategy.TEMPLATE_MATCHING,
    max_plan_steps=25,
    max_planning_time=120,
    cost_optimization=True,
    risk_threshold=0.7
)

executor_config = ExecutorConfig(
    strategy=ExecutionStrategy.PARALLEL,
    max_concurrent_steps=3,
    step_timeout=600,
    retry_policy={
        "max_retries": 5,
        "backoff_factor": 1.5,
        "max_delay": 120
    },
    enable_rollback=True
)

# Initialize with custom config
planner = TemplatePlanner(planner_config)
executor = ConcreteExecutor(executor_config)
```

## Real-World Scenarios

### Microservices Architecture

```bash
# Create multiple services
python -m src.cli.main create "User authentication service" \
    --framework fastapi \
    --database postgresql \
    --output ./services/auth

python -m src.cli.main create "Product catalog service" \
    --framework nodejs \
    --database mongodb \
    --output ./services/catalog

python -m src.cli.main create "Order processing service" \
    --framework fastapi \
    --database postgresql \
    --output ./services/orders
```

### Full-Stack Application

```bash
# Backend API
python -m src.cli.main create "REST API backend with authentication" \
    --framework fastapi \
    --database postgresql \
    --output ./backend

# Frontend
python -m src.cli.main create "React frontend with TypeScript" \
    --framework react \
    --output ./frontend
```

### Development Environment Setup

```bash
# Complete development setup
python -m src.cli.main create "Development environment with Docker, Git, and testing setup" \
    --framework nodejs \
    --database mongodb \
    --cloud-provider aws \
    --output ./my-project
```

## Troubleshooting Examples

### Debug Mode

```bash
# Enable verbose logging
python -m src.cli.main --verbose create "Node.js app"

# Check logs
python -m src.cli.main logs --lines 100
```

### Dry Run Testing

```bash
# Test plan generation without execution
python -m src.cli.main create "complex application" --dry-run
```

### Tool Diagnostics

```bash
# Check tool availability
python -m src.cli.main tools

# Test specific operations
python -c "
import asyncio
from src.tools.git import GitTool
from src.tools.base import ToolConfig

async def test_git():
    config = ToolConfig(name='git', version='1.0.0')
    tool = GitTool(config)
    await tool.initialize()
    schema = await tool.get_schema()
    print(f'Git tool available: {schema.name}')

asyncio.run(test_git())
"
```
