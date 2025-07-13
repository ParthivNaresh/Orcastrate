# User Guide

## Overview

Orcastrate transforms natural language descriptions into fully functional development environments through intelligent planning and automated execution.

## Core Workflow

### 1. Requirements Processing

Convert natural language into structured requirements:

```python
requirements = Requirements(
    description="Create a Node.js web application with Express",
    framework="nodejs",
    database="postgresql",
    cloud_provider="aws"
)
```

### 2. Plan Generation

Generate detailed execution plans:

```python
planner = TemplatePlanner(config)
plan = await planner.create_plan(requirements)
```

### 3. Plan Validation

Validate plan requirements and dependencies:

```python
executor = ConcreteExecutor(config)
validation = await executor.validate_plan_requirements(plan)
```

### 4. Execution

Execute the plan with real tools:

```python
result = await executor.execute_plan(plan)
```

## Command Line Interface

### Basic Commands

#### List Templates

```bash
python -m src.cli.main templates
```

#### Create Environment

```bash
python -m src.cli.main create "description" [options]
```

Options:

- `--framework`: Specify technology framework
- `--database`: Database requirement
- `--cloud-provider`: Cloud deployment target
- `--output`: Output directory
- `--dry-run`: Show plan without execution

#### Check Tools

```bash
python -m src.cli.main tools
```

#### View Logs

```bash
python -m src.cli.main logs [--lines N]
```

#### Version Information

```bash
python -m src.cli.main version
```

## Templates

### Node.js Web Application

Creates a complete Node.js web application with:

- Express.js framework
- Package.json configuration
- Git repository initialization
- Dockerfile for containerization
- Development server setup

**Example:**

```bash
python -m src.cli.main create "Node.js web app with Express and MongoDB"
```

### Python FastAPI Application

Creates a FastAPI REST API with:

- FastAPI framework setup
- Requirements.txt configuration
- Git repository initialization
- Dockerfile for containerization
- Health check endpoints

**Example:**

```bash
python -m src.cli.main create "FastAPI REST API with PostgreSQL"
```

## Tools

### Filesystem Tool

Handles file and directory operations with security validation:

**Actions:**

- `create_directory`: Create directories with permissions
- `write_file`: Write files with JSON or text content
- `read_file`: Read files with size limits
- `copy_file`: Copy files preserving permissions
- `move_file`: Move files and directories
- `delete_file`: Remove files and directories
- `list_directory`: List directory contents
- `get_info`: Get file/directory metadata
- `set_permissions`: Modify file permissions

**Security Features:**

- Path traversal prevention
- Permission validation
- Size limit enforcement
- Allowed pattern matching

### Git Tool

Provides version control operations:

**Actions:**

- `init`: Initialize repository
- `clone`: Clone remote repository
- `add`: Stage files for commit
- `commit`: Create commits
- `branch`: Branch operations
- `merge`: Merge branches
- `pull`: Pull remote changes
- `push`: Push local changes
- `status`: Check repository status
- `log`: View commit history
- `tag`: Manage tags
- `remote`: Manage remotes
- `diff`: View differences

### Docker Tool

Handles containerization and deployment:

**Actions:**

- `build_image`: Build Docker images
- `run_container`: Run containers
- `stop_container`: Stop running containers
- `list_containers`: List containers
- `list_images`: List available images
- `remove_container`: Remove containers
- `remove_image`: Remove images
- `pull_image`: Pull images from registry
- `push_image`: Push images to registry

## Configuration

### Planner Configuration

```python
from src.planners.base import PlannerConfig, PlanningStrategy

config = PlannerConfig(
    strategy=PlanningStrategy.TEMPLATE_MATCHING,
    max_plan_steps=20,
    max_planning_time=60,
    cost_optimization=True,
    risk_threshold=0.8
)
```

### Executor Configuration

```python
from src.executors.base import ExecutorConfig, ExecutionStrategy

config = ExecutorConfig(
    strategy=ExecutionStrategy.SEQUENTIAL,
    max_concurrent_steps=5,
    step_timeout=300,
    retry_policy={
        "max_retries": 3,
        "backoff_factor": 2.0,
        "max_delay": 60
    },
    enable_rollback=True
)
```

### Tool Configuration

```python
from src.tools.base import ToolConfig

config = ToolConfig(
    name="filesystem",
    version="1.0.0",
    timeout=300,
    retry_count=3
)
```

## Error Handling

### Plan Validation Errors

- Missing required tools
- Invalid action parameters
- Circular dependencies
- Resource constraints

### Execution Errors

- Tool initialization failures
- Step execution timeouts
- Resource unavailability
- Permission denied

### Recovery Mechanisms

- Automatic retry with backoff
- Rollback to previous state
- Graceful degradation
- Detailed error reporting

## Best Practices

### Requirements Writing

- Be specific about technology choices
- Include deployment preferences
- Specify database requirements
- Mention security considerations

### Template Selection

- Use framework-specific templates
- Consider deployment targets
- Factor in team expertise
- Plan for scalability

### Tool Management

- Verify tool availability
- Configure appropriate timeouts
- Set reasonable retry policies
- Monitor resource usage

### Security

- Use path validation
- Implement permission checks
- Validate input parameters
- Monitor file operations
