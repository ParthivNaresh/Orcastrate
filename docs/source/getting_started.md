# Getting Started

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for version control operations)
- Docker (optional, for containerization features)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## First Steps

### 1. Explore Available Templates

```bash
python -m src.cli.main templates
```

This will show you the available project templates:

- Node.js Web Application
- Python FastAPI Application

### 2. Create Your First Environment

```bash
python -m src.cli.main create "Node.js web application with Express"
```

### 3. Check Tool Status

```bash
python -m src.cli.main tools
```

This shows which tools are available and their status.

### 4. Run the Interactive Demo

```bash
python demo.py
```

The demo showcases the complete workflow from natural language requirements to running applications.

## Basic Concepts

### Requirements

Natural language descriptions of what you want to build:

- "Create a Node.js web application"
- "Build a FastAPI REST API service"
- "Set up a React frontend with TypeScript"

### Templates

Pre-built patterns for common development scenarios:

- Technology stack configuration
- Step-by-step execution plans
- Dependency resolution
- Cost estimation

### Tools

Concrete implementations for specific operations:

- **Filesystem**: File and directory operations
- **Git**: Version control management
- **Docker**: Containerization and deployment

### Plans

Generated execution plans containing:

- Sequential or parallel steps
- Tool assignments and parameters
- Dependency resolution
- Risk assessment and cost estimation

## Configuration

Orcastrate uses configuration files and environment variables:

```python
# Planner Configuration
planner_config = PlannerConfig(
    strategy=PlanningStrategy.TEMPLATE_MATCHING,
    max_plan_steps=20,
    cost_optimization=True
)

# Executor Configuration  
executor_config = ExecutorConfig(
    strategy=ExecutionStrategy.SEQUENTIAL,
    max_concurrent_steps=5,
    retry_policy={"max_retries": 3}
)
```

## Next Steps

- Read the [User Guide](user_guide.md) for detailed usage information
- Check out [Examples](examples.md) for common use cases
- Explore the [CLI Reference](cli_reference.md) for all available commands
- Learn about the [Architecture](architecture.md) for customization
