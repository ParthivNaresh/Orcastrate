# CLI Reference

## Synopsis

```bash
python -m src.cli.main [OPTIONS] COMMAND [ARGS]...
```

## Global Options

- `--verbose, -v`: Enable verbose logging
- `--help`: Show help message and exit

## Commands

### create

Create a development environment from natural language description.

```bash
python -m src.cli.main create [OPTIONS] DESCRIPTION
```

**Arguments:**

- `DESCRIPTION`: Natural language description of the environment to create

**Options:**

- `--framework TEXT`: Technology framework (nodejs, fastapi, react, etc.)
- `--database TEXT`: Database requirement (postgresql, mongodb, mysql, etc.)
- `--cloud-provider TEXT`: Cloud deployment target (aws, gcp, azure, etc.)
- `--output PATH`: Output directory for the generated environment
- `--dry-run`: Show execution plan without executing it

**Examples:**

```bash
# Basic Node.js application
python -m src.cli.main create "Node.js web application"

# FastAPI with PostgreSQL
python -m src.cli.main create "REST API service" --framework fastapi --database postgresql

# Dry run to see the plan
python -m src.cli.main create "React frontend" --dry-run

# Custom output directory
python -m src.cli.main create "Express server" --output ~/projects/my-app
```

### templates

List available environment templates with details.

```bash
python -m src.cli.main templates
```

**Output includes:**

- Template name and description
- Supported framework
- Estimated duration
- Estimated cost
- Required tools

**Example Output:**

```
📚 Available Templates:

🏗️  Node.js Web Application
   Description: Node.js web application with npm dependencies
   Framework: nodejs
   Duration: ~3.5 minutes
   Cost: $0.15

🏗️  Python FastAPI Application
   Description: FastAPI web application with Python
   Framework: fastapi
   Duration: ~4.7 minutes
   Cost: $0.20
```

### tools

Show status of available tools and their capabilities.

```bash
python -m src.cli.main tools
```

**Output includes:**

- Tool name and status
- Version information
- Supported actions
- Error details (if unavailable)

**Example Output:**

```
🔧 Available Tools:

✅ GIT
   📝 Description: Git version control system tool
   🔧 Version: 1.0.0
   ⚡ Actions: init, clone, status, add, commit, branch, merge

❌ DOCKER: Docker is not available or not running
```

### logs

Display recent Orcastrate logs.

```bash
python -m src.cli.main logs [OPTIONS]
```

**Options:**

- `--lines INTEGER`: Number of log lines to display (default: 50)

**Examples:**

```bash
# Show last 50 lines
python -m src.cli.main logs

# Show last 100 lines
python -m src.cli.main logs --lines 100
```

### version

Show version information and current phase.

```bash
python -m src.cli.main version
```

**Output:**

```
Orcastrate Development Environment Agent
Version: 1.0.0
Phase: 2 - Concrete Implementations
```

## Environment Variables

### ORCASTRATE_LOG_LEVEL

Set logging level for Orcastrate operations.

**Values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`
**Default:** `INFO`

```bash
export ORCASTRATE_LOG_LEVEL=DEBUG
python -m src.cli.main create "Node.js app"
```

### ORCASTRATE_CONFIG_PATH

Path to custom configuration file.

**Default:** `~/.orcastrate/config.yaml`

```bash
export ORCASTRATE_CONFIG_PATH=/path/to/custom/config.yaml
```

### ORCASTRATE_WORK_DIR

Default working directory for generated environments.

**Default:** `/tmp/orcastrate/`

```bash
export ORCASTRATE_WORK_DIR=$HOME/orcastrate-projects
```

### ORCASTRATE_CACHE_DIR

Directory for caching templates and metadata.

**Default:** `~/.orcastrate/cache/`

```bash
export ORCASTRATE_CACHE_DIR=/var/cache/orcastrate
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid command line arguments
- `3`: Configuration error
- `4`: Tool initialization failure
- `5`: Plan generation failure
- `6`: Plan validation failure
- `7`: Execution failure

## Configuration File

Orcastrate can be configured using YAML files:

```yaml
# ~/.orcastrate/config.yaml
planner:
  strategy: template_matching
  max_plan_steps: 20
  cost_optimization: true
  risk_threshold: 0.8

executor:
  strategy: sequential
  max_concurrent_steps: 5
  step_timeout: 300
  retry_policy:
    max_retries: 3
    backoff_factor: 2.0
    max_delay: 60
  enable_rollback: true

tools:
  filesystem:
    base_path: /tmp/orcastrate
    max_file_size: 10485760  # 10MB
  
  git:
    default_branch: main
    auto_commit: true
  
  docker:
    default_registry: docker.io
    build_timeout: 600
```

## Troubleshooting

### Tool Not Available

If tools show as unavailable:

1. **Docker**: Ensure Docker is installed and running

   ```bash
   docker --version
   sudo systemctl start docker  # Linux
   ```

2. **Git**: Install Git if missing

   ```bash
   git --version
   # Install via package manager if needed
   ```

### Permission Errors

For filesystem permission errors:

```bash
# Ensure proper permissions for work directory
chmod 755 /tmp/orcastrate/
chown $USER:$USER /tmp/orcastrate/
```

### Log Analysis

Enable verbose logging for troubleshooting:

```bash
python -m src.cli.main --verbose create "description"
```

Check logs:

```bash
python -m src.cli.main logs --lines 200
```

### Common Issues

1. **"No suitable template found"**
   - Use more specific framework descriptions
   - Check available templates with `templates` command

2. **"Plan validation failed"**
   - Verify required tools are available
   - Check tool status with `tools` command

3. **"Execution timeout"**
   - Increase timeout in configuration
   - Check system resources
