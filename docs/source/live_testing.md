# Live Integration Testing

This document explains how to run live integration tests that use real services instead of mocks.

## Overview

Orcastrate includes comprehensive live integration tests that run against real services:

- **LocalStack** - AWS services simulation
- **Docker** - Real Docker daemon
- **Databases** - PostgreSQL, MySQL, MongoDB, Redis
- **Storage** - MinIO (S3-compatible)
- **Search** - Elasticsearch

These tests verify that our tools work correctly with actual APIs and services.

## Prerequisites

### Required Software

- Docker Desktop or Docker Engine
- Docker Compose V2
- Python 3.8+ with Poetry
- At least 4GB available RAM
- 10GB available disk space

### System Requirements

- macOS, Linux, or Windows with WSL2
- Docker daemon running and accessible
- Ports 4566, 5432, 3306, 6379, 27017, 9000, 9200 available

## Quick Start

### 1. Start Test Infrastructure

```bash
# Start all test services
docker compose -f docker-compose.test.yml up -d --wait

# Verify services are healthy
docker compose -f docker-compose.test.yml ps
```

### 2. Run Live Tests

```bash
# Run all live tests
python scripts/run_live_tests.py

# Run specific test categories
pytest tests/live/ --live -v

# Run only AWS tests
pytest tests/live/test_aws_live.py --live -v

# Run only Docker tests  
pytest tests/live/test_docker_live.py --live -v
```

### 3. Cleanup

```bash
# Stop and remove all test services
docker compose -f docker-compose.test.yml down -v --remove-orphans
```

## Test Infrastructure

### Services Overview

| Service | Port | Purpose | Health Check |
|---------|------|---------|--------------|
| LocalStack | 4566 | AWS services simulation | `curl http://localhost:4566/_localstack/health` |
| PostgreSQL | 5432 | Relational database testing | `pg_isready -h localhost` |
| MySQL | 3306 | Alternative SQL database | `mysqladmin ping -h localhost` |
| Redis | 6379 | Cache and pub/sub testing | `redis-cli ping` |
| MongoDB | 27017 | NoSQL database testing | `mongosh --eval "db.runCommand('ping')"` |
| MinIO | 9000 | S3-compatible storage | `curl http://localhost:9000/minio/health/live` |
| Elasticsearch | 9200 | Search engine testing | `curl http://localhost:9200/_cluster/health` |

### Container Networking

All test services run in the `orcastrate-test-network` Docker network and can communicate with each other using service names as hostnames.

## Running Tests

### Using the Helper Script

The `scripts/run_live_tests.py` script provides convenient test execution with infrastructure management:

```bash
# Run all tests with automatic infrastructure management
python scripts/run_live_tests.py

# Skip infrastructure startup (if already running)
python scripts/run_live_tests.py --no-infrastructure

# Keep infrastructure running after tests
python scripts/run_live_tests.py --keep-infrastructure

# Run specific test patterns
python scripts/run_live_tests.py --test-filter "aws and ec2"

# Run specific test file
python scripts/run_live_tests.py --test-file test_aws_live.py
```

### Direct Pytest Execution

You can also run tests directly with pytest:

```bash
# Install test dependencies
poetry install --with dev

# Set environment variables
export LOCALSTACK_ENDPOINT="http://localhost:4566"
export POSTGRES_HOST="localhost"
export MYSQL_HOST="localhost"
export REDIS_HOST="localhost"
export MONGODB_HOST="localhost"

# Run tests
pytest tests/live/ --live -v
```

### Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only LocalStack tests
pytest -m "localstack" tests/live/ --live -v

# Run only Docker tests
pytest -m "docker_required" tests/live/ --live -v

# Run integration tests but exclude live tests
pytest -m "integration and not live" tests/ -v

# Run all tests including live tests
pytest --live -v
```

## Test Categories

### AWS Live Tests (`test_aws_live.py`)

Tests the AWS Cloud Provider tool against LocalStack:

- **Account Management**: Get account info, identity verification
- **EC2 Lifecycle**: Create, start, stop, terminate instances
- **Security Groups**: Create groups, authorize ingress rules
- **IAM Management**: Create roles, attach/detach policies
- **Lambda Functions**: Create, invoke, delete functions
- **RDS Databases**: Create, list, delete database instances
- **Cost Estimation**: Multi-resource cost calculations
- **End-to-End Workflows**: Complete application deployments

Example test execution:

```bash
pytest tests/live/test_aws_live.py::TestAWSLiveIntegration::test_ec2_instance_lifecycle_live --live -v
```

### Docker Live Tests (`test_docker_live.py`)

Tests the Docker tool against a real Docker daemon:

- **Container Lifecycle**: Create, start, stop, remove containers
- **Image Management**: Pull, build, inspect, remove images
- **Network Management**: Create networks, container connectivity
- **Volume Management**: Create volumes, data persistence
- **Multi-Container Apps**: Simulate docker-compose deployments
- **Dockerfile Builds**: Build custom images from Dockerfiles

Example test execution:

```bash
pytest tests/live/test_docker_live.py::TestDockerLiveIntegration::test_container_lifecycle_live --live -v
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALSTACK_ENDPOINT` | `http://localhost:4566` | LocalStack API endpoint |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `MYSQL_HOST` | `localhost` | MySQL host |
| `REDIS_HOST` | `localhost` | Redis host |
| `MONGODB_HOST` | `localhost` | MongoDB host |
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO S3 endpoint |
| `ELASTICSEARCH_HOST` | `localhost` | Elasticsearch host |

### Docker Compose Override

Create `docker-compose.test.override.yml` to customize the test environment:

```yaml
version: '3.8'
services:
  localstack:
    environment:
      - DEBUG=1
      - SERVICES=ec2,rds,lambda,iam,s3,sts,cloudformation
    ports:
      - "4566:4566"
      - "8080:8080"  # LocalStack dashboard
```

## Troubleshooting

### Common Issues

#### Port Conflicts

```bash
# Check what's using a port
lsof -i :4566

# Kill processes using the port
sudo lsof -ti :4566 | xargs kill -9
```

#### Docker Issues

```bash
# Check Docker daemon status
docker info

# Restart Docker Desktop
# On macOS: restart Docker Desktop app
# On Linux: sudo systemctl restart docker

# Clean up Docker resources
docker system prune -a --volumes
```

#### LocalStack Issues

```bash
# Check LocalStack health
curl http://localhost:4566/_localstack/health

# View LocalStack logs
docker compose -f docker-compose.test.yml logs localstack

# Reset LocalStack data
docker compose -f docker-compose.test.yml down -v
docker compose -f docker-compose.test.yml up -d localstack
```

#### Memory Issues

```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Increase Docker memory limit in Docker Desktop settings
# Recommended: 4GB minimum, 8GB preferred
```

### Test Isolation

Each test should clean up its resources, but if tests fail:

```bash
# Clean up Docker resources
docker container prune -f
docker image prune -f
docker network prune -f
docker volume prune -f

# Restart test infrastructure
docker compose -f docker-compose.test.yml down -v
docker compose -f docker-compose.test.yml up -d --wait
```

### Debugging Failed Tests

```bash
# Run single test with detailed output
pytest tests/live/test_aws_live.py::test_name --live -vvv --tb=long

# Run with pdb debugger
pytest tests/live/test_aws_live.py::test_name --live --pdb

# Capture test output
pytest tests/live/ --live -v --capture=no
```

## CI/CD Integration

### GitHub Actions

Add live tests to your CI pipeline:

```yaml
name: Live Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  live-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Start test infrastructure
        run: |
          docker compose -f docker-compose.test.yml up -d --wait
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with dev
      
      - name: Run live tests
        run: |
          poetry run pytest tests/live/ --live -v
      
      - name: Cleanup
        if: always()
        run: |
          docker compose -f docker-compose.test.yml down -v --remove-orphans
```

### Performance Considerations

- Live tests are slower than unit tests (2-5 minutes vs seconds)
- Run live tests less frequently (e.g., on merge to main)
- Use parallel test execution carefully to avoid resource conflicts
- Consider running different test categories on different CI jobs

## Best Practices

### Writing Live Tests

1. **Resource Cleanup**: Always clean up resources in test teardown
2. **Unique Names**: Use `generate_unique_name()` for test resources
3. **Timeouts**: Set appropriate timeouts for service operations
4. **Error Handling**: Test both success and failure scenarios
5. **Isolation**: Tests should not depend on each other
6. **Documentation**: Clearly document what each test verifies

### Test Organization

```python
@pytest.mark.live
@pytest.mark.localstack
class TestAWSLiveIntegration:
    """Live integration tests for AWS Cloud Provider tool."""
    
    @pytest.mark.asyncio
    async def test_feature_workflow(self, aws_live_tool):
        """Test complete workflow for a specific feature."""
        # Setup
        resource_name = generate_unique_name("test-resource")
        
        try:
            # Test operations
            result = await aws_live_tool.execute("action", {...})
            assert result.success
            
        finally:
            # Cleanup
            await cleanup_resources()
```

### Monitoring Test Performance

Track test execution times and resource usage:

```bash
# Run with timing
pytest tests/live/ --live -v --durations=10

# Profile memory usage
pytest tests/live/ --live --profile

# Generate test report
pytest tests/live/ --live --html=report.html
```

## Security Considerations

- Live tests use test credentials and isolated environments
- LocalStack simulates AWS but doesn't charge real money
- Test data is ephemeral and cleaned up automatically
- Use strong passwords for test databases
- Ensure test environments are isolated from production

## Contributing

When adding new live tests:

1. Follow the existing test patterns and naming conventions
2. Add appropriate pytest markers (`@pytest.mark.live`, etc.)
3. Include comprehensive cleanup in `finally` blocks
4. Test both success and error scenarios
5. Update this documentation if adding new services
6. Ensure tests pass in CI environment

For questions or issues with live testing, please create an issue in the repository.
