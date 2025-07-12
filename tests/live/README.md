# Live Integration Tests

This directory contains live integration tests that run against real services instead of mocks.

## Quick Start

```bash
# Start test infrastructure
docker compose -f docker-compose.test.yml up -d --wait

# Run all live tests
python scripts/run_live_tests.py

# Or run directly with pytest
pytest tests/live/ --live -v

# Cleanup
docker compose -f docker-compose.test.yml down -v
```

## Test Categories

- **`test_aws_live.py`** - AWS Cloud Provider tests using LocalStack
- **`test_docker_live.py`** - Docker tool tests using real Docker daemon

## Services Used

- **LocalStack** (port 4566) - AWS services simulation
- **PostgreSQL** (port 5432) - Database testing
- **MySQL** (port 3306) - Alternative database
- **Redis** (port 6379) - Cache testing
- **MongoDB** (port 27017) - NoSQL testing
- **MinIO** (port 9000) - S3-compatible storage
- **Elasticsearch** (port 9200) - Search engine

## Requirements

- Docker Desktop or Docker Engine
- Docker Compose V2
- 4GB+ available RAM
- 10GB+ available disk space

## Documentation

See [docs/source/live_testing.md](../../docs/source/live_testing.md) for complete documentation.

## Notes

- Live tests are marked with `@pytest.mark.live`
- Tests are skipped by default unless `--live` flag is used
- Each test cleans up its resources automatically
- Tests use unique names to avoid conflicts
