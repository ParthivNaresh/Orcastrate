# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          ORCASTRATE DEVELOPMENT TASKS                       ║
# ║                                                                              ║
# ║  Production-grade development environment orchestration system              ║
# ║  Run `just --list` to see all available commands                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Show all available commands by default
default:
    @just --list

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                         🌍  ENVIRONMENT MANAGEMENT                          │
# └──────────────────────────────────────────────────────────────────────────────┘

# Load environment from specific file and export variables
load-env file="local":
    #!/usr/bin/env bash
    if [ -f "scripts/env/{{file}}.env" ]; then
        set -o allexport
        source scripts/env/{{file}}.env
        set +o allexport
        echo "✅ Loaded environment: {{file}}"
    else
        echo "❌ Environment file not found: scripts/env/{{file}}.env"
        echo "📋 Available environments:"
        ls scripts/env/*.env 2>/dev/null | sed 's/scripts\/env\///g' | sed 's/\.env//g' | sed 's/^/  - /'
        exit 1
    fi

# Show current configuration (with sensitive values masked)
show-config env="local":
    @echo "📋 Configuration for environment: {{env}}"
    @echo "═══════════════════════════════════════════"
    @just load-env {{env}} > /dev/null
    @python scripts/show_config.py

# Validate environment configuration and dependencies
validate-env env="local":
    @echo "🔍 Validating environment: {{env}}"
    @echo "════════════════════════════════════"
    @just load-env {{env}} > /dev/null
    @echo "├── 🔧 Checking configuration validity..."
    @python scripts/validate_config.py
    @echo "└── ✅ Environment validation completed"

# List all available environment configurations
list-envs:
    @echo "🌍 Available Environment Configurations:"
    @echo "═══════════════════════════════════════"
    @echo "📁 Local environments:"
    @ls scripts/env/*.env 2>/dev/null | sed 's/scripts\/env\///g' | sed 's/\.env//g' | sed 's/^/  🔧 /' || echo "  No local environments found"
    @echo ""
    @echo "📁 Docker environments:"
    @ls docker/*.env 2>/dev/null | sed 's/docker\///g' | sed 's/\.env\.//g' | sed 's/^/  🐳 /' || echo "  No docker environments found"
    @echo ""
    @echo "📋 Usage examples:"
    @echo "  just load-env local        # Load local development"
    @echo "  just validate-env testing  # Validate test environment"
    @echo "  just show-config staging   # Show staging config"

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                            🏗️  PROJECT SETUP                                │
# └──────────────────────────────────────────────────────────────────────────────┘

# Install all dependencies for development (including dev dependencies)
install:
    @echo "📦 Installing project dependencies..."
    @echo "├── 🔒 Updating poetry lock file..."
    poetry lock
    @echo "├── 📚 Installing dependencies..."
    poetry install --with dev
    @echo "✅ Dependencies installed successfully"

# Upgrade all dependencies to latest compatible versions
upgrade:
    @echo "⬆️  Upgrading dependencies..."
    poetry update
    @echo "✅ Dependencies upgraded successfully"

# Show project development status and environment health
status:
    @echo "🚀 Orcastrate Development Status"
    @echo "================================"
    @python --version 2>/dev/null || echo "❌ Python: Not available"
    @poetry --version 2>/dev/null || echo "❌ Poetry: Not installed"
    @docker --version 2>/dev/null || echo "⚠️  Docker: Not available (needed for live tests)"
    @echo ""
    @echo "📊 Dependencies Status:"
    @poetry check 2>/dev/null && echo "✅ Poetry dependencies OK" || echo "❌ Poetry issues detected"
    @echo ""
    @echo "🧪 Test Discovery:"
    @poetry run pytest --co -q tests/unit/ tests/integration/ 2>/dev/null | tail -1 || echo "Cannot count tests"
    @echo ""
    @echo "🐳 Live Test Dependencies:"
    @poetry run python -c "try: import testcontainers, localstack, docker; print('✅ All live test dependencies available')" 2>/dev/null || echo "❌ Missing live test dependencies"

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                           🧹  CODE QUALITY & LINTING                        │
# └──────────────────────────────────────────────────────────────────────────────┘

# Run all linting and code quality checks (used by CI/CD)
lint:
    @echo "🔍 Running comprehensive code quality checks..."
    @echo "├── 🧼 Running autoflake (remove unused imports)..."
    poetry run python -m autoflake --check --recursive --remove-all-unused-imports --remove-unused-variables --quiet src/ tests/
    @echo "├── ⚫ Running black (code formatting)..."
    poetry run python -m black --check --diff src/ tests/
    @echo "├── 📚 Running isort (import sorting)..."
    poetry run python -m isort --check-only --diff src/ tests/
    @echo "├── 🐍 Running flake8 (style checking)..."
    poetry run python -m flake8 src/ tests/
    @echo "└── 🔬 Running mypy (static type checking)..."
    poetry run python -m mypy src/ --ignore-missing-imports
    @echo "✅ All linting checks passed!"

# Auto-fix all fixable linting issues
lint-fix:
    @echo "🔧 Auto-fixing code quality issues..."
    @echo "├── 🧼 Running autoflake (removing unused imports)..."
    poetry run python -m autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables src/ tests/
    @echo "├── ⚫ Running black (formatting code)..."
    poetry run python -m black src/ tests/
    @echo "├── 📚 Running isort (sorting imports)..."
    poetry run python -m isort src/ tests/
    @echo "└── ✅ Code formatting completed!"
    @echo ""
    @echo "🔍 Re-running checks to verify fixes..."
    @just lint-verify

# Quick verification of linting status (no diffs, just pass/fail)
lint-verify:
    @echo "🔍 Verifying code quality..."
    poetry run python -m autoflake --check --recursive --remove-all-unused-imports --remove-unused-variables src/ tests/ > /dev/null
    poetry run python -m black --check src/ tests/ > /dev/null
    poetry run python -m isort --check-only src/ tests/ > /dev/null
    poetry run python -m flake8 src/ tests/ > /dev/null
    poetry run python -m mypy src/ --ignore-missing-imports > /dev/null
    @echo "✅ All linting checks passed!"

# Run security analysis tools
security:
    @echo "🔒 Running security analysis..."
    @echo "├── 🛡️  Running bandit (security linting)..."
    poetry run python -m bandit -r src/ -f json -o bandit-report.json || true
    @echo "└── 🔐 Running safety (dependency vulnerability check)..."
    poetry run python -m safety scan || true
    @echo "✅ Security analysis completed (check bandit-report.json)"

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                              🧪  TESTING SUITE                              │
# └──────────────────────────────────────────────────────────────────────────────┘

# Run all tests (unit + integration, excludes live tests)
test env="testing":
    @echo "🧪 Running unit and integration tests..."
    poetry run python -m pytest tests/unit/ tests/integration/ -v --tb=short
    @echo "✅ All tests completed!"

# Run tests with detailed coverage report
test-coverage env="testing":
    @echo "🧪 Running tests with coverage analysis..."
    poetry run python -m pytest tests/unit/ tests/integration/ \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-report=xml:coverage.xml \
        --cov-fail-under=5 \
        -v
    @echo "📊 Coverage report generated: htmlcov/index.html"

# Run only unit tests (fast feedback loop)
test-unit env="testing":
    @echo "🧪 Running unit tests only..."
    poetry run python -m pytest tests/unit/ -v --tb=short
    @echo "✅ Unit tests completed!"

# Run unit tests with detailed coverage report
test-unit-coverage env="testing":
    @echo "🧪 Running unit tests with coverage analysis..."
    poetry run python -m pytest tests/unit/ \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-report=xml:coverage.xml \
        --cov-fail-under=5 \
        -v
    @echo "📊 Unit test coverage report generated: htmlcov/index.html"

# Run only integration tests (component interactions)
test-integration env="testing":
    @echo "🧪 Running integration tests only..."
    poetry run python -m pytest tests/integration/ -v --tb=short
    @echo "✅ Integration tests completed!"

# Run integration tests with coverage
test-integration-coverage env="testing":
    @echo "🧪 Running integration tests with coverage analysis..."
    poetry run python -m pytest tests/integration/ \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov-integration \
        --cov-report=xml:coverage-integration.xml \
        --cov-fail-under=5 \
        -v
    @echo "📊 Integration test coverage report generated: htmlcov-integration/index.html"

# Run tests in watch mode for development
test-watch env="testing":
    @echo "👀 Starting test watch mode (Ctrl+C to stop)..."
    poetry run python -m pytest tests/unit/ tests/integration/ --lf -x -v --tb=short -f

# Run specific test file or pattern
test-file file:
    @echo "🧪 Running specific test: {{file}}..."
    poetry run python -m pytest {{file}} -v --tb=short

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                        🐳  DOCKER & LIVE TESTING                            │
# └──────────────────────────────────────────────────────────────────────────────┘

# Start Docker infrastructure for live testing (LocalStack + dependencies)
docker-start env="docker":
    @echo "🐳 Starting live test infrastructure..."
    @echo "├── 🚀 Starting LocalStack and dependencies..."
    docker compose -f docker-compose.test.yml --env-file docker/.env.{{env}} up -d --wait
    @echo "├── ⏳ Waiting for services to be ready..."
    @sleep 5
    @echo "└── ✅ Live test infrastructure is ready!"
    @just docker-status

# Stop Docker infrastructure and clean up
docker-stop:
    @echo "🐳 Stopping live test infrastructure..."
    docker compose -f docker-compose.test.yml down -v --remove-orphans
    @echo "✅ Infrastructure stopped and cleaned up!"

# Check status of Docker infrastructure
docker-status:
    @echo "🐳 Live Test Infrastructure Status:"
    @echo "═══════════════════════════════════"
    @docker compose -f docker-compose.test.yml ps
    @echo ""
    @echo "🔗 LocalStack Health:"
    @curl -s http://localhost:4566/_localstack/health 2>/dev/null | python -m json.tool || echo "❌ LocalStack not responding"

# Restart Docker infrastructure (stop + start)
docker-restart env="docker": docker-stop
    @just docker-start {{env}}

# Start full Docker stack for development (app + dependencies)
stack-up env="docker":
    @echo "🐳 Starting full Docker development stack..."
    @echo "├── 🚀 Starting all services..."
    docker compose --env-file docker/.env.{{env}} up -d --wait
    @echo "├── ⏳ Waiting for services to be ready..."
    @sleep 10
    @echo "└── ✅ Development stack is ready!"
    @echo ""
    @echo "📋 Service URLs:"
    @echo "  🌐 Application: http://localhost:8000"
    @echo "  🗄️  PostgreSQL: localhost:5432"
    @echo "  🍃 MongoDB: localhost:27017"
    @echo "  🔗 Redis: localhost:6379"
    @echo "  ☁️  LocalStack: http://localhost:4566"

# Stop full Docker stack
stack-down:
    @echo "🐳 Stopping Docker development stack..."
    docker compose down -v --remove-orphans
    @echo "✅ Development stack stopped!"

# View logs from Docker infrastructure
docker-logs:
    @echo "📋 Viewing live test infrastructure logs..."
    docker compose -f docker-compose.test.yml logs -f

# Run all live integration tests (requires Docker infrastructure)
test-live env="testing":
    @echo "🐳 Running live integration tests..."
    @echo "├── 🔍 Checking Docker infrastructure..."
    @just docker-status > /dev/null || (echo "❌ Docker infrastructure not running. Start with 'just docker-start'" && exit 1)
    @echo "├── 🧪 Running live tests..."
    poetry run python -m pytest tests/live/ --live -v --tb=short
    @echo "└── ✅ Live tests completed!"

# Run live tests with coverage
test-live-coverage env="testing":
    @echo "🐳 Running live tests with coverage analysis..."
    @echo "├── 🔍 Checking Docker infrastructure..."
    @just docker-status > /dev/null || (echo "❌ Docker infrastructure not running. Start with 'just docker-start'" && exit 1)
    @echo "├── 🧪 Running live tests with coverage..."
    poetry run python -m pytest tests/live/ \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov-live \
        --cov-report=xml:coverage-live.xml \
        --cov-fail-under=5 \
        --live \
        -v
    @echo "└── 📊 Live test coverage report generated: htmlcov-live/index.html"

# Run live tests for specific AWS services
test-live-aws:
    @echo "🐳 Running AWS live integration tests..."
    poetry run python -m pytest tests/live/test_aws_live.py --live -v --tb=short

# Run live tests for multi-cloud operations
test-live-multicloud:
    @echo "🐳 Running multi-cloud live integration tests..."
    poetry run python -m pytest tests/live/test_multicloud_live.py --live -v --tb=short

# Run comprehensive live test suite with infrastructure management
test-live-full:
    @echo "🐳 Running full live test suite with infrastructure management..."
    @just docker-start
    @just test-live
    @just docker-stop

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                            🚀  CI/CD COMMANDS                               │
# └──────────────────────────────────────────────────────────────────────────────┘

# Run all CI checks (matches GitHub Actions pipeline)
ci-check env="ci":
    @echo "🚀 Running complete CI/CD pipeline checks..."
    @echo "├── 🧹 Code linting and formatting..."
    @just lint
    @echo "├── 🔒 Security analysis..."
    @just security
    @echo "├── 📚 Documentation build..."
    @just docs-ci
    @echo "├── 🧪 Unit and integration tests (after quality gates)..."
    @just test-coverage {{env}}
    @echo "└── ✅ All CI checks passed!"

# Run full CI pipeline including live tests (for comprehensive validation)
ci-full env="ci":
    @echo "🚀 Running full CI/CD pipeline with live tests..."
    @echo "├── 🧹 Code linting and formatting..."
    @just lint
    @echo "├── 🔒 Security analysis..."
    @just security
    @echo "├── 📚 Documentation build..."
    @just docs-ci
    @echo "├── 🧪 Unit and integration tests (after quality gates)..."
    @just test-coverage {{env}}
    @echo "├── 🐳 Starting live test infrastructure..."
    @just docker-start docker
    @echo "├── 🧪 Running live integration tests..."
    @just test-live {{env}}
    @echo "├── 🐳 Cleaning up infrastructure..."
    @just docker-stop
    @echo "└── ✅ Full CI pipeline completed!"

# Quick pre-commit checks (fast feedback for developers)
pre-commit env="testing":
    @echo "⚡ Running quick pre-commit checks..."
    @just lint-verify
    @just test-unit {{env}}
    @echo "✅ Pre-commit checks passed!"

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                          📚  DOCUMENTATION                                  │
# └──────────────────────────────────────────────────────────────────────────────┘

# Build project documentation
docs-build:
    @echo "📚 Building project documentation..."
    poetry run sphinx-build -b html docs/source docs/build/html
    @echo "✅ Documentation built at docs/build/html/index.html"

# Build documentation with dependencies and link checking (for CI)
docs-ci:
    @echo "📚 Building documentation for CI..."
    @echo "├── 📦 Installing documentation dependencies..."
    poetry install --with dev
    poetry run pip install -r docs/requirements.txt
    @echo "├── 🏗️  Building HTML documentation..."
    poetry run sphinx-build -b html docs/source docs/build/html
    @echo "├── 🔗 Checking links in documentation..."
    poetry run sphinx-build -b linkcheck docs/source docs/build/linkcheck
    @echo "✅ Documentation CI build completed!"

# Serve documentation locally for development
docs-serve:
    @echo "📚 Serving documentation at http://localhost:8000"
    @echo "Press Ctrl+C to stop the server"
    cd docs/build/html && python -m http.server 8000

# Build and serve documentation in one command
docs: docs-build docs-serve

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                           📦  PACKAGING & RELEASE                           │
# └──────────────────────────────────────────────────────────────────────────────┘

# Build package for release
build-package:
    @echo "📦 Building package for release..."
    @echo "├── 📚 Installing dependencies..."
    poetry install --with dev
    pip install twine
    @echo "├── 🏗️  Building package..."
    poetry build
    @echo "├── ✅ Checking package..."
    twine check dist/*
    @echo "✅ Package built and validated successfully!"

# Publish package to PyPI
publish-package:
    @echo "🚀 Publishing package to PyPI..."
    twine upload dist/*
    @echo "✅ Package published successfully!"

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                         🎮  DEVELOPMENT UTILITIES                           │
# └──────────────────────────────────────────────────────────────────────────────┘

# Run the demo application for testing
demo:
    @echo "🎮 Running Orcastrate demo application..."
    poetry run python src/cli/main.py create-fastapi-app --name demo-app --database postgres

# Start development environment with hot reloading
dev env="local":
    @echo "🔥 Starting development environment..."
    @echo "Use this for active development with file watching"
    @just load-env {{env}} > /dev/null
    poetry run python -m src.cli.main

# Clean all build artifacts, caches, and temporary files
clean:
    @echo "🧹 Cleaning build artifacts and caches..."
    rm -rf dist/ build/ docs/build/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "bandit-report.json" -delete
    find . -type f -name "coverage.xml" -delete
    @echo "✅ Cleanup completed!"

# Reset environment completely (clean + reinstall)
reset: clean
    @echo "🔄 Resetting development environment..."
    poetry install --with dev
    @echo "✅ Environment reset completed!"

# Show help for common development workflows
help:
    @echo ""
    @echo "🚀 ORCASTRATE DEVELOPMENT WORKFLOWS"
    @echo "═══════════════════════════════════"
    @echo ""
    @echo "📋 GETTING STARTED:"
    @echo "  just install                    # Install dependencies"
    @echo "  just status                     # Check environment health"
    @echo "  just list-envs                  # List available environments"
    @echo ""
    @echo "🌍 ENVIRONMENT MANAGEMENT:"
    @echo "  just load-env local             # Load local development environment"
    @echo "  just validate-env testing       # Validate test environment config"
    @echo "  just show-config staging        # Show staging configuration"
    @echo ""
    @echo "⚡ QUICK DEVELOPMENT:"
    @echo "  just dev local                  # Start development with local env"
    @echo "  just pre-commit testing         # Fast pre-commit checks"
    @echo "  just test-watch testing         # Run tests in watch mode"
    @echo "  just lint-fix                   # Auto-fix code issues"
    @echo ""
    @echo "🧪 TESTING:"
    @echo "  just test testing               # Unit + integration tests"
    @echo "  just test-coverage testing      # Tests with coverage"
    @echo "  just docker-start docker        # Start live test infrastructure"
    @echo "  just test-live testing          # Run live integration tests"
    @echo ""
    @echo "🐳 DOCKER STACK:"
    @echo "  just stack-up docker            # Start full development stack"
    @echo "  just stack-down                 # Stop development stack"
    @echo ""
    @echo "🚀 CI/CD:"
    @echo "  just ci-check ci                # Full CI pipeline (no live tests)"
    @echo "  just ci-full ci                 # Complete pipeline with live tests"
    @echo ""
    @echo "🔧 UTILITIES:"
    @echo "  just clean                      # Clean build artifacts"
    @echo "  just docs                       # Build and serve documentation"
    @echo "  just --list                     # Show all available commands"
    @echo ""
