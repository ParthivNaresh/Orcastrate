#!/usr/bin/env python3
"""
Script to run live integration tests with proper infrastructure setup.

This script manages the test infrastructure lifecycle and runs live tests
against real services like LocalStack and Docker.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list, cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"🔧 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.stdout:
        print(f"📤 Output: {result.stdout}")
    if result.stderr:
        print(f"❌ Error: {result.stderr}")

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

    return result


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = run_command(["docker", "version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_docker_compose_available() -> bool:
    """Check if Docker Compose is available."""
    try:
        result = run_command(["docker", "compose", "version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def start_test_infrastructure(project_dir: Path, timeout: int = 300) -> bool:
    """Start the test infrastructure using Docker Compose."""
    print("🚀 Starting test infrastructure...")

    compose_file = project_dir / "docker-compose.test.yml"
    if not compose_file.exists():
        raise FileNotFoundError(f"Docker Compose file not found: {compose_file}")

    # Start services
    run_command([
        "docker", "compose", "-f", str(compose_file),
        "up", "-d", "--wait"
    ], cwd=str(project_dir))

    print("⏳ Waiting for services to be healthy...")

    # Wait for services to be healthy
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = run_command([
            "docker", "compose", "-f", str(compose_file), "ps", "--format", "json"
        ], cwd=str(project_dir), check=False)

        if result.returncode == 0:
            # Check if all services are healthy
            try:
                import json
                services = json.loads(result.stdout)
                if isinstance(services, dict):
                    services = [services]

                all_healthy = True
                required_services = 0
                healthy_services = 0

                for service in services:
                    state = service.get("State", "")
                    status = service.get("Status", "")
                    name = service.get("Name", "")

                    # Skip non-essential services
                    if not any(essential in name for essential in ["localstack", "postgres", "mysql", "redis", "mongodb", "test-coordinator"]):
                        continue

                    required_services += 1

                    # Check if service is running
                    if state != "running":
                        print(f"⏳ Service {name} is {state}")
                        all_healthy = False
                        continue

                    # Check health status if present
                    if "healthy" in status.lower():
                        healthy_services += 1
                    elif "unhealthy" in status.lower():
                        print(f"❌ Service {name} is unhealthy: {status}")
                        all_healthy = False
                    else:
                        # Service is running but no health info - assume ok for now
                        healthy_services += 1

                # All required services must be running and healthy
                if all_healthy and required_services > 0 and healthy_services >= required_services:
                    all_healthy = True
                else:
                    all_healthy = False

                if all_healthy:
                    print("✅ All services are healthy!")
                    return True

            except json.JSONDecodeError:
                pass

        print("⏳ Services still starting...")
        time.sleep(10)

    print("❌ Timeout waiting for services to be healthy")
    return False


def stop_test_infrastructure(project_dir: Path):
    """Stop the test infrastructure."""
    print("🛑 Stopping test infrastructure...")

    compose_file = project_dir / "docker-compose.test.yml"
    run_command([
        "docker", "compose", "-f", str(compose_file),
        "down", "-v", "--remove-orphans"
    ], cwd=str(project_dir), check=False)


def run_live_tests(project_dir: Path, test_args: list = None) -> int:
    """Run the live integration tests."""
    print("🧪 Running live integration tests...")

    if test_args is None:
        test_args = []

    # Set environment variables for tests
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(project_dir),
        "LOCALSTACK_ENDPOINT": "http://localhost:4566",
        "POSTGRES_HOST": "localhost",
        "MYSQL_HOST": "localhost",
        "REDIS_HOST": "localhost",
        "MONGODB_HOST": "localhost",
    })

    # Run pytest with live tests
    cmd = [
        "python", "-m", "pytest",
        "tests/live/",
        "-v",
        "--live",  # Enable live tests
        "-x",  # Stop on first failure
        "--tb=short",
    ] + test_args

    result = subprocess.run(cmd, cwd=str(project_dir), env=env)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run live integration tests")
    parser.add_argument(
        "--no-infrastructure",
        action="store_true",
        help="Skip infrastructure startup (assume it's already running)"
    )
    parser.add_argument(
        "--keep-infrastructure",
        action="store_true",
        help="Keep infrastructure running after tests"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for infrastructure startup (default: 300s)"
    )
    parser.add_argument(
        "--test-filter",
        type=str,
        help="Filter tests to run (pytest -k argument)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Specific test file to run"
    )

    args = parser.parse_args()

    # Get project directory
    project_dir = Path(__file__).parent.parent.absolute()

    # Check prerequisites
    if not check_docker_available():
        print("❌ Docker is not available. Please install and start Docker.")
        return 1

    if not check_docker_compose_available():
        print("❌ Docker Compose is not available. Please install Docker Compose.")
        return 1

    print(f"📁 Project directory: {project_dir}")

    infrastructure_started = False
    try:
        # Start infrastructure if needed
        if not args.no_infrastructure:
            if not start_test_infrastructure(project_dir, args.timeout):
                print("❌ Failed to start test infrastructure")
                return 1
            infrastructure_started = True

        # Prepare test arguments
        test_args = []
        if args.test_filter:
            test_args.extend(["-k", args.test_filter])
        if args.test_file:
            test_args.append(f"tests/live/{args.test_file}")

        # Run tests
        exit_code = run_live_tests(project_dir, test_args)

        if exit_code == 0:
            print("✅ All live tests passed!")
        else:
            print(f"❌ Tests failed with exit code {exit_code}")

        return exit_code

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 130

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    finally:
        # Cleanup infrastructure if we started it
        if infrastructure_started and not args.keep_infrastructure:
            stop_test_infrastructure(project_dir)


if __name__ == "__main__":
    sys.exit(main())
