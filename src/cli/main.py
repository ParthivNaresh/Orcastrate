"""
Main CLI interface for Orcastrate development environment agent.

This module provides a command-line interface for creating and managing
development environments using natural language requirements.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
import click

from ..agent.base import Requirements
from ..agent.coordinator import AgentCoordinator
from ..executors.base import ExecutionStrategy, ExecutorConfig
from ..executors.concrete_executor import ConcreteExecutor
from ..planners.base import PlannerConfig, PlanningStrategy
from ..planners.template_planner import TemplatePlanner

# Ensure log directory exists before configuring logging
LOG_DIR = Path("/tmp/orcastrate")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "orcastrate.log"),
    ],
)


class OrcastrateAgent:
    """Main Orcastrate agent that coordinates the entire workflow."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner: Optional[TemplatePlanner] = None
        self.executor: Optional[ConcreteExecutor] = None
        self.coordinator: Optional[AgentCoordinator] = None

    async def initialize(self) -> None:
        """Initialize the agent and its components."""
        try:
            self.logger.info("🚀 Initializing Orcastrate Agent...")

            # Create configurations
            planner_config = PlannerConfig(
                strategy=PlanningStrategy.TEMPLATE_MATCHING,
                max_plan_steps=20,
                max_planning_time=60,
                cost_optimization=True,
                risk_threshold=0.8,
            )

            executor_config = ExecutorConfig(
                strategy=ExecutionStrategy.SEQUENTIAL,
                max_concurrent_steps=5,
                step_timeout=300,
                retry_policy={
                    "max_retries": 3,
                    "backoff_factor": 2.0,
                    "max_delay": 60,
                },
                enable_rollback=True,
            )

            # Initialize components
            self.planner = TemplatePlanner(planner_config)
            await self.planner.initialize()

            self.executor = ConcreteExecutor(executor_config)
            await self.executor.initialize()

            self.logger.info("✅ Orcastrate Agent initialized successfully!")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agent: {e}")
            raise

    async def create_environment(self, requirements: Requirements) -> Dict[str, Any]:
        """Create a development environment based on requirements."""
        if not self.planner or not self.executor:
            raise RuntimeError("Agent not initialized")

        try:
            self.logger.info(f"📋 Creating environment: {requirements.description}")

            # Generate plan
            self.logger.info("🧠 Generating execution plan...")
            plan = await self.planner.create_plan(requirements)

            self.logger.info(f"📊 Plan created with {len(plan.steps)} steps:")
            for i, step in enumerate(plan.steps, 1):
                self.logger.info(f"  {i}. {step.get('name', step['id'])}")

            # Validate plan
            self.logger.info("🔍 Validating plan requirements...")
            validation = await self.executor.validate_plan_requirements(plan)

            if not validation["valid"]:
                error_details = {
                    "missing_tools": validation.get("missing_tools", []),
                    "invalid_actions": validation.get("invalid_actions", []),
                }
                raise RuntimeError(f"Plan validation failed: {error_details}")

            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    self.logger.warning(f"⚠️  {warning}")

            # Execute plan
            self.logger.info("⚡ Executing plan...")
            result = await self.executor.execute_plan(plan)

            # Return comprehensive result
            return {
                "success": result.success,
                "execution_id": result.execution_id,
                "plan": {
                    "id": plan.id,
                    "steps": len(plan.steps),
                    "estimated_duration": plan.estimated_duration,
                    "estimated_cost": plan.estimated_cost,
                },
                "execution": {
                    "duration": result.duration,
                    "artifacts": result.artifacts,
                    "metrics": result.metrics,
                },
                "error": result.error,
                "requirements": requirements.model_dump(),
            }

        except Exception as e:
            self.logger.error(f"❌ Environment creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "requirements": requirements.model_dump(),
            }

    async def list_templates(self) -> Dict[str, Any]:
        """List available templates."""
        if not self.planner:
            raise RuntimeError("Agent not initialized")

        templates = await self.planner.get_available_templates()
        return {"templates": templates}

    async def get_tools_status(self) -> Dict[str, Any]:
        """Get status of available tools."""
        if not self.executor:
            raise RuntimeError("Agent not initialized")

        tools = await self.executor.get_available_tools()
        return {"tools": tools}


# CLI Commands


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """Orcastrate - Production-Grade Development Environment Agent"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Log directory already created at module level

    ctx.ensure_object(dict)


@cli.command()
@click.argument("description")
@click.option("--framework", "-f", help="Preferred framework (nodejs, fastapi, etc.)")
@click.option("--database", "-d", help="Database type (postgresql, mysql, redis, etc.)")
@click.option("--cloud-provider", "-c", help="Cloud provider (aws, gcp, azure)")
@click.option("--output", "-o", type=click.Path(), help="Save result to JSON file")
@click.option("--dry-run", is_flag=True, help="Generate plan without executing")
def create(description, framework, database, cloud_provider, output, dry_run):
    """Create a development environment from description.

    Examples:
      orcastrate create "Node.js web application with Express"
      orcastrate create "FastAPI REST API" --framework fastapi
      orcastrate create "Full-stack app" --framework nodejs --database postgresql
    """

    async def _create():
        agent = OrcastrateAgent()

        try:
            await agent.initialize()

            # Create requirements
            requirements = Requirements(
                description=description,
                framework=framework,
                database=database,
                cloud_provider=cloud_provider,
                metadata={"created_at": datetime.utcnow().isoformat()},
            )

            if dry_run:
                click.echo("🔍 Dry run mode - generating plan only...")
                assert (
                    agent.planner is not None
                ), "Planner must be initialized before use"
                plan = await agent.planner.create_plan(requirements)

                result: Dict[str, Any] = {
                    "success": True,
                    "dry_run": True,
                    "plan": {
                        "id": plan.id,
                        "steps": [
                            {
                                "id": step["id"],
                                "name": step.get("name"),
                                "description": step.get("description"),
                                "tool": step.get("tool"),
                                "action": step.get("action"),
                                "estimated_duration": step.get("estimated_duration"),
                            }
                            for step in plan.steps
                        ],
                        "estimated_duration": plan.estimated_duration,
                        "estimated_cost": plan.estimated_cost,
                        "risk_assessment": plan.risk_assessment,
                    },
                    "requirements": requirements.model_dump(),
                }
            else:
                result = await agent.create_environment(requirements)

            # Output result
            if output:
                async with aiofiles.open(output, "w") as f:
                    await f.write(json.dumps(result, indent=2, default=str))
                click.echo(f"📄 Result saved to {output}")

            # Display summary
            if result["success"]:
                click.echo("🎉 Environment created successfully!")
                if not dry_run:
                    metrics = result.get("execution", {}).get("metrics", {})
                    duration = result.get("execution", {}).get("duration", 0)
                    click.echo(f"⏱️  Duration: {duration:.1f} seconds")
                    if "successful_steps" in metrics:
                        click.echo(
                            f"✅ Steps completed: {metrics['successful_steps']}/{metrics.get('total_steps', 0)}"
                        )
            else:
                click.echo(f"❌ Failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error: {e}")
            sys.exit(1)

    asyncio.run(_create())


@cli.command()
def templates():
    """List available environment templates."""

    async def _templates():
        agent = OrcastrateAgent()
        await agent.initialize()

        result = await agent.list_templates()

        click.echo("📚 Available Templates:")
        click.echo()

        for template in result["templates"]:
            click.echo(f"🏗️  {template['name']}")
            click.echo(f"   Description: {template['description']}")
            click.echo(f"   Framework: {template.get('framework', 'N/A')}")
            click.echo(
                f"   Duration: ~{template.get('estimated_duration', 0)/60:.1f} minutes"
            )
            click.echo(f"   Cost: ${template.get('estimated_cost', 0):.2f}")
            click.echo()

    asyncio.run(_templates())


@cli.command()
def tools():
    """Show status of available tools."""

    async def _tools():
        agent = OrcastrateAgent()
        await agent.initialize()

        result = await agent.get_tools_status()

        click.echo("🔧 Tool Status:")
        click.echo()

        for tool_name, tool_info in result["tools"].items():
            status = tool_info.get("status", "unknown")
            if status == "available":
                click.echo(
                    f"✅ {tool_name}: {tool_info.get('description', 'No description')}"
                )
                click.echo(f"   Version: {tool_info.get('version', 'Unknown')}")
                click.echo(f"   Actions: {', '.join(tool_info.get('actions', []))}")
            else:
                click.echo(f"❌ {tool_name}: {tool_info.get('error', 'Not available')}")
            click.echo()

    asyncio.run(_tools())


@cli.command()
@click.option("--lines", "-n", default=50, help="Number of log lines to show")
def logs(lines):
    """Show recent logs."""

    async def _logs():
        log_file = LOG_DIR / "orcastrate.log"

        if not log_file.exists():
            click.echo("📄 No logs found")
            return

        try:
            async with aiofiles.open(log_file, "r") as f:
                content = await f.read()
                log_lines = content.splitlines()

            # Show last N lines
            recent_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines

            click.echo(f"📄 Recent logs (last {len(recent_lines)} lines):")
            click.echo()

            for line in recent_lines:
                click.echo(line)

        except Exception as e:
            click.echo(f"❌ Error reading logs: {e}")

    asyncio.run(_logs())


@cli.command()
def version():
    """Show version information."""
    click.echo("Orcastrate Development Environment Agent")
    click.echo("Version: 1.0.0")
    click.echo("Phase: 2 - Concrete Implementations")


if __name__ == "__main__":
    cli()
