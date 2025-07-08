#!/usr/bin/env python3
"""
Orcastrate End-to-End Demonstration

This script demonstrates the complete workflow of the Orcastrate development
environment agent, from high-level requirements to running applications.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.base import Requirements
from src.executors.concrete_executor import ConcreteExecutor
from src.executors.base import ExecutorConfig, ExecutionStrategy
from src.planners.template_planner import TemplatePlanner
from src.planners.base import PlannerConfig, PlanningStrategy


# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("OrcastrateDemo")


class DemoRunner:
    """Demo runner that showcases Orcastrate capabilities."""

    def __init__(self):
        self.planner: Optional[TemplatePlanner] = None
        self.executor: Optional[ConcreteExecutor] = None

    async def initialize(self):
        """Initialize the demo environment."""
        logger.info("🚀 Initializing Orcastrate Demo Environment...")

        try:
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

            # Initialize planner
            logger.info("📋 Initializing Template Planner...")
            self.planner = TemplatePlanner(planner_config)
            await self.planner.initialize()

            # Initialize executor
            logger.info("⚡ Initializing Concrete Executor...")
            self.executor = ConcreteExecutor(executor_config)
            await self.executor.initialize()

            logger.info("✅ Demo environment initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Demo initialization failed: {e}")
            raise

    async def show_available_templates(self):
        """Display available templates."""
        logger.info("\n" + "="*60)
        logger.info("📚 AVAILABLE TEMPLATES")
        logger.info("="*60)

        templates = await self.planner.get_available_templates()

        for i, template in enumerate(templates, 1):
            logger.info(f"\n{i}. {template['name']}")
            logger.info(f"   📝 Description: {template['description']}")
            logger.info(f"   🏗️  Framework: {template.get('framework', 'N/A')}")
            logger.info(f"   ⏱️  Duration: ~{template.get('estimated_duration', 0)/60:.1f} minutes")
            logger.info(f"   💰 Cost: ${template.get('estimated_cost', 0):.2f}")

    async def show_available_tools(self):
        """Display available tools."""
        logger.info("\n" + "="*60)
        logger.info("🔧 AVAILABLE TOOLS")
        logger.info("="*60)

        tools = await self.executor.get_available_tools()

        for tool_name, tool_info in tools.items():
            status = tool_info.get("status", "unknown")
            if status == "available":
                logger.info(f"\n✅ {tool_name.upper()}")
                logger.info(f"   📝 Description: {tool_info.get('description', 'No description')}")
                logger.info(f"   🔧 Version: {tool_info.get('version', 'Unknown')}")
                logger.info(f"   ⚡ Actions: {', '.join(tool_info.get('actions', []))}")
            else:
                logger.info(f"\n❌ {tool_name.upper()}: {tool_info.get('error', 'Not available')}")

    async def demonstrate_workflow(self, requirements: Requirements, demo_name: str):
        """Demonstrate the complete workflow for given requirements."""
        logger.info("\n" + "="*80)
        logger.info(f"🎬 DEMO: {demo_name}")
        logger.info("="*80)
        logger.info(f"📋 Requirements: {requirements.description}")
        if requirements.framework:
            logger.info(f"🏗️  Framework: {requirements.framework}")
        if requirements.database:
            logger.info(f"🗄️  Database: {requirements.database}")

        try:
            # Step 1: Generate Plan
            logger.info("\n📋 STEP 1: GENERATING EXECUTION PLAN")
            logger.info("-" * 40)

            start_time = time.time()
            plan = await self.planner.create_plan(requirements)
            planning_duration = time.time() - start_time

            logger.info(f"✅ Plan generated in {planning_duration:.2f} seconds")
            logger.info(f"📊 Plan ID: {plan.id}")
            logger.info(f"📈 Steps: {len(plan.steps)}")
            logger.info(f"⏱️  Estimated Duration: {plan.estimated_duration:.0f} seconds")
            logger.info(f"💰 Estimated Cost: ${plan.estimated_cost:.2f}")

            # Show plan steps
            logger.info("\n📝 PLAN STEPS:")
            for i, step in enumerate(plan.steps, 1):
                logger.info(f"  {i:2d}. {step.get('name', step['id'])}")
                logger.info(f"      🔧 Tool: {step.get('tool')} | Action: {step.get('action')}")
                if step.get("dependencies"):
                    logger.info(f"      🔗 Dependencies: {', '.join(step['dependencies'])}")

            # Step 2: Validate Plan
            logger.info("\n🔍 STEP 2: VALIDATING PLAN")
            logger.info("-" * 40)

            validation = await self.executor.validate_plan_requirements(plan)

            if validation["valid"]:
                logger.info("✅ Plan validation passed")
            else:
                logger.error("❌ Plan validation failed:")
                if validation.get("missing_tools"):
                    logger.error(f"   Missing tools: {validation['missing_tools']}")
                if validation.get("invalid_actions"):
                    logger.error(f"   Invalid actions: {validation['invalid_actions']}")
                return False

            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    logger.warning(f"⚠️  {warning}")

            # Step 3: Execute Plan
            logger.info("\n⚡ STEP 3: EXECUTING PLAN")
            logger.info("-" * 40)

            start_time = time.time()
            result = await self.executor.execute_plan(plan)
            execution_duration = time.time() - start_time

            # Step 4: Show Results
            logger.info("\n📊 STEP 4: EXECUTION RESULTS")
            logger.info("-" * 40)

            if result.success:
                logger.info("🎉 EXECUTION SUCCESSFUL!")
                logger.info(f"⏱️  Total Duration: {execution_duration:.2f} seconds")

                metrics = result.metrics or {}
                if "successful_steps" in metrics:
                    logger.info(f"✅ Steps Completed: {metrics['successful_steps']}/{metrics.get('total_steps', 0)}")

                logger.info(f"📦 Artifacts Created: {len(result.artifacts)} items")

                if result.artifacts:
                    logger.info("🎯 Key Artifacts:")
                    for artifact_id, artifact_data in result.artifacts.items():
                        if isinstance(artifact_data, dict):
                            if "path" in artifact_data:
                                logger.info(f"  📁 {artifact_id}: {artifact_data['path']}")
                            elif "image" in artifact_data:
                                logger.info(f"  🐳 {artifact_id}: {artifact_data['image']}")
                            elif "container_id" in artifact_data:
                                logger.info(f"  📦 {artifact_id}: Container {artifact_data['container_id'][:12]}")

                # Show application URLs if containers were created
                for step in plan.steps:
                    if step.get("action") == "run_container" and step.get("parameters", {}).get("ports"):
                        ports = step["parameters"]["ports"]
                        for port_mapping in ports:
                            if ":" in port_mapping:
                                host_port = port_mapping.split(":")[0]
                                logger.info(f"🌐 Application available at: http://localhost:{host_port}")

            else:
                logger.error("❌ EXECUTION FAILED!")
                logger.error(f"Error: {result.error}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Demo failed with exception: {e}")
            return False

    async def run_demo_scenarios(self):
        """Run multiple demo scenarios."""
        scenarios = [
            {
                "name": "Node.js Web Application",
                "requirements": Requirements(
                    description="Create a Node.js web application with Express framework",
                    framework="nodejs",
                    metadata={"demo": True}
                )
            },
            {
                "name": "Python FastAPI Service",
                "requirements": Requirements(
                    description="Build a FastAPI REST API service with Python",
                    framework="fastapi",
                    metadata={"demo": True}
                )
            },
        ]

        results = []

        for scenario in scenarios:
            success = await self.demonstrate_workflow(
                scenario["requirements"],
                scenario["name"]
            )
            results.append({"name": scenario["name"], "success": success})

            # Pause between demos
            if scenario != scenarios[-1]:  # Not the last scenario
                logger.info("\n⏸️  Pausing 5 seconds before next demo...")
                await asyncio.sleep(5)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("📈 DEMO SUMMARY")
        logger.info("="*80)

        successful = sum(1 for r in results if r["success"])
        total = len(results)

        logger.info(f"✅ Successful Demos: {successful}/{total}")

        for result in results:
            status = "✅" if result["success"] else "❌"
            logger.info(f"  {status} {result['name']}")

        if successful == total:
            logger.info("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
            logger.info("🚀 Orcastrate is ready for production use!")
        else:
            logger.info(f"\n⚠️  {total - successful} demos failed. Check logs for details.")

    async def cleanup_demo_artifacts(self):
        """Clean up demo artifacts."""
        logger.info("\n🧹 Cleaning up demo artifacts...")

        try:
            # This would stop any running containers and clean up files
            # For now, just log the cleanup intention
            logger.info("ℹ️  Demo artifacts can be found in /tmp/orcastrate/")
            logger.info("ℹ️  To clean up manually:")
            logger.info("   - Stop containers: docker stop $(docker ps -q --filter name=*-app)")
            logger.info("   - Remove containers: docker rm $(docker ps -aq --filter name=*-app)")
            logger.info("   - Remove images: docker rmi $(docker images '*-*' -q)")
            logger.info("   - Remove files: rm -rf /tmp/orcastrate/")

        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


async def main():
    """Main demo function."""
    demo = DemoRunner()

    try:
        # Initialize
        await demo.initialize()

        # Show capabilities
        await demo.show_available_templates()
        await demo.show_available_tools()

        # Run demo scenarios
        await demo.run_demo_scenarios()

        # Cleanup
        await demo.cleanup_demo_artifacts()

    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║               🚀 ORCASTRATE DEMO 🚀                          ║
    ║                                                              ║
    ║     Production-Grade Development Environment Agent           ║
    ║                                                              ║
    ║  This demo showcases the complete workflow from natural      ║
    ║  language requirements to running development environments   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())
