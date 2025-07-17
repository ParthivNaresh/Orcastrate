"""
Enhanced Template-based planner implementation.

This planner uses modular template components that can be composed together
to create sophisticated multi-technology development environments without AI.
Supports detection and combination of frameworks, databases, caching, and infrastructure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import (
    Planner,
    PlannerConfig,
    PlannerError,
    PlanStep,
    PlanStructure,
    Requirements,
)


@dataclass
class DetectedTechnologies:
    """Container for detected technologies from user input."""

    framework: Optional[str] = None
    databases: List[str] = field(default_factory=list)
    cache: List[str] = field(default_factory=list)
    infrastructure: List[str] = field(default_factory=list)


@dataclass
class TemplateComponent:
    """A reusable template component for a specific technology."""

    name: str
    technology: str
    category: str  # 'framework', 'database', 'cache', 'infrastructure'
    steps: List[Dict[str, Any]]
    dependencies: List[str] = field(
        default_factory=list
    )  # Other components this depends on
    provides: List[str] = field(default_factory=list)  # What this component provides


class TemplatePlanner(Planner):
    """
    Enhanced template-based planner using modular components.

    This planner can detect multiple technologies in user requests and compose
    relevant components together to create sophisticated multi-service environments.
    No AI required - uses pattern matching and rule-based composition.
    """

    def __init__(self, config: PlannerConfig):
        super().__init__(config)
        # Modular component system
        self._components: Dict[str, TemplateComponent] = {}
        self._technology_patterns: Dict[str, Dict[str, List[str]]] = {}
        self._loaded = False

    async def initialize(self) -> None:
        """Initialize the enhanced template planner."""
        await super().initialize()
        await self._load_technology_patterns()
        self._loaded = True

    def detect_technologies(self, description: str) -> DetectedTechnologies:
        """
        Analyze user input to detect multiple technologies.

        This is the core enhancement - instead of picking one template,
        we identify ALL technologies mentioned and compose them together.
        """
        description_lower = description.lower()
        detected = DetectedTechnologies()

        # Framework detection
        for framework, patterns in self._technology_patterns.get(
            "frameworks", {}
        ).items():
            if any(pattern in description_lower for pattern in patterns):
                detected.framework = framework
                break

        # Database detection (can be multiple)
        for database, patterns in self._technology_patterns.get(
            "databases", {}
        ).items():
            if any(pattern in description_lower for pattern in patterns):
                detected.databases.append(database)

        # Cache detection
        for cache, patterns in self._technology_patterns.get("cache", {}).items():
            if any(pattern in description_lower for pattern in patterns):
                detected.cache.append(cache)

        # Infrastructure detection
        for infra, patterns in self._technology_patterns.get(
            "infrastructure", {}
        ).items():
            if any(pattern in description_lower for pattern in patterns):
                detected.infrastructure.append(infra)

        return detected

    async def _load_technology_patterns(self) -> None:
        """
        Load keyword patterns for technology detection.

        This maps user-friendly terms to our internal technology names.
        Makes the planner understand terms like 'postgres' -> 'postgresql'
        """
        self._technology_patterns = {
            "frameworks": {
                "nodejs": ["node.js", "nodejs", "node", "express", "npm"],
                "fastapi": ["fastapi", "fast api"],
                "django": ["django"],
                "flask": ["flask"],
                "react": ["react", "reactjs"],
                "vue": ["vue", "vuejs"],
                "angular": ["angular"],
            },
            "databases": {
                "postgresql": ["postgresql", "postgres", "pg"],
                "mysql": ["mysql"],
                "mongodb": ["mongodb", "mongo"],
                "sqlite": ["sqlite"],
                "mariadb": ["mariadb"],
            },
            "cache": {"redis": ["redis"], "memcached": ["memcached", "memcache"]},
            "infrastructure": {
                "docker": ["docker", "container", "containerized"],
                "kubernetes": ["kubernetes", "k8s"],
                "aws": ["aws", "amazon web services"],
                "terraform": ["terraform", "infrastructure as code"],
            },
        }

    def compose_plan_from_technologies(
        self, technologies: DetectedTechnologies
    ) -> List[Dict[str, Any]]:
        """
        Compose plan steps from detected technologies.

        This is the core composition logic - it combines modular components
        based on what technologies were detected in the user's request.
        """
        composed_steps = []
        step_counter = 1

        # Always start with base project setup
        composed_steps.extend(self._get_base_steps(step_counter))
        step_counter = len(composed_steps) + 1

        # Add security documentation for projects with databases
        if technologies.databases or technologies.cache:
            security_steps = self._get_security_documentation_steps(step_counter)
            composed_steps.extend(security_steps)
            step_counter = len(composed_steps) + 1

        # Add framework steps
        if technologies.framework:
            framework_steps = self._get_framework_steps(
                technologies.framework, step_counter
            )
            composed_steps.extend(framework_steps)
            step_counter = len(composed_steps) + 1

        # Add database steps (can be multiple)
        for database in technologies.databases:
            db_steps = self._get_database_steps(database, step_counter)
            composed_steps.extend(db_steps)
            step_counter = len(composed_steps) + 1

        # Add cache steps
        for cache in technologies.cache:
            cache_steps = self._get_cache_steps(cache, step_counter)
            composed_steps.extend(cache_steps)
            step_counter = len(composed_steps) + 1

        # Add infrastructure steps (docker, compose file, etc.)
        if technologies.infrastructure or technologies.databases or technologies.cache:
            # If we have databases or cache, we need docker-compose
            infra_steps = self._get_infrastructure_steps(technologies, step_counter)
            composed_steps.extend(infra_steps)

        return composed_steps

    def _get_base_steps(self, start_counter: int) -> List[Dict[str, Any]]:
        """Get base project setup steps - always included."""
        return [
            {
                "id": "setup_directory",
                "name": "Setup Project Directory",
                "description": "Create and initialize project directory",
                "tool": "filesystem",
                "action": "create_directory",
                "parameters": {"path": "{project_path}", "mode": "755"},
                "dependencies": [],
                "estimated_duration": 5.0,
                "estimated_cost": 0.0,
            },
            {
                "id": "init_git",
                "name": "Initialize Git Repository",
                "description": "Initialize Git repository for version control",
                "tool": "git",
                "action": "init",
                "parameters": {
                    "path": "{project_path}",
                    "initial_branch": "main",
                },
                "dependencies": ["setup_directory"],
                "estimated_duration": 10.0,
                "estimated_cost": 0.0,
            },
        ]

    def _get_framework_steps(
        self, framework: str, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Get framework-specific steps."""
        if framework == "nodejs":
            return [
                {
                    "id": "create_package_json",
                    "name": "Create package.json",
                    "description": "Create Node.js package.json file",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/package.json",
                        "content": {
                            "name": "{project_name}",
                            "version": "1.0.0",
                            "description": "{description}",
                            "main": "index.js",
                            "scripts": {
                                "start": "node index.js",
                                "dev": "nodemon index.js",
                            },
                            "dependencies": {"express": "^4.18.0"},
                            "devDependencies": {"nodemon": "^2.0.0"},
                        },
                    },
                    "dependencies": ["setup_directory"],
                    "estimated_duration": 15.0,
                    "estimated_cost": 0.0,
                },
                {
                    "id": "create_main_file",
                    "name": "Create Main Application File",
                    "description": "Create the main Node.js application file",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/index.js",
                        "content": "const express = require('express');\nconst app = express();\nconst port = process.env.PORT || 3000;\n\napp.get('/', (req, res) => {\n  res.json({ message: 'Hello from {project_name}!' });\n});\n\napp.listen(port, () => {\n  console.log(`Server running on port ${port}`);\n});",
                    },
                    "dependencies": ["setup_directory"],
                    "estimated_duration": 20.0,
                    "estimated_cost": 0.0,
                },
            ]
        elif framework == "fastapi":
            return [
                {
                    "id": "create_requirements",
                    "name": "Create requirements.txt",
                    "description": "Create Python requirements file",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/requirements.txt",
                        "content": "fastapi==0.104.1\nuvicorn[standard]==0.24.0\npydantic==2.5.0",
                    },
                    "dependencies": ["setup_directory"],
                    "estimated_duration": 10.0,
                    "estimated_cost": 0.0,
                },
                {
                    "id": "create_main_app",
                    "name": "Create Main FastAPI Application",
                    "description": "Create the main FastAPI application file",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/main.py",
                        "content": 'from fastapi import FastAPI\n\napp = FastAPI(title="{project_name}", description="{description}")\n\n@app.get("/")\nasync def root():\n    return {"message": "Hello from {project_name}!"}\n\n@app.get("/health")\nasync def health_check():\n    return {"status": "healthy"}\n\nif __name__ == "__main__":\n    import uvicorn\n    uvicorn.run(app, host="0.0.0.0", port=8000)',
                    },
                    "dependencies": ["setup_directory"],
                    "estimated_duration": 30.0,
                    "estimated_cost": 0.0,
                },
            ]
        return []

    def _get_database_steps(
        self, database: str, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Get database-specific steps."""
        if database == "postgresql":
            return [
                {
                    "id": f"create_postgres_service_{start_counter}",
                    "name": "Setup PostgreSQL Service",
                    "description": "Add PostgreSQL service to docker-compose",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/.env",
                        "content": "# Database Configuration\n# Copy from .env.example and update with secure values\nPOSTGRES_DB={project_name}_db\nPOSTGRES_USER=${POSTGRES_USER:-postgres}\nPOSTGRES_PASSWORD=${POSTGRES_PASSWORD:-your-postgres-password}\nPOSTGRES_HOST=${POSTGRES_HOST:-localhost}\nPOSTGRES_PORT=${POSTGRES_PORT:-5432}\nDATABASE_URL=postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-your-postgres-password}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/{project_name}_db\n\n# SECURITY WARNING: Change POSTGRES_PASSWORD in production!\n# Generate secure password: python -c 'import secrets; print(secrets.token_urlsafe(32))'\n",
                    },
                    "dependencies": ["setup_directory"],
                    "estimated_duration": 10.0,
                    "estimated_cost": 0.0,
                },
            ]
        elif database == "redis":
            return [
                {
                    "id": f"create_redis_service_{start_counter}",
                    "name": "Setup Redis Service",
                    "description": "Add Redis service configuration",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/.env.redis",
                        "content": "# Redis Configuration\nREDIS_URL=${REDIS_URL:-redis://localhost:6379}\nREDIS_DB=${REDIS_DB:-0}\nREDIS_HOST=${REDIS_HOST:-localhost}\nREDIS_PORT=${REDIS_PORT:-6379}\n",
                    },
                    "dependencies": ["setup_directory"],
                    "estimated_duration": 5.0,
                    "estimated_cost": 0.0,
                },
            ]
        return []

    def _get_cache_steps(self, cache: str, start_counter: int) -> List[Dict[str, Any]]:
        """Get cache-specific steps."""
        return self._get_database_steps(
            cache, start_counter
        )  # Redis is handled as database

    def _get_infrastructure_steps(
        self, technologies: DetectedTechnologies, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Get infrastructure steps like Docker, docker-compose."""
        steps = []

        # If we have multiple services (app + databases), create docker-compose
        if technologies.databases or technologies.cache:
            # Create Dockerfile for the main app
            if technologies.framework == "nodejs":
                dockerfile_content = 'FROM node:18-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nEXPOSE 3000\nCMD ["npm", "start"]'
            elif technologies.framework == "fastapi":
                dockerfile_content = 'FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]'
            else:
                dockerfile_content = 'FROM alpine:latest\nWORKDIR /app\nCOPY . .\nEXPOSE 3000\nCMD ["echo", "No specific framework Dockerfile"]'

            steps.append(
                {
                    "id": "create_dockerfile",
                    "name": "Create Dockerfile",
                    "description": "Create Dockerfile for the application",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/Dockerfile",
                        "content": dockerfile_content,
                    },
                    "dependencies": ["setup_directory"],
                    "estimated_duration": 10.0,
                    "estimated_cost": 0.0,
                }
            )

            # Create docker-compose.yml with multiple services
            compose_content = self._generate_docker_compose_content(technologies)
            steps.append(
                {
                    "id": "create_docker_compose",
                    "name": "Create Docker Compose Configuration",
                    "description": "Create multi-service docker-compose file",
                    "tool": "filesystem",
                    "action": "write_file",
                    "parameters": {
                        "path": "{project_path}/docker-compose.yml",
                        "content": compose_content,
                    },
                    "dependencies": ["create_dockerfile"],
                    "estimated_duration": 15.0,
                    "estimated_cost": 0.0,
                }
            )

            # Start the multi-service stack
            steps.append(
                {
                    "id": "start_services",
                    "name": "Start All Services",
                    "description": "Start application and database services",
                    "tool": "docker",
                    "action": "compose_up",
                    "parameters": {
                        "compose_file": "{project_path}/docker-compose.yml",
                        "build": True,
                        "detach": True,
                    },
                    "dependencies": ["create_docker_compose"],
                    "estimated_duration": 60.0,
                    "estimated_cost": 0.1,
                }
            )
        else:
            # Single service - just containerize the app
            steps.extend(self._get_simple_docker_steps(technologies))

        return steps

    def _generate_docker_compose_content(
        self, technologies: DetectedTechnologies
    ) -> str:
        """Generate docker-compose.yml content based on detected technologies."""
        app_port = "3000" if technologies.framework == "nodejs" else "8000"

        compose = f"""# WARNING: Update environment variables in production!
# See .env.example for secure configuration
# Generate secure passwords: python -c 'import secrets; print(secrets.token_urlsafe(32))'
version: '3.8'
services:
  app:
    build: .
    ports:
      - "{app_port}:{app_port}"
    environment:
      - NODE_ENV=development
    env_file:
      - .env  # Create from .env.example
    depends_on:"""

        services = []
        if "postgresql" in technologies.databases:
            services.append("postgres")
            compose += "\n      - postgres"
        if "redis" in technologies.cache:
            services.append("redis")
            compose += "\n      - redis"

        # Add database services
        if "postgresql" in technologies.databases:
            compose += """

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB={project_name}_db
      - POSTGRES_USER=${POSTGRES_USER:-postgres}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-your-postgres-password}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data"""

        if "redis" in technologies.cache:
            compose += """

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data"""

        # Add volumes section if needed
        if services:
            compose += "\n\nvolumes:"
            if "postgres" in services:
                compose += "\n  postgres_data:"
            if "redis" in services:
                compose += "\n  redis_data:"

        return compose

    def _get_simple_docker_steps(
        self, technologies: DetectedTechnologies
    ) -> List[Dict[str, Any]]:
        """Get simple Docker steps for single-service applications."""
        if technologies.framework == "nodejs":
            dockerfile_content = 'FROM node:18-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nEXPOSE 3000\nCMD ["npm", "start"]'
            port = "3000"
        elif technologies.framework == "fastapi":
            dockerfile_content = 'FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]'
            port = "8000"
        else:
            return []

        return [
            {
                "id": "create_dockerfile",
                "name": "Create Dockerfile",
                "description": "Create Dockerfile for containerization",
                "tool": "filesystem",
                "action": "write_file",
                "parameters": {
                    "path": "{project_path}/Dockerfile",
                    "content": dockerfile_content,
                },
                "dependencies": ["setup_directory"],
                "estimated_duration": 10.0,
                "estimated_cost": 0.0,
            },
            {
                "id": "build_docker_image",
                "name": "Build Docker Image",
                "description": "Build Docker image for the application",
                "tool": "docker",
                "action": "build_image",
                "parameters": {
                    "context_path": "{project_path}",
                    "image_name": "{project_name}",
                    "tag": "latest",
                },
                "dependencies": ["create_dockerfile"],
                "estimated_duration": 120.0,
                "estimated_cost": 0.1,
            },
            {
                "id": "run_container",
                "name": "Run Application Container",
                "description": "Run the containerized application",
                "tool": "docker",
                "action": "run_container",
                "parameters": {
                    "image": "{project_name}:latest",
                    "name": "{project_name}-app",
                    "ports": [f"{port}:{port}"],
                    "detached": True,
                },
                "dependencies": ["build_docker_image"],
                "estimated_duration": 30.0,
                "estimated_cost": 0.05,
            },
        ]

    async def _generate_initial_plan(self, context: Dict[str, Any]) -> PlanStructure:
        """Generate enhanced plan using modular components."""
        requirements_data = context.get("requirements")
        if not requirements_data:
            raise PlannerError("Requirements not found in context")

        # Convert dict back to Requirements object if needed
        if isinstance(requirements_data, dict):
            from ..agent.base import Requirements

            requirements = Requirements(**requirements_data)
        else:
            requirements = requirements_data

        # NEW ENHANCED LOGIC: Detect multiple technologies
        detected_technologies = self.detect_technologies(requirements.description)

        # Handle explicit framework/database from requirements object
        if requirements.framework and not detected_technologies.framework:
            detected_technologies.framework = requirements.framework.lower()
        if (
            requirements.database
            and requirements.database not in detected_technologies.databases
        ):
            detected_technologies.databases.append(requirements.database.lower())

        # NEW: Compose plan from multiple technologies
        composed_steps = self.compose_plan_from_technologies(detected_technologies)

        if not composed_steps:
            raise PlannerError("No suitable components found for requirements")

        # Generate plan steps with proper variable substitution
        steps = await self._process_composed_steps(composed_steps, requirements)

        return PlanStructure(
            steps=steps,
            metadata={
                "composition_type": "multi_technology",
                "detected_technologies": {
                    "framework": detected_technologies.framework,
                    "databases": detected_technologies.databases,
                    "cache": detected_technologies.cache,
                    "infrastructure": detected_technologies.infrastructure,
                },
                "generated_at": context.get("timestamp"),
            },
        )

    async def _gather_context(self, requirements: Requirements) -> Dict[str, Any]:
        """Gather context for planning."""
        return {
            "requirements": requirements.model_dump(),
            "timestamp": "2023-01-01T00:00:00Z",
            "planner_type": "enhanced_template_based",
        }

    async def _process_composed_steps(
        self, composed_steps: List[Dict[str, Any]], requirements: Requirements
    ) -> List[PlanStep]:
        """Process composed steps with variable substitution and convert to PlanStep objects."""
        processed_steps = []

        # Generate project variables
        project_name = self._generate_project_name(requirements.description)
        project_path = f"/tmp/orcastrate/{project_name}"

        # Variable substitution context
        substitution_vars = {
            "project_name": project_name,
            "project_path": project_path,
            "description": requirements.description,
        }

        for step_data in composed_steps:
            # Perform variable substitution
            processed_step = self._substitute_variables(step_data, substitution_vars)

            # Convert to PlanStep object
            plan_step = PlanStep(
                id=processed_step["id"],
                name=processed_step["name"],
                description=processed_step["description"],
                tool=processed_step["tool"],
                action=processed_step["action"],
                parameters=processed_step["parameters"],
                dependencies=processed_step.get("dependencies", []),
                estimated_duration=processed_step.get("estimated_duration", 300.0),
                estimated_cost=processed_step.get("estimated_cost", 0.0),
            )

            processed_steps.append(plan_step)

        return processed_steps

    def _generate_project_name(self, description: str) -> str:
        """Generate a project name from description."""
        # Clean description and create a slug
        import re

        clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", description.lower())
        words = clean_name.split()[:3]  # Take first 3 words
        return "-".join(words) if words else "orcastrate-project"

    def _get_security_documentation_steps(
        self, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Generate security documentation for database projects."""
        return [
            {
                "id": "create_security_readme",
                "name": "Create Security Documentation",
                "description": "Create security best practices documentation",
                "tool": "filesystem",
                "action": "write_file",
                "parameters": {
                    "path": "{project_path}/SECURITY.md",
                    "content": "# Security Configuration\n\n## Database Security\n\n⚠️ **IMPORTANT**: This project uses database services that require secure configuration.\n\n### Environment Variables\n\nThis project uses environment variables for configuration. See `.env.example` for all variables.\n\n### Database Passwords\n\n1. **Never use default passwords in production**\n2. **Generate secure passwords**:\n   ```bash\n   python -c 'import secrets; print(secrets.token_urlsafe(32))'\n   ```\n3. **Set environment variables**:\n   ```bash\n   export POSTGRES_PASSWORD=\"your-secure-password\"\n   ```\n\n### Docker Compose Security\n\n- Change default passwords in docker-compose.yml\n- Use environment variable substitution: `${POSTGRES_PASSWORD}`\n- Mount secrets as files (recommended for production)\n\n### Production Checklist\n\n- [ ] Change all default passwords\n- [ ] Use environment variables for all secrets\n- [ ] Enable SSL/TLS for database connections\n- [ ] Restrict database network access\n- [ ] Regular security updates\n- [ ] Monitor for suspicious activity\n\n### References\n\n- [OWASP Database Security](https://owasp.org/www-community/vulnerabilities/Insecure_Storage)\n- [Docker Secrets Management](https://docs.docker.com/engine/swarm/secrets/)\n",
                },
                "dependencies": ["setup_directory"],
                "estimated_duration": 5.0,
                "estimated_cost": 0.0,
            }
        ]

    def _substitute_variables(
        self, step_data: Dict[str, Any], variables: Dict[str, str]
    ) -> Dict[str, Any]:
        """Substitute variables in step data."""
        import json

        # Convert to JSON string for easy substitution
        step_json = json.dumps(step_data)

        # Perform substitutions
        for var_name, var_value in variables.items():
            step_json = step_json.replace(f"{{{var_name}}}", var_value)

        # Convert back to dict
        result = json.loads(step_json)
        assert isinstance(result, dict), "Expected dict from JSON parsing"
        return result

    async def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available templates for CLI display."""
        # Return information about available technology combinations
        return [
            {
                "id": "nodejs_web_app",
                "name": "Node.js Web Application",
                "description": "Node.js web application with Express framework",
                "framework": "nodejs",
                "estimated_duration": 120.0,
                "estimated_cost": 0.05,
            },
            {
                "id": "fastapi_app",
                "name": "FastAPI REST API",
                "description": "Python FastAPI application with async support",
                "framework": "fastapi",
                "estimated_duration": 150.0,
                "estimated_cost": 0.1,
            },
            {
                "id": "multi_tech_stack",
                "name": "Multi-Technology Stack",
                "description": "Compose multiple technologies: frameworks + databases + caching",
                "framework": "composable",
                "estimated_duration": 180.0,
                "estimated_cost": 0.15,
            },
        ]
