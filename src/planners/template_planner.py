"""
Template-based planner implementation.

This planner uses predefined templates to generate execution plans for common
development environment patterns like web applications, APIs, and databases.
"""

import json
from typing import Any, Dict, List, Optional

from .base import (
    Planner,
    PlannerConfig,
    PlannerError,
    PlanStep,
    PlanStructure,
    Requirements,
    RiskAssessment,
)


class TemplatePlanner(Planner):
    """
    Template-based planner for common development environment patterns.

    This planner maintains a library of templates for different technology stacks
    and uses them to generate concrete execution plans.
    """

    def __init__(self, config: PlannerConfig):
        super().__init__(config)
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    async def initialize(self) -> None:
        """Initialize the template planner."""
        await super().initialize()
        await self._load_builtin_templates()
        self._loaded = True

    async def _load_builtin_templates(self) -> None:
        """Load built-in templates for common patterns."""
        self._templates = {
            "nodejs_web_app": {
                "name": "Node.js Web Application",
                "description": "Node.js web application with npm dependencies",
                "framework": "nodejs",
                "steps": [
                    {
                        "id": "setup_directory",
                        "name": "Setup Project Directory",
                        "description": "Create and initialize project directory",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {"path": "{project_path}", "mode": "755"},
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
                    {
                        "id": "create_dockerfile",
                        "name": "Create Dockerfile",
                        "description": "Create Dockerfile for containerization",
                        "tool": "filesystem",
                        "action": "write_file",
                        "parameters": {
                            "path": "{project_path}/Dockerfile",
                            "content": 'FROM node:18-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nEXPOSE 3000\nCMD ["npm", "start"]',
                        },
                        "dependencies": ["create_package_json"],
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
                        "dependencies": ["create_dockerfile", "create_main_file"],
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
                            "ports": ["3000:3000"],
                            "detached": True,
                        },
                        "dependencies": ["build_docker_image"],
                        "estimated_duration": 30.0,
                        "estimated_cost": 0.05,
                    },
                ],
                "variables": ["project_name", "project_path", "description"],
                "risk_factors": ["docker_availability", "port_conflicts"],
                "estimated_total_duration": 210.0,
                "estimated_total_cost": 0.15,
            },
            "python_fastapi": {
                "name": "Python FastAPI Application",
                "description": "FastAPI web application with Python",
                "framework": "fastapi",
                "steps": [
                    {
                        "id": "setup_directory",
                        "name": "Setup Project Directory",
                        "description": "Create and initialize project directory",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {"path": "{project_path}", "mode": "755"},
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
                    {
                        "id": "create_dockerfile",
                        "name": "Create Dockerfile",
                        "description": "Create Dockerfile for containerization",
                        "tool": "filesystem",
                        "action": "write_file",
                        "parameters": {
                            "path": "{project_path}/Dockerfile",
                            "content": 'FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]',
                        },
                        "dependencies": ["create_requirements"],
                        "estimated_duration": 15.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "build_docker_image",
                        "name": "Build Docker Image",
                        "description": "Build Docker image for the FastAPI application",
                        "tool": "docker",
                        "action": "build_image",
                        "parameters": {
                            "context_path": "{project_path}",
                            "image_name": "{project_name}",
                            "tag": "latest",
                        },
                        "dependencies": ["create_dockerfile", "create_main_app"],
                        "estimated_duration": 180.0,
                        "estimated_cost": 0.15,
                    },
                    {
                        "id": "run_container",
                        "name": "Run FastAPI Container",
                        "description": "Run the containerized FastAPI application",
                        "tool": "docker",
                        "action": "run_container",
                        "parameters": {
                            "image": "{project_name}:latest",
                            "name": "{project_name}-api",
                            "ports": ["8000:8000"],
                            "detached": True,
                        },
                        "dependencies": ["build_docker_image"],
                        "estimated_duration": 30.0,
                        "estimated_cost": 0.05,
                    },
                ],
                "variables": ["project_name", "project_path", "description"],
                "risk_factors": [
                    "docker_availability",
                    "port_conflicts",
                    "python_version",
                ],
                "estimated_total_duration": 280.0,
                "estimated_total_cost": 0.20,
            },
            "postgresql_database": {
                "name": "PostgreSQL Database Environment",
                "description": "PostgreSQL database with configuration and schema setup",
                "framework": "postgresql",
                "steps": [
                    {
                        "id": "setup_database_directory",
                        "name": "Setup Database Directory",
                        "description": "Create directory for database configuration",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {
                            "path": "{project_path}/database",
                            "mode": "755",
                        },
                        "estimated_duration": 5.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "create_database",
                        "name": "Create PostgreSQL Database",
                        "description": "Create new PostgreSQL database",
                        "tool": "postgresql",
                        "action": "create_database",
                        "parameters": {"database_name": "{database_name}"},
                        "dependencies": ["setup_database_directory"],
                        "estimated_duration": 30.0,
                        "estimated_cost": 0.01,
                    },
                    {
                        "id": "create_schema_file",
                        "name": "Create Database Schema",
                        "description": "Create SQL schema file",
                        "tool": "filesystem",
                        "action": "write_file",
                        "parameters": {
                            "path": "{project_path}/database/schema.sql",
                            "content": "-- Database schema for {project_name}\n-- Created by Orcastrate\n\n-- Create users table\nCREATE TABLE IF NOT EXISTS users (\n    id SERIAL PRIMARY KEY,\n    username VARCHAR(50) UNIQUE NOT NULL,\n    email VARCHAR(100) UNIQUE NOT NULL,\n    password_hash VARCHAR(255) NOT NULL,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);\n\n-- Create indexes\nCREATE INDEX IF NOT EXISTS idx_users_username ON users(username);\nCREATE INDEX IF NOT EXISTS idx_users_email ON users(email);\n",
                        },
                        "dependencies": ["setup_database_directory"],
                        "estimated_duration": 15.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "execute_schema",
                        "name": "Execute Database Schema",
                        "description": "Execute schema SQL file",
                        "tool": "postgresql",
                        "action": "execute_query",
                        "parameters": {
                            "query": "-- Schema will be loaded from file",
                            "timeout": 60,
                        },
                        "dependencies": ["create_database", "create_schema_file"],
                        "estimated_duration": 45.0,
                        "estimated_cost": 0.02,
                    },
                ],
                "variables": ["project_name", "project_path", "database_name"],
                "risk_factors": ["postgresql_availability", "connection_issues"],
                "estimated_total_duration": 95.0,
                "estimated_total_cost": 0.03,
            },
            "mysql_database": {
                "name": "MySQL Database Environment",
                "description": "MySQL database with configuration and schema setup",
                "framework": "mysql",
                "steps": [
                    {
                        "id": "setup_database_directory",
                        "name": "Setup Database Directory",
                        "description": "Create directory for database configuration",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {
                            "path": "{project_path}/database",
                            "mode": "755",
                        },
                        "estimated_duration": 5.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "create_database",
                        "name": "Create MySQL Database",
                        "description": "Create new MySQL database",
                        "tool": "mysql",
                        "action": "create_database",
                        "parameters": {
                            "database_name": "{database_name}",
                            "charset": "utf8mb4",
                            "collation": "utf8mb4_unicode_ci",
                        },
                        "dependencies": ["setup_database_directory"],
                        "estimated_duration": 30.0,
                        "estimated_cost": 0.01,
                    },
                    {
                        "id": "create_schema_file",
                        "name": "Create Database Schema",
                        "description": "Create SQL schema file",
                        "tool": "filesystem",
                        "action": "write_file",
                        "parameters": {
                            "path": "{project_path}/database/schema.sql",
                            "content": "-- Database schema for {project_name}\n-- Created by Orcastrate\n\n-- Create users table\nCREATE TABLE IF NOT EXISTS users (\n    id INT AUTO_INCREMENT PRIMARY KEY,\n    username VARCHAR(50) UNIQUE NOT NULL,\n    email VARCHAR(100) UNIQUE NOT NULL,\n    password_hash VARCHAR(255) NOT NULL,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;\n\n-- Create indexes\nCREATE INDEX idx_users_username ON users(username);\nCREATE INDEX idx_users_email ON users(email);\n",
                        },
                        "dependencies": ["setup_database_directory"],
                        "estimated_duration": 15.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "create_table",
                        "name": "Create Users Table",
                        "description": "Create initial users table",
                        "tool": "mysql",
                        "action": "create_table",
                        "parameters": {
                            "table_name": "users",
                            "columns": [
                                {
                                    "name": "id",
                                    "type": "INT AUTO_INCREMENT PRIMARY KEY",
                                },
                                {
                                    "name": "username",
                                    "type": "VARCHAR(50)",
                                    "not_null": True,
                                    "unique": True,
                                },
                                {
                                    "name": "email",
                                    "type": "VARCHAR(100)",
                                    "not_null": True,
                                    "unique": True,
                                },
                                {
                                    "name": "password_hash",
                                    "type": "VARCHAR(255)",
                                    "not_null": True,
                                },
                                {
                                    "name": "created_at",
                                    "type": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                                },
                                {
                                    "name": "updated_at",
                                    "type": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
                                },
                            ],
                            "engine": "InnoDB",
                            "charset": "utf8mb4",
                        },
                        "dependencies": ["create_database"],
                        "estimated_duration": 20.0,
                        "estimated_cost": 0.01,
                    },
                    {
                        "id": "create_indexes",
                        "name": "Create Database Indexes",
                        "description": "Create performance indexes",
                        "tool": "mysql",
                        "action": "execute_query",
                        "parameters": {
                            "query": "CREATE INDEX idx_users_username ON users(username); CREATE INDEX idx_users_email ON users(email);",
                        },
                        "dependencies": ["create_table"],
                        "estimated_duration": 15.0,
                        "estimated_cost": 0.01,
                    },
                ],
                "variables": ["project_name", "project_path", "database_name"],
                "risk_factors": ["mysql_availability", "connection_issues"],
                "estimated_total_duration": 85.0,
                "estimated_total_cost": 0.03,
            },
            "redis_database": {
                "name": "Redis Cache Environment",
                "description": "Redis cache with configuration and key-value operations",
                "framework": "redis",
                "steps": [
                    {
                        "id": "setup_cache_directory",
                        "name": "Setup Cache Directory",
                        "description": "Create directory for cache configuration",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {
                            "path": "{project_path}/cache",
                            "mode": "755",
                        },
                        "estimated_duration": 5.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "create_redis_config",
                        "name": "Create Redis Configuration",
                        "description": "Create Redis configuration file",
                        "tool": "filesystem",
                        "action": "write_file",
                        "parameters": {
                            "path": "{project_path}/cache/redis.conf",
                            "content": "# Redis configuration for {project_name}\n# Created by Orcastrate\n\n# Basic settings\nport 6379\nbind 127.0.0.1\ntimeout 300\ntcp-keepalive 60\n\n# Memory management\nmaxmemory 256mb\nmaxmemory-policy allkeys-lru\n\n# Persistence\nsave 900 1\nsave 300 10\nsave 60 10000\n\n# Security\n# requirepass your_password_here\n\n# Logging\nloglevel notice\nlogfile /var/log/redis/redis-server.log\n",
                        },
                        "dependencies": ["setup_cache_directory"],
                        "estimated_duration": 15.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "test_redis_connection",
                        "name": "Test Redis Connection",
                        "description": "Test Redis connectivity and basic operations",
                        "tool": "redis",
                        "action": "set",
                        "parameters": {
                            "key": "test_key",
                            "value": "Hello from {project_name}!",
                            "ttl": 300,
                        },
                        "dependencies": ["create_redis_config"],
                        "estimated_duration": 10.0,
                        "estimated_cost": 0.001,
                    },
                    {
                        "id": "verify_redis_get",
                        "name": "Verify Redis Get Operation",
                        "description": "Verify that Redis can retrieve stored values",
                        "tool": "redis",
                        "action": "get",
                        "parameters": {
                            "key": "test_key",
                        },
                        "dependencies": ["test_redis_connection"],
                        "estimated_duration": 5.0,
                        "estimated_cost": 0.001,
                    },
                    {
                        "id": "create_cache_helpers",
                        "name": "Create Cache Helper Functions",
                        "description": "Create utility functions for cache operations",
                        "tool": "filesystem",
                        "action": "write_file",
                        "parameters": {
                            "path": "{project_path}/cache/cache_utils.py",
                            "content": '#!/usr/bin/env python3\n"""\nCache utilities for {project_name}\nGenerated by Orcastrate\n"""\n\nimport redis\nimport json\nfrom typing import Any, Optional\n\nclass CacheManager:\n    def __init__(self, host=\'localhost\', port=6379, db=0, password=None):\n        self.redis_client = redis.Redis(\n            host=host, \n            port=port, \n            db=db, \n            password=password,\n            decode_responses=True\n        )\n    \n    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:\n        """Set a value in cache with optional TTL."""\n        try:\n            if isinstance(value, (dict, list)):\n                value = json.dumps(value)\n            return self.redis_client.set(key, value, ex=ttl)\n        except Exception as e:\n            print(f"Cache set error: {e}")\n            return False\n    \n    def get(self, key: str) -> Any:\n        """Get a value from cache."""\n        try:\n            value = self.redis_client.get(key)\n            if value:\n                try:\n                    return json.loads(value)\n                except json.JSONDecodeError:\n                    return value\n            return None\n        except Exception as e:\n            print(f"Cache get error: {e}")\n            return None\n    \n    def delete(self, key: str) -> bool:\n        """Delete a key from cache."""\n        try:\n            return bool(self.redis_client.delete(key))\n        except Exception as e:\n            print(f"Cache delete error: {e}")\n            return False\n    \n    def exists(self, key: str) -> bool:\n        """Check if key exists in cache."""\n        try:\n            return bool(self.redis_client.exists(key))\n        except Exception as e:\n            print(f"Cache exists error: {e}")\n            return False\n',
                        },
                        "dependencies": ["setup_cache_directory"],
                        "estimated_duration": 20.0,
                        "estimated_cost": 0.0,
                    },
                ],
                "variables": ["project_name", "project_path"],
                "risk_factors": ["redis_availability", "connection_issues"],
                "estimated_total_duration": 55.0,
                "estimated_total_cost": 0.002,
            },
            "mongodb_database": {
                "name": "MongoDB Database Environment",
                "description": "MongoDB database with collections and indexes",
                "framework": "mongodb",
                "steps": [
                    {
                        "id": "setup_database_directory",
                        "name": "Setup Database Directory",
                        "description": "Create directory for database configuration",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {
                            "path": "{project_path}/database",
                            "mode": "755",
                        },
                        "estimated_duration": 5.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "create_collection",
                        "name": "Create MongoDB Collection",
                        "description": "Create main collection",
                        "tool": "mongodb",
                        "action": "create_collection",
                        "parameters": {"collection_name": "{collection_name}"},
                        "dependencies": ["setup_database_directory"],
                        "estimated_duration": 20.0,
                        "estimated_cost": 0.01,
                    },
                    {
                        "id": "create_indexes",
                        "name": "Create Database Indexes",
                        "description": "Create indexes for performance",
                        "tool": "mongodb",
                        "action": "create_index",
                        "parameters": {
                            "collection_name": "{collection_name}",
                            "index_spec": {"username": 1},
                            "unique": True,
                        },
                        "dependencies": ["create_collection"],
                        "estimated_duration": 30.0,
                        "estimated_cost": 0.02,
                    },
                    {
                        "id": "seed_data",
                        "name": "Insert Sample Data",
                        "description": "Insert sample documents",
                        "tool": "mongodb",
                        "action": "insert_document",
                        "parameters": {
                            "collection_name": "{collection_name}",
                            "document": {
                                "username": "admin",
                                "email": "admin@example.com",
                                "role": "administrator",
                                "created_at": "2024-01-01T00:00:00Z",
                            },
                        },
                        "dependencies": ["create_indexes"],
                        "estimated_duration": 15.0,
                        "estimated_cost": 0.01,
                    },
                ],
                "variables": ["project_name", "project_path", "collection_name"],
                "risk_factors": ["mongodb_availability", "connection_issues"],
                "estimated_total_duration": 70.0,
                "estimated_total_cost": 0.04,
            },
            "web_app_with_database": {
                "name": "Web Application with Database",
                "description": "Full-stack web application with database integration",
                "framework": "fullstack",
                "steps": [
                    {
                        "id": "setup_project",
                        "name": "Setup Project Structure",
                        "description": "Create project directory structure",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {"path": "{project_path}", "mode": "755"},
                        "estimated_duration": 5.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "setup_database",
                        "name": "Setup Database",
                        "description": "Create and configure database",
                        "tool": "{database_type}",
                        "action": "create_database",
                        "parameters": {"database_name": "{database_name}"},
                        "dependencies": ["setup_project"],
                        "estimated_duration": 30.0,
                        "estimated_cost": 0.01,
                    },
                    {
                        "id": "create_app_structure",
                        "name": "Create Application Structure",
                        "description": "Create web application files",
                        "tool": "filesystem",
                        "action": "create_directory",
                        "parameters": {"path": "{project_path}/src", "mode": "755"},
                        "dependencies": ["setup_project"],
                        "estimated_duration": 10.0,
                        "estimated_cost": 0.0,
                    },
                    {
                        "id": "create_config_file",
                        "name": "Create Database Configuration",
                        "description": "Create database connection configuration",
                        "tool": "filesystem",
                        "action": "write_file",
                        "parameters": {
                            "path": "{project_path}/config/database.json",
                            "content": {
                                "database": {
                                    "type": "{database_type}",
                                    "host": "localhost",
                                    "port": "{database_port}",
                                    "database": "{database_name}",
                                    "username": "{database_username}",
                                    "password": "{database_password}",
                                },
                            },
                        },
                        "dependencies": ["setup_database"],
                        "estimated_duration": 15.0,
                        "estimated_cost": 0.0,
                    },
                ],
                "variables": [
                    "project_name",
                    "project_path",
                    "database_type",
                    "database_name",
                    "database_port",
                    "database_username",
                    "database_password",
                ],
                "risk_factors": [
                    "database_availability",
                    "connection_issues",
                    "port_conflicts",
                ],
                "estimated_total_duration": 60.0,
                "estimated_total_cost": 0.01,
            },
        }

    async def _generate_initial_plan(self, context: Dict[str, Any]) -> PlanStructure:
        """Generate initial plan using templates."""
        requirements_data = context.get("requirements")
        if not requirements_data:
            raise PlannerError("Requirements not found in context")

        # Convert dict back to Requirements object if needed
        if isinstance(requirements_data, dict):
            from ..agent.base import Requirements

            requirements = Requirements(**requirements_data)
        else:
            requirements = requirements_data

        # Select template based on requirements
        template = await self._select_template(requirements)
        if not template:
            raise PlannerError("No suitable template found for requirements")

        # Generate plan steps from template
        steps = await self._generate_steps_from_template(template, requirements)

        return PlanStructure(
            steps=steps,
            metadata={
                "template_name": template["name"],
                "framework": template.get("framework"),
                "generated_at": context.get("timestamp"),
            },
        )

    async def _gather_context(self, requirements: Requirements) -> Dict[str, Any]:
        """Gather context for planning."""
        return {
            "requirements": requirements.model_dump(),
            "timestamp": "2023-01-01T00:00:00Z",  # In real implementation, use actual timestamp
            "available_templates": list(self._templates.keys()),
            "planner_type": "template_based",
        }

    async def _select_template(
        self, requirements: Requirements
    ) -> Optional[Dict[str, Any]]:
        """Select the best template based on requirements."""
        framework = requirements.framework
        database = requirements.database
        description = requirements.description.lower()

        # Database + Framework combination (highest priority)
        if database and framework:
            database_lower = database.lower()
            framework_lower = framework.lower()

            # Full-stack combinations
            if framework_lower in [
                "nodejs",
                "node",
                "javascript",
            ] and database_lower in [
                "postgresql",
                "postgres",
                "mysql",
                "mongodb",
                "mongo",
            ]:
                return self._templates.get("web_app_with_database")
            elif framework_lower in ["fastapi", "python", "api"] and database_lower in [
                "postgresql",
                "postgres",
                "mysql",
                "mongodb",
                "mongo",
            ]:
                return self._templates.get("web_app_with_database")

        # Database-first selection (second priority)
        if database:
            database_lower = database.lower()
            if database_lower in ["postgresql", "postgres"]:
                return self._templates.get("postgresql_database")
            elif database_lower in ["mysql"]:
                return self._templates.get("mysql_database")
            elif database_lower in ["mongodb", "mongo"]:
                return self._templates.get("mongodb_database")
            elif database_lower in ["redis"]:
                # Redis template will be available once implemented
                return self._templates.get("redis_database")

        # Framework-based selection (third priority)
        if framework:
            framework_lower = framework.lower()
            if framework_lower in ["nodejs", "node", "javascript"]:
                return self._templates.get("nodejs_web_app")
            elif framework_lower in ["fastapi", "python", "api"]:
                return self._templates.get("python_fastapi")
            elif framework_lower in ["postgresql", "postgres"]:
                return self._templates.get("postgresql_database")
            elif framework_lower in ["mongodb", "mongo"]:
                return self._templates.get("mongodb_database")
            elif framework_lower in ["fullstack", "web_app_db"]:
                return self._templates.get("web_app_with_database")

        # Description-based selection (fourth priority)
        if any(keyword in description for keyword in ["node", "express", "javascript"]):
            return self._templates.get("nodejs_web_app")
        elif any(keyword in description for keyword in ["fastapi", "python", "api"]):
            return self._templates.get("python_fastapi")
        elif any(
            keyword in description for keyword in ["postgresql", "postgres", "pg"]
        ):
            return self._templates.get("postgresql_database")
        elif any(keyword in description for keyword in ["mysql"]):
            return self._templates.get("mysql_database")
        elif any(keyword in description for keyword in ["mongodb", "mongo", "nosql"]):
            return self._templates.get("mongodb_database")
        elif any(keyword in description for keyword in ["redis", "cache"]):
            return self._templates.get("redis_database")
        elif any(keyword in description for keyword in ["database", "db"]) and any(
            keyword in description for keyword in ["web", "app", "application"]
        ):
            return self._templates.get("web_app_with_database")

        # Default to Node.js web app for web applications only
        if "web" in description and "app" in description:
            return self._templates.get("nodejs_web_app")

        return None

    async def _generate_steps_from_template(
        self, template: Dict[str, Any], requirements: Requirements
    ) -> List[PlanStep]:
        """Generate concrete plan steps from template."""
        steps = []
        variables = self._extract_variables(requirements)

        for step_template in template["steps"]:
            # Replace variables in step template
            step_data = self._replace_variables(step_template, variables)

            # Create PlanStep instance
            step = PlanStep(
                id=step_data["id"],
                name=step_data["name"],
                description=step_data["description"],
                tool=step_data["tool"],
                action=step_data["action"],
                parameters=step_data["parameters"],
                dependencies=step_data.get("dependencies", []),
                estimated_duration=step_data.get("estimated_duration", 60.0),
                estimated_cost=step_data.get("estimated_cost", 0.0),
                retry_count=3,
                timeout=300,
                metadata={"template_generated": True},
            )
            steps.append(step)

        return steps

    def _extract_variables(self, requirements: Requirements) -> Dict[str, str]:
        """Extract variables from requirements for template substitution."""
        # Generate project name from description
        project_name = self._generate_project_name(requirements.description)

        # Determine project path
        project_path = f"/tmp/orcastrate/{project_name}"

        # Base variables
        variables = {
            "project_name": project_name,
            "project_path": project_path,
            "description": requirements.description,
            "framework": requirements.framework or "unknown",
        }

        # Add database-specific variables
        framework = requirements.framework or ""
        description = requirements.description.lower()

        if "postgresql" in framework.lower() or "postgresql" in description:
            variables.update(
                {
                    "database_name": f"{project_name}_db",
                    "database_type": "postgresql",
                    "database_port": "5432",
                    "database_username": "postgres",
                    "database_password": "postgres",
                }
            )
        elif "mongodb" in framework.lower() or "mongodb" in description:
            variables.update(
                {
                    "collection_name": f"{project_name}_collection",
                    "database_type": "mongodb",
                    "database_port": "27017",
                    "database_username": "",
                    "database_password": "",
                }
            )
        elif "mysql" in framework.lower() or "mysql" in description:
            variables.update(
                {
                    "database_name": f"{project_name}_db",
                    "database_type": "mysql",
                    "database_port": "3306",
                    "database_username": "root",
                    "database_password": "mysql",
                }
            )
        elif "database" in description or "db" in description:
            # Default to PostgreSQL for generic database requirements
            variables.update(
                {
                    "database_name": f"{project_name}_db",
                    "database_type": "postgresql",
                    "database_port": "5432",
                    "database_username": "postgres",
                    "database_password": "postgres",
                }
            )

        return variables

    def _generate_project_name(self, description: str) -> str:
        """Generate a project name from description."""
        # Simple implementation - clean description and make it a valid name
        name = description.lower()
        # Remove common words and clean up
        words = name.split()
        filtered_words = [
            w for w in words if w not in ["a", "an", "the", "with", "for", "using"]
        ]

        if filtered_words:
            # Take first few meaningful words
            name_parts = filtered_words[:3]
            project_name = "-".join(name_parts)
            # Clean non-alphanumeric characters except hyphens
            project_name = "".join(
                c if c.isalnum() or c == "-" else "" for c in project_name
            )
            # Remove consecutive hyphens
            while "--" in project_name:
                project_name = project_name.replace("--", "-")
            return project_name.strip("-") or "app"

        return "app"

    def _replace_variables(
        self, template_data: Dict[str, Any], variables: Dict[str, str]
    ) -> Dict[str, Any]:
        """Replace variable placeholders in template data."""
        # Convert to JSON string, replace variables, then parse back
        json_str = json.dumps(template_data)

        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            json_str = json_str.replace(placeholder, var_value)

        data: Dict[str, Any] = json.loads(json_str)
        return data

    async def _assess_risks(self, plan: PlanStructure) -> RiskAssessment:
        """Assess risks for the plan."""
        risk_factors = []
        mitigation_strategies = []
        overall_risk = 0.2  # Default low risk for template-based plans

        # Check for Docker dependencies
        docker_steps = [step for step in plan.steps if step.tool == "docker"]
        if docker_steps:
            risk_factors.append("Docker availability required")
            mitigation_strategies.append("Verify Docker is installed and running")
            overall_risk += 0.1

        # Check for file system operations
        file_steps = [step for step in plan.steps if step.tool == "filesystem"]
        if file_steps:
            risk_factors.append("File system write permissions required")
            mitigation_strategies.append("Ensure write permissions to target directory")
            overall_risk += 0.05

        # Check for port conflicts
        container_steps = [
            step for step in plan.steps if step.action == "run_container"
        ]
        if container_steps:
            risk_factors.append("Potential port conflicts")
            mitigation_strategies.append(
                "Check port availability before container startup"
            )
            overall_risk += 0.1

        return RiskAssessment(
            overall_risk=min(overall_risk, 1.0),
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            failure_probability=overall_risk * 0.3,
            recovery_time=300.0,  # 5 minutes
            impact_assessment={
                "cost": "low",
                "timeline": "low",
                "complexity": "medium",
            },
        )

    async def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available templates."""
        if not self._loaded:
            await self.initialize()

        return [
            {
                "id": template_id,
                "name": template["name"],
                "description": template["description"],
                "framework": template.get("framework"),
                "estimated_duration": template.get("estimated_total_duration", 0),
                "estimated_cost": template.get("estimated_total_cost", 0),
            }
            for template_id, template in self._templates.items()
        ]

    def add_custom_template(
        self, template_id: str, template_data: Dict[str, Any]
    ) -> None:
        """Add a custom template to the planner."""
        # Validate template structure
        required_fields = ["name", "description", "steps"]
        for field in required_fields:
            if field not in template_data:
                raise PlannerError(f"Template missing required field: {field}")

        self._templates[template_id] = template_data
        self.logger.info(f"Added custom template: {template_id}")
