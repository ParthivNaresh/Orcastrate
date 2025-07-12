"""
Prompt template management for intelligent planning.

This module provides structured prompt templates and template management
for consistent LLM interactions across different planning operations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import PlanningPrompt


class PromptType(Enum):
    """Types of planning prompts."""

    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    PLAN_GENERATION = "plan_generation"
    PLAN_OPTIMIZATION = "plan_optimization"
    COST_ESTIMATION = "cost_estimation"
    RISK_ASSESSMENT = "risk_assessment"
    PLAN_VALIDATION = "plan_validation"
    PLAN_EXPLANATION = "plan_explanation"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    SECURITY_REVIEW = "security_review"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    TECHNOLOGY_RECOMMENDATION = "technology_recommendation"
    ARCHITECTURE_REVIEW = "architecture_review"
    GENERAL_PLANNING = "general_planning"


@dataclass
class TemplateConfig:
    """Configuration for prompt templates."""

    include_examples: bool = True
    include_constraints: bool = True
    include_context: bool = True
    output_format: str = "json"
    temperature: float = 0.7
    max_tokens: int = 4000
    custom_instructions: str = ""


class PromptTemplateManager:
    """Manages prompt templates for LLM-based planning operations."""

    def __init__(self):
        self._templates = {}
        self._examples = {}
        self._constraints = {}
        self._initialize_templates()
        self._initialize_examples()
        self._initialize_constraints()

    def get_prompt(
        self,
        prompt_type: PromptType,
        context: Dict[str, Any],
        config: Optional[TemplateConfig] = None,
    ) -> PlanningPrompt:
        """Generate a prompt for the specified type and context."""
        if config is None:
            config = TemplateConfig()

        template = self._templates.get(prompt_type)
        if not template:
            raise ValueError(f"No template found for prompt type: {prompt_type}")

        # Create a safe context with defaults for missing variables
        safe_context = self._create_safe_context(context, prompt_type)

        # Build system and user prompts
        system_prompt = template["system"].format(**safe_context)
        user_prompt = template["user"].format(**safe_context)

        # Add custom instructions if provided
        if config.custom_instructions:
            system_prompt += (
                f"\n\nAdditional Instructions:\n{config.custom_instructions}"
            )

        # Add examples if requested
        examples = []
        if config.include_examples and prompt_type in self._examples:
            examples = self._examples[prompt_type]
            # Also mention examples in the prompt text
            if examples:
                user_prompt += "\n\nExample interactions are provided to guide your response format."

        # Add constraints if requested
        constraints = []
        if config.include_constraints and prompt_type in self._constraints:
            constraints = self._constraints[prompt_type]
            # Also mention constraints in the prompt text
            if constraints:
                user_prompt += "\n\nPlease follow these constraints:\n" + "\n".join(
                    f"- {c}" for c in constraints
                )

        return PlanningPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            examples=examples,
            constraints=constraints,
        )

    def _create_safe_context(
        self, context: Dict[str, Any], prompt_type: PromptType
    ) -> Dict[str, Any]:
        """Create a safe context with default values for missing variables."""
        safe_context = context.copy()

        # Define default values for different prompt types
        defaults = {
            PromptType.REQUIREMENTS_ANALYSIS: {
                "requirements": safe_context.get(
                    "requirements", "Build a web application"
                ),
                "additional_context": safe_context.get(
                    "additional_context",
                    (
                        f"Known technologies: {safe_context.get('technologies', [])}"
                        if safe_context.get("technologies")
                        else ""
                    ),
                ),
            },
            PromptType.PLAN_GENERATION: {
                "available_tools": safe_context.get(
                    "available_tools", ["aws", "docker", "postgresql"]
                ),
                "requirements_description": safe_context.get(
                    "requirements_description",
                    safe_context.get("requirements", "Build a web application"),
                ),
                "analysis_summary": safe_context.get(
                    "analysis_summary",
                    f"Technologies: {safe_context.get('technologies', [])}, Budget: {safe_context.get('budget', {})}",
                ),
                "constraints_summary": safe_context.get(
                    "constraints_summary",
                    f"Timeline: {safe_context.get('timeline', 'flexible')}, Team size: {safe_context.get('team_size', 'small')}",
                ),
            },
            PromptType.PLAN_OPTIMIZATION: {
                "optimization_goals": safe_context.get(
                    "optimization_goals", ["cost", "performance"]
                ),
                "constraints": safe_context.get("constraints", []),
                "optimization_priority": safe_context.get(
                    "optimization_priority", "cost"
                ),
                "original_plan": safe_context.get("original_plan", {}),
            },
            PromptType.COST_ESTIMATION: {
                "plan": safe_context.get("plan", safe_context.get("plan_steps", [])),
                "region": safe_context.get("region", "us-east-1"),
                "usage_pattern": safe_context.get("usage_pattern", "development"),
                "team_size": safe_context.get("team_size", 1),
                "environment_type": safe_context.get("environment_type", "development"),
            },
            PromptType.RISK_ASSESSMENT: {
                "plan": safe_context.get(
                    "plan",
                    {
                        "complexity": safe_context.get("complexity", 5.0),
                        "identified_patterns": safe_context.get(
                            "identified_patterns", []
                        ),
                    },
                ),
                "industry": safe_context.get("industry", "technology"),
                "compliance_requirements": safe_context.get(
                    "compliance_requirements", []
                ),
                "data_sensitivity": safe_context.get("data_sensitivity", "medium"),
                "team_experience": safe_context.get("team_experience", "intermediate"),
            },
            PromptType.PLAN_VALIDATION: {
                "requirements": safe_context.get(
                    "requirements", "Build a web application"
                ),
                "plan": safe_context.get("plan", {}),
            },
            PromptType.PLAN_EXPLANATION: {
                "plan": safe_context.get("plan", {}),
                "audience_level": safe_context.get("audience_level", "technical"),
            },
            PromptType.DEPENDENCY_ANALYSIS: {
                "plan": safe_context.get("plan", {}),
            },
            PromptType.SECURITY_REVIEW: {
                "plan": safe_context.get("plan", {}),
                "security_requirements": safe_context.get("security_requirements", []),
            },
            PromptType.PERFORMANCE_ANALYSIS: {
                "plan": safe_context.get("plan", {}),
                "performance_requirements": safe_context.get(
                    "performance_requirements", []
                ),
            },
            PromptType.TECHNOLOGY_RECOMMENDATION: {
                "project_description": safe_context.get(
                    "project_description",
                    safe_context.get("requirements", "Build a web application"),
                ),
                "requirements": safe_context.get(
                    "requirements", "Build a web application"
                ),
                "team_size": safe_context.get("team_size", 1),
                "experience_level": safe_context.get(
                    "experience_level", "intermediate"
                ),
                "existing_expertise": safe_context.get("existing_expertise", []),
                "constraints": safe_context.get("constraints", []),
            },
            PromptType.ARCHITECTURE_REVIEW: {
                "architecture_description": safe_context.get(
                    "architecture_description",
                    f"{safe_context.get('requirements', 'Web application architecture')} - Pattern: {safe_context.get('architecture_pattern', 'N/A')}",
                ),
                "current_implementation": safe_context.get(
                    "current_implementation",
                    f"Initial implementation with {safe_context.get('technology_stack', {})}",
                ),
                "project_scale": safe_context.get("project_scale", "medium"),
                "performance_requirements": safe_context.get(
                    "performance_requirements", []
                ),
                "security_requirements": safe_context.get("security_requirements", []),
                "team_structure": safe_context.get("team_structure", "small team"),
            },
            PromptType.GENERAL_PLANNING: {
                "project_description": safe_context.get(
                    "project_description",
                    safe_context.get("requirements", "Build a web application"),
                ),
                "requirements": safe_context.get(
                    "requirements", "Build a web application"
                ),
                "timeline": safe_context.get("timeline", "3 months"),
                "budget": safe_context.get("budget", "$5000"),
                "team_description": safe_context.get(
                    "team_description", "small development team"
                ),
                "technology_preferences": safe_context.get(
                    "technology_preferences", "modern web technologies"
                ),
            },
        }

        # Apply defaults for the specific prompt type
        if prompt_type in defaults:
            for key, value in defaults[prompt_type].items():
                safe_context.setdefault(key, value)

        return safe_context

    def _initialize_templates(self):
        """Initialize all prompt templates."""
        self._templates = {
            PromptType.REQUIREMENTS_ANALYSIS: {
                "system": """You are an expert DevOps architect analyzing user requirements for development environment setup.

Analyze the following requirements and extract:
1. Technologies mentioned (frameworks, databases, cloud providers, etc.)
2. Architecture pattern (microservices, monolithic, serverless, etc.)
3. Complexity score (1-10, where 10 is most complex)
4. Estimated duration in hours
5. Constraints (budget, security, performance, timeline)
6. Security requirements
7. Performance requirements
8. Scalability requirements

Be thorough but concise. Focus on actionable insights that will guide plan generation.
Return your analysis in JSON format with the specified structure.""",
                "user": """Requirements to analyze:
{requirements}

{additional_context}

Please provide a detailed analysis in JSON format with the following structure:
{{
    "technologies": ["list", "of", "technologies"],
    "architecture_pattern": "pattern_name",
    "complexity_score": 5.5,
    "estimated_duration": 40.0,
    "identified_constraints": ["constraint1", "constraint2"],
    "security_requirements": ["requirement1", "requirement2"],
    "performance_requirements": ["requirement1", "requirement2"],
    "scalability_requirements": ["requirement1", "requirement2"],
    "budget_constraints": {{"max_monthly_cost": 500.0}},
    "confidence_score": 0.85
}}""",
            },
            PromptType.PLAN_GENERATION: {
                "system": """You are an expert DevOps architect creating detailed execution plans for development environment setup.

Available tools: {available_tools}

Create a step-by-step execution plan that:
1. Uses only available tools
2. Follows best practices for security and performance
3. Optimizes for cost efficiency
4. Handles dependencies correctly
5. Includes proper error handling and rollback strategies
6. Considers scalability and maintainability
7. Implements monitoring and logging

Your plan should be production-ready and follow DevOps best practices.
Return the plan in JSON format with clear step definitions, dependencies, and estimated costs/durations.""",
                "user": """Requirements:
{requirements_description}

{analysis_summary}

{constraints_summary}

Please create a detailed execution plan in JSON format with this structure:
{{
    "steps": [
        {{
            "id": "step_1",
            "name": "Step Name",
            "description": "Detailed description",
            "tool": "tool_name",
            "action": "action_name",
            "parameters": {{}},
            "dependencies": [],
            "estimated_duration": 300.0,
            "estimated_cost": 50.0,
            "rollback_strategy": "Description of rollback",
            "validation_criteria": ["criterion1", "criterion2"]
        }}
    ],
    "dependencies": {{"step_2": ["step_1"]}},
    "estimated_cost": 225.0,
    "estimated_duration": 1300.0,
    "risk_assessment": {{
        "overall_risk": 0.3,
        "risk_factors": ["Network configuration"],
        "mitigation_strategies": ["Use VPN", "Backup configs"]
    }},
    "monitoring": {{
        "metrics": ["cpu_usage", "memory_usage"],
        "alerts": ["high_cost", "deployment_failure"]
    }}
}}""",
            },
            PromptType.PLAN_OPTIMIZATION: {
                "system": """You are an expert DevOps architect optimizing execution plans for development environments.

Optimization goals: {optimization_goals}
Constraints: {constraints}

Optimize the given plan by:
1. Reducing costs where possible without compromising functionality
2. Improving performance and reliability
3. Enhancing security posture
4. Minimizing execution time
5. Following industry best practices
6. Improving resource utilization
7. Enhancing maintainability

Focus on {optimization_priority} optimization.
Return the optimized plan in the same JSON format, highlighting the changes made.""",
                "user": """Original plan to optimize:
{original_plan}

Please provide an optimized version of this plan that addresses the specified goals while respecting constraints.

Include:
1. The optimized plan in JSON format
2. A summary of changes made
3. Expected improvements (cost savings, performance gains, etc.)
4. Any trade-offs or considerations""",
            },
            PromptType.COST_ESTIMATION: {
                "system": """You are a cloud cost optimization expert analyzing development environment costs.

Provide detailed cost estimates for the given plan considering:
1. Resource costs (compute, storage, network)
2. Service costs (managed services, APIs)
3. Data transfer costs
4. Operational overhead
5. Scaling costs
6. Hidden costs and gotchas

Use current market rates and provide estimates in USD.
Consider both initial setup costs and ongoing monthly costs.""",
                "user": """Plan to estimate costs for:
{plan}

Additional context:
- Target region: {region}
- Expected usage: {usage_pattern}
- Team size: {team_size}
- Environment type: {environment_type}

Please provide detailed cost estimates in JSON format:
{{
    "initial_setup_cost": 500.0,
    "monthly_recurring_cost": 200.0,
    "cost_breakdown": {{
        "compute": 150.0,
        "storage": 30.0,
        "network": 20.0
    }},
    "scaling_projections": {{
        "low_usage": 180.0,
        "medium_usage": 200.0,
        "high_usage": 350.0
    }},
    "cost_optimization_suggestions": [
        "Use spot instances for non-critical workloads",
        "Implement auto-scaling policies"
    ]
}}""",
            },
            PromptType.RISK_ASSESSMENT: {
                "system": """You are a DevOps security and risk assessment expert evaluating deployment plans.

Assess risks in the following categories:
1. Security risks (data exposure, access control, vulnerabilities)
2. Operational risks (downtime, performance, reliability)
3. Financial risks (cost overruns, resource waste)
4. Compliance risks (regulatory, policy violations)
5. Technical risks (compatibility, scalability, maintainability)

Provide specific, actionable risk assessments with mitigation strategies.
Rate risks on a scale of 1-10 where 10 is highest risk.""",
                "user": """Plan to assess for risks:
{plan}

Context:
- Industry: {industry}
- Compliance requirements: {compliance_requirements}
- Data sensitivity: {data_sensitivity}
- Team experience: {team_experience}

Please provide risk assessment in JSON format:
{{
    "overall_risk_score": 4.2,
    "risk_categories": {{
        "security": {{
            "score": 5.0,
            "risks": ["Exposed database", "Weak authentication"],
            "mitigations": ["Enable encryption", "Add MFA"]
        }},
        "operational": {{
            "score": 3.0,
            "risks": ["Single point of failure"],
            "mitigations": ["Add redundancy"]
        }}
    }},
    "critical_risks": ["risk1", "risk2"],
    "recommended_actions": ["action1", "action2"]
}}""",
            },
            PromptType.PLAN_VALIDATION: {
                "system": """You are an expert DevOps architect validating execution plans against user requirements.

Validate the plan by checking:
1. Does it meet all stated requirements?
2. Are the technologies appropriate for the use case?
3. Is the architecture sound and scalable?
4. Are security best practices followed?
5. Is the cost estimation reasonable?
6. Are there any missing components?
7. Are dependencies correctly specified?
8. Is the timeline realistic?

Provide specific, constructive feedback with actionable suggestions.
Return validation results in JSON format with detailed feedback.""",
                "user": """Original requirements:
{requirements}

Plan to validate:
{plan}

Please validate this plan and provide feedback in JSON format:
{{
    "is_valid": true,
    "completeness_score": 0.95,
    "requirement_coverage": 0.90,
    "issues": [],
    "suggestions": [],
    "missing_components": [],
    "estimated_success_probability": 0.90,
    "validation_details": {{
        "requirements_met": ["req1", "req2"],
        "requirements_missing": [],
        "architecture_soundness": 0.85,
        "security_compliance": 0.92
    }}
}}""",
            },
            PromptType.PLAN_EXPLANATION: {
                "system": """You are an expert DevOps architect explaining technical implementation plans to users.

Provide a clear, accessible explanation of the execution plan that:
1. Explains what will be created/configured
2. Describes the purpose of each major component
3. Highlights important considerations and dependencies
4. Estimates timeline and costs in user-friendly terms
5. Mentions any risks or important considerations
6. Explains the benefits and value delivered

Write in a friendly, professional tone suitable for both technical and non-technical audiences.
Use analogies and examples where helpful.""",
                "user": """Plan to explain:
{plan}

Target audience: {audience_level}

Please provide a clear, comprehensive explanation of this execution plan that helps the user understand:
- What will be built
- Why each component is needed
- How everything works together
- What to expect during execution
- What the final result will look like""",
            },
            PromptType.DEPENDENCY_ANALYSIS: {
                "system": """You are a systems architect analyzing dependencies in development environment plans.

Analyze the plan for:
1. Direct dependencies between steps
2. Implicit dependencies and ordering requirements
3. Circular dependencies
4. Critical path analysis
5. Parallel execution opportunities
6. Dependency conflicts
7. External service dependencies

Provide optimization recommendations for dependency management.
Focus on execution efficiency and risk reduction.""",
                "user": """Plan to analyze dependencies for:
{plan}

Please provide dependency analysis in JSON format:
{{
    "dependency_graph": {{}},
    "critical_path": ["step1", "step2"],
    "parallel_groups": [["step3", "step4"]],
    "circular_dependencies": [],
    "external_dependencies": ["aws", "github"],
    "optimization_opportunities": ["Parallelize database setup"],
    "risk_factors": ["External API dependency"]
}}""",
            },
            PromptType.SECURITY_REVIEW: {
                "system": """You are a cybersecurity expert reviewing development environment plans for security vulnerabilities and compliance.

Review the plan for:
1. Authentication and authorization mechanisms
2. Data encryption (in transit and at rest)
3. Network security and segmentation
4. Access controls and least privilege
5. Secrets management
6. Logging and monitoring for security
7. Compliance with security frameworks
8. Vulnerability management

Provide specific security recommendations and identify potential vulnerabilities.
Rate security posture on a scale of 1-10.""",
                "user": """Plan to review for security:
{plan}

Security requirements:
{security_requirements}

Please provide security review in JSON format:
{{
    "security_score": 7.5,
    "vulnerabilities": [
        {{
            "severity": "high",
            "description": "Database exposed to internet",
            "remediation": "Add VPC and security groups"
        }}
    ],
    "compliance_status": {{
        "gdpr": "compliant",
        "sox": "needs_review"
    }},
    "recommendations": ["Enable MFA", "Use secrets manager"],
    "security_controls": ["encryption", "access_logs"]
}}""",
            },
            PromptType.PERFORMANCE_ANALYSIS: {
                "system": """You are a performance engineering expert analyzing development environment plans for performance optimization.

Analyze the plan for:
1. Resource allocation and sizing
2. Performance bottlenecks
3. Scalability considerations
4. Caching strategies
5. Database optimization
6. Network performance
7. Monitoring and observability
8. Load testing requirements

Provide specific performance optimization recommendations.
Estimate performance characteristics and scaling limits.""",
                "user": """Plan to analyze for performance:
{plan}

Performance requirements:
{performance_requirements}

Please provide performance analysis in JSON format:
{{
    "performance_score": 8.0,
    "bottlenecks": ["Database queries", "Network latency"],
    "optimization_recommendations": [
        "Add Redis caching",
        "Use CDN for static assets"
    ],
    "resource_sizing": {{
        "cpu": "appropriate",
        "memory": "oversized",
        "storage": "undersized"
    }},
    "scalability_assessment": {{
        "current_capacity": "100 concurrent users",
        "scaling_strategy": "horizontal",
        "breaking_points": ["Database connections at 1000 users"]
    }}
}}""",
            },
            PromptType.TECHNOLOGY_RECOMMENDATION: {
                "system": """You are a technology advisory expert helping teams choose the best technologies for their development environments.

Provide technology recommendations based on:
1. Project requirements and constraints
2. Team expertise and learning curve
3. Community support and ecosystem maturity
4. Performance and scalability characteristics
5. Cost considerations (licensing, infrastructure)
6. Integration capabilities
7. Long-term maintenance and support
8. Industry standards and best practices

Focus on practical, proven technologies that align with the project goals.
Consider both current needs and future growth requirements.""",
                "user": """Project context:
{project_description}

Current requirements:
{requirements}

Team context:
- Team size: {team_size}
- Experience level: {experience_level}
- Existing expertise: {existing_expertise}

Constraints:
{constraints}

Please provide technology recommendations in JSON format:
{{
    "recommended_technologies": {{
        "frontend": {{
            "primary": "React",
            "rationale": "Good team expertise and ecosystem",
            "alternatives": ["Vue.js", "Angular"]
        }},
        "backend": {{
            "primary": "Node.js",
            "rationale": "JavaScript consistency with frontend",
            "alternatives": ["Python", "Java"]
        }}
    }},
    "technology_stack": {{
        "languages": ["JavaScript", "TypeScript"],
        "frameworks": ["React", "Express.js"],
        "databases": ["PostgreSQL", "Redis"],
        "infrastructure": ["AWS", "Docker"]
    }},
    "implementation_roadmap": [
        "Phase 1: Core backend API",
        "Phase 2: Frontend development",
        "Phase 3: Database optimization"
    ],
    "learning_requirements": {{
        "new_technologies": ["Docker"],
        "estimated_learning_time": "2 weeks",
        "training_resources": ["Official docs", "Online courses"]
    }},
    "risk_assessment": {{
        "technology_risks": ["Docker learning curve"],
        "mitigation_strategies": ["Team training", "Gradual adoption"]
    }}
}}""",
            },
            PromptType.ARCHITECTURE_REVIEW: {
                "system": """You are a senior software architect reviewing development environment architectures for best practices and optimization opportunities.

Review the architecture for:
1. Design patterns and architectural principles
2. Scalability and performance considerations
3. Security architecture and data protection
4. Maintainability and code organization
5. Technology choices and integration patterns
6. Deployment and infrastructure design
7. Monitoring and observability
8. Cost optimization opportunities

Provide constructive feedback with specific recommendations.
Focus on long-term sustainability and industry best practices.""",
                "user": """Architecture to review:
{architecture_description}

Current implementation:
{current_implementation}

Context:
- Project scale: {project_scale}
- Performance requirements: {performance_requirements}
- Security requirements: {security_requirements}
- Team structure: {team_structure}

Please provide architecture review in JSON format:
{{
    "overall_assessment": {{
        "architecture_score": 7.5,
        "strengths": ["Well-defined API boundaries", "Good separation of concerns"],
        "weaknesses": ["Lack of caching strategy", "No disaster recovery plan"]
    }},
    "design_patterns": {{
        "current_patterns": ["MVC", "Repository"],
        "recommended_patterns": ["CQRS", "Event Sourcing"],
        "anti_patterns_found": ["God object in main controller"]
    }},
    "scalability_analysis": {{
        "current_capacity": "1000 concurrent users",
        "bottlenecks": ["Database queries", "File uploads"],
        "scaling_recommendations": ["Add read replicas", "Implement CDN"]
    }},
    "security_review": {{
        "security_score": 8.0,
        "vulnerabilities": ["Weak password policy"],
        "recommendations": ["Implement OAuth2", "Add rate limiting"]
    }},
    "technology_assessment": {{
        "appropriate_choices": ["PostgreSQL for data integrity"],
        "questionable_choices": ["Monolithic structure for this scale"],
        "upgrade_recommendations": ["Migrate to microservices"]
    }},
    "action_items": [
        "Implement caching layer",
        "Add comprehensive monitoring",
        "Create disaster recovery plan"
    ]
}}""",
            },
            PromptType.GENERAL_PLANNING: {
                "system": """You are an expert project planning consultant helping teams create comprehensive development environment plans.

Create detailed plans that address:
1. Project scope and objectives
2. Timeline and milestone planning
3. Resource allocation and team structure
4. Technology stack and infrastructure
5. Risk management and mitigation
6. Quality assurance and testing
7. Deployment and go-live strategy
8. Maintenance and support planning

Focus on creating realistic, achievable plans with clear deliverables.
Consider both technical and business requirements.""",
                "user": """Project to plan:
{project_description}

Requirements:
{requirements}

Constraints:
- Timeline: {timeline}
- Budget: {budget}
- Team: {team_description}
- Technology preferences: {technology_preferences}

Please provide a comprehensive plan in JSON format:
{{
    "project_overview": {{
        "scope": "Development environment for e-commerce platform",
        "objectives": ["Create scalable infrastructure", "Ensure security"],
        "success_criteria": ["Handle 10k users", "99.9% uptime"]
    }},
    "timeline": {{
        "total_duration": "12 weeks",
        "phases": [
            {{
                "name": "Planning & Setup",
                "duration": "2 weeks",
                "deliverables": ["Architecture design", "Tool selection"]
            }},
            {{
                "name": "Infrastructure",
                "duration": "4 weeks",
                "deliverables": ["VPC setup", "Database deployment"]
            }}
        ],
        "milestones": [
            {{
                "name": "MVP Release",
                "date": "Week 8",
                "criteria": ["Basic functionality working"]
            }}
        ]
    }},
    "resource_plan": {{
        "team_structure": {{
            "developers": 3,
            "devops": 1,
            "qa": 1
        }},
        "skill_requirements": ["AWS", "Docker", "PostgreSQL"],
        "training_needs": ["Kubernetes basics"]
    }},
    "technology_stack": {{
        "frontend": ["React", "TypeScript"],
        "backend": ["Node.js", "Express"],
        "database": ["PostgreSQL", "Redis"],
        "infrastructure": ["AWS", "Docker", "Kubernetes"]
    }},
    "risk_management": {{
        "identified_risks": [
            {{
                "risk": "Database performance",
                "probability": "medium",
                "impact": "high",
                "mitigation": "Performance testing early"
            }}
        ],
        "contingency_plans": ["Cloud provider backup"]
    }},
    "quality_assurance": {{
        "testing_strategy": ["Unit tests", "Integration tests"],
        "code_review_process": "Pull request reviews",
        "deployment_gates": ["All tests pass", "Security scan clean"]
    }}
}}""",
            },
        }

    def _initialize_examples(self):
        """Initialize example conversations for few-shot learning."""
        self._examples = {
            PromptType.REQUIREMENTS_ANALYSIS: [
                {
                    "user": "I need a Python web app with PostgreSQL and Redis, deployed on AWS",
                    "assistant": """{{
    "technologies": ["Python", "PostgreSQL", "Redis", "AWS"],
    "architecture_pattern": "three-tier",
    "complexity_score": 6.0,
    "estimated_duration": 24.0,
    "identified_constraints": ["AWS only"],
    "security_requirements": ["Database encryption", "HTTPS"],
    "performance_requirements": ["Sub-second response time"],
    "scalability_requirements": ["Auto-scaling"],
    "budget_constraints": {{"max_monthly_cost": 300.0}},
    "confidence_score": 0.9
}}""",
                }
            ],
            PromptType.PLAN_GENERATION: [
                {
                    "user": "Simple Node.js API with MongoDB",
                    "assistant": """{{
    "steps": [
        {{
            "id": "vpc_setup",
            "name": "Create VPC",
            "description": "Set up isolated network",
            "tool": "aws",
            "action": "create_vpc",
            "parameters": {{"cidr": "10.0.0.0/16"}},
            "dependencies": [],
            "estimated_duration": 300.0,
            "estimated_cost": 0.0
        }}
    ],
    "estimated_cost": 150.0,
    "estimated_duration": 1800.0,
    "risk_assessment": {{
        "overall_risk": 0.2,
        "risk_factors": ["MongoDB configuration"]
    }}
}}""",
                }
            ],
        }

    def _initialize_constraints(self):
        """Initialize common constraints for different prompt types."""
        self._constraints = {
            PromptType.REQUIREMENTS_ANALYSIS: [
                "Analyze only the explicit requirements provided",
                "Do not make assumptions about unstated requirements",
                "Provide confidence scores based on clarity of requirements",
            ],
            PromptType.PLAN_GENERATION: [
                "Use only the specified available tools",
                "Include proper error handling and rollback strategies",
                "Optimize for cost efficiency while meeting requirements",
                "Follow security best practices",
            ],
            PromptType.PLAN_OPTIMIZATION: [
                "Maintain the original functionality",
                "Justify all optimization changes",
                "Consider long-term maintainability",
                "Respect budget and timeline constraints",
            ],
            PromptType.COST_ESTIMATION: [
                "Use current market rates",
                "Include all relevant cost components",
                "Provide estimates in USD",
                "Consider both setup and operational costs",
            ],
            PromptType.RISK_ASSESSMENT: [
                "Focus on actionable risks",
                "Provide specific mitigation strategies",
                "Consider probability and impact",
                "Prioritize critical risks",
            ],
        }

    def add_custom_template(
        self, prompt_type: PromptType, system_template: str, user_template: str
    ):
        """Add or update a custom template."""
        self._templates[prompt_type] = {
            "system": system_template,
            "user": user_template,
        }

    def add_examples(self, prompt_type: PromptType, examples: List[Dict[str, str]]):
        """Add examples for a prompt type."""
        if prompt_type not in self._examples:
            self._examples[prompt_type] = []
        self._examples[prompt_type].extend(examples)

    def add_constraints(self, prompt_type: PromptType, constraints: List[str]):
        """Add constraints for a prompt type."""
        if prompt_type not in self._constraints:
            self._constraints[prompt_type] = []
        self._constraints[prompt_type].extend(constraints)

    def get_available_prompt_types(self) -> List[PromptType]:
        """Get list of available prompt types."""
        return list(self._templates.keys())

    def validate_template(
        self, prompt_type: PromptType, context: Dict[str, Any]
    ) -> bool:
        """Validate that a template can be rendered with the given context."""
        try:
            template = self._templates.get(prompt_type)
            if not template:
                return False

            # Try to format both system and user templates
            template["system"].format(**context)
            template["user"].format(**context)
            return True
        except (KeyError, ValueError):
            return False
