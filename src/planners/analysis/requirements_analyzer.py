"""
Requirements analysis engine for intelligent planning.

This module provides sophisticated analysis of user requirements using
both rule-based and LLM-powered techniques.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..llm.base import LLMClient, RequirementsAnalysis
from ..llm.prompt_templates import PromptTemplateManager
from .constraint_extractor import ConstraintExtractor, ConstraintType
from .technology_detector import TechnologyDetector, TechnologyMatch


class RequirementType(Enum):
    """Types of requirements that can be detected."""

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    TECHNICAL = "technical"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    COMPLIANCE = "compliance"


class AnalysisPriority(Enum):
    """Priority levels for analysis results."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Requirement:
    """Individual requirement extracted from user input."""

    id: str
    type: RequirementType
    priority: AnalysisPriority
    description: str
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source_text: str = ""
    implications: List[str] = field(default_factory=list)


class AnalysisResult(BaseModel):
    """Complete analysis result combining multiple analysis techniques."""

    # Original input
    original_requirements: str

    # LLM-powered analysis
    llm_analysis: Optional[RequirementsAnalysis] = None

    # Rule-based analysis
    detected_technologies: List[TechnologyMatch] = field(default_factory=list)
    extracted_requirements: List[Requirement] = field(default_factory=list)
    identified_patterns: List[str] = field(default_factory=list)

    # Combined insights
    architecture_recommendations: List[str] = field(default_factory=list)
    technology_stack: Dict[str, List[str]] = field(default_factory=dict)
    estimated_complexity: float = 0.0
    estimated_effort_hours: Optional[float] = None

    # Risk and constraint analysis
    potential_risks: List[str] = field(default_factory=list)
    constraints: Dict[ConstraintType, Any] = field(default_factory=dict)

    # Quality metrics
    analysis_confidence: float = 0.0
    completeness_score: float = 0.0
    ambiguity_score: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class RequirementsAnalyzer:
    """Sophisticated requirements analysis engine."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.prompt_manager = PromptTemplateManager()
        self.technology_detector = TechnologyDetector()
        self.constraint_extractor = ConstraintExtractor()
        self.logger = logging.getLogger(f"{__name__}.RequirementsAnalyzer")

        # Initialize rule-based patterns
        self._initialize_patterns()

    async def analyze(
        self,
        requirements: str,
        context: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> AnalysisResult:
        """Perform comprehensive requirements analysis."""
        if context is None:
            context = {}

        self.logger.info(
            f"Starting requirements analysis for {len(requirements)} characters"
        )

        # Initialize result
        result = AnalysisResult(original_requirements=requirements)

        try:
            # Rule-based analysis (always performed)
            await self._perform_rule_based_analysis(requirements, result, context)

            # LLM-powered analysis (if available and requested)
            if use_llm and self.llm_client:
                await self._perform_llm_analysis(requirements, result, context)

            # Combine and synthesize results
            await self._synthesize_analysis(result, context)

            self.logger.info(
                f"Requirements analysis completed with confidence {result.analysis_confidence:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Requirements analysis failed: {e}")
            # Return partial results if possible
            result.analysis_confidence = 0.0

        return result

    async def _perform_rule_based_analysis(
        self, requirements: str, result: AnalysisResult, context: Dict[str, Any]
    ) -> None:
        """Perform rule-based requirements analysis."""
        self.logger.debug("Performing rule-based analysis")

        # Detect technologies
        result.detected_technologies = self.technology_detector.detect_technologies(
            requirements
        )

        # Extract constraints
        constraints = self.constraint_extractor.extract_constraints(requirements)
        result.constraints = {
            constraint.type: constraint.value for constraint in constraints
        }

        # Extract individual requirements
        result.extracted_requirements = self._extract_requirements(requirements)

        # Identify patterns
        result.identified_patterns = self._identify_patterns(requirements)

        # Calculate rule-based metrics
        result.completeness_score = self._calculate_completeness(requirements, result)
        result.ambiguity_score = self._calculate_ambiguity(requirements)

    async def _perform_llm_analysis(
        self, requirements: str, result: AnalysisResult, context: Dict[str, Any]
    ) -> None:
        """Perform LLM-powered requirements analysis."""
        self.logger.debug("Performing LLM-powered analysis")

        try:
            # Create analysis prompt (future enhancement)
            # prompt_context = {"requirements": requirements, **context}
            # template_config = TemplateConfig(
            #     include_examples=True,
            #     include_constraints=True,
            #     temperature=0.3,  # Lower temperature for more consistent analysis
            # )
            # prompt = self.prompt_manager.get_prompt(
            #     PromptType.REQUIREMENTS_ANALYSIS, prompt_context, template_config
            # )

            # Perform analysis
            if self.llm_client is not None:
                result.llm_analysis = await self.llm_client.analyze_requirements(
                    requirements
                )

            if result.llm_analysis:
                self.logger.debug(
                    f"LLM analysis completed with confidence {result.llm_analysis.confidence_score}"
                )

        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}")
            result.llm_analysis = None

    async def _synthesize_analysis(
        self, result: AnalysisResult, context: Dict[str, Any]
    ) -> None:
        """Combine rule-based and LLM analysis into final insights."""
        self.logger.debug("Synthesizing analysis results")

        # Combine technology detections
        self._synthesize_technology_stack(result)

        # Generate architecture recommendations
        self._generate_architecture_recommendations(result)

        # Estimate complexity and effort
        self._estimate_complexity_and_effort(result)

        # Identify potential risks
        self._identify_potential_risks(result)

        # Calculate overall confidence
        self._calculate_overall_confidence(result)

    def _extract_requirements(self, requirements: str) -> List[Requirement]:
        """Extract individual requirements using pattern matching."""
        extracted = []

        # Split into sentences and analyze each
        sentences = re.split(r"[.!?]+", requirements)

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            req_type = self._classify_requirement(sentence)
            priority = self._determine_priority(sentence)
            keywords = self._extract_keywords(sentence)
            confidence = self._calculate_sentence_confidence(sentence)

            requirement = Requirement(
                id=f"req_{i+1}",
                type=req_type,
                priority=priority,
                description=sentence,
                keywords=keywords,
                confidence=confidence,
                source_text=sentence,
            )

            extracted.append(requirement)

        return extracted

    def _classify_requirement(self, sentence: str) -> RequirementType:
        """Classify requirement type based on content patterns."""
        sentence_lower = sentence.lower()

        # Security patterns
        security_patterns = [
            "security",
            "encrypt",
            "auth",
            "permission",
            "secure",
            "ssl",
            "tls",
            "certificate",
        ]
        if any(pattern in sentence_lower for pattern in security_patterns):
            return RequirementType.SECURITY

        # Performance patterns
        performance_patterns = [
            "performance",
            "fast",
            "speed",
            "latency",
            "throughput",
            "response time",
        ]
        if any(pattern in sentence_lower for pattern in performance_patterns):
            return RequirementType.PERFORMANCE

        # Scalability patterns
        scalability_patterns = [
            "scale",
            "load",
            "users",
            "concurrent",
            "horizontal",
            "vertical",
        ]
        if any(pattern in sentence_lower for pattern in scalability_patterns):
            return RequirementType.SCALABILITY

        # Technical patterns
        technical_patterns = [
            "database",
            "api",
            "service",
            "deploy",
            "configure",
            "install",
        ]
        if any(pattern in sentence_lower for pattern in technical_patterns):
            return RequirementType.TECHNICAL

        # Compliance patterns
        compliance_patterns = [
            "gdpr",
            "hipaa",
            "sox",
            "compliance",
            "regulation",
            "audit",
        ]
        if any(pattern in sentence_lower for pattern in compliance_patterns):
            return RequirementType.COMPLIANCE

        return RequirementType.FUNCTIONAL

    def _determine_priority(self, sentence: str) -> AnalysisPriority:
        """Determine requirement priority based on language patterns."""
        sentence_lower = sentence.lower()

        # Critical indicators
        critical_patterns = ["must", "required", "critical", "essential", "mandatory"]
        if any(pattern in sentence_lower for pattern in critical_patterns):
            return AnalysisPriority.CRITICAL

        # High priority indicators
        high_patterns = ["important", "should", "need", "necessary"]
        if any(pattern in sentence_lower for pattern in high_patterns):
            return AnalysisPriority.HIGH

        # Low priority indicators
        low_patterns = ["nice to have", "optional", "if possible", "maybe"]
        if any(pattern in sentence_lower for pattern in low_patterns):
            return AnalysisPriority.LOW

        return AnalysisPriority.MEDIUM

    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract key technical terms from sentence."""
        # Simple keyword extraction using common technical terms
        technical_terms = re.findall(
            r"\b(?:AWS|GCP|Azure|Docker|Kubernetes|PostgreSQL|MySQL|MongoDB|Redis|Python|Node\.js|React|Angular|Vue|Django|Flask|Express|REST|GraphQL|API|CI/CD|Git|GitHub|GitLab|Jenkins|Terraform|Ansible|Nginx|Apache|Linux|Ubuntu|CentOS|SSL|HTTPS|OAuth|JWT|SAML|LDAP|VPC|ECS|EC2|RDS|S3|Lambda|CDN|DNS|Load Balancer|Auto Scaling|Monitoring|Logging|Prometheus|Grafana|ElasticSearch|Kibana|Splunk)\b",
            sentence,
            re.IGNORECASE,
        )

        return list(set(term.lower() for term in technical_terms))

    def _calculate_sentence_confidence(self, sentence: str) -> float:
        """Calculate confidence score for a requirement sentence."""
        confidence = 0.5  # Base confidence

        # Increase confidence for specific terms
        if re.search(r"\b(?:must|should|will|need)\b", sentence, re.IGNORECASE):
            confidence += 0.2

        # Increase confidence for technical specificity
        tech_terms = self._extract_keywords(sentence)
        confidence += min(len(tech_terms) * 0.1, 0.3)

        # Decrease confidence for vague language
        vague_terms = [
            "maybe",
            "possibly",
            "might",
            "unclear",
            "tbd",
            "to be determined",
        ]
        if any(term in sentence.lower() for term in vague_terms):
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def _identify_patterns(self, requirements: str) -> List[str]:
        """Identify architectural and deployment patterns from requirements."""
        patterns = []
        req_lower = requirements.lower()

        # Architecture patterns
        if "microservice" in req_lower:
            patterns.append("microservices_architecture")
        elif "api" in req_lower and "service" in req_lower:
            patterns.append("service_oriented_architecture")
        elif "web application" in req_lower or "web app" in req_lower:
            patterns.append("web_application_architecture")

        if "serverless" in req_lower or "lambda" in req_lower:
            patterns.append("serverless_architecture")

        # Authentication patterns
        if "authentication" in req_lower or "auth" in req_lower:
            patterns.append("user_authentication")

        # Security patterns
        if "ssl" in req_lower or "tls" in req_lower or "encryption" in req_lower:
            patterns.append("encryption_security")

        if "gdpr" in req_lower or "compliant" in req_lower:
            patterns.append("compliance_requirements")

        # Deployment patterns
        if "docker" in req_lower or "container" in req_lower:
            patterns.append("containerized_deployment")

        if "kubernetes" in req_lower or "k8s" in req_lower:
            patterns.append("kubernetes_orchestration")

        if "aws" in req_lower or "azure" in req_lower or "gcp" in req_lower:
            patterns.append("cloud_deployment")

        # Data patterns
        if "database" in req_lower:
            patterns.append("database_integration")
            if "read" in req_lower or "write" in req_lower:
                patterns.append("read_write_separation")

        if "cache" in req_lower or "redis" in req_lower:
            patterns.append("caching_layer")

        # Performance patterns
        if "concurrent" in req_lower or "users" in req_lower:
            patterns.append("scalability_requirements")

        # Infrastructure patterns
        if "load balancer" in req_lower or "lb" in req_lower:
            patterns.append("load_balancing")

        if "auto scal" in req_lower:
            patterns.append("auto_scaling")

        # Budget and timeline patterns
        if "budget" in req_lower or "$" in req_lower or "cost" in req_lower:
            patterns.append("budget_constraints")

        if "month" in req_lower or "week" in req_lower or "timeline" in req_lower:
            patterns.append("timeline_constraints")

        return patterns

    def _calculate_completeness(
        self, requirements: str, result: AnalysisResult
    ) -> float:
        """Calculate how complete the requirements specification is."""
        completeness = 0.0

        # Check for different types of requirements
        req_types_present = set(req.type for req in result.extracted_requirements)
        completeness += len(req_types_present) / len(RequirementType) * 0.4

        # Check for technology specifications
        if result.detected_technologies:
            completeness += 0.3

        # Check for constraints
        if result.constraints:
            completeness += 0.2

        # Check for deployment specifics
        deployment_terms = ["deploy", "production", "environment", "staging"]
        if any(term in requirements.lower() for term in deployment_terms):
            completeness += 0.1

        return min(1.0, completeness)

    def _calculate_ambiguity(self, requirements: str) -> float:
        """Calculate ambiguity score (higher = more ambiguous)."""
        ambiguity = 0.0
        req_lower = requirements.lower()

        # Vague terms increase ambiguity
        vague_terms = [
            "some",
            "maybe",
            "possibly",
            "might",
            "unclear",
            "flexible",
            "various",
        ]
        ambiguity += sum(req_lower.count(term) for term in vague_terms) * 0.1

        # Lack of specifics increases ambiguity
        if not re.search(r"\d+", requirements):  # No numbers
            ambiguity += 0.2

        # Multiple interpretation possibilities
        conditional_terms = ["or", "either", "maybe", "depends"]
        ambiguity += sum(req_lower.count(term) for term in conditional_terms) * 0.15

        return min(1.0, ambiguity)

    def _synthesize_technology_stack(self, result: AnalysisResult) -> None:
        """Combine rule-based and LLM technology detection."""
        tech_stack: Dict[str, List[str]] = {
            "frontend": [],
            "backend": [],
            "database": [],
            "infrastructure": [],
            "tools": [],
        }

        # Add rule-based detections
        for tech_match in result.detected_technologies:
            category = tech_match.category.value.lower()  # Get enum value
            if category in tech_stack:
                tech_stack[category].append(tech_match.technology)

        # Add LLM detections if available
        if result.llm_analysis and result.llm_analysis.technologies:
            for tech in result.llm_analysis.technologies:
                # Categorize based on known technology types
                category = self._categorize_technology(tech)
                if tech not in tech_stack[category]:
                    tech_stack[category].append(tech)

        result.technology_stack = tech_stack

    def _categorize_technology(self, tech: str) -> str:
        """Categorize a technology into stack layer."""
        tech_lower = tech.lower()

        frontend_techs = [
            "react",
            "angular",
            "vue",
            "svelte",
            "html",
            "css",
            "javascript",
            "typescript",
        ]
        backend_techs = [
            "python",
            "node.js",
            "java",
            "go",
            "ruby",
            "php",
            "django",
            "flask",
            "express",
            "spring",
        ]
        database_techs = [
            "postgresql",
            "mysql",
            "mongodb",
            "redis",
            "cassandra",
            "dynamodb",
            "sqlite",
        ]
        infrastructure_techs = [
            "aws",
            "gcp",
            "azure",
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
        ]

        if any(t in tech_lower for t in frontend_techs):
            return "frontend"
        elif any(t in tech_lower for t in backend_techs):
            return "backend"
        elif any(t in tech_lower for t in database_techs):
            return "database"
        elif any(t in tech_lower for t in infrastructure_techs):
            return "infrastructure"
        else:
            return "tools"

    def _generate_architecture_recommendations(self, result: AnalysisResult) -> None:
        """Generate architecture recommendations based on analysis."""
        recommendations = []

        # Based on detected patterns
        if "microservices_architecture" in result.identified_patterns:
            recommendations.append(
                "Consider API gateway for microservices coordination"
            )
            recommendations.append("Implement service discovery mechanism")

        if "containerized_deployment" in result.identified_patterns:
            recommendations.append("Use multi-stage Docker builds for optimization")
            recommendations.append("Implement container health checks")

        # Based on technology stack
        if (
            "database" in result.technology_stack
            and len(result.technology_stack["database"]) > 1
        ):
            recommendations.append("Consider database per service pattern")

        # Based on LLM analysis
        if result.llm_analysis and result.llm_analysis.architecture_pattern:
            pattern = result.llm_analysis.architecture_pattern
            if pattern == "microservices":
                recommendations.append(
                    "Implement distributed tracing for microservices"
                )
            elif pattern == "monolithic":
                recommendations.append("Plan for modular architecture within monolith")

        result.architecture_recommendations = recommendations

    def _estimate_complexity_and_effort(self, result: AnalysisResult) -> None:
        """Estimate project complexity and effort."""
        complexity = 1.0  # Base complexity

        # Technology stack complexity
        total_technologies = sum(
            len(techs) for techs in result.technology_stack.values()
        )
        complexity += total_technologies * 0.5

        # Pattern complexity
        complexity += len(result.identified_patterns) * 0.3

        # Requirements complexity
        critical_reqs = sum(
            1
            for req in result.extracted_requirements
            if req.priority == AnalysisPriority.CRITICAL
        )
        complexity += critical_reqs * 0.2

        # LLM complexity if available
        if result.llm_analysis and result.llm_analysis.complexity_score:
            complexity = (complexity + result.llm_analysis.complexity_score) / 2

        result.estimated_complexity = min(10.0, complexity)

        # Estimate effort (rough calculation)
        base_hours = 8  # Minimum setup time
        complexity_hours = result.estimated_complexity * 4
        integration_hours = total_technologies * 2

        result.estimated_effort_hours = (
            base_hours + complexity_hours + integration_hours
        )

        # Use LLM estimate if available and reasonable
        if (
            result.llm_analysis
            and result.llm_analysis.estimated_duration
            and result.llm_analysis.estimated_duration > 0
        ):
            # Average with our estimate
            result.estimated_effort_hours = (
                result.estimated_effort_hours + result.llm_analysis.estimated_duration
            ) / 2

    def _identify_potential_risks(self, result: AnalysisResult) -> None:
        """Identify potential implementation risks."""
        risks = []

        # Technology complexity risks
        if result.estimated_complexity > 7:
            risks.append("High complexity may lead to implementation challenges")

        # Multiple database risk
        if len(result.technology_stack.get("database", [])) > 2:
            risks.append("Multiple databases increase operational complexity")

        # Microservices risks
        if "microservices_architecture" in result.identified_patterns:
            risks.append("Microservices architecture requires sophisticated monitoring")
            risks.append("Network communication complexity between services")

        # Security risks
        security_reqs = [
            req
            for req in result.extracted_requirements
            if req.type == RequirementType.SECURITY
        ]
        if not security_reqs:
            risks.append("No explicit security requirements identified")

        # Ambiguity risks
        if result.ambiguity_score > 0.5:
            risks.append("High ambiguity in requirements may cause scope creep")

        result.potential_risks = risks

    def _calculate_overall_confidence(self, result: AnalysisResult) -> None:
        """Calculate overall analysis confidence score."""
        confidence_factors = []

        # Rule-based confidence
        rule_confidence = (1.0 - result.ambiguity_score) * result.completeness_score
        confidence_factors.append(rule_confidence)

        # LLM confidence if available
        if result.llm_analysis and result.llm_analysis.confidence_score:
            confidence_factors.append(result.llm_analysis.confidence_score)

        # Technology detection confidence
        if result.detected_technologies:
            avg_tech_confidence = sum(
                tech.confidence for tech in result.detected_technologies
            ) / len(result.detected_technologies)
            confidence_factors.append(avg_tech_confidence)

        # Requirements extraction confidence
        if result.extracted_requirements:
            avg_req_confidence = sum(
                req.confidence for req in result.extracted_requirements
            ) / len(result.extracted_requirements)
            confidence_factors.append(avg_req_confidence)

        # Calculate weighted average
        result.analysis_confidence = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.0
        )

    def _initialize_patterns(self) -> None:
        """Initialize pattern recognition rules."""
        # This could be expanded with more sophisticated pattern matching
