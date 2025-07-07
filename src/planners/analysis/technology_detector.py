"""
Technology detection engine for requirements analysis.

This module provides pattern-based detection of technologies mentioned
in user requirements text.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class TechnologyCategory(Enum):
    """Categories of technologies that can be detected."""

    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"
    CLOUD_PROVIDER = "cloud_provider"
    CONTAINER = "container"
    ORCHESTRATION = "orchestration"
    CI_CD = "ci_cd"
    MONITORING = "monitoring"
    MESSAGING = "messaging"
    CACHING = "caching"
    SECURITY = "security"
    ANALYTICS = "analytics"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class TechnologyMatch:
    """A detected technology with metadata."""

    technology: str
    category: TechnologyCategory
    confidence: float
    aliases_found: List[str]
    context: str
    version_hints: List[str]


class TechnologyDetector:
    """Advanced technology detection using pattern matching."""

    def __init__(self):
        self._technology_patterns = {}
        self._version_patterns = {}
        self._context_patterns = {}
        self._initialize_patterns()

    def detect_technologies(self, text: str) -> List[TechnologyMatch]:
        """Detect all technologies mentioned in the text."""
        text_lower = text.lower()
        detected = []

        for tech_name, tech_info in self._technology_patterns.items():
            matches = self._find_technology_matches(
                text, text_lower, tech_name, tech_info
            )
            detected.extend(matches)

        # Remove duplicates and sort by confidence
        unique_detected = self._deduplicate_matches(detected)
        return sorted(unique_detected, key=lambda x: x.confidence, reverse=True)

    def _find_technology_matches(
        self, original_text: str, text_lower: str, tech_name: str, tech_info: Dict
    ) -> List[TechnologyMatch]:
        """Find all matches for a specific technology."""
        matches = []

        # Check main name and aliases
        all_names = [tech_name] + tech_info.get("aliases", [])

        for name in all_names:
            name_lower = name.lower()

            # Find all occurrences
            for match in re.finditer(re.escape(name_lower), text_lower):
                start, end = match.span()

                # Extract context around the match
                context_start = max(0, start - 50)
                context_end = min(len(original_text), end + 50)
                context = original_text[context_start:context_end].strip()

                # Calculate confidence
                confidence = self._calculate_confidence(
                    name, context, tech_info, original_text
                )

                # Find version hints
                version_hints = self._extract_version_hints(context, tech_name)

                # Find other aliases in context
                aliases_found = self._find_aliases_in_context(context, all_names)

                match_obj = TechnologyMatch(
                    technology=tech_name,
                    category=tech_info["category"],
                    confidence=confidence,
                    aliases_found=aliases_found,
                    context=context,
                    version_hints=version_hints,
                )

                matches.append(match_obj)

        return matches

    def _calculate_confidence(
        self, matched_name: str, context: str, tech_info: Dict, full_text: str
    ) -> float:
        """Calculate confidence score for a technology match."""
        confidence = 0.5  # Base confidence

        context_lower = context.lower()
        matched_lower = matched_name.lower()

        # Boost confidence for exact name matches vs aliases
        if matched_lower == tech_info.get("canonical_name", "").lower():
            confidence += 0.2

        # Boost confidence for contextual indicators
        context_indicators = tech_info.get("context_indicators", [])
        for indicator in context_indicators:
            if indicator.lower() in context_lower:
                confidence += 0.1

        # Boost confidence for version mentions
        if self._has_version_indicators(
            context, tech_info.get("canonical_name", matched_name)
        ):
            confidence += 0.15

        # Reduce confidence for common words that might be false positives
        if matched_lower in [
            "go",
            "rust",
            "dart",
            "java",
        ] and not self._has_programming_context(context_lower):
            confidence -= 0.3

        # Boost confidence for explicit technology lists
        if self._in_technology_list_context(context_lower):
            confidence += 0.2

        # Boost confidence for configuration or setup contexts
        setup_keywords = ["install", "setup", "configure", "deploy", "use", "with"]
        if any(keyword in context_lower for keyword in setup_keywords):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _has_version_indicators(self, context: str, tech_name: str) -> bool:
        """Check if context contains version indicators for the technology."""
        version_patterns = [
            rf"{re.escape(tech_name)}\s*v?\d+(\.\d+)*",
            rf"{re.escape(tech_name)}\s*version\s*\d+",
            rf"{re.escape(tech_name)}\s*\d+(\.\d+)*",
        ]

        for pattern in version_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True

        return False

    def _has_programming_context(self, context: str) -> bool:
        """Check if context suggests programming/technical usage."""
        programming_indicators = [
            "language",
            "framework",
            "library",
            "package",
            "install",
            "import",
            "require",
            "dependency",
            "code",
            "development",
            "programming",
            "script",
            "application",
            "service",
            "api",
        ]

        return any(indicator in context for indicator in programming_indicators)

    def _in_technology_list_context(self, context: str) -> bool:
        """Check if the match appears in a technology list context."""
        list_indicators = [
            "stack:",
            "technologies:",
            "using:",
            "with:",
            "includes:",
            "requirements:",
            "dependencies:",
            "tools:",
            "frameworks:",
        ]

        return any(indicator in context for indicator in list_indicators)

    def _extract_version_hints(self, context: str, tech_name: str) -> List[str]:
        """Extract version information from context."""
        version_hints = []

        # Look for version patterns near the technology name
        version_patterns = [
            rf"{re.escape(tech_name)}\s*v?(\d+(?:\.\d+)*)",
            rf"{re.escape(tech_name)}\s*version\s*(\d+(?:\.\d+)*)",
            rf"{re.escape(tech_name)}\s*(\d+(?:\.\d+)*)",
            rf"v?(\d+(?:\.\d+)*)\s*{re.escape(tech_name)}",
        ]

        for pattern in version_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            version_hints.extend(matches)

        return list(set(version_hints))

    def _find_aliases_in_context(self, context: str, all_names: List[str]) -> List[str]:
        """Find which aliases were found in the context."""
        context_lower = context.lower()
        found_aliases = []

        for name in all_names:
            if name.lower() in context_lower:
                found_aliases.append(name)

        return found_aliases

    def _deduplicate_matches(
        self, matches: List[TechnologyMatch]
    ) -> List[TechnologyMatch]:
        """Remove duplicate matches for the same technology."""
        tech_to_best_match: Dict[str, TechnologyMatch] = {}

        for match in matches:
            tech_key = match.technology.lower()
            if (
                tech_key not in tech_to_best_match
                or match.confidence > tech_to_best_match[tech_key].confidence
            ):
                tech_to_best_match[tech_key] = match

        return list(tech_to_best_match.values())

    def _initialize_patterns(self):
        """Initialize technology detection patterns."""
        self._technology_patterns = {
            # Frontend Technologies
            "React": {
                "category": TechnologyCategory.FRONTEND,
                "canonical_name": "React",
                "aliases": ["ReactJS", "React.js"],
                "context_indicators": ["component", "jsx", "hooks", "state management"],
            },
            "Angular": {
                "category": TechnologyCategory.FRONTEND,
                "canonical_name": "Angular",
                "aliases": ["AngularJS", "Angular.js"],
                "context_indicators": ["component", "typescript", "cli", "directive"],
            },
            "Vue": {
                "category": TechnologyCategory.FRONTEND,
                "canonical_name": "Vue.js",
                "aliases": ["Vue.js", "VueJS", "Vuejs"],
                "context_indicators": ["component", "vue-cli", "nuxt"],
            },
            "Svelte": {
                "category": TechnologyCategory.FRONTEND,
                "canonical_name": "Svelte",
                "aliases": ["SvelteKit"],
                "context_indicators": ["component", "reactive"],
            },
            # Backend Technologies
            "Python": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Python",
                "aliases": ["py"],
                "context_indicators": ["django", "flask", "fastapi", "pip", "conda"],
            },
            "Node.js": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Node.js",
                "aliases": ["NodeJS", "Node", "npm"],
                "context_indicators": ["express", "javascript", "npm", "yarn"],
            },
            "Java": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Java",
                "aliases": ["JDK", "JVM"],
                "context_indicators": ["spring", "maven", "gradle", "jar"],
            },
            "Go": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Go",
                "aliases": ["Golang"],
                "context_indicators": ["goroutine", "gin", "fiber", "go mod"],
            },
            "Rust": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Rust",
                "aliases": ["rustc", "cargo"],
                "context_indicators": ["cargo", "crate", "actix"],
            },
            "PHP": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "PHP",
                "aliases": ["php"],
                "context_indicators": ["laravel", "symfony", "composer"],
            },
            "Ruby": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Ruby",
                "aliases": ["ruby"],
                "context_indicators": ["rails", "gem", "bundler"],
            },
            # Frameworks
            "Django": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Django",
                "aliases": ["django"],
                "context_indicators": ["python", "orm", "admin", "middleware"],
            },
            "Flask": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Flask",
                "aliases": ["flask"],
                "context_indicators": ["python", "route", "jinja2"],
            },
            "FastAPI": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "FastAPI",
                "aliases": ["fastapi"],
                "context_indicators": ["python", "async", "pydantic", "openapi"],
            },
            "Express": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Express.js",
                "aliases": ["Express", "ExpressJS"],
                "context_indicators": ["node", "middleware", "route"],
            },
            "Spring": {
                "category": TechnologyCategory.BACKEND,
                "canonical_name": "Spring",
                "aliases": ["Spring Boot", "SpringBoot"],
                "context_indicators": ["java", "bean", "annotation", "mvc"],
            },
            # Databases
            "PostgreSQL": {
                "category": TechnologyCategory.DATABASE,
                "canonical_name": "PostgreSQL",
                "aliases": ["Postgres", "psql"],
                "context_indicators": ["sql", "relational", "acid", "jsonb"],
            },
            "MySQL": {
                "category": TechnologyCategory.DATABASE,
                "canonical_name": "MySQL",
                "aliases": ["mysql"],
                "context_indicators": ["sql", "relational", "innodb", "mariadb"],
            },
            "MongoDB": {
                "category": TechnologyCategory.DATABASE,
                "canonical_name": "MongoDB",
                "aliases": ["Mongo", "mongo"],
                "context_indicators": ["nosql", "document", "bson", "collection"],
            },
            "Redis": {
                "category": TechnologyCategory.CACHING,
                "canonical_name": "Redis",
                "aliases": ["redis"],
                "context_indicators": ["cache", "key-value", "memory", "pub-sub"],
            },
            "Cassandra": {
                "category": TechnologyCategory.DATABASE,
                "canonical_name": "Cassandra",
                "aliases": ["cassandra"],
                "context_indicators": ["nosql", "column", "distributed", "wide-column"],
            },
            "DynamoDB": {
                "category": TechnologyCategory.DATABASE,
                "canonical_name": "DynamoDB",
                "aliases": ["dynamodb"],
                "context_indicators": ["aws", "nosql", "key-value", "serverless"],
            },
            # Cloud Providers
            "AWS": {
                "category": TechnologyCategory.CLOUD_PROVIDER,
                "canonical_name": "Amazon Web Services",
                "aliases": ["Amazon Web Services", "aws"],
                "context_indicators": ["ec2", "s3", "rds", "lambda", "cloudformation"],
            },
            "GCP": {
                "category": TechnologyCategory.CLOUD_PROVIDER,
                "canonical_name": "Google Cloud Platform",
                "aliases": ["Google Cloud", "Google Cloud Platform", "gcp"],
                "context_indicators": ["compute engine", "cloud storage", "bigquery"],
            },
            "Azure": {
                "category": TechnologyCategory.CLOUD_PROVIDER,
                "canonical_name": "Microsoft Azure",
                "aliases": ["Microsoft Azure", "azure"],
                "context_indicators": ["vm", "blob storage", "active directory"],
            },
            # Containers & Orchestration
            "Docker": {
                "category": TechnologyCategory.CONTAINER,
                "canonical_name": "Docker",
                "aliases": ["docker"],
                "context_indicators": ["container", "image", "dockerfile", "compose"],
            },
            "Kubernetes": {
                "category": TechnologyCategory.ORCHESTRATION,
                "canonical_name": "Kubernetes",
                "aliases": ["k8s", "kube"],
                "context_indicators": [
                    "pod",
                    "deployment",
                    "service",
                    "ingress",
                    "helm",
                ],
            },
            # Infrastructure as Code
            "Terraform": {
                "category": TechnologyCategory.INFRASTRUCTURE,
                "canonical_name": "Terraform",
                "aliases": ["terraform"],
                "context_indicators": ["infrastructure", "iac", "hcl", "state"],
            },
            "Ansible": {
                "category": TechnologyCategory.INFRASTRUCTURE,
                "canonical_name": "Ansible",
                "aliases": ["ansible"],
                "context_indicators": ["playbook", "automation", "configuration"],
            },
            # CI/CD
            "Jenkins": {
                "category": TechnologyCategory.CI_CD,
                "canonical_name": "Jenkins",
                "aliases": ["jenkins"],
                "context_indicators": ["pipeline", "build", "ci/cd", "automation"],
            },
            "GitHub Actions": {
                "category": TechnologyCategory.CI_CD,
                "canonical_name": "GitHub Actions",
                "aliases": ["github actions"],
                "context_indicators": ["workflow", "github", "ci/cd", "yaml"],
            },
            "GitLab CI": {
                "category": TechnologyCategory.CI_CD,
                "canonical_name": "GitLab CI",
                "aliases": ["gitlab ci", "gitlab"],
                "context_indicators": ["pipeline", "runner", "ci/cd"],
            },
            # Monitoring & Observability
            "Prometheus": {
                "category": TechnologyCategory.MONITORING,
                "canonical_name": "Prometheus",
                "aliases": ["prometheus"],
                "context_indicators": ["metrics", "monitoring", "alerting", "grafana"],
            },
            "Grafana": {
                "category": TechnologyCategory.MONITORING,
                "canonical_name": "Grafana",
                "aliases": ["grafana"],
                "context_indicators": ["dashboard", "visualization", "prometheus"],
            },
            "Elasticsearch": {
                "category": TechnologyCategory.ANALYTICS,
                "canonical_name": "Elasticsearch",
                "aliases": ["elasticsearch", "elastic"],
                "context_indicators": ["search", "index", "kibana", "logstash"],
            },
            "Kibana": {
                "category": TechnologyCategory.ANALYTICS,
                "canonical_name": "Kibana",
                "aliases": ["kibana"],
                "context_indicators": ["elasticsearch", "visualization", "dashboard"],
            },
            # Message Queues
            "RabbitMQ": {
                "category": TechnologyCategory.MESSAGING,
                "canonical_name": "RabbitMQ",
                "aliases": ["rabbitmq"],
                "context_indicators": ["queue", "message", "amqp", "broker"],
            },
            "Apache Kafka": {
                "category": TechnologyCategory.MESSAGING,
                "canonical_name": "Apache Kafka",
                "aliases": ["Kafka", "kafka"],
                "context_indicators": ["stream", "topic", "producer", "consumer"],
            },
            # Web Servers
            "Nginx": {
                "category": TechnologyCategory.INFRASTRUCTURE,
                "canonical_name": "Nginx",
                "aliases": ["nginx"],
                "context_indicators": ["web server", "reverse proxy", "load balancer"],
            },
            "Apache": {
                "category": TechnologyCategory.INFRASTRUCTURE,
                "canonical_name": "Apache HTTP Server",
                "aliases": ["Apache HTTP Server", "httpd"],
                "context_indicators": ["web server", "mod_", "virtual host"],
            },
        }
