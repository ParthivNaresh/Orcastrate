"""
Constraint extraction engine for requirements analysis.

This module provides pattern-based extraction of constraints from
user requirements text including budget, timeline, technical, and business constraints.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List


class ConstraintType(Enum):
    """Types of constraints that can be extracted."""

    BUDGET = "budget"
    TIMELINE = "timeline"
    TECHNICAL = "technical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    COMPLIANCE = "compliance"
    GEOGRAPHIC = "geographic"
    TEAM = "team"
    BUSINESS = "business"
    OPERATIONAL = "operational"


@dataclass
class Constraint:
    """A detected constraint with metadata."""

    type: ConstraintType
    value: Any
    confidence: float
    source_text: str
    description: str
    priority: str = "medium"  # low, medium, high, critical


class ConstraintExtractor:
    """Advanced constraint extraction using pattern matching and NLP."""

    def __init__(self):
        self._constraint_patterns = {}
        self._initialize_patterns()

    def extract_constraints(self, text: str) -> List[Constraint]:
        """Extract all constraints mentioned in the text."""
        constraints = []

        # Extract budget constraints
        constraints.extend(self._extract_budget_constraints(text))

        # Extract timeline constraints
        constraints.extend(self._extract_timeline_constraints(text))

        # Extract technical constraints
        constraints.extend(self._extract_technical_constraints(text))

        # Extract security constraints
        constraints.extend(self._extract_security_constraints(text))

        # Extract performance constraints
        constraints.extend(self._extract_performance_constraints(text))

        # Extract scalability constraints
        constraints.extend(self._extract_scalability_constraints(text))

        # Extract compliance constraints
        constraints.extend(self._extract_compliance_constraints(text))

        # Extract geographic constraints
        constraints.extend(self._extract_geographic_constraints(text))

        # Extract team constraints
        constraints.extend(self._extract_team_constraints(text))

        # Extract business constraints
        constraints.extend(self._extract_business_constraints(text))

        # Extract operational constraints
        constraints.extend(self._extract_operational_constraints(text))

        return sorted(constraints, key=lambda x: x.confidence, reverse=True)

    def _extract_budget_constraints(self, text: str) -> List[Constraint]:
        """Extract budget-related constraints."""
        constraints = []
        text_lower = text.lower()

        # Monetary amounts
        money_patterns = [
            r"budget[:\s]*\$?([0-9,]+(?:\.[0-9]{2})?)",
            r"cost[:\s]*\$?([0-9,]+(?:\.[0-9]{2})?)",
            r"spend[:\s]*\$?([0-9,]+(?:\.[0-9]{2})?)",
            r"maximum[:\s]*\$?([0-9,]+(?:\.[0-9]{2})?)",
            r"limit[:\s]*\$?([0-9,]+(?:\.[0-9]{2})?)",
            r"\$([0-9,]+(?:\.[0-9]{2})?)",
        ]

        for pattern in money_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(",", "")
                try:
                    amount = float(amount_str)

                    # Determine if it's monthly, yearly, or one-time
                    context = self._get_context(text, match.start(), match.end())
                    period = self._determine_budget_period(context)

                    constraint = Constraint(
                        type=ConstraintType.BUDGET,
                        value={
                            "amount": amount,
                            "currency": "USD",
                            "period": period,
                            "type": "maximum",
                        },
                        confidence=0.8,
                        source_text=match.group(0),
                        description=f"Budget constraint: ${amount:,.2f} {period}",
                        priority=self._determine_budget_priority(amount, period),
                    )
                    constraints.append(constraint)
                except ValueError:
                    continue

        # Cost consciousness indicators
        cost_conscious_patterns = [
            "low cost",
            "cheap",
            "budget-friendly",
            "cost-effective",
            "minimal cost",
            "free tier",
            "open source",
        ]

        for pattern in cost_conscious_patterns:
            if pattern in text_lower:
                constraint = Constraint(
                    type=ConstraintType.BUDGET,
                    value={"preference": "low_cost"},
                    confidence=0.6,
                    source_text=pattern,
                    description="Preference for low-cost solutions",
                    priority="medium",
                )
                constraints.append(constraint)
                break

        return constraints

    def _extract_timeline_constraints(self, text: str) -> List[Constraint]:
        """Extract timeline-related constraints."""
        constraints = []

        # Time duration patterns
        duration_patterns = [
            r"(?:deadline|due|complete|finish|deliver).*?(\d+)\s*(day|week|month|year)s?",
            r"(\d+)\s*(day|week|month|year)s?.*?(?:deadline|timeline|schedule)",
            r"within\s+(\d+)\s*(day|week|month|year)s?",
            r"asap|as soon as possible",
            r"urgent|rush|immediate",
            r"by\s+(\w+\s+\d+|\d+/\d+|\d+-\d+-\d+)",
        ]

        for pattern in duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if (
                    "asap" in match.group(0).lower()
                    or "urgent" in match.group(0).lower()
                ):
                    constraint = Constraint(
                        type=ConstraintType.TIMELINE,
                        value={"urgency": "high", "timeline": "asap"},
                        confidence=0.9,
                        source_text=match.group(0),
                        description="Urgent timeline requirement",
                        priority="critical",
                    )
                elif len(match.groups()) >= 2 and match.group(1) and match.group(2):
                    duration = int(match.group(1))
                    unit = match.group(2)

                    constraint = Constraint(
                        type=ConstraintType.TIMELINE,
                        value={"duration": duration, "unit": unit, "type": "deadline"},
                        confidence=0.8,
                        source_text=match.group(0),
                        description=f"Timeline constraint: {duration} {unit}(s)",
                        priority=self._determine_timeline_priority(duration, unit),
                    )
                elif len(match.groups()) >= 1 and match.group(1):
                    # Date constraint
                    constraint = Constraint(
                        type=ConstraintType.TIMELINE,
                        value={"deadline": match.group(1), "type": "date"},
                        confidence=0.7,
                        source_text=match.group(0),
                        description="Specific deadline mentioned",
                        priority="high",
                    )
                else:
                    # Generic urgency constraint
                    constraint = Constraint(
                        type=ConstraintType.TIMELINE,
                        value={"urgency": "medium", "timeline": "general"},
                        confidence=0.6,
                        source_text=match.group(0),
                        description="General timeline requirement",
                        priority="medium",
                    )
                constraints.append(constraint)

        return constraints

    def _extract_technical_constraints(self, text: str) -> List[Constraint]:
        """Extract technical constraints."""
        constraints = []
        text_lower = text.lower()

        # Technology restrictions (future enhancement)
        # restriction_patterns = [
        #     r"(?:only|must use|required?)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        #     r"no\s+([a-zA-Z]+)",
        #     r"cannot use\s+([a-zA-Z]+)",
        #     r"avoid\s+([a-zA-Z]+)",
        # ]

        # Platform constraints
        platform_constraints = {
            "windows only": {"platform": "windows", "restriction": "only"},
            "linux only": {"platform": "linux", "restriction": "only"},
            "mac only": {"platform": "macos", "restriction": "only"},
            "cross platform": {"platform": "cross_platform", "restriction": "required"},
            "cloud only": {"deployment": "cloud_only", "restriction": "required"},
            "on-premises": {"deployment": "on_premises", "restriction": "required"},
            "hybrid": {"deployment": "hybrid", "restriction": "allowed"},
        }

        for phrase, constraint_data in platform_constraints.items():
            if phrase in text_lower:
                constraint = Constraint(
                    type=ConstraintType.TECHNICAL,
                    value=constraint_data,
                    confidence=0.8,
                    source_text=phrase,
                    description=f"Platform constraint: {phrase}",
                    priority="high" if "only" in phrase else "medium",
                )
                constraints.append(constraint)

        # Architecture constraints
        architecture_constraints = [
            "microservices",
            "monolithic",
            "serverless",
            "event-driven",
            "restful",
            "graphql",
            "soap",
            "stateless",
            "stateful",
        ]

        for arch in architecture_constraints:
            if arch in text_lower:
                constraint = Constraint(
                    type=ConstraintType.TECHNICAL,
                    value={"architecture": arch, "type": "preference"},
                    confidence=0.7,
                    source_text=arch,
                    description=f"Architecture preference: {arch}",
                    priority="medium",
                )
                constraints.append(constraint)

        return constraints

    def _extract_security_constraints(self, text: str) -> List[Constraint]:
        """Extract security-related constraints."""
        constraints = []
        text_lower = text.lower()

        # Security requirements
        security_requirements = {
            "gdpr": {"compliance": "GDPR", "type": "regulatory"},
            "hipaa": {"compliance": "HIPAA", "type": "regulatory"},
            "sox": {"compliance": "SOX", "type": "regulatory"},
            "pci": {"compliance": "PCI-DSS", "type": "regulatory"},
            "encryption": {"requirement": "encryption", "type": "technical"},
            "ssl": {"requirement": "SSL/TLS", "type": "technical"},
            "tls": {"requirement": "SSL/TLS", "type": "technical"},
            "authentication": {
                "requirement": "authentication",
                "type": "access_control",
            },
            "authorization": {"requirement": "authorization", "type": "access_control"},
            "mfa": {"requirement": "multi_factor_auth", "type": "access_control"},
            "2fa": {"requirement": "two_factor_auth", "type": "access_control"},
            "multi-factor": {"requirement": "multi_factor", "type": "access_control"},
            "multi factor": {"requirement": "multi_factor", "type": "access_control"},
            "audit log": {"requirement": "audit_logging", "type": "monitoring"},
            "security scan": {"requirement": "security_scanning", "type": "testing"},
        }

        for keyword, constraint_data in security_requirements.items():
            if keyword in text_lower:
                constraint = Constraint(
                    type=ConstraintType.SECURITY,
                    value=constraint_data,
                    confidence=0.8,
                    source_text=keyword,
                    description=f"Security requirement: {keyword}",
                    priority=(
                        "high" if constraint_data["type"] == "regulatory" else "medium"
                    ),
                )
                constraints.append(constraint)

        return constraints

    def _extract_performance_constraints(self, text: str) -> List[Constraint]:
        """Extract performance-related constraints."""
        constraints = []

        # Response time constraints
        response_time_patterns = [
            r"response time.*?(\d+)\s*(ms|millisecond|second|sec)s?",
            r"latency.*?(\d+)\s*(ms|millisecond|second|sec)s?",
            r"(\d+)\s*(ms|millisecond|second|sec)s?.*?response",
        ]

        for pattern in response_time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = int(match.group(1))
                unit = match.group(2).lower()

                # Convert to milliseconds
                if unit.startswith("sec"):
                    value_ms = value * 1000
                else:
                    value_ms = value

                constraint = Constraint(
                    type=ConstraintType.PERFORMANCE,
                    value={
                        "metric": "response_time",
                        "value": value_ms,
                        "unit": "ms",
                        "type": "maximum",
                    },
                    confidence=0.8,
                    source_text=match.group(0),
                    description=f"Response time requirement: {value_ms}ms maximum",
                    priority="high" if value_ms < 1000 else "medium",
                )
                constraints.append(constraint)

        # Throughput constraints
        throughput_patterns = [
            r"(\d+)\s*(?:requests?|rps|qps).*?(?:per|/)\s*(?:second|sec)",
            r"throughput.*?(\d+)",
        ]

        for pattern in throughput_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = int(match.group(1))

                constraint = Constraint(
                    type=ConstraintType.PERFORMANCE,
                    value={
                        "metric": "throughput",
                        "value": value,
                        "unit": "rps",
                        "type": "minimum",
                    },
                    confidence=0.8,
                    source_text=match.group(0),
                    description=f"Throughput requirement: {value} requests/second minimum",
                    priority="high",
                )
                constraints.append(constraint)

        return constraints

    def _extract_scalability_constraints(self, text: str) -> List[Constraint]:
        """Extract scalability-related constraints."""
        constraints = []
        text_lower = text.lower()

        # User load patterns
        user_patterns = [
            r"(\d+)\s*(?:concurrent\s*)?users?",
            r"(\d+)\s*(?:simultaneous\s*)?connections?",
            r"scale.*?(\d+)",
        ]

        for pattern in user_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                users = int(match.group(1))

                # Higher confidence for explicit requirements
                confidence = 0.7
                source_text = match.group(0)
                context_start = max(0, match.start() - 20)
                context_end = min(len(text), match.end() + 20)
                context = text[context_start:context_end].lower()

                if any(
                    word in context
                    for word in ["must", "exactly", "requirement", "support"]
                ):
                    confidence = 0.8

                constraint = Constraint(
                    type=ConstraintType.SCALABILITY,
                    value={
                        "metric": "concurrent_users",
                        "value": users,
                        "type": "target",
                    },
                    confidence=confidence,
                    source_text=source_text,
                    description=f"Scalability target: {users} concurrent users",
                    priority="high" if users > 1000 else "medium",
                )
                constraints.append(constraint)

        # Scaling type preferences
        scaling_types = {
            "auto scal": {"type": "auto_scaling", "preference": "required"},
            "horizontal scal": {
                "type": "horizontal_scaling",
                "preference": "preferred",
            },
            "vertical scal": {"type": "vertical_scaling", "preference": "preferred"},
            "elastic": {"type": "elastic_scaling", "preference": "preferred"},
        }

        for keyword, constraint_data in scaling_types.items():
            if keyword in text_lower:
                constraint = Constraint(
                    type=ConstraintType.SCALABILITY,
                    value=constraint_data,
                    confidence=0.8,
                    source_text=keyword,
                    description=f"Scaling requirement: {keyword}",
                    priority="medium",
                )
                constraints.append(constraint)

        return constraints

    def _extract_compliance_constraints(self, text: str) -> List[Constraint]:
        """Extract compliance-related constraints."""
        constraints = []
        text_lower = text.lower()

        compliance_frameworks = {
            "iso 27001": {"framework": "ISO_27001", "type": "security"},
            "soc 2": {"framework": "SOC_2", "type": "operational"},
            "fedramp": {"framework": "FedRAMP", "type": "government"},
            "nist": {"framework": "NIST", "type": "security"},
        }

        for keyword, constraint_data in compliance_frameworks.items():
            if keyword in text_lower:
                constraint = Constraint(
                    type=ConstraintType.COMPLIANCE,
                    value=constraint_data,
                    confidence=0.9,
                    source_text=keyword,
                    description=f"Compliance requirement: {keyword}",
                    priority="critical",
                )
                constraints.append(constraint)

        return constraints

    def _extract_geographic_constraints(self, text: str) -> List[Constraint]:
        """Extract geographic/regional constraints."""
        constraints = []
        text_lower = text.lower()

        # Region preferences
        regions = {
            "us-east": {"region": "us-east", "provider": "aws"},
            "us-west": {"region": "us-west", "provider": "aws"},
            "europe": {"region": "europe", "provider": "generic"},
            "asia": {"region": "asia", "provider": "generic"},
            "eu-west": {"region": "eu-west", "provider": "aws"},
            "data residency": {"requirement": "data_residency", "type": "legal"},
        }

        for keyword, constraint_data in regions.items():
            if keyword in text_lower:
                constraint = Constraint(
                    type=ConstraintType.GEOGRAPHIC,
                    value=constraint_data,
                    confidence=0.8,
                    source_text=keyword,
                    description=f"Geographic constraint: {keyword}",
                    priority="high" if "residency" in keyword else "medium",
                )
                constraints.append(constraint)

        return constraints

    def _extract_team_constraints(self, text: str) -> List[Constraint]:
        """Extract team-related constraints."""
        constraints = []

        # Team size patterns
        team_patterns = [
            r"team.*?(\d+)\s*(?:people|person|developer|engineer)",
            r"(\d+)\s*(?:people|person|developer|engineer).*?team",
        ]

        for pattern in team_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                size = int(match.group(1))

                constraint = Constraint(
                    type=ConstraintType.TEAM,
                    value={"size": size, "type": "development_team"},
                    confidence=0.7,
                    source_text=match.group(0),
                    description=f"Team size: {size} members",
                    priority="medium",
                )
                constraints.append(constraint)

        return constraints

    def _extract_business_constraints(self, text: str) -> List[Constraint]:
        """Extract business-related constraints."""
        constraints = []
        text_lower = text.lower()

        business_requirements = {
            "mvp": {"type": "mvp", "scope": "minimal"},
            "minimum viable product": {"type": "mvp", "scope": "minimal"},
            "proof of concept": {"type": "poc", "scope": "experimental"},
            "poc": {"type": "poc", "scope": "experimental"},
            "production ready": {"type": "production", "scope": "full"},
            "enterprise": {"type": "enterprise", "scope": "full"},
        }

        for keyword, constraint_data in business_requirements.items():
            if keyword in text_lower:
                constraint = Constraint(
                    type=ConstraintType.BUSINESS,
                    value=constraint_data,
                    confidence=0.8,
                    source_text=keyword,
                    description=f"Business requirement: {keyword}",
                    priority="high",
                )
                constraints.append(constraint)

        return constraints

    def _extract_operational_constraints(self, text: str) -> List[Constraint]:
        """Extract operational constraints."""
        constraints = []
        text_lower = text.lower()

        operational_requirements = {
            "high availability": {"requirement": "high_availability", "level": "99.9%"},
            "disaster recovery": {"requirement": "disaster_recovery", "type": "backup"},
            "backup": {"requirement": "backup", "type": "data_protection"},
            "monitoring": {"requirement": "monitoring", "type": "observability"},
            "logging": {"requirement": "logging", "type": "observability"},
            "alerting": {"requirement": "alerting", "type": "notification"},
        }

        for keyword, constraint_data in operational_requirements.items():
            if keyword in text_lower:
                constraint = Constraint(
                    type=ConstraintType.OPERATIONAL,
                    value=constraint_data,
                    confidence=0.8,
                    source_text=keyword,
                    description=f"Operational requirement: {keyword}",
                    priority=(
                        "high"
                        if keyword in ["high availability", "disaster recovery"]
                        else "medium"
                    ),
                )
                constraints.append(constraint)

        return constraints

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _determine_budget_period(self, context: str) -> str:
        """Determine if budget is monthly, yearly, or one-time."""
        context_lower = context.lower()

        if any(word in context_lower for word in ["month", "monthly", "/month"]):
            return "monthly"
        elif any(
            word in context_lower for word in ["year", "yearly", "annual", "/year"]
        ):
            return "yearly"
        elif any(word in context_lower for word in ["total", "one-time", "setup"]):
            return "one_time"
        else:
            return "monthly"  # Default assumption

    def _determine_budget_priority(self, amount: float, period: str) -> str:
        """Determine priority based on budget amount."""
        monthly_amount = amount
        if period == "yearly":
            monthly_amount = amount / 12
        elif period == "one_time":
            monthly_amount = amount / 6  # Amortize over 6 months

        if monthly_amount < 100:
            return "low"
        elif monthly_amount < 500:
            return "medium"
        elif monthly_amount < 2000:
            return "high"
        else:
            return "critical"

    def _determine_timeline_priority(self, duration: int, unit: str) -> str:
        """Determine priority based on timeline urgency."""
        days = duration
        if unit == "week":
            days = duration * 7
        elif unit == "month":
            days = duration * 30
        elif unit == "year":
            days = duration * 365

        if days <= 7:
            return "critical"
        elif days <= 30:
            return "high"
        elif days <= 90:
            return "medium"
        else:
            return "low"

    def _initialize_patterns(self):
        """Initialize constraint detection patterns."""
        # This method can be expanded with more sophisticated patterns
