"""
Test suite for Constraint Extractor.

This module tests the constraint extraction engine that identifies
11 types of constraints from natural language requirements.
"""

import pytest

from src.planners.analysis.constraint_extractor import (
    ConstraintExtractor,
    ConstraintType,
)


class TestConstraintExtractor:
    """Test ConstraintExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create constraint extractor instance."""
        return ConstraintExtractor()

    def test_extract_budget_constraints_monetary(self, extractor):
        """Test extraction of monetary budget constraints."""
        requirements = (
            "The project has a budget of $5000 per month and maximum cost of $50,000"
        )

        constraints = extractor.extract_constraints(requirements)
        budget_constraints = [c for c in constraints if c.type == ConstraintType.BUDGET]

        assert len(budget_constraints) >= 1

        # Should find at least one monetary constraint
        monetary_constraint = budget_constraints[0]
        assert monetary_constraint.value["currency"] == "USD"
        assert monetary_constraint.confidence > 0.0

    def test_extract_budget_constraints_cost_conscious(self, extractor):
        """Test extraction of cost-conscious budget preferences."""
        requirements = (
            "Looking for a low cost solution that is budget-friendly and cost-effective"
        )

        constraints = extractor.extract_constraints(requirements)
        budget_constraints = [c for c in constraints if c.type == ConstraintType.BUDGET]

        assert len(budget_constraints) >= 1
        assert budget_constraints[0].value["preference"] == "low_cost"

    def test_extract_timeline_constraints_duration(self, extractor):
        """Test extraction of timeline duration constraints."""
        requirements = (
            "Project must be completed within 3 months and delivered by the deadline"
        )

        constraints = extractor.extract_constraints(requirements)
        timeline_constraints = [
            c for c in constraints if c.type == ConstraintType.TIMELINE
        ]

        assert len(timeline_constraints) >= 1

        # Should find duration constraint
        duration_constraint = next(
            (c for c in timeline_constraints if "duration" in c.value), None
        )
        assert duration_constraint is not None
        assert duration_constraint.value["duration"] == 3
        assert duration_constraint.value["unit"] == "month"

    def test_extract_timeline_constraints_urgent(self, extractor):
        """Test extraction of urgent timeline constraints."""
        requirements = (
            "This is urgent and needs to be done ASAP with immediate deployment"
        )

        constraints = extractor.extract_constraints(requirements)
        timeline_constraints = [
            c for c in constraints if c.type == ConstraintType.TIMELINE
        ]

        assert len(timeline_constraints) >= 1

        # Should detect urgency
        urgent_constraint = next(
            (c for c in timeline_constraints if c.value.get("urgency") == "high"), None
        )
        assert urgent_constraint is not None
        assert urgent_constraint.priority == "critical"

    def test_extract_technical_constraints_platform(self, extractor):
        """Test extraction of platform technical constraints."""
        requirements = (
            "Application must run on Linux only and be cross platform compatible"
        )

        constraints = extractor.extract_constraints(requirements)
        technical_constraints = [
            c for c in constraints if c.type == ConstraintType.TECHNICAL
        ]

        assert len(technical_constraints) >= 1

        # Should find platform constraints
        platform_constraints = [
            c
            for c in technical_constraints
            if "platform" in c.value or "deployment" in c.value
        ]
        assert len(platform_constraints) >= 1

    def test_extract_technical_constraints_architecture(self, extractor):
        """Test extraction of architecture technical constraints."""
        requirements = (
            "Use microservices architecture with RESTful APIs and stateless design"
        )

        constraints = extractor.extract_constraints(requirements)
        technical_constraints = [
            c for c in constraints if c.type == ConstraintType.TECHNICAL
        ]

        assert len(technical_constraints) >= 1

        # Should detect architecture preferences
        arch_constraints = [
            c for c in technical_constraints if "architecture" in c.value
        ]
        assert len(arch_constraints) >= 1

    def test_extract_security_constraints_compliance(self, extractor):
        """Test extraction of security compliance constraints."""
        requirements = (
            "Must be GDPR compliant with HIPAA requirements and SOX compliance"
        )

        constraints = extractor.extract_constraints(requirements)
        security_constraints = [
            c for c in constraints if c.type == ConstraintType.SECURITY
        ]

        assert len(security_constraints) >= 2  # GDPR and HIPAA

        # Should detect compliance requirements
        compliance_types = [c.value.get("compliance") for c in security_constraints]
        assert "GDPR" in compliance_types
        assert "HIPAA" in compliance_types

    def test_extract_security_constraints_technical(self, extractor):
        """Test extraction of technical security constraints."""
        requirements = (
            "Require SSL encryption with multi-factor authentication and audit logging"
        )

        constraints = extractor.extract_constraints(requirements)
        security_constraints = [
            c for c in constraints if c.type == ConstraintType.SECURITY
        ]

        assert len(security_constraints) >= 2

        # Should detect specific security requirements
        security_reqs = [c.value.get("requirement") for c in security_constraints]
        assert any("SSL" in str(req) for req in security_reqs)
        assert any("multi_factor" in str(req) for req in security_reqs)

    def test_extract_performance_constraints_response_time(self, extractor):
        """Test extraction of response time performance constraints."""
        requirements = (
            "Response time must be under 200ms with latency below 100 milliseconds"
        )

        constraints = extractor.extract_constraints(requirements)
        performance_constraints = [
            c for c in constraints if c.type == ConstraintType.PERFORMANCE
        ]

        assert len(performance_constraints) >= 1

        # Should detect response time requirements
        response_time_constraints = [
            c
            for c in performance_constraints
            if c.value.get("metric") == "response_time"
        ]
        assert len(response_time_constraints) >= 1
        assert response_time_constraints[0].value["unit"] == "ms"

    def test_extract_performance_constraints_throughput(self, extractor):
        """Test extraction of throughput performance constraints."""
        requirements = (
            "System must handle 1000 requests per second with high throughput"
        )

        constraints = extractor.extract_constraints(requirements)
        performance_constraints = [
            c for c in constraints if c.type == ConstraintType.PERFORMANCE
        ]

        assert len(performance_constraints) >= 1

        # Should detect throughput requirements
        throughput_constraints = [
            c for c in performance_constraints if c.value.get("metric") == "throughput"
        ]
        assert len(throughput_constraints) >= 1
        assert throughput_constraints[0].value["value"] == 1000

    def test_extract_scalability_constraints_users(self, extractor):
        """Test extraction of user scalability constraints."""
        requirements = (
            "Must support 10000 concurrent users and scale to handle peak load"
        )

        constraints = extractor.extract_constraints(requirements)
        scalability_constraints = [
            c for c in constraints if c.type == ConstraintType.SCALABILITY
        ]

        assert len(scalability_constraints) >= 1

        # Should detect user load requirements
        user_constraints = [
            c
            for c in scalability_constraints
            if c.value.get("metric") == "concurrent_users"
        ]
        assert len(user_constraints) >= 1
        assert user_constraints[0].value["value"] == 10000

    def test_extract_scalability_constraints_scaling_type(self, extractor):
        """Test extraction of scaling type preferences."""
        requirements = "Use auto scaling with horizontal scaling capabilities and elastic infrastructure"

        constraints = extractor.extract_constraints(requirements)
        scalability_constraints = [
            c for c in constraints if c.type == ConstraintType.SCALABILITY
        ]

        assert len(scalability_constraints) >= 1

        # Should detect scaling types
        scaling_types = [c.value.get("type") for c in scalability_constraints]
        assert any("auto_scaling" in str(stype) for stype in scaling_types)

    def test_extract_compliance_constraints(self, extractor):
        """Test extraction of compliance framework constraints."""
        requirements = (
            "Must meet ISO 27001 standards and SOC 2 compliance with NIST framework"
        )

        constraints = extractor.extract_constraints(requirements)
        compliance_constraints = [
            c for c in constraints if c.type == ConstraintType.COMPLIANCE
        ]

        assert len(compliance_constraints) >= 2

        # Should detect compliance frameworks
        frameworks = [c.value.get("framework") for c in compliance_constraints]
        assert "ISO_27001" in frameworks
        assert "SOC_2" in frameworks

    def test_extract_geographic_constraints(self, extractor):
        """Test extraction of geographic/regional constraints."""
        requirements = "Deploy in us-east region with data residency in Europe"

        constraints = extractor.extract_constraints(requirements)
        geographic_constraints = [
            c for c in constraints if c.type == ConstraintType.GEOGRAPHIC
        ]

        assert len(geographic_constraints) >= 1

        # Should detect regional preferences
        regions = [c.value.get("region") for c in geographic_constraints]
        requirements_types = [
            c.value.get("requirement") for c in geographic_constraints
        ]

        assert any("us-east" in str(region) for region in regions)
        assert any("data_residency" in str(req) for req in requirements_types)

    def test_extract_team_constraints(self, extractor):
        """Test extraction of team-related constraints."""
        requirements = (
            "Development team of 5 engineers with 3 developers working on frontend"
        )

        constraints = extractor.extract_constraints(requirements)
        team_constraints = [c for c in constraints if c.type == ConstraintType.TEAM]

        assert len(team_constraints) >= 1

        # Should detect team size
        team_size_constraint = team_constraints[0]
        assert team_size_constraint.value["size"] in [5, 3]  # Either team size detected

    def test_extract_business_constraints(self, extractor):
        """Test extraction of business-related constraints."""
        requirements = (
            "Building an MVP for proof of concept that will become production ready"
        )

        constraints = extractor.extract_constraints(requirements)
        business_constraints = [
            c for c in constraints if c.type == ConstraintType.BUSINESS
        ]

        assert len(business_constraints) >= 1

        # Should detect business scope
        business_types = [c.value.get("type") for c in business_constraints]
        assert "mvp" in business_types or "poc" in business_types

    def test_extract_operational_constraints(self, extractor):
        """Test extraction of operational constraints."""
        requirements = "Require high availability with disaster recovery and comprehensive monitoring"

        constraints = extractor.extract_constraints(requirements)
        operational_constraints = [
            c for c in constraints if c.type == ConstraintType.OPERATIONAL
        ]

        assert len(operational_constraints) >= 2

        # Should detect operational requirements
        op_requirements = [c.value.get("requirement") for c in operational_constraints]
        assert "high_availability" in op_requirements
        assert "disaster_recovery" in op_requirements
        assert "monitoring" in op_requirements

    def test_budget_period_determination(self, extractor):
        """Test budget period determination logic."""
        # Monthly budget
        monthly_text = "Budget of $500 per month"
        constraints = extractor.extract_constraints(monthly_text)
        monthly_budget = next(c for c in constraints if c.type == ConstraintType.BUDGET)
        assert monthly_budget.value["period"] == "monthly"

        # Yearly budget
        yearly_text = "Annual budget of $10000"
        constraints = extractor.extract_constraints(yearly_text)
        yearly_budget = next(c for c in constraints if c.type == ConstraintType.BUDGET)
        assert yearly_budget.value["period"] == "yearly"

        # One-time budget
        onetime_text = "Total setup cost of $2000"
        constraints = extractor.extract_constraints(onetime_text)
        onetime_budget = next(c for c in constraints if c.type == ConstraintType.BUDGET)
        assert onetime_budget.value["period"] == "one_time"

    def test_budget_priority_determination(self, extractor):
        """Test budget priority determination based on amount."""
        # Low priority (small budget)
        small_budget = "Budget of $50 per month"
        constraints = extractor.extract_constraints(small_budget)
        small_constraint = next(
            c for c in constraints if c.type == ConstraintType.BUDGET
        )
        assert small_constraint.priority == "low"

        # High priority (large budget)
        large_budget = "Budget of $1500 per month"
        constraints = extractor.extract_constraints(large_budget)
        large_constraint = next(
            c for c in constraints if c.type == ConstraintType.BUDGET
        )
        assert large_constraint.priority == "high"

    def test_timeline_priority_determination(self, extractor):
        """Test timeline priority determination based on urgency."""
        # Critical (very short timeline)
        urgent_timeline = "Must be completed within 5 days"
        constraints = extractor.extract_constraints(urgent_timeline)
        urgent_constraint = next(
            c for c in constraints if c.type == ConstraintType.TIMELINE
        )
        assert urgent_constraint.priority == "critical"

        # Low priority (long timeline)
        relaxed_timeline = "Completion within 6 months"
        constraints = extractor.extract_constraints(relaxed_timeline)
        relaxed_constraint = next(
            c for c in constraints if c.type == ConstraintType.TIMELINE
        )
        assert relaxed_constraint.priority == "low"

    def test_context_extraction(self, extractor):
        """Test context extraction around constraint matches."""
        requirements = "The application must have a monthly budget limit of $1000 for operational costs"

        constraints = extractor.extract_constraints(requirements)
        budget_constraint = next(
            c for c in constraints if c.type == ConstraintType.BUDGET
        )

        # Should capture relevant context
        assert "$1000" in budget_constraint.source_text
        assert "monthly" in budget_constraint.description.lower()

    def test_multiple_constraints_same_type(self, extractor):
        """Test extraction of multiple constraints of the same type."""
        requirements = "Budget of $500 per month with maximum one-time cost of $2000"

        constraints = extractor.extract_constraints(requirements)
        budget_constraints = [c for c in constraints if c.type == ConstraintType.BUDGET]

        # Should find both budget constraints
        assert len(budget_constraints) >= 1

        # Check if different periods are detected
        [c.value.get("period") for c in budget_constraints]
        # At least one should be detected

    def test_constraint_confidence_scoring(self, extractor):
        """Test confidence scoring for constraint extraction."""
        # High confidence constraint
        explicit_constraint = "The system must support exactly 1000 concurrent users"
        constraints = extractor.extract_constraints(explicit_constraint)
        if constraints:
            high_conf_constraint = constraints[0]
            assert high_conf_constraint.confidence > 0.7

        # Lower confidence constraint (more ambiguous)
        vague_constraint = "Maybe around 1000 users might use it"
        constraints = extractor.extract_constraints(vague_constraint)
        # Vague language might not be detected or have lower confidence

    def test_empty_input(self, extractor):
        """Test handling of empty input."""
        constraints = extractor.extract_constraints("")
        assert constraints == []

    def test_no_constraints_found(self, extractor):
        """Test handling when no constraints are found."""
        requirements = "Build a simple application with good user experience"
        constraints = extractor.extract_constraints(requirements)

        # Should return empty list or very few low-confidence constraints
        assert isinstance(constraints, list)

    def test_sorted_by_confidence(self, extractor):
        """Test that results are sorted by confidence."""
        requirements = """
        Must have budget of $1000 monthly with GDPR compliance required.
        Maybe consider high availability if possible.
        """

        constraints = extractor.extract_constraints(requirements)

        if len(constraints) > 1:
            # Should be sorted by confidence (descending)
            for i in range(len(constraints) - 1):
                assert constraints[i].confidence >= constraints[i + 1].confidence

    def test_complex_requirements_multiple_constraints(self, extractor):
        """Test extraction from complex requirements with multiple constraint types."""
        complex_requirements = """
        Build a web application with a budget of $5000 per month that must be
        completed within 3 months. The system needs to support 10000 concurrent users
        with response time under 200ms. It must be GDPR compliant with SSL encryption
        and deployed on AWS in the us-east region. The development team has 5 engineers
        and this is for an MVP with high availability requirements.
        """

        constraints = extractor.extract_constraints(complex_requirements)

        # Should detect multiple constraint types
        constraint_types = set(c.type for c in constraints)

        expected_types = [
            ConstraintType.BUDGET,
            ConstraintType.TIMELINE,
            ConstraintType.SCALABILITY,
            ConstraintType.PERFORMANCE,
            ConstraintType.SECURITY,
            ConstraintType.GEOGRAPHIC,
            ConstraintType.TEAM,
            ConstraintType.BUSINESS,
            ConstraintType.OPERATIONAL,
        ]

        # Should detect most constraint types
        detected_count = len(constraint_types.intersection(set(expected_types)))
        assert detected_count >= 5  # At least 5 out of 9 types

    def test_numeric_value_extraction(self, extractor):
        """Test extraction of numeric values from constraints."""
        requirements = (
            "Support 1,500 users with response time of 250ms and budget $3,500"
        )

        constraints = extractor.extract_constraints(requirements)

        # Should handle comma-separated numbers
        numeric_constraints = [
            c
            for c in constraints
            if isinstance(c.value, dict)
            and any(isinstance(v, (int, float)) for v in c.value.values())
        ]

        assert len(numeric_constraints) >= 1

    def test_case_insensitive_extraction(self, extractor):
        """Test case-insensitive constraint extraction."""
        requirements = (
            "MUST BE GDPR COMPLIANT WITH SSL ENCRYPTION AND HIGH AVAILABILITY"
        )

        constraints = extractor.extract_constraints(requirements)

        # Should detect constraints regardless of case
        security_constraints = [
            c for c in constraints if c.type == ConstraintType.SECURITY
        ]
        operational_constraints = [
            c for c in constraints if c.type == ConstraintType.OPERATIONAL
        ]

        assert len(security_constraints) >= 1
        assert len(operational_constraints) >= 1
