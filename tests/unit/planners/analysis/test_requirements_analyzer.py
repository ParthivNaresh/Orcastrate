"""
Test suite for Requirements Analyzer.

This module tests the sophisticated requirements analysis engine that combines
rule-based and LLM-powered analysis techniques.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.planners.analysis.constraint_extractor import Constraint, ConstraintType
from src.planners.analysis.requirements_analyzer import (
    AnalysisPriority,
    AnalysisResult,
    Requirement,
    RequirementsAnalyzer,
    RequirementType,
)
from src.planners.analysis.technology_detector import (
    TechnologyCategory,
    TechnologyMatch,
)
from src.planners.llm.base import LLMClient, RequirementsAnalysis


class TestRequirementsAnalyzer:
    """Test RequirementsAnalyzer functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock(spec=LLMClient)
        client.analyze_requirements = AsyncMock()
        return client

    @pytest.fixture
    def analyzer_with_llm(self, mock_llm_client):
        """Create analyzer with LLM client."""
        return RequirementsAnalyzer(llm_client=mock_llm_client)

    @pytest.fixture
    def analyzer_without_llm(self):
        """Create analyzer without LLM client."""
        return RequirementsAnalyzer(llm_client=None)

    @pytest.fixture
    def sample_requirements(self):
        """Sample requirements text for testing."""
        return (
            "Build a web application with user authentication using React and Node.js. "
            "The application must support 1000 concurrent users and have a budget of $500 per month. "
            "It should be deployed on AWS with PostgreSQL database and include SSL encryption. "
            "The project needs to be completed within 3 months and must be GDPR compliant."
        )

    @pytest.fixture
    def mock_technology_matches(self):
        """Mock technology detection results."""
        return [
            TechnologyMatch(
                technology="React",
                category=TechnologyCategory.FRONTEND,
                confidence=0.9,
                aliases_found=["React"],
                context="web application with user authentication using React",
                version_hints=[],
            ),
            TechnologyMatch(
                technology="Node.js",
                category=TechnologyCategory.BACKEND,
                confidence=0.9,
                aliases_found=["Node.js"],
                context="React and Node.js",
                version_hints=[],
            ),
            TechnologyMatch(
                technology="PostgreSQL",
                category=TechnologyCategory.DATABASE,
                confidence=0.8,
                aliases_found=["PostgreSQL"],
                context="PostgreSQL database",
                version_hints=[],
            ),
            TechnologyMatch(
                technology="AWS",
                category=TechnologyCategory.INFRASTRUCTURE,
                confidence=0.8,
                aliases_found=["AWS"],
                context="deployed on AWS",
                version_hints=[],
            ),
        ]

    @pytest.fixture
    def mock_constraints(self):
        """Mock constraint extraction results."""
        return [
            Constraint(
                type=ConstraintType.BUDGET,
                value={"amount": 500.0, "currency": "USD", "period": "monthly"},
                confidence=0.9,
                source_text="budget of $500 per month",
                description="Budget constraint: $500.00 monthly",
                priority="medium",
            ),
            Constraint(
                type=ConstraintType.TIMELINE,
                value={"duration": 3, "unit": "month", "type": "deadline"},
                confidence=0.8,
                source_text="completed within 3 months",
                description="Timeline constraint: 3 month(s)",
                priority="high",
            ),
            Constraint(
                type=ConstraintType.SCALABILITY,
                value={"metric": "concurrent_users", "value": 1000},
                confidence=0.8,
                source_text="1000 concurrent users",
                description="Scalability target: 1000 concurrent users",
                priority="high",
            ),
        ]

    @pytest.fixture
    def mock_llm_analysis(self):
        """Mock LLM analysis result."""
        return RequirementsAnalysis(
            technologies=["React", "Node.js", "PostgreSQL", "AWS"],
            architecture_pattern="three_tier",
            complexity_score=7.0,
            estimated_duration=480.0,  # 3 months in hours
            confidence_score=0.85,
        )

    @pytest.mark.asyncio
    async def test_analyze_with_llm(
        self,
        analyzer_with_llm,
        sample_requirements,
        mock_technology_matches,
        mock_constraints,
        mock_llm_analysis,
    ):
        """Test comprehensive analysis with LLM support."""
        # Mock the component methods
        with patch.object(
            analyzer_with_llm.technology_detector,
            "detect_technologies",
            return_value=mock_technology_matches,
        ):
            with patch.object(
                analyzer_with_llm.constraint_extractor,
                "extract_constraints",
                return_value=mock_constraints,
            ):
                with patch.object(
                    analyzer_with_llm.llm_client,
                    "analyze_requirements",
                    new_callable=AsyncMock,
                    return_value=mock_llm_analysis,
                ):
                    result = await analyzer_with_llm.analyze(
                        sample_requirements, use_llm=True
                    )

        assert isinstance(result, AnalysisResult)
        assert result.original_requirements == sample_requirements
        assert result.llm_analysis == mock_llm_analysis
        assert len(result.detected_technologies) == 4
        assert len(result.constraints) == 3
        assert result.analysis_confidence > 0.0

    @pytest.mark.asyncio
    async def test_analyze_without_llm(
        self,
        analyzer_without_llm,
        sample_requirements,
        mock_technology_matches,
        mock_constraints,
    ):
        """Test analysis without LLM (rule-based only)."""
        # Mock the component methods
        with patch.object(
            analyzer_without_llm.technology_detector,
            "detect_technologies",
            return_value=mock_technology_matches,
        ):
            with patch.object(
                analyzer_without_llm.constraint_extractor,
                "extract_constraints",
                return_value=mock_constraints,
            ):
                result = await analyzer_without_llm.analyze(
                    sample_requirements, use_llm=False
                )

        assert isinstance(result, AnalysisResult)
        assert result.llm_analysis is None
        assert len(result.detected_technologies) == 4
        assert len(result.constraints) == 3
        assert result.analysis_confidence > 0.0

    @pytest.mark.asyncio
    async def test_rule_based_analysis(self, analyzer_without_llm, sample_requirements):
        """Test rule-based analysis components."""
        result = AnalysisResult(original_requirements=sample_requirements)

        await analyzer_without_llm._perform_rule_based_analysis(
            sample_requirements, result, {}
        )

        # Should have detected technologies
        assert len(result.detected_technologies) > 0

        # Should have extracted requirements
        assert len(result.extracted_requirements) > 0

        # Should have identified patterns
        assert len(result.identified_patterns) > 0

        # Should have calculated metrics
        assert 0.0 <= result.completeness_score <= 1.0
        assert 0.0 <= result.ambiguity_score <= 1.0

    @pytest.mark.asyncio
    async def test_llm_analysis_with_error(
        self, analyzer_with_llm, sample_requirements
    ):
        """Test LLM analysis when LLM call fails."""
        # Mock LLM to raise an exception
        with patch.object(
            analyzer_with_llm.llm_client,
            "analyze_requirements",
            new_callable=AsyncMock,
            side_effect=Exception("LLM service unavailable"),
        ):
            result = await analyzer_with_llm.analyze(sample_requirements)

        # Should fall back gracefully
        assert result.llm_analysis is None
        assert result.analysis_confidence >= 0.0

    def test_extract_requirements(self, analyzer_without_llm):
        """Test individual requirement extraction."""
        requirements_text = (
            "The system must authenticate users. "
            "It should support high performance. "
            "Security is critical for this application."
        )

        extracted = analyzer_without_llm._extract_requirements(requirements_text)

        assert len(extracted) == 3

        # Check requirement types
        requirement_types = [req.type for req in extracted]
        assert RequirementType.SECURITY in requirement_types
        assert RequirementType.PERFORMANCE in requirement_types

    def test_classify_requirement_security(self, analyzer_without_llm):
        """Test security requirement classification."""
        sentence = "The application must use SSL encryption"
        req_type = analyzer_without_llm._classify_requirement(sentence)
        assert req_type == RequirementType.SECURITY

    def test_classify_requirement_performance(self, analyzer_without_llm):
        """Test performance requirement classification."""
        sentence = "Response time should be under 200ms"
        req_type = analyzer_without_llm._classify_requirement(sentence)
        assert req_type == RequirementType.PERFORMANCE

    def test_classify_requirement_scalability(self, analyzer_without_llm):
        """Test scalability requirement classification."""
        sentence = "System must scale to handle 10000 concurrent users"
        req_type = analyzer_without_llm._classify_requirement(sentence)
        assert req_type == RequirementType.SCALABILITY

    def test_determine_priority_critical(self, analyzer_without_llm):
        """Test critical priority determination."""
        sentence = "Authentication is required for all users"
        priority = analyzer_without_llm._determine_priority(sentence)
        assert priority == AnalysisPriority.CRITICAL

    def test_determine_priority_high(self, analyzer_without_llm):
        """Test high priority determination."""
        sentence = "The system should have good performance"
        priority = analyzer_without_llm._determine_priority(sentence)
        assert priority == AnalysisPriority.HIGH

    def test_determine_priority_low(self, analyzer_without_llm):
        """Test low priority determination."""
        sentence = "It would be nice to have dark mode"
        priority = analyzer_without_llm._determine_priority(sentence)
        assert priority == AnalysisPriority.LOW

    def test_extract_keywords(self, analyzer_without_llm):
        """Test keyword extraction from requirements."""
        sentence = "Deploy using Docker and Kubernetes on AWS with PostgreSQL"
        keywords = analyzer_without_llm._extract_keywords(sentence)

        expected_keywords = ["docker", "kubernetes", "aws", "postgresql"]
        for keyword in expected_keywords:
            assert keyword in keywords

    def test_calculate_sentence_confidence(self, analyzer_without_llm):
        """Test sentence confidence calculation."""
        # High confidence sentence
        high_conf_sentence = "Must use PostgreSQL database with SSL encryption"
        high_conf = analyzer_without_llm._calculate_sentence_confidence(
            high_conf_sentence
        )

        # Low confidence sentence
        low_conf_sentence = "Maybe we might possibly use some database"
        low_conf = analyzer_without_llm._calculate_sentence_confidence(
            low_conf_sentence
        )

        assert high_conf > low_conf
        assert 0.0 <= high_conf <= 1.0
        assert 0.0 <= low_conf <= 1.0

    def test_identify_patterns(self, analyzer_without_llm):
        """Test architectural pattern identification."""
        requirements = (
            "Build microservices with Docker containers using Kubernetes. "
            "Include load balancer and auto scaling. "
            "Use Redis for caching and database read/write separation."
        )

        patterns = analyzer_without_llm._identify_patterns(requirements)

        expected_patterns = [
            "microservices_architecture",
            "containerized_deployment",
            "kubernetes_orchestration",
            "load_balancing",
            "auto_scaling",
            "caching_layer",
            "read_write_separation",
        ]

        for pattern in expected_patterns:
            assert pattern in patterns

    def test_calculate_completeness_high(self, analyzer_without_llm):
        """Test completeness calculation for comprehensive requirements."""
        requirements = (
            "Build secure web application with authentication. "
            "Deploy on AWS with auto-scaling and monitoring. "
            "Use PostgreSQL database with backup and SSL encryption."
        )

        # Create result with diverse requirement types
        result = AnalysisResult(original_requirements=requirements)
        result.extracted_requirements = [
            Requirement(
                "1", RequirementType.SECURITY, AnalysisPriority.HIGH, "security"
            ),
            Requirement(
                "2", RequirementType.TECHNICAL, AnalysisPriority.HIGH, "deploy"
            ),
            Requirement(
                "3", RequirementType.SCALABILITY, AnalysisPriority.MEDIUM, "scaling"
            ),
        ]
        result.detected_technologies = [MagicMock()]  # Mock tech detection
        result.constraints = {ConstraintType.SECURITY: {}}  # Mock constraints

        completeness = analyzer_without_llm._calculate_completeness(
            requirements, result
        )
        assert completeness > 0.5

    def test_calculate_ambiguity_high(self, analyzer_without_llm):
        """Test ambiguity calculation for vague requirements."""
        vague_requirements = (
            "Maybe build some application that might possibly handle various users. "
            "Unclear what database to use or performance requirements."
        )

        ambiguity = analyzer_without_llm._calculate_ambiguity(vague_requirements)
        assert ambiguity > 0.5

    def test_calculate_ambiguity_low(self, analyzer_without_llm):
        """Test ambiguity calculation for specific requirements."""
        specific_requirements = (
            "Build React web application with Node.js backend. "
            "Use PostgreSQL database. Deploy on AWS with 500ms response time limit."
        )

        ambiguity = analyzer_without_llm._calculate_ambiguity(specific_requirements)
        assert ambiguity is not None and ambiguity < 0.5

    def test_synthesize_technology_stack(self, analyzer_without_llm):
        """Test technology stack synthesis."""
        result = AnalysisResult(original_requirements="test")
        result.detected_technologies = [
            TechnologyMatch(
                "React", TechnologyCategory.FRONTEND, 0.9, ["React"], "", []
            ),
            TechnologyMatch(
                "Node.js", TechnologyCategory.BACKEND, 0.9, ["Node.js"], "", []
            ),
            TechnologyMatch(
                "PostgreSQL", TechnologyCategory.DATABASE, 0.8, ["PostgreSQL"], "", []
            ),
            TechnologyMatch(
                "AWS", TechnologyCategory.INFRASTRUCTURE, 0.8, ["AWS"], "", []
            ),
        ]

        # Mock LLM analysis
        result.llm_analysis = RequirementsAnalysis(
            technologies=["Django"],  # Additional technology from LLM
            architecture_pattern="mvc",
            complexity_score=5.0,
            estimated_duration=100.0,
            confidence_score=0.8,
        )

        analyzer_without_llm._synthesize_technology_stack(result)

        assert "React" in result.technology_stack["frontend"]
        assert "Node.js" in result.technology_stack["backend"]
        assert "PostgreSQL" in result.technology_stack["database"]
        assert "AWS" in result.technology_stack["infrastructure"]
        assert "Django" in result.technology_stack["backend"]  # From LLM

    def test_categorize_technology(self, analyzer_without_llm):
        """Test technology categorization."""
        assert analyzer_without_llm._categorize_technology("React") == "frontend"
        assert analyzer_without_llm._categorize_technology("Node.js") == "backend"
        assert analyzer_without_llm._categorize_technology("PostgreSQL") == "database"
        assert analyzer_without_llm._categorize_technology("AWS") == "infrastructure"
        assert analyzer_without_llm._categorize_technology("Unknown Tech") == "tools"

    def test_generate_architecture_recommendations(self, analyzer_without_llm):
        """Test architecture recommendation generation."""
        result = AnalysisResult(original_requirements="test")
        result.identified_patterns = [
            "microservices_architecture",
            "containerized_deployment",
        ]
        result.technology_stack = {
            "database": ["PostgreSQL", "MongoDB"]  # Multiple databases
        }
        result.llm_analysis = RequirementsAnalysis(
            technologies=[],
            architecture_pattern="microservices",
            complexity_score=5.0,
            estimated_duration=100.0,
            confidence_score=0.8,
        )

        analyzer_without_llm._generate_architecture_recommendations(result)

        assert len(result.architecture_recommendations) > 0
        # Should recommend API gateway for microservices
        assert any("API gateway" in rec for rec in result.architecture_recommendations)
        # Should recommend health checks for containers
        assert any(
            "health checks" in rec for rec in result.architecture_recommendations
        )

    def test_estimate_complexity_and_effort(self, analyzer_without_llm):
        """Test complexity and effort estimation."""
        result = AnalysisResult(original_requirements="test")
        result.technology_stack = {
            "frontend": ["React"],
            "backend": ["Node.js", "Django"],
            "database": ["PostgreSQL"],
            "infrastructure": ["AWS", "Docker"],
        }
        result.identified_patterns = [
            "microservices_architecture",
            "containerized_deployment",
        ]
        result.extracted_requirements = [
            Requirement(
                "1", RequirementType.SECURITY, AnalysisPriority.CRITICAL, "security"
            ),
            Requirement(
                "2", RequirementType.TECHNICAL, AnalysisPriority.HIGH, "deploy"
            ),
        ]

        analyzer_without_llm._estimate_complexity_and_effort(result)

        assert result.estimated_complexity > 1.0
        assert result.estimated_effort_hours > 0

    def test_identify_potential_risks(self, analyzer_without_llm):
        """Test risk identification."""
        result = AnalysisResult(original_requirements="test")
        result.estimated_complexity = 8.0  # High complexity
        result.technology_stack = {
            "database": ["PostgreSQL", "MySQL", "MongoDB"]  # Multiple databases
        }
        result.identified_patterns = ["microservices_architecture"]
        result.extracted_requirements = []  # No security requirements
        result.ambiguity_score = 0.7  # High ambiguity

        analyzer_without_llm._identify_potential_risks(result)

        assert len(result.potential_risks) > 0
        # Should identify specific risks
        assert any("complexity" in risk for risk in result.potential_risks)
        assert any("databases" in risk for risk in result.potential_risks)
        assert any("security" in risk for risk in result.potential_risks)
        assert any("ambiguity" in risk for risk in result.potential_risks)

    def test_calculate_overall_confidence(self, analyzer_without_llm):
        """Test overall confidence calculation."""
        result = AnalysisResult(original_requirements="test")
        result.ambiguity_score = 0.2  # Low ambiguity
        result.completeness_score = 0.8  # High completeness
        result.detected_technologies = [
            TechnologyMatch(
                "React", TechnologyCategory.FRONTEND, 0.9, ["React"], "", []
            )
        ]
        result.extracted_requirements = [
            Requirement(
                "1",
                RequirementType.SECURITY,
                AnalysisPriority.HIGH,
                "security",
                confidence=0.8,
            )
        ]
        result.llm_analysis = RequirementsAnalysis(
            technologies=[],
            architecture_pattern="mvc",
            complexity_score=5.0,
            estimated_duration=100.0,
            confidence_score=0.85,
        )

        analyzer_without_llm._calculate_overall_confidence(result)

        assert 0.0 <= result.analysis_confidence <= 1.0
        assert result.analysis_confidence > 0.5  # Should be reasonably confident

    @pytest.mark.asyncio
    async def test_analyze_error_handling(self, analyzer_without_llm):
        """Test error handling during analysis."""
        # Test with problematic requirements that might cause errors
        problematic_requirements = ""  # Empty requirements

        result = await analyzer_without_llm.analyze(problematic_requirements)

        # Should handle gracefully
        assert isinstance(result, AnalysisResult)
        assert result.analysis_confidence >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_with_context(
        self, analyzer_without_llm, sample_requirements
    ):
        """Test analysis with additional context."""
        context = {
            "team_size": 5,
            "experience_level": "intermediate",
            "preferred_stack": "JavaScript",
        }

        result = await analyzer_without_llm.analyze(
            sample_requirements, context=context, use_llm=False
        )

        assert isinstance(result, AnalysisResult)
        # Context should be passed through to component analyzers
        assert result.analysis_confidence > 0.0
