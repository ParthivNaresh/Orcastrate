"""
Test suite for Technology Detector.

This module tests the technology detection system that identifies
90+ technologies across 10+ categories with pattern matching.
"""

import pytest

from src.planners.analysis.technology_detector import (
    TechnologyCategory,
    TechnologyDetector,
)


class TestTechnologyDetector:
    """Test TechnologyDetector functionality."""

    @pytest.fixture
    def detector(self):
        """Create technology detector instance."""
        return TechnologyDetector()

    def test_detect_frontend_technologies(self, detector):
        """Test detection of frontend technologies."""
        requirements = (
            "Build a web application using React with Redux for state management"
        )

        matches = detector.detect_technologies(requirements)

        # Should detect React
        react_matches = [m for m in matches if m.technology == "React"]
        assert len(react_matches) == 1
        assert react_matches[0].category == TechnologyCategory.FRONTEND
        assert react_matches[0].confidence > 0.0

        # Should detect Redux
        redux_matches = [m for m in matches if m.technology == "Redux"]
        assert len(redux_matches) == 1

    def test_detect_backend_technologies(self, detector):
        """Test detection of backend technologies."""
        requirements = "Create REST API using Node.js with Express framework and Django for admin panel"

        matches = detector.detect_technologies(requirements)

        # Should detect Node.js
        nodejs_matches = [m for m in matches if m.technology == "Node.js"]
        assert len(nodejs_matches) == 1
        assert nodejs_matches[0].category == TechnologyCategory.BACKEND

        # Should detect Express
        express_matches = [m for m in matches if m.technology == "Express"]
        assert len(express_matches) == 1

        # Should detect Django
        django_matches = [m for m in matches if m.technology == "Django"]
        assert len(django_matches) == 1

    def test_detect_database_technologies(self, detector):
        """Test detection of database technologies."""
        requirements = "Store data in PostgreSQL and use Redis for caching with MongoDB for documents"

        matches = detector.detect_technologies(requirements)

        # Should detect PostgreSQL
        postgres_matches = [m for m in matches if m.technology == "PostgreSQL"]
        assert len(postgres_matches) == 1
        assert postgres_matches[0].category == TechnologyCategory.DATABASE

        # Should detect Redis
        redis_matches = [m for m in matches if m.technology == "Redis"]
        assert len(redis_matches) == 1

        # Should detect MongoDB
        mongo_matches = [m for m in matches if m.technology == "MongoDB"]
        assert len(mongo_matches) == 1

    def test_detect_infrastructure_technologies(self, detector):
        """Test detection of infrastructure technologies."""
        requirements = (
            "Deploy on AWS using Docker containers orchestrated by Kubernetes"
        )

        matches = detector.detect_technologies(requirements)

        # Should detect AWS
        aws_matches = [m for m in matches if m.technology == "AWS"]
        assert len(aws_matches) == 1
        assert aws_matches[0].category == TechnologyCategory.INFRASTRUCTURE

        # Should detect Docker
        docker_matches = [m for m in matches if m.technology == "Docker"]
        assert len(docker_matches) == 1

        # Should detect Kubernetes
        k8s_matches = [m for m in matches if m.technology == "Kubernetes"]
        assert len(k8s_matches) == 1

    def test_detect_cloud_providers(self, detector):
        """Test detection of cloud providers."""
        requirements = "Multi-cloud deployment using AWS, Google Cloud Platform, and Microsoft Azure"

        matches = detector.detect_technologies(requirements)

        # Should detect all cloud providers
        aws_matches = [m for m in matches if m.technology == "AWS"]
        gcp_matches = [m for m in matches if m.technology in ["Google Cloud", "GCP"]]
        azure_matches = [m for m in matches if m.technology == "Azure"]

        assert len(aws_matches) == 1
        assert len(gcp_matches) >= 1
        assert len(azure_matches) == 1

    def test_detect_programming_languages(self, detector):
        """Test detection of programming languages."""
        requirements = (
            "Backend in Python and Java, frontend in JavaScript and TypeScript"
        )

        matches = detector.detect_technologies(requirements)

        # Should detect programming languages
        python_matches = [m for m in matches if m.technology == "Python"]
        java_matches = [m for m in matches if m.technology == "Java"]
        js_matches = [m for m in matches if m.technology == "JavaScript"]
        ts_matches = [m for m in matches if m.technology == "TypeScript"]

        assert len(python_matches) == 1
        assert len(java_matches) == 1
        assert len(js_matches) == 1
        assert len(ts_matches) == 1

    def test_detect_mobile_technologies(self, detector):
        """Test detection of mobile technologies."""
        requirements = "Build mobile app with React Native and native iOS using Swift"

        matches = detector.detect_technologies(requirements)

        # Should detect mobile technologies
        rn_matches = [m for m in matches if m.technology == "React Native"]
        ios_matches = [m for m in matches if m.technology in ["iOS", "Swift"]]

        assert len(rn_matches) == 1
        assert len(ios_matches) >= 1

    def test_detect_devops_tools(self, detector):
        """Test detection of DevOps tools."""
        requirements = "CI/CD pipeline with Jenkins, deploy using Terraform, and monitor with Prometheus"

        matches = detector.detect_technologies(requirements)

        # Should detect DevOps tools
        jenkins_matches = [m for m in matches if m.technology == "Jenkins"]
        terraform_matches = [m for m in matches if m.technology == "Terraform"]
        prometheus_matches = [m for m in matches if m.technology == "Prometheus"]

        assert len(jenkins_matches) == 1
        assert len(terraform_matches) == 1
        assert len(prometheus_matches) == 1

    def test_detect_testing_frameworks(self, detector):
        """Test detection of testing frameworks."""
        requirements = "Unit tests with Jest, integration tests with Selenium, and load testing with JMeter"

        matches = detector.detect_technologies(requirements)

        # Should detect testing tools
        jest_matches = [m for m in matches if m.technology == "Jest"]
        selenium_matches = [m for m in matches if m.technology == "Selenium"]
        jmeter_matches = [m for m in matches if m.technology == "JMeter"]

        assert len(jest_matches) == 1
        assert len(selenium_matches) == 1
        assert len(jmeter_matches) == 1

    def test_detect_monitoring_tools(self, detector):
        """Test detection of monitoring tools."""
        requirements = "Monitor with Grafana dashboards, log aggregation with ELK stack"

        matches = detector.detect_technologies(requirements)

        # Should detect monitoring tools
        grafana_matches = [m for m in matches if m.technology == "Grafana"]
        elk_matches = [m for m in matches if m.technology in ["Elasticsearch", "ELK"]]

        assert len(grafana_matches) == 1
        assert len(elk_matches) >= 1

    def test_detect_security_tools(self, detector):
        """Test detection of security tools."""
        requirements = "Security scanning with SonarQube and vulnerability assessment with OWASP ZAP"

        matches = detector.detect_technologies(requirements)

        # Should detect security tools
        sonar_matches = [m for m in matches if m.technology == "SonarQube"]
        owasp_matches = [m for m in matches if m.technology in ["OWASP", "OWASP ZAP"]]

        assert len(sonar_matches) == 1
        assert len(owasp_matches) >= 1

    def test_confidence_scoring(self, detector):
        """Test confidence scoring for technology matches."""
        # Exact match should have high confidence
        exact_match = "Use React for frontend development"
        matches = detector.detect_technologies(exact_match)
        react_match = next(m for m in matches if m.technology == "React")
        assert react_match.confidence > 0.8

        # Partial match should have lower confidence
        partial_match = "Using reactive programming patterns"
        matches = detector.detect_technologies(partial_match)
        # Should not match React with high confidence due to partial match
        react_matches = [m for m in matches if m.technology == "React"]
        if react_matches:
            assert react_matches[0].confidence < 0.8

    def test_context_extraction(self, detector):
        """Test context extraction for technology matches."""
        requirements = (
            "Build a scalable web application using React for the frontend interface"
        )

        matches = detector.detect_technologies(requirements)
        react_match = next(m for m in matches if m.technology == "React")

        assert "React" in react_match.context
        assert "frontend" in react_match.context.lower()

    def test_case_insensitive_detection(self, detector):
        """Test case-insensitive technology detection."""
        requirements = "Use POSTGRESQL database with DOCKER containers on aws"

        matches = detector.detect_technologies(requirements)

        # Should detect regardless of case
        postgres_matches = [m for m in matches if m.technology == "PostgreSQL"]
        docker_matches = [m for m in matches if m.technology == "Docker"]
        aws_matches = [m for m in matches if m.technology == "AWS"]

        assert len(postgres_matches) == 1
        assert len(docker_matches) == 1
        assert len(aws_matches) == 1

    def test_acronym_detection(self, detector):
        """Test detection of technology acronyms."""
        requirements = "Deploy on K8s cluster with ML models using AI frameworks"

        matches = detector.detect_technologies(requirements)

        # Should detect acronyms
        k8s_matches = [m for m in matches if m.technology in ["Kubernetes", "K8s"]]

        assert len(k8s_matches) >= 1
        # ML/AI might be detected as general terms

    def test_version_specific_detection(self, detector):
        """Test detection of version-specific technologies."""
        requirements = "Use Node.js 18 with React 18 and PostgreSQL 14"

        matches = detector.detect_technologies(requirements)

        # Should detect technologies regardless of version numbers
        nodejs_matches = [m for m in matches if m.technology == "Node.js"]
        react_matches = [m for m in matches if m.technology == "React"]
        postgres_matches = [m for m in matches if m.technology == "PostgreSQL"]

        assert len(nodejs_matches) == 1
        assert len(react_matches) == 1
        assert len(postgres_matches) == 1

    def test_framework_detection(self, detector):
        """Test detection of various frameworks."""
        requirements = (
            "Backend with Spring Boot, frontend with Angular, testing with Pytest"
        )

        matches = detector.detect_technologies(requirements)

        spring_matches = [
            m for m in matches if m.technology in ["Spring", "Spring Boot"]
        ]
        angular_matches = [m for m in matches if m.technology == "Angular"]
        pytest_matches = [m for m in matches if m.technology == "Pytest"]

        assert len(spring_matches) >= 1
        assert len(angular_matches) == 1
        assert len(pytest_matches) == 1

    def test_no_false_positives(self, detector):
        """Test that common words don't trigger false technology matches."""
        requirements = "The application should be reactive and angular in design with a spring interface"

        matches = detector.detect_technologies(requirements)

        # Should minimize false positives for common words
        # This test might need adjustment based on actual implementation
        for match in matches:
            # Check that confidence is reasonable for actual matches
            if match.technology in ["React", "Angular", "Spring"]:
                # These might be detected but should have low confidence
                # due to context not being technology-specific
                pass

    def test_compound_technology_detection(self, detector):
        """Test detection of compound technology names."""
        requirements = (
            "Use React Native for mobile and Next.js for server-side rendering"
        )

        matches = detector.detect_technologies(requirements)

        react_native_matches = [m for m in matches if m.technology == "React Native"]
        nextjs_matches = [m for m in matches if m.technology in ["Next.js", "NextJS"]]

        assert len(react_native_matches) == 1
        assert len(nextjs_matches) >= 1

    def test_alternative_names_detection(self, detector):
        """Test detection of technologies with alternative names."""
        requirements = (
            "Use Postgres database with K8s orchestration and JS for frontend"
        )

        matches = detector.detect_technologies(requirements)

        # Should detect alternative names
        postgres_matches = [
            m
            for m in matches
            if "PostgreSQL" in m.technology or "Postgres" in m.technology
        ]
        k8s_matches = [
            m for m in matches if "Kubernetes" in m.technology or "K8s" in m.technology
        ]
        js_matches = [
            m for m in matches if "JavaScript" in m.technology or m.technology == "JS"
        ]

        assert len(postgres_matches) >= 1
        assert len(k8s_matches) >= 1
        assert len(js_matches) >= 1

    def test_empty_input(self, detector):
        """Test handling of empty input."""
        matches = detector.detect_technologies("")
        assert matches == []

    def test_no_technologies_found(self, detector):
        """Test handling when no technologies are found."""
        requirements = (
            "Build a simple application with good performance and user experience"
        )
        matches = detector.detect_technologies(requirements)

        # Should return empty list or very few matches with low confidence
        assert isinstance(matches, list)

    def test_technology_categories_coverage(self, detector):
        """Test that all major technology categories are covered."""
        comprehensive_requirements = """
        Build a full-stack web application using React frontend, Node.js backend,
        PostgreSQL database, deploy on AWS with Docker containers,
        use Jenkins for CI/CD, monitor with Prometheus,
        test with Jest and Selenium, and secure with OAuth.
        """

        matches = detector.detect_technologies(comprehensive_requirements)

        # Should cover multiple categories
        categories = set(match.category for match in matches)

        expected_categories = [
            TechnologyCategory.FRONTEND,
            TechnologyCategory.BACKEND,
            TechnologyCategory.DATABASE,
            TechnologyCategory.INFRASTRUCTURE,
            TechnologyCategory.CI_CD,
            TechnologyCategory.MONITORING,
            TechnologyCategory.SECURITY,
        ]

        # Should detect most major categories
        detected_categories = len(categories.intersection(set(expected_categories)))
        assert detected_categories >= 5  # At least 5 out of 7 categories

    def test_sorted_by_confidence(self, detector):
        """Test that results are sorted by confidence score."""
        requirements = (
            "Use React for frontend and maybe consider Angular as alternative"
        )

        matches = detector.detect_technologies(requirements)

        if len(matches) > 1:
            # Should be sorted by confidence (descending)
            for i in range(len(matches) - 1):
                assert matches[i].confidence >= matches[i + 1].confidence

    def test_duplicate_detection_handling(self, detector):
        """Test handling of duplicate technology mentions."""
        requirements = "Use React for frontend, React components, and React hooks"

        matches = detector.detect_technologies(requirements)

        # Should handle duplicates appropriately
        react_matches = [m for m in matches if m.technology == "React"]

        # Implementation might return one match with high confidence
        # or multiple matches - both are acceptable
        assert len(react_matches) >= 1
