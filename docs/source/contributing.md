# Contributing to Orcastrate

We welcome contributions to Orcastrate! This guide will help you get started with contributing to the project.

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub/GitLab
2. Clone your fork locally:

```bash
git clone https://github.com/yourusername/orcastrate.git
cd orcastrate
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/original/orcastrate.git
```

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Contribution Types

### Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Logs or error messages** if applicable

### Feature Requests

For new features, please:

- **Describe the use case** and problem it solves
- **Provide examples** of how it would be used
- **Consider the scope** and complexity
- **Discuss alternatives** if applicable

### Code Contributions

We accept contributions for:

- **New tools** (Docker, Kubernetes, cloud providers, etc.)
- **New planners** (AI-based, rule-based, etc.)
- **Template improvements** (new templates, better existing ones)
- **Bug fixes** and improvements
- **Documentation** updates
- **Test coverage** improvements

## Development Workflow

### 1. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

```bash
# Run tests
pytest

# Check code quality
make lint
make format

# Test specific areas
pytest tests/unit/tools/  # If you added a tool
pytest tests/integration/  # Integration tests
```

### 4. Submit Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request via GitHub/GitLab UI
```

## Coding Standards

### Python Style

- **Follow PEP 8** with 88-character line limit (Black formatter)
- **Use type hints** for all function parameters and return values
- **Add docstrings** for all public methods and classes
- **Use descriptive names** for variables and functions

### Code Quality Tools

We use the following tools to maintain code quality:

```bash
# Automatic formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Example Code Style

```python
from typing import List, Optional, Dict, Any
import logging


class ExampleClass:
    """Example class demonstrating coding standards.
    
    This class shows the preferred style for Orcastrate contributions
    including docstrings, type hints, and error handling.
    
    Args:
        config: Configuration object for the class.
        optional_param: Optional parameter with default value.
    """
    
    def __init__(self, config: ExampleConfig, optional_param: Optional[str] = None):
        self.config = config
        self.optional_param = optional_param
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process_data(self, data: List[Dict[str, Any]]) -> List[ProcessedItem]:
        """Process a list of data items.
        
        Args:
            data: List of data dictionaries to process.
            
        Returns:
            List of processed items.
            
        Raises:
            ProcessingError: If data processing fails.
            ValidationError: If input data is invalid.
        """
        if not data:
            raise ValidationError("Data list cannot be empty")
        
        processed_items = []
        
        for item in data:
            try:
                processed_item = await self._process_single_item(item)
                processed_items.append(processed_item)
            except Exception as e:
                self.logger.error(f"Failed to process item {item}: {e}")
                raise ProcessingError(f"Processing failed: {e}") from e
        
        return processed_items
    
    async def _process_single_item(self, item: Dict[str, Any]) -> ProcessedItem:
        """Process a single data item."""
        # Implementation details
        pass
```

## Testing Guidelines

### Test Requirements

All contributions must include appropriate tests:

- **Unit tests** for individual components
- **Integration tests** for component interactions
- **End-to-end tests** for complete workflows (when applicable)

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, patch

from src.tools.example_tool import ExampleTool
from src.tools.base import ToolConfig, ToolError


class TestExampleTool:
    """Test suite for ExampleTool."""
    
    @pytest.fixture
    def tool_config(self):
        """Create tool configuration for testing."""
        return ToolConfig(
            name="example_tool",
            version="1.0.0",
            timeout=30,
            retry_count=3
        )
    
    @pytest.fixture
    async def tool(self, tool_config):
        """Create and initialize tool instance."""
        tool = ExampleTool(tool_config)
        await tool.initialize()
        return tool
    
    @pytest.mark.asyncio
    async def test_successful_operation(self, tool):
        """Test successful tool operation."""
        result = await tool.execute("example_action", {"param": "value"})
        
        assert result.success is True
        assert "expected_output" in result.output
    
    @pytest.mark.asyncio
    async def test_error_handling(self, tool):
        """Test tool error handling."""
        with pytest.raises(ToolError, match="Invalid parameter"):
            await tool.execute("example_action", {"invalid": "param"})
    
    @pytest.mark.asyncio
    @patch('src.tools.example_tool.external_dependency')
    async def test_with_mocked_dependency(self, mock_dependency, tool):
        """Test tool with mocked external dependency."""
        mock_dependency.return_value = "mocked_result"
        
        result = await tool.execute("example_action", {"param": "value"})
        
        assert result.success is True
        mock_dependency.assert_called_once_with("value")
```

### Test Coverage

- Aim for **90%+ test coverage** for new code
- Test both **success and failure scenarios**
- Include **edge cases** and **boundary conditions**
- Mock **external dependencies** appropriately

## Documentation

### Documentation Requirements

- **Update user documentation** for new features
- **Add API documentation** for new classes/methods
- **Include examples** for complex functionality
- **Update CLI reference** for new commands

### Documentation Style

- Use **clear, concise language**
- Include **code examples** where helpful
- Follow **existing documentation structure**
- Use **proper markdown formatting**

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs
make html

# View documentation
open build/html/index.html
```

## Submitting Changes

### Pull Request Guidelines

1. **Clear title** describing the change
2. **Detailed description** of what and why
3. **Link to related issues** if applicable
4. **Test coverage** maintained or improved
5. **Documentation** updated as needed

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing
- [ ] Test coverage maintained

## Documentation
- [ ] Documentation updated
- [ ] Examples added/updated
- [ ] API documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Changes tested locally
- [ ] Related issues linked

## Additional Notes
Any additional information or context.
```

### Review Process

1. **Automated checks** must pass (tests, linting, etc.)
2. **Code review** by maintainers
3. **Integration testing** in CI environment
4. **Documentation review** for user-facing changes
5. **Final approval** and merge

## Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Focus on constructive feedback**
- **Help others learn** and grow
- **Assume good intentions**

### Communication

- **Use GitHub/GitLab issues** for bug reports and feature requests
- **Use pull requests** for code discussions
- **Be patient** with response times
- **Ask questions** if anything is unclear

### Getting Help

- **Check existing issues** before creating new ones
- **Search documentation** for answers
- **Provide context** when asking questions
- **Share relevant logs** and error messages

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version numbers**
2. **Update changelog**
3. **Run full test suite**
4. **Build and test documentation**
5. **Create release tag**
6. **Deploy to package repositories**

## Recognition

Contributors are recognized in:

- **Changelog** for each release
- **Contributors file** in the repository
- **Release notes** for significant contributions

Thank you for contributing to Orcastrate! 🚀
