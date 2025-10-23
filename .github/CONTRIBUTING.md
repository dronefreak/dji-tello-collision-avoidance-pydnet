# Contributing to DJI Tello Collision Avoidance

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior via GitHub issues.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of computer vision and/or drone programming
- (Optional) DJI Tello drone for testing

### Areas for Contribution

We welcome contributions in these areas:

- **Bug fixes**: Fix issues reported in GitHub Issues
- **New features**: Depth estimation improvements, new collision avoidance algorithms
- **Documentation**: Improve README, add tutorials, fix typos
- **Tests**: Add or improve unit tests
- **Performance**: Optimize inference speed or memory usage
- **Hardware support**: Add support for other drones or cameras
- **Examples**: Create example scripts or notebooks

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/dji-tello-collision-avoidance-pydnet.git
cd dji-tello-collision-avoidance-pydnet

# Add upstream remote
git remote add upstream https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet.git
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### 3. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

**Required Information:**

- Python version: `python --version`
- OS and version
- TensorFlow version: `pip show tensorflow`
- Error message and full traceback
- Steps to reproduce

**Template:**

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:

1. Run command '...'
2. With config '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**

- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.9.0]
- TensorFlow: [e.g., 2.12.0]

**Additional context**
Add any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- Clear and descriptive title
- Detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any alternative solutions you've considered
- Mockups or examples if applicable

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```bash
# Format code with black
black --line-length 100 src/ tests/

# Check style with flake8
flake8 src/ tests/ --max-line-length=100

# Sort imports with isort
isort src/ tests/
```

### Code Quality Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] Functions and classes have docstrings
- [ ] Complex logic has inline comments
- [ ] No debugging print statements
- [ ] No commented-out code blocks
- [ ] Variable names are descriptive
- [ ] No hardcoded paths or values (use config)

### Docstring Format

Use Google-style docstrings:

```python
def estimate_depth(image: np.ndarray, model: DepthEstimator) -> np.ndarray:
    """
    Estimate depth from an RGB image.

    Args:
        image: Input RGB image as numpy array (H, W, 3)
        model: Depth estimation model instance

    Returns:
        Depth map as numpy array (H, W)

    Raises:
        ValueError: If image shape is invalid

    Example:
        >>> image = np.random.rand(256, 512, 3)
        >>> depth = estimate_depth(image, model)
    """
    pass
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests/test_depth_estimator.py

# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # or start htmlcov/index.html on Windows
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test both success and failure cases
- Use mocks for hardware (webcam, drone)

**Example Test:**

```python
def test_depth_estimation_with_valid_input(self):
    """Test depth estimation with valid input."""
    config = Config()
    estimator = DepthEstimator(config)

    # Create valid input
    image = np.random.rand(256, 512, 3).astype(np.float32)

    # Test
    depth_map = estimator.predict(image)

    # Assert
    self.assertIsInstance(depth_map, np.ndarray)
    self.assertEqual(depth_map.shape, (256, 512))
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(depth): add support for different resolution outputs

fix(tello): handle connection timeout gracefully

docs(readme): update installation instructions

test(utils): add tests for colormap function
```

### Pull Request Process

1. **Update your branch**

   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Ensure quality**

   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/

   # Run tests
   python -m unittest discover tests/

   # Check style
   flake8 src/ tests/
   ```

3. **Push changes**

   ```bash
   git push origin your-feature-branch
   ```

4. **Create Pull Request**
   - Go to GitHub and create a PR from your branch
   - Fill out the PR template
   - Link related issues
   - Request review

### Pull Request Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented complex code
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues

Fixes #(issue number)

## Screenshots (if applicable)

Add screenshots for UI changes
```

### Review Process

- Maintainers will review your PR
- Address review comments
- Once approved, maintainer will merge
- PR will be closed automatically on merge

## Development Tips

### Testing Without Hardware

```python
# Use webcam demo for testing without drone
python webcam_demo.py

# Mock hardware in tests
from unittest.mock import Mock, patch

@patch('cv2.VideoCapture')
def test_webcam(mock_capture):
    # Your test here
    pass
```

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use IPython for interactive debugging
import IPython; IPython.embed()
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── depth_estimator.py # Depth estimation
│   ├── utils.py           # Utilities
│   ├── camera_interface.py
│   ├── webcam_source.py
│   ├── tello_source.py
│   └── collision_avoidance.py
├── tests/                 # Unit tests
├── webcam_demo.py         # Webcam demo
├── tello_demo.py          # Tello demo
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
└── README.md             # Documentation
```

## Release Process

(For maintainers)

1. Update version in `setup.py`, `pyproject.toml`, and `src/__init__.py`
2. Update CHANGELOG.md
3. Create release branch: `git checkout -b release/v2.x.x`
4. Run full test suite
5. Create GitHub release with tag
6. Update documentation

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: See [SECURITY.md](SECURITY.md)
- **General**: Check existing issues and discussions first

## Recognition

Contributors will be:

- Listed in the project's contributor graph
- Mentioned in release notes for significant contributions
- Credited in the README for major features

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing! Your efforts help make this project better for everyone. 🚀
