# Contributing to Smart Image Search

Thank you for your interest in contributing to Smart Image Search! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of machine learning and computer vision

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/smartimagesearch.git
   cd smartimagesearch
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. Copy configuration files:
   ```bash
   cp .env.example .env
   cp config/scan_paths.example.yaml config/scan_paths.yaml
   cp config/model_config.example.yaml config/model_config.yaml
   cp config/search_config.example.yaml config/search_config.yaml
   ```

## 🔧 Development Workflow

### Branch Naming
- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Refactor: `refactor/description`

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use Black for formatting
- Use isort for import sorting

Run formatters:
```bash
black src/ tests/
isort src/ tests/
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_embeddings.py

# Run with verbose output
pytest -v
```

### Code Quality Checks
```bash
# Type checking
mypy src/

# Linting
flake8 src/ tests/

# All checks
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/
pytest
```

## 📝 Making Changes

### 1. Create a New Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Write clean, readable code
- Add docstrings to functions and classes
- Update documentation if needed
- Add tests for new features

### 3. Commit Your Changes
Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```bash
git commit -m "feat(search): add multi-modal image+text search"
```

### 4. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional embedding model support (BLIP-2, LLaVA)
- [ ] GPU optimization and batching improvements
- [ ] Web UI enhancements
- [ ] Mobile app development
- [ ] Advanced filtering and search features

### Documentation
- [ ] API documentation
- [ ] Tutorial videos
- [ ] Code examples
- [ ] Architecture diagrams

### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Edge case coverage

### Features
- [ ] Face recognition and clustering
- [ ] OCR text extraction
- [ ] Video frame indexing
- [ ] Duplicate detection improvements
- [ ] Cloud sync (optional)

## 🐛 Reporting Bugs

When reporting bugs, please include:
- Description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, GPU)
- Error messages and logs
- Screenshots if applicable

## 💡 Suggesting Features

When suggesting features:
- Describe the feature clearly
- Explain the use case
- Provide examples if possible
- Consider implementation complexity
- Check if similar features exist

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings for all public functions/classes
- Update configuration examples if needed
- Add inline comments for complex logic

## 🔍 Code Review Process

All submissions require review. We use GitHub pull requests for this:
1. Ensure all tests pass
2. Ensure code follows style guidelines
3. Update documentation
4. Request review from maintainers
5. Address feedback
6. Wait for approval and merge

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

## 📫 Contact

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and general discussion
- Email: contact@memohsin.com (for security issues)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
