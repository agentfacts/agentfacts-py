# Contributing to AgentFacts

Thank you for your interest in contributing to AgentFacts! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [existing issues](https://github.com/agentfacts/agentfacts-py/issues) to avoid duplicates
2. Collect information about the bug:
   - Stack trace
   - OS, Python version
   - AgentFacts version (`agentfacts --version`)
   - Steps to reproduce

Create an issue with:
- Clear, descriptive title
- Detailed description of the problem
- Minimal code example to reproduce
- Expected vs actual behavior

### Suggesting Features

We welcome feature suggestions! Please:
1. Check existing issues and discussions
2. Describe the use case clearly
3. Explain why existing features don't solve the problem
4. Consider implementation complexity

### Pull Requests

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/agentfacts/agentfacts-py.git
cd agentfacts-py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with dev dependencies
pip install -e ".[all]"
pip install pytest pytest-asyncio pytest-cov mypy ruff black

# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

#### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Follow the coding style (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**
   ```bash
   # Run all tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=src/agentfacts --cov-report=html

   # Run specific test file
   pytest tests/test_crypto.py -v
   ```

4. **Run linters**
   ```bash
   # Format code
   black src/ tests/ examples/

   # Lint
   ruff check src/ tests/

   # Type check
   mypy src/agentfacts
   ```

5. **Build docs (optional)**
   ```bash
   pip install mkdocs mkdocs-material "mkdocstrings[python]" pymdown-extensions
   mkdocs serve
   # or
   mkdocs build --strict
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature X"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation only
   - `test:` adding tests
   - `refactor:` code refactoring
   - `chore:` maintenance tasks

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

#### PR Requirements

- [ ] Tests pass (`pytest tests/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Type checking passes (`mypy src/agentfacts`)
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Commit messages follow conventional commits

## Coding Style

### Python Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with these specifics:

- **Line length**: 100 characters
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with `isort` (handled by `ruff`)
- **Type hints**: Required for all public functions

```python
# Good
def sign_message(message: bytes, key_pair: KeyPair) -> str:
    """Sign a message and return base64 signature."""
    return key_pair.sign_base64(message)

# Bad
def sign_message(message, key_pair):
    return key_pair.sign_base64(message)
```

### Documentation Style

- Use Google-style docstrings
- Document all public classes and functions
- Include examples for complex functionality

```python
def from_agent(
    cls,
    agent: Any,
    name: str,
    description: str = "",
) -> "AgentFacts":
    """
    Create AgentFacts from any supported framework agent.

    Args:
        agent: A framework agent, executor, chain, or graph object
        name: Human-readable name for the agent
        description: Agent description

    Returns:
        Configured AgentFacts instance

    Raises:
        ValueError: If the framework cannot be detected or is unsupported

    Example:
        ```python
        facts = AgentFacts.from_agent(executor, name="My Agent")
        ```
    """
```

### Test Style

- One test class per module/concept
- Descriptive test names
- Use fixtures for common setup
- Test edge cases

```python
class TestKeyPair:
    """Tests for Ed25519 key pair management."""

    def test_generate_creates_valid_keypair(self):
        """Generated key pair should be able to sign and verify."""
        kp = KeyPair.generate()
        message = b"test"
        signature = kp.sign(message)
        assert kp.verify(message, signature)

    def test_verification_fails_with_wrong_message(self):
        """Verification should fail if message is modified."""
        kp = KeyPair.generate()
        signature = kp.sign(b"original")
        assert not kp.verify(b"modified", signature)
```

## Project Structure

```
agentfacts/
├── src/agentfacts/          # Main package
│   ├── __init__.py          # Public API exports
│   ├── core.py              # AgentFacts main class
│   ├── models.py            # Pydantic data models
│   ├── exceptions.py        # Custom exceptions
│   ├── crypto/              # Cryptographic primitives
│   ├── merkle/              # Merkle tree implementation
│   ├── integrations/        # Framework integrations (LangChain, CrewAI, etc.)
│   ├── plugins/             # Verification plugins (DID, attestations, status)
│   ├── policy/              # Policy engine
│   ├── middleware/          # HTTP middleware
│   └── cli/                 # Command-line interface
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation
└── pyproject.toml           # Package configuration
```

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue`:
- Documentation improvements
- Adding test coverage
- Fixing typos
- Small bug fixes

### Help Wanted

Look for issues labeled `help wanted`:
- New framework integrations (Semantic Kernel, Haystack, Vercel AI SDK, DSPy)
- Additional policy rules
- Performance improvements
- Documentation

### Feature Development

Major features we'd love help with:
- **Async API**: Add async variants of core methods
- **DID resolver plugins**: did:web, did:ion, or registry-backed resolvers
- **Attestation verifiers**: SD-JWT-VC, JSON-LD, or other VC formats
- **W3C VC alignment**: Align with Verifiable Credentials formats/terminology
- **NANDA Registry**: Integration with agent registry

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/agentfacts/agentfacts-py/discussions)
- **Bugs**: Open an [Issue](https://github.com/agentfacts/agentfacts-py/issues)
- **Security**: Email security@agentfacts.dev (see [SECURITY.md](SECURITY.md))

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AgentFacts!
