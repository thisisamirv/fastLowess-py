# Contributing to fastLowess-py

Thank you for your interest in contributing to `fastLowess-py`! We welcome bug reports, feature suggestions, documentation improvements, and code contributions.

## Quick Links

- 🐛 [Report a bug](https://github.com/thisisamirv/fastLowess-py/issues/new?labels=bug)
- 💡 [Request a feature](https://github.com/thisisamirv/fastLowess-py/issues/new?labels=enhancement)
- 📖 [Documentation](https://github.com/thisisamirv/fastLowess-py)
- 💬 [Discussions](https://github.com/thisisamirv/fastLowess-py/discussions)

## Code of Conduct

Be respectful, inclusive, and constructive. We're here to build great software together.

## Reporting Bugs

**Before submitting**, search existing issues to avoid duplicates.

Please include:

- Clear description of the problem
- Minimal reproducible example
- Expected vs actual behavior
- Environment details (OS, Python version, NumPy version)
- Full traceback if applicable

**Example:**

```python
import numpy as np
import fastLowess

# This produces unexpected output
x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 3.0])
result = fastLowess.smooth(x, y, fraction=0.5)
# Expected: [1.0, 2.0, 3.0]
# Actual: [0.9, 2.1, 2.9]
```

## Suggesting Features

Feature requests are welcome! Please:

- **Check existing issues** first
- **Explain the use case** - why is this needed?
- **Provide examples** of how it would work
- **Consider alternatives** - have you tried existing features?

Areas of particular interest:

- Performance optimizations
- Better error messages
- Real-world use case examples
- New kernels or robustness methods

## Pull Requests

### Process

1. **Fork** the repository and create a feature branch

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** with clear, focused commits

3. **Add tests** for new functionality

4. **Update documentation** (docstrings, README, CHANGELOG)

5. **Ensure quality**

   ```bash
   make check
   ```

6. **Submit PR** with clear description of changes

### PR Checklist

- [ ] Tests added/updated and passing
- [ ] Documentation updated (if applicable)
- [ ] `cargo fmt` applied (Rust code)
- [ ] `cargo clippy` passes with no warnings
- [ ] Python tests pass (`pytest tests/`)
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Commit messages follow [conventional commits](https://www.conventionalcommits.org/)

### What Makes a Good PR?

✅ **Do:**

- Keep changes focused and atomic
- Write descriptive commit messages
- Add tests that fail without your changes
- Update documentation for API changes
- Consider backward compatibility

❌ **Avoid:**

- Mixing unrelated changes
- Breaking existing APIs without discussion
- Adding dependencies without justification
- Submitting untested code

## Development Setup

This project uses [maturin](https://github.com/PyO3/maturin) to build Python bindings for the Rust code via [PyO3](https://pyo3.rs/).

### Prerequisites

- **Rust**: Latest stable (1.85.0+)
- **Python**: 3.9+ with pip
- **maturin**: `pip install maturin`

### Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fastLowess-py.git
cd fastLowess-py

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install development dependencies
pip install maturin pytest numpy

# Build and install in development mode
maturin develop

# Verify installation
python -c "import fastLowess; print(fastLowess.__version__)"
```

### Development Commands

```bash
# Build and install (development mode, unoptimized)
maturin develop

# Build optimized release
maturin develop --release

# Run Python tests
pytest tests/ -v

# Run Rust tests
cargo test

# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy

# Build wheel for distribution
maturin build --release
```

### Using the Makefile

The project includes a `Makefile` for common tasks:

```bash
# Install from git (latest develop branch)
make install

# Build development version
make dev

# Run all tests
make test

# Run all quality checks
make check

# Format code
make fmt
```

## Project Structure

```text
fastLowess-py/
├── src/
│   └── lib.rs              # Rust source with PyO3 bindings
├── fastLowess/
│   └── __init__.py         # Python package initialization & docs
├── tests/
│   └── test_fastlowess.py  # Python tests
├── examples/
│   ├── batch_smoothing.py
│   ├── online_smoothing.py
│   └── streaming_smoothing.py
├── Cargo.toml              # Rust dependencies
├── pyproject.toml          # Python package configuration
└── README.md
```

### Key Files

| File                           | Purpose                                     |
|--------------------------------|---------------------------------------------|
| `src/lib.rs`                   | Rust code with PyO3 bindings                |
| `fastLowess/__init__.py`       | Python module documentation and re-exports  |
| `tests/test_fastlowess.py`     | Comprehensive Python test suite             |
| `Cargo.toml`                   | Rust dependencies (fastLowess, pyo3, numpy) |
| `pyproject.toml`               | Python build configuration (maturin)        |

## Testing Guidelines

### Running Tests

```bash
# All Python tests
pytest tests/ -v

# Specific test class
pytest tests/test_fastlowess.py::TestSmooth -v

# Specific test function
pytest tests/test_fastlowess.py::TestSmooth::test_basic_smooth -v

# With output
pytest tests/ -v -s

# Rust tests
cargo test
```

### Writing Tests

Place tests in `tests/test_fastlowess.py`:

```python
import numpy as np
import pytest
import fastLowess

class TestMyFeature:
    """Tests for my new feature."""

    def test_descriptive_name(self):
        """Test that describes what it's testing."""
        # Arrange
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

        # Act
        result = fastLowess.smooth(x, y, fraction=0.5)

        # Assert
        assert len(result.y) == len(x)
        assert result.fraction_used == pytest.approx(0.5)

    def test_error_handling(self):
        """Test that errors are raised correctly."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0])  # Mismatched length

        with pytest.raises(ValueError):
            fastLowess.smooth(x, y)
```

## Code Style

### Rust Code

```bash
# Format all Rust code
cargo fmt

# Check without modifying
cargo fmt --check

# Run clippy lints
cargo clippy
```

### Python Code

Follow [PEP 8](https://pep8.org/) style guidelines. Use descriptive variable names and add docstrings to functions.

### Documentation

All public APIs must have documentation:

**Rust (lib.rs):**

```rust
/// LOWESS smoothing with the batch adapter.
///
/// Parameters
/// ----------
/// x : array_like
///     Independent variable values.
/// y : array_like
///     Dependent variable values.
/// fraction : float, optional
///     Smoothing fraction (default: 0.67).
///
/// Returns
/// -------
/// LowessResult
///     Result object with smoothed values.
#[pyfunction]
fn smooth(...) -> PyResult<PyLowessResult> {
    // ...
}
```

**Python (**init**.py):**

```python
"""fastLowess: High-Performance LOWESS Smoothing for Python

smooth(x, y, fraction=0.67, iterations=3, ...)
    LOWESS smoothing with the batch adapter.
    
    Parameters
    ----------
    x : array_like
        Independent variable values.
    ...
"""
```

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting)
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Build, dependencies, or maintenance

**Examples:**

```text
feat(bindings): add auto_converge parameter to smooth()

fix(streaming): handle empty chunks gracefully

docs: update README with new API examples

test: add cross-validation tests
```

## Adding Examples

Examples are located in `examples/` and demonstrate key features:

- `batch_smoothing.py` - Basic smoothing, intervals, cross-validation
- `online_smoothing.py` - Real-time streaming, sliding window
- `streaming_smoothing.py` - Large dataset chunked processing

**To add a new example:**

1. Create `examples/your_example.py`
2. Follow the established pattern with clear print statements
3. Test it: `python examples/your_example.py`
4. Document it in the README if it's a key feature

## Release Process

(For maintainers)

1. Update version in `Cargo.toml` and `pyproject.toml`
2. Update `CHANGELOG.md` with user-facing changes
3. Build and test: `maturin develop --release && pytest tests/`
4. Build wheels: `maturin build --release`
5. Create git tag: `git tag -a v0.x.y -m "Release v0.x.y"`
6. Push tag: `git push origin v0.x.y`
7. Publish to PyPI: `maturin publish`
8. Create GitHub release with changelog

## Getting Help

- 📖 **Documentation**: [GitHub README](https://github.com/thisisamirv/fastLowess-py)
- 💬 **Discussions**: Ask questions in GitHub Discussions
- 📧 **Email**: <thisisamirv@gmail.com> (for private inquiries)
- 🐛 **Issues**: Report bugs via GitHub Issues

## Related Projects

- **[lowess](https://crates.io/crates/lowess)**: Core LOWESS Rust implementation
- **[fastLowess](https://crates.io/crates/fastLowess)**: High-level Rust crate with parallel execution

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.

---

Thank you for contributing! Every improvement, no matter how small, helps make `fastLowess` better for everyone. 🎉
