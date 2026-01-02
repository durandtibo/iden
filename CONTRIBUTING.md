# Contributing to `iden`

We want to make contributing to this project as easy and transparent as possible.

## Overview

We welcome contributions from anyone, even if you are new to open source.

- If you are planning to contribute back bug-fixes, please do so without any further discussion.
- If you plan to contribute new features, utility functions, or extensions to the core, please first
  open an issue and discuss the feature with us.

Once you implement and test your feature or bug-fix, please submit a Pull Request.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Git

### Setting up your development environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```shell
   git clone git@github.com:YOUR_USERNAME/iden.git
   cd iden
   ```

3. **Set up the development environment**:
   ```shell
   # Install uv if you haven't already
   pip install uv
   
   # Create and activate virtual environment with dependencies
   make setup-venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

   Alternatively, use conda:
   ```shell
   make conda
   conda activate iden
   make install-all
   ```

4. **Install pre-commit hooks**:
   ```shell
   pre-commit install
   ```

### Running tests

```shell
# Run unit tests
make unit-test

# Run unit tests with coverage
make unit-test-cov

# Run integration tests
make integration-test
```

### Code quality checks

```shell
# Check code formatting
make format

# Check linting
make lint

# Run all pre-commit hooks
pre-commit run --all-files
```

### Building documentation

```shell
# Install documentation dependencies (if not already installed)
make install-all

# Build and serve documentation locally
cd docs
mkdocs serve
```

Then visit http://localhost:8000 in your browser.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes. You can use the following command to run the tests:
   ```shell
   make unit-test-cov
   ```
5. Make sure your code lints. The following commands can help you to format the code:
   ```shell
   pre-commit run --all-files
   ```
6. If you've added a new feature, add an example to the `examples/` directory.
7. Update the documentation in the `docs/` directory if needed.

### Code Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://github.com/psf/black) for code formatting (line length: 100)
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Follow [Google style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings) for docstrings
- Add type hints to all functions and methods

### Commit Message Guidelines

- Use clear and descriptive commit messages
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Reference issues and pull requests when relevant

Example:
```
Add support for HDF5 shard format

- Implement HDF5Shard class
- Add HDF5ShardLoader
- Update documentation with HDF5 examples
- Add tests for HDF5 functionality

Fixes #123
```

## Issues

We use GitHub issues to track public bugs or feature requests.
For bugs, please ensure your description is clear and concise description, and has sufficient
information to be easily reproducible.
For feature request, please add a clear and concise description of the feature proposal.
Please outline the motivation for the proposal.

## License

By contributing to `iden`, you agree that your contributions will be licensed under the LICENSE
file in the root directory of this source tree.
