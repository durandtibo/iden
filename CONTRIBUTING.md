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
- [`uv`](https://docs.astral.sh/uv/) for dependency management (see [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- Git

### Setting up your development environment

Please refer to the [Get Started guide](https://durandtibo.github.io/iden/get_started) for detailed instructions on setting up your development environment, including:

- Creating and activating virtual environments
- Installing dependencies
- Verifying the installation

**Quick setup:**

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```shell
   git clone git@github.com:YOUR_USERNAME/iden.git
   cd iden
   ```

3. **Set up the development environment** (requires `uv` to be installed):
   ```shell
   # Create and activate virtual environment with dependencies
   inv create-venv
   inv install --docs-deps
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. **Install pre-commit hooks**:
   ```shell
   pre-commit install
   ```

### Running tests

```shell
# Run unit tests
inv unit-test

# Run unit tests with coverage
inv unit-test --cov

# Run integration tests
inv integration-test
```

### Code quality checks

```shell
# Check code formatting
inv check-format

# Check linting
inv check-lint

# Run all pre-commit hooks
pre-commit run --all-files
```

### Building documentation

```shell
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
   inv unit-test --cov
   ```
5. Make sure your code lints. The following commands can help you to format the code:
   ```shell
   pre-commit run --all-files
   ```
6. If you've added a new feature, add an example to the `examples/` directory.
7. Update the documentation in the `docs/` directory if needed.

## Issues

We use GitHub issues to track public bugs or feature requests.
For bugs, please ensure your description is clear and concise description, and has sufficient
information to be easily reproducible.
For feature request, please add a clear and concise description of the feature proposal.
Please outline the motivation for the proposal.

## License

By contributing to `iden`, you agree that your contributions will be licensed under the LICENSE
file in the root directory of this source tree.
