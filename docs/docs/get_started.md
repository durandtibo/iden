# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep the system in order.

## Installing with `uv pip` (recommended)

The following command installs the latest version of the library:

```shell
uv pip install iden
```

To make the package as slim as possible, only the packages required to use `iden` are installed.
It is possible to install all the optional dependencies by running the following command:

```shell
uv pip install 'iden[all]'
```

This command also installs NumPy and PyTorch.
It is also possible to install the optional packages manually or to select specific packages to install.

### Installing specific optional dependencies

You can install individual optional dependencies as needed:

```shell
# Install with NumPy support
uv pip install 'iden[numpy]'

# Install with PyTorch support
uv pip install 'iden[torch]'

# Install with YAML support
uv pip install 'iden[pyyaml]'

# Install with safetensors support
uv pip install 'iden[safetensors]'

# Install with cloudpickle support
uv pip install 'iden[cloudpickle]'

# Install with joblib support
uv pip install 'iden[joblib]'
```

## Installing from source

To install `iden` from source, you'll need [`uv`](https://docs.astral.sh/uv/) for dependency management.
If `uv` is not already installed, please refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Then, clone the git repository:

```shell
git clone git@github.com:durandtibo/iden.git
cd iden
```

It is recommended to create a Python 3.10+ virtual environment. This step is optional and
can be skipped. To create a virtual environment, the following command can be used:

```shell
inv conda
```

This command automatically creates a conda virtual environment. When the virtual environment is created, it
can be activated with the following command:

```shell
conda activate iden
```

Alternatively, you can use `uv` to create a virtual environment:

```shell
inv create-venv
```

This will create a virtual environment using `uv` and install invoke for task management.

## Verifying the installation

After installation, the required packages can be installed or updated with the following
command:

```shell
inv install
```

Finally, the installation can be verified with the following command:

```shell
inv unit-test --cov
```

## Development workflow

For contributors, the following commands are commonly used:

### Running tests

```shell
# Run unit tests
inv unit-test

# Run unit tests with coverage
inv unit-test --cov

# Run integration tests
inv integration-test

# Run all tests
inv all-test
```

### Code quality checks

```shell
# Check code formatting
inv check-format

# Check linting
inv check-lint

# Check type hints
inv check-types

# Format docstrings
inv docformat
```

### Building and testing documentation

```shell
# Install documentation dependencies (if not already installed)
inv install --docs-deps

# Build and serve documentation locally
cd docs
mkdocs serve
```

Then visit http://localhost:8000 in your browser.

### Managing dependencies

```shell
# Update all dependencies to latest versions
inv update

# Show installed packages
inv show-installed-packages
```

For more detailed contribution guidelines, please refer to [CONTRIBUTING.md](https://github.com/durandtibo/iden/blob/main/CONTRIBUTING.md).
