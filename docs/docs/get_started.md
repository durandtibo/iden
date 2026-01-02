# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep the system in order.

## Installing with `pip` (recommended)

The following command installs the latest version of the library:

```shell
pip install iden
```

To make the package as slim as possible, only the packages required to use `iden` are installed.
It is possible to install all the optional dependencies by running the following command:

```shell
pip install 'iden[all]'
```

This command also installs NumPy and PyTorch.
It is also possible to install the optional packages manually or to select specific packages to install.

### Installing specific optional dependencies

You can install individual optional dependencies as needed:

```shell
# Install with NumPy support
pip install 'iden[numpy]'

# Install with PyTorch support
pip install 'iden[torch]'

# Install with YAML support
pip install 'iden[pyyaml]'

# Install with safetensors support
pip install 'iden[safetensors]'

# Install with cloudpickle support
pip install 'iden[cloudpickle]'

# Install with joblib support
pip install 'iden[joblib]'
```

## Installing from source

To install `iden` from source, you'll need [`uv`](https://docs.astral.sh/uv/) for dependency management.
If `uv` is not already installed, you can install it using:

```shell
pip install uv
```

Then, clone the git repository:

```shell
git clone git@github.com:durandtibo/iden.git
cd iden
```

It is recommended to create a Python 3.10+ virtual environment. This step is optional and
can be skipped. To create a virtual environment, the following command can be used:

```shell
make conda
```

This command automatically creates a conda virtual environment. When the virtual environment is created, it
can be activated with the following command:

```shell
conda activate iden
```

Alternatively, you can use `uv` to create a virtual environment:

```shell
make setup-venv
```

This will create a virtual environment using `uv` and install all development dependencies.

## Verifying the installation

After installation, the required packages can be installed or updated with the following
command:

```shell
make install
```

Finally, the installation can be verified with the following command:

```shell
make unit-test-cov
```
