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

This command also installed NumPy and PyTorch.
It is also possible to install the optional packages manually or to select the packages to install.
In the following example, only NumPy is installed:

```shell
pip install iden numpy
```

## Installing from source

To install `iden` from source, the steps below can be followed. First, [`poetry`](https://python-poetry.org/docs/master/) needs to be installed. `poetry` is used to manage and install
the dependencies.
If `poetry` is already installed on the machine, this step can be skipped. There are several ways to
install `poetry`, and any preferred method can be used. The `poetry` installation can be checked by
running the following command:

```shell
poetry --version
```

Then, the git repository can be cloned:

```shell
git clone git@github.com:durandtibo/iden.git
```

It is recommended to create a Python 3.8+ virtual environment. This step is optional and
can be skipped. To create a virtual environment, the following command can be used:

```shell
make conda
```

This command automatically creates a conda virtual environment. When the virtual environment is created, it
can be activated with the following command:

```shell
conda activate iden
```

This example uses `conda` to create a virtual environment, but other tools or
configurations can be used. Then, the required packages to use `iden` should be installed with the following
command:

```shell
make install
```

This command will install all the required packages. This command can also be used to update the
required packages. This command will check if there is a more recent package available and will
install it. Finally, the installation can be tested with the following command:

```shell
make unit-test-cov
```
