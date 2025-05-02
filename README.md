[![PyPi](https://img.shields.io/pypi/v/ir-datasets-longeval?style=flat-square)](https://pypi.org/project/ir-datasets-longeval/)
[![CI](https://img.shields.io/github/actions/workflow/status/jueri/ir-datasets-longeval/ci.yml?branch=main&style=flat-square)](https://github.com/jueri/ir-datasets-longeval/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/jueri/ir-datasets-longeval?style=flat-square)](https://codecov.io/github/jueri/ir-datasets-longeval/)
[![Python](https://img.shields.io/pypi/pyversions/ir-datasets-longeval?style=flat-square)](https://pypi.org/project/ir-datasets-longeval/)
[![Issues](https://img.shields.io/github/issues/jueri/ir-datasets-longeval?style=flat-square)](https://github.com/jueri/ir-datasets-longeval/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jueri/ir-datasets-longeval?style=flat-square)](https://github.com/jueri/ir-datasets-longeval/commits)
[![Downloads](https://img.shields.io/pypi/dm/ir-datasets-longeval?style=flat-square)](https://pypi.org/project/ir-datasets-longeval/)
[![License](https://img.shields.io/github/license/jueri/ir-datasets-longeval?style=flat-square)](LICENSE)

# 💾 ir-datasets-longeval

Extension for accessing the [LongEval](https://clef-longeval.github.io/) datasets via [ir_datasets](https://ir-datasets.com/).


## Installation

Install the package from PyPI:

```shell
pip install ir-datasets-longeval
```

## Usage

The `ir_datasets_longeval` extension provides an `load` method that returns a LongEval `ir_dataset` that allows to load official versions of the LongEval datasets as well as modified versions that you have on your local filesystem:

```python
from ir_datasets_longeval import load

# load an official version of the LongEval dataset.
dataset = load("longeval-web/2022-06")

# load a local copy of a LongEval dataset.
# E.g., so that you can easily run your approach on modified data.
dataset = load("<PATH-TO-A-DIRECTORY-ON-YOUR-MACHINE>")

# From now on, you can use dataset as any ir_dataset
```

LongEval datasets have a set of temporal specifics that you can use:

```Python
# At what time does/did a dataset take place?
dataset.get_timestamp()

# Each dataset can have a list of zero or more past datasets/interactions.
# You can incorporate them in your retrieval system:
for past_dataset in dataset.get_prior_datasets():
    # `past_dataset` is an LongEval `ir_dataset` with the same functionality as the `dataset`
    past_dataset.get_timestamp()
```


If you want to use the [CLI](https://ir-datasets.com/cli.html), just use the `ir_datasets_longeval` instead of `ir_datasets`. All CLI commands will work as usual, e.g., to list the officially available datasets:

```shell
ir_datasets_longeval list
```

## Development

To build this package and contribute to its development you need to install the `build`, `setuptools`, and `wheel` packages (pre-installed on most systems):

```shell
pip install build setuptools wheel
```

Create and activate a virtual environment:

```shell
python3.10 -m venv venv/
source venv/bin/activate
```

### Dependencies

Install the package and test dependencies:

```shell
pip install -e .[tests]
```

### Testing

Verify your changes against the test suite to verify.

```shell
ruff check .                   # Code format and LINT
mypy .                         # Static typing
bandit -c pyproject.toml -r .  # Security
pytest .                       # Unit tests
```

Please also add tests for your newly developed code.

### Build wheels

Wheels for this package can be built with:

```shell
python -m build
```

## Support

If you have any problems using this package, please file an [issue](https://github.com/jueri/ir-datasets-longeval/issues/new).
We're happy to help!

## Fork Notice

This repository is a fork of [ir-datasets-clueweb22](https://github.com/janheinrichmerker/ir-datasets-clueweb22), originally developed by Jan Heinrich Merker. All credit for the original work goes to him, and this fork retains the original MIT License. The changes made in this fork include an adaptation from the clueweb22 dataset to the LongEval datasets.


## License

This repository is released under the [MIT license](LICENSE).
