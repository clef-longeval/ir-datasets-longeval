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

Using this extension is simple. Just register the additional datasets by calling `register()`. Then you can load the datasets with [ir_datasets](https://ir-datasets.com/python.html) as usual:

```python
from ir_datasets import load
from ir_datasets_longeval import register

# Register the longeval datasets.
register()
# Use ir_datasets as usual.
dataset = load("longeval-web/2022-06")
```

You can also register only the `longeval-web` or `longeval-sci` dataset:

```Python
from ir_datasets import load
from ir_datasets_longeval import register

# Register the longeval datasets.
register("longeval-web")
```


If you want to use the [CLI](https://ir-datasets.com/cli.html), just use the `ir_datasets_longeval` instead of `ir_datasets`. All CLI commands will work as usual, e.g., to list the available datasets:

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
