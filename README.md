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

Install the package from [PyPI](https://pypi.org/project/ir-datasets-longeval/):

```shell
pip install ir-datasets-longeval
```

## Usage

> [!TIP]
> [LongEval 2026](https://clef-longeval.github.io/): The new `longeval-sci-2026` test collection is now available and we included extra tags for the shared task: `longeval-sci-2026/clef-2026/sci` and `longeval-sci-2026/clef-2026/rag`.


The `ir_datasets_longeval` extension provides a `load` method that returns a LongEval `ir_dataset` that allows to load official versions of the LongEval datasets as well as modified versions that you have on your local file system:

```python
from ir_datasets_longeval import load

# load an official version of the LongEval dataset.
dataset = load("longeval-sci-2026/snapshot-3")

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

## Datasets

### LongEval 2026
<details>
  <summary> Details </summary>

The fourth LongEval Lab in 2026 introduced a new LongEval-Sci test collection. It contains three snapshots that each span three months. `snapshot-1` from March to May, `snapshot-2` from June to August, `snapshot-3` from September to November, all in 2025. Additionally, training queries and qrels are provided for `snapshot-1`. Each snapshot contains different qrels sets: `raw` qrels mark all clicked documents as relevant and `dctr` qrels use the Document Click Through Rate (DCTR) as pseudo relevance label. Additionally, a set of RAG questions is available for the most recent snapshot `snapshot-3`.

#### Meta Tags:
- `longeval-sci-2026/*`
- `longeval-sci-2026/clef-2026/sci`
- `longeval-sci-2026/clef-2026/sci/raw`
- `longeval-sci-2026/clef-2026/sci/dctr`


#### Tags:
- `longeval-sci/snapshot-1`
- `longeval-sci/snapshot-1/raw`
- `longeval-sci/snapshot-1/dctr`
- `longeval-sci/snapshot-2`
- `longeval-sci/snapshot-2/raw`
- `longeval-sci/snapshot-2/dctr`
- `longeval-sci/snapshot-3`
- `longeval-sci/snapshot-3/raw`
- `longeval-sci/snapshot-3/dctr`
- `longeval-sci/snapshot-3/rag`
- `longeval-sci-2026/clef-2026/rag`


</details>


### LongEval Sci
<details>
  <summary> Details </summary>

The third LongEval Lab introduced the first LongEval-Sci test collection. It contains the two snapshots `2024-11` and `2025-01` and additional training queries and qrels for `2024-11`.

#### Meta Tags:
- `longeval-sci/*`
- `longeval-sci/clef-2025-test`


#### Tags:
- `longeval-sci/2024-11/train`
- `longeval-sci/2024-11`
- `longeval-sci/2025-01`




#### Citation:
```bibtex
@inproceedings{DBLP:conf/ecir/AlkhalifaBDEAFSGGILMMMPPSZ24,
  author       = {Rabab Alkhalifa and
                  Hsuvas Borkakoty and
                  Romain Deveaud and
                  Alaa El{-}Ebshihy and
                  Luis Espinosa Anke and
                  Tobias Fink and
                  Gabriela Gonz{\'{a}}lez S{\'{a}}ez and
                  Petra Galusc{\'{a}}kov{\'{a}} and
                  Lorraine Goeuriot and
                  David Iommi and
                  Maria Liakata and
                  Harish Tayyar Madabushi and
                  Pablo Medina{-}Alias and
                  Philippe Mulhem and
                  Florina Piroi and
                  Martin Popel and
                  Christophe Servan and
                  Arkaitz Zubiaga},
  editor       = {Nazli Goharian and
                  Nicola Tonellotto and
                  Yulan He and
                  Aldo Lipani and
                  Graham McDonald and
                  Craig Macdonald and
                  Iadh Ounis},
  title        = {LongEval: Longitudinal Evaluation of Model Performance at {CLEF} 2024},
  booktitle    = {Advances in Information Retrieval - 46th European Conference on Information
                  Retrieval, {ECIR} 2024, Glasgow, UK, March 24-28, 2024, Proceedings,
                  Part {VI}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14613},
  pages        = {60--66},
  publisher    = {Springer},
  year         = {2024},
  url          = {https://doi.org/10.1007/978-3-031-56072-9\_8},
  doi          = {10.1007/978-3-031-56072-9\_8},
  timestamp    = {Mon, 15 Apr 2024 08:25:15 +0200},
  biburl       = {https://dblp.org/rec/conf/ecir/AlkhalifaBDEAFSGGILMMMPPSZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
</details>


### LongEval Web
<details>
  <summary> Details </summary>

The third LongEval Lab continued the LongEval test collection and made many more snapshots available. It is only available in French and contains the monthly snapshots `2022-06`, `2022-07`, `2022-08`, `2022-09`, `2022-10`, `2022-11`, `2022-12`, `2023-01`, `2023-02`, `2023-03`, `2023-04`, `2023-05`, `2023-06`, `2023-07`, `2023-08`.

#### Meta Tags:
- `longeval-web/*`
- `longeval-web/clef-2025-test`


#### Tags:
- `longeval-web/2022-06`
- `longeval-web/2022-07`
- `longeval-web/2022-08`
- `longeval-web/2022-09`
- `longeval-web/2022-10`
- `longeval-web/2022-11`
- `longeval-web/2022-12`
- `longeval-web/2023-01`
- `longeval-web/2023-02`
- `longeval-web/2023-03`
- `longeval-web/2023-04`
- `longeval-web/2023-05`
- `longeval-web/2023-06`
- `longeval-web/2023-07`
- `longeval-web/2023-08`



#### Citation:
```bibtex
@inproceedings{DBLP:conf/ecir/AlkhalifaBDEAFSGGILMMMPPSZ24,
  author       = {Rabab Alkhalifa and
                  Hsuvas Borkakoty and
                  Romain Deveaud and
                  Alaa El{-}Ebshihy and
                  Luis Espinosa Anke and
                  Tobias Fink and
                  Gabriela Gonz{\'{a}}lez S{\'{a}}ez and
                  Petra Galusc{\'{a}}kov{\'{a}} and
                  Lorraine Goeuriot and
                  David Iommi and
                  Maria Liakata and
                  Harish Tayyar Madabushi and
                  Pablo Medina{-}Alias and
                  Philippe Mulhem and
                  Florina Piroi and
                  Martin Popel and
                  Christophe Servan and
                  Arkaitz Zubiaga},
  editor       = {Nazli Goharian and
                  Nicola Tonellotto and
                  Yulan He and
                  Aldo Lipani and
                  Graham McDonald and
                  Craig Macdonald and
                  Iadh Ounis},
  title        = {LongEval: Longitudinal Evaluation of Model Performance at {CLEF} 2024},
  booktitle    = {Advances in Information Retrieval - 46th European Conference on Information
                  Retrieval, {ECIR} 2024, Glasgow, UK, March 24-28, 2024, Proceedings,
                  Part {VI}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14613},
  pages        = {60--66},
  publisher    = {Springer},
  year         = {2024},
  url          = {https://doi.org/10.1007/978-3-031-56072-9\_8},
  doi          = {10.1007/978-3-031-56072-9\_8},
  timestamp    = {Mon, 15 Apr 2024 08:25:15 +0200},
  biburl       = {https://dblp.org/rec/conf/ecir/AlkhalifaBDEAFSGGILMMMPPSZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
</details>



### LongEval 2023
The dataset is hosted at [Lindat](https://lindat.mff.cuni.cz/repository/items/3b505342-6bad-48f8-8bc3-d0ae09d3f6e4) and a local copy is needed to be placed in the ir_dataset directory. 

<details>
  <summary> Details </summary>

The original LongEval test collection is available in French and English and comprises the three snapshots `2022-06` (WT), `2022-07` (ST), and `2022-08` (LT). The initial version assigned different IDs to documents and queries present in multiple snapshots. The original IDs are available in the `non-unified` tags.

#### Tags:
- `longeval-2023`
- `longeval-2023/2022-06/fr`
- `longeval-2023/2022-07/fr`
- `longeval-2023/2022-09/fr`
- `longeval-2023/2022-06/en`
- `longeval-2023/2022-07/en`
- `longeval-2023/2022-09/en`
- `longeval-2023/2022-06/fr/non-unified`
- `longeval-2023/2022-07/fr/non-unified`
- `longeval-2023/2022-09/fr/non-unified`
- `longeval-2023/2022-06/en/non-unified`
- `longeval-2023/2022-07/en/non-unified`
- `longeval-2023/2022-09/en/non-unified`


#### Citation:
```bibtex
@inproceedings{DBLP:conf/sigir/GaluscakovaDSMG23,
  author       = {Petra Galusc{\'{a}}kov{\'{a}} and
                  Romain Deveaud and
                  Gabriela Gonz{\'{a}}lez S{\'{a}}ez and
                  Philippe Mulhem and
                  Lorraine Goeuriot and
                  Florina Piroi and
                  Martin Popel},
  editor       = {Hsin{-}Hsi Chen and
                  Wei{-}Jou (Edward) Duh and
                  Hen{-}Hsen Huang and
                  Makoto P. Kato and
                  Josiane Mothe and
                  Barbara Poblete},
  title        = {LongEval-Retrieval: French-English Dynamic Test Collection for Continuous
                  Web Search Evaluation},
  booktitle    = {Proceedings of the 46th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2023, Taipei,
                  Taiwan, July 23-27, 2023},
  pages        = {3086--3094},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3539618.3591921},
  doi          = {10.1145/3539618.3591921},
  timestamp    = {Wed, 25 Feb 2026 08:28:08 +0100},
  biburl       = {https://dblp.org/rec/conf/sigir/GaluscakovaDSMG23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


</details>

## Citation

If you use this package, please cite the original ir_datasets paper and this extension:

```
@inproceedings{ir_datasets_longeval,
  author       = {J{\"{u}}ri Keller and Maik Fr{\"{o}}be and Gijs Hendriksen and Daria Alexander and Martin Potthast and Philipp Schaer},
  title        = {Simplified Longitudinal Retrieval Experiments: A Case Study on Query Expansion and Document Boosting},
  booktitle    = {Experimental {IR} Meets Multilinguality, Multimodality, and Interaction - 16th International Conference of the {CLEF} Association, {CLEF} 2024, Madrid, Spain, September 9-12, 2025, Proceedings, Part {I}},
  series       = {Lecture Notes in Computer Science},
  publisher    = {Springer},
  year         = {2025}
}
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
