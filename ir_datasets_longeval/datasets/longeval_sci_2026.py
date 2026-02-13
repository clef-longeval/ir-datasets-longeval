import contextlib
import json
import os
from datetime import datetime
from pathlib import Path
from pkgutil import get_data
from typing import Dict, List, NamedTuple, Optional

from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import JsonlDocs, TrecQrels, TsvQueries
from ir_datasets.util import ZipExtractCache, home_path

from ir_datasets_longeval.formats import MetaDataset
from ir_datasets_longeval.util import DownloadConfig, YamlDocumentation

NAME = "longeval-sci-2026"
QREL_DEFS = {
    2: "highly relevant",
    1: "relevant",
    0: "not relevant",
}
SUB_COLLECTIONS = ["2024-11"]
MAPPING = {
    "doc_id": "id",
    "title": "title",
    "abstract": "abstract",
    "authors": "authors",
    "createdDate": "createdDate",
    "doi": "doi",
    "arxivId": "arxivId",
    "pubmedId": "pubmedId",
    "magId": "magId",
    "oaiIds": "oaiIds",
    "links": "links",
    "publishedDate": "publishedDate",
    "updatedDate": "updatedDate",
}


class LongEvalSciDoc(NamedTuple):
    doc_id: str
    title: str
    abstract: str
    authors: List[Dict[str, str]]
    createdDate: Optional[str]
    doi: Optional[str]
    arxivId: Optional[str]
    pubmedId: Optional[str]
    magId: Optional[str]
    oaiIds: Optional[List[str]]
    links: List[Dict[str, str]]
    publishedDate: str
    updatedDate: str

    def default_text(self):
        ret = self.title
        if self.abstract:
            ret += " " + self.abstract
        return ret


class ExtractedPath:
    def __init__(self, path):
        self._path = path

    def path(self, force=True):
        if force and not self._path.exists():
            raise FileNotFoundError(self._path)
        return self._path

    @contextlib.contextmanager
    def stream(self):
        with open(self._path, "rb") as f:
            yield f


def docs_iter_without_duplicates(iterator):
    returned = set()
    for doc in iterator:
        if doc.doc_id not in returned and doc.default_text():
            returned.add(doc.doc_id)
            yield doc


class LongEvalSciDataset(Dataset):
    def __init__(
        self,
        base_path: Path,
        yaml_documentation: str = "longeval_sci_2026.yaml",
        timestamp: Optional[str] = None,
        prior_datasets: Optional[List[str]] = None,
        snapshot: Optional[str] = None,
        qrels_path: Optional[Path] = None,
    ):
        """LongEvalSciDataset class

        Args:
            base_path (Path): Base path to the root of a typical longeval sci dataset directory structure.
            yaml_documentation (str, optional): Documentation file. Defaults to "longeval_sci_2026.yaml".
            timestamp (Optional[str], optional): Timestamp in the YYYY-MM format of the snapshot. Defaults to None.
            prior_datasets (Optional[List[str]], optional): List of all snapshot that come before this snapshot. Defaults to None.
            snapshot (Optional[str], optional): Name of the sub-collection. This was previously also referred to as lag and is most likely similar to the dataset_id. Defaults to None.
        """
        documentation = YamlDocumentation(yaml_documentation)
        self.base_path = base_path

        if not base_path or not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(
                f"I expected that the directory {base_path} exists. But the directory does not exist."
            )

        if not timestamp:
            timestamp = self.read_property_from_metadata("timestamp")

        if timestamp is None:
            raise ValueError("Timestamp cannot be None.")
        self.timestamp = datetime.strptime(timestamp, "%Y-%m")

        if not snapshot:
            try:
                snapshot = self.read_property_from_metadata("snapshot")
            except KeyError:
                snapshot = None
        self.snapshot = snapshot

        if prior_datasets is None:
            prior_datasets = self.read_property_from_metadata("prior-datasets")

        self.prior_datasets = prior_datasets

        docs_path = base_path / snapshot.strip("-train") / "documents"

        if not docs_path.exists() or not docs_path.is_dir():
            raise FileNotFoundError(
                f"I expected that the directory {docs_path} exists. But the directory does not exist."
            )

        jsonl_doc_files = os.listdir(docs_path)
        if len(jsonl_doc_files) == 0:
            raise ValueError(
                f"The directory {docs_path} has no jsonl files. This is likely an error."
            )
        docs_paths = []
        for file in jsonl_doc_files:
            if file.endswith(".jsonl"):
                docs_paths.append(ExtractedPath(docs_path / file))
        docs = JsonlDocs(
            docs_paths,
            doc_cls=LongEvalSciDoc,
            docstore_path=f"{docs_path}/docstore.pklz4",
            mapping=MAPPING,
        )

        if "train" in snapshot:
            queries = TsvQueries(ExtractedPath(base_path / "queries-train.tsv"))
        else:
            queries = TsvQueries(ExtractedPath(base_path / "queries-test.tsv"))

        qrels = None
        if qrels_path is not None and qrels_path.exists() and qrels_path.is_file():
            qrels = TrecQrels(ExtractedPath(qrels_path), QREL_DEFS)

        super().__init__(docs, queries, qrels, documentation)
        original_iter = self.docs_iter
        self.docs_iter = lambda: docs_iter_without_duplicates(original_iter())

    def get_timestamp(self):
        return self.timestamp

    def get_snapshot(self):
        return self.snapshot

    def get_datasets(self):
        return None

    def get_prior_datasets(self):
        if not self.prior_datasets:
            return []
        elif isinstance(self.prior_datasets[0], str):
            # prior_datasets = []
            # for snapshot in self.prior_datasets:
            #     prior_datasets.append(
            #         LongEvalSciDataset(
            #             base_path=self.base_path ,
            #             prior_datasets=self.prior_datasets[:-1],
            #             snapshot=self.snapshot[:4],
            #             timestamp="2025-" + self.snapshot[:2],
            #         )
            #     )
            return [
                LongEvalSciDataset(self.base_path, snapshot=i)
                for i in self.prior_datasets
            ]
        else:
            return self.prior_datasets

    def read_property_from_metadata(self, property):
        try:
            return json.load(open(self.base_path / "metadata.json", "r"))[property]
        except FileNotFoundError:
            metadata = json.loads(get_data("ir_datasets_longeval", "etc/metadata.json"))
            return metadata[f"longeval-sci/{self.timestamp.strftime('%Y-%m')}"][
                property
            ]


from shutil import copyfile


def register_spot_check_datasets():
    if f"{NAME}/spot-check/no-prior-data" in registry:
        return

    base_path = home_path() / NAME / "spot-check"
    dlc = DownloadConfig.context(NAME, base_path)

    def extract_from_tira(name):
        return ZipExtractCache(dlc[name], base_path / name).path()

    no_prior_data_inputs = extract_from_tira(
        "sci-spot-check-no-prior-data-20260210-training-inputs"
    )
    with_prior_data_inputs = extract_from_tira(
        "sci-spot-check-with-prior-data-20260210-training-inputs"
    )
    no_prior_data_truths = extract_from_tira(
        "sci-spot-check-no-prior-data-20260210-training-truths"
    )
    with_prior_data_truths = extract_from_tira(
        "sci-spot-check-with-prior-data-20260210-training-truths"
    )

    if not (no_prior_data_inputs / "qrels.txt").exists():
        copyfile(no_prior_data_truths / "qrels.txt", no_prior_data_inputs / "qrels.txt")

    if not (with_prior_data_inputs / "qrels.txt").exists():
        copyfile(
            with_prior_data_truths / "qrels.txt", with_prior_data_inputs / "qrels.txt"
        )

    no_prior = LongEvalSciDataset(
        no_prior_data_inputs,
        snapshot="03-05",
        qrels_path=no_prior_data_inputs / "qrels.txt",
    )
    prior_data = LongEvalSciDataset(
        with_prior_data_inputs,
        snapshot="03-05",
        qrels_path=with_prior_data_inputs / "qrels.txt",
    )

    registry.register(f"{NAME}/spot-check/no-prior-data", no_prior)
    registry.register(f"{NAME}/spot-check/with-prior-data", prior_data)
    registry.register(f"{NAME}/spot-check/*", MetaDataset([no_prior, prior_data]))


def register():
    if f"{NAME}/03-05" in registry:
        # Already registered.
        return
    base_path = home_path() / NAME
    dlc = DownloadConfig.context(NAME, base_path)

    subsets = {}

    # 2026 train
    data_path = ZipExtractCache(dlc["longeval_sci_2026"], base_path).path()

    subsets["03-05-train"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="03-05-train",
    )
    subsets["03-05-train/raw"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="03-05-train",
        qrels_path=data_path / "03-05" / "qrels-snapshot-train-raw.txt",
    )
    subsets["03-05-train/dctr"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="03-05-train",
        qrels_path=data_path / "03-05" / "qrels-snapshot-train-dctr.txt",
    )

    # 2026 test

    ### 03-05
    subsets["03-05"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="03-05",
    )
    subsets["03-05/raw"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="03-05",
        qrels_path=data_path / "03-05" / "qrels-snapshot-1-raw.txt",
    )
    subsets["03-05/dctr"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="03-05",
        qrels_path=data_path / "03-05" / "qrels-snapshot-1-dctr.txt",
    )

    ### 06-08
    subsets["06-08"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-06",
        prior_datasets=[subsets["03-05"]],
        snapshot="06-08",
    )
    subsets["06-08/raw"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-06",
        prior_datasets=[subsets["03-05/raw"]],
        snapshot="06-08",
        qrels_path=data_path / "06-08" / "qrels-snapshot-2-raw.txt",
    )
    subsets["06-08/dctr"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-06",
        prior_datasets=[subsets["03-05/dctr"]],
        snapshot="06-08",
        qrels_path=data_path / "06-08" / "qrels-snapshot-2-dctr.txt",
    )

    ### 09-11
    subsets["09-11"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-09",
        prior_datasets=[subsets["06-08/raw"], subsets["03-05/raw"]],
        snapshot="09-11",
    )
    subsets["09-11/raw"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-09",
        prior_datasets=[subsets["06-08/raw"], subsets["03-05/raw"]],
        snapshot="09-11",
        qrels_path=data_path / "09-11" / "qrels-snapshot-3-raw.txt",
    )
    subsets["09-11/dctr"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-09",
        prior_datasets=[subsets["06-08/dctr"], subsets["03-05/dctr"]],
        snapshot="09-11",
        qrels_path=data_path / "09-11" / "qrels-snapshot-3-dctr.txt",
    )

    for s in sorted(subsets):
        registry.register(f"{NAME}/{s}", subsets[s])

    if f"{NAME}/*" in registry:
        return
    registry.register(
        f"{NAME}/*", MetaDataset([subsets["03-05"], subsets["06-08"], subsets["09-11"]])
    )

    ### Meta datasets
    if f"{NAME}/clef-2026" in registry:
        return
    registry.register(
        f"{NAME}/clef-2026",
        MetaDataset([subsets["03-05"], subsets["06-08"], subsets["09-11"]]),
    )
    registry.register(
        f"{NAME}/clef-2026/raw",
        MetaDataset([subsets["03-05/raw"], subsets["06-08/raw"], subsets["09-11/raw"]]),
    )
    registry.register(
        f"{NAME}/clef-2026/dctr",
        MetaDataset(
            [subsets["03-05/dctr"], subsets["06-08/dctr"], subsets["09-11/dctr"]]
        ),
    )
