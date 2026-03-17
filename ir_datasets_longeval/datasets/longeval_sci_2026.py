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
        queries_path: Optional[Path] = None,
    ):
        """LongEvalSciDataset class

        Args:
            base_path (Path): Base path to the root of a typical longeval sci dataset directory structure.
            yaml_documentation (str, optional): Documentation file. Defaults to "longeval_sci_2026.yaml".
            timestamp (Optional[str], optional): Timestamp in the YYYY-MM format of the snapshot. Defaults to None.
            prior_datasets (Optional[List[str]], optional): List of all snapshot that come before this snapshot. Defaults to None.
            snapshot (Optional[str], optional): Name of the sub-collection. This was previously also referred to as lag and is most likely similar to the dataset_id. Defaults to None.
            qrels_path (Optional[Path], optional): Path to the qrels file. Defaults to None.
            queries_path (Optional[Path], optional): Path to the queries file. Defaults to None.
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

        docs_path = base_path

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

        if not queries_path:
            queries_path = base_path / "queries-test.tsv"

        # if not queries_path.is_file():
        #     raise FileNotFoundError(
        #         f"I expected that the file {queries_path} exists. But the directory does not exist."
        #     )

        queries = TsvQueries(queries_path)

        qrels = None
        # if qrels_path is not None and qrels_path.is_file():
        if qrels_path:
            qrels = TrecQrels(qrels_path, QREL_DEFS)

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


def register():
    if f"{NAME}/snapshot-1" in registry:
        # Already registered.
        return
    base_path = home_path() / NAME
    dlc = DownloadConfig.context(NAME, base_path)

    subsets = {}

    # 2026 train
    data_path_train = (
        ZipExtractCache(
            dlc["longeval_sci_training_2026_documents"],
            base_path / "longeval_sci_training_2026",
        ).path()
        / "data"
        / "processed"
        / "doc_collection_09032026_abstract_2"
        / "snapshot-1"
        / "longeval_sci_training_2026_abstract"
        / "documents"
    )

    queries_path_train = dlc["longeval_sci_training_2026_queries"]

    subsets["snapshot-1/train"] = LongEvalSciDataset(
        base_path=data_path_train,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="snapshot-1",
        queries_path=queries_path_train,
    )
    subsets["snapshot-1/train/raw"] = LongEvalSciDataset(
        base_path=data_path_train,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="snapshot-1",
        qrels_path=dlc["longeval_sci_training_2026_qrels_raw"],
        queries_path=queries_path_train,
    )
    subsets["snapshot-1/train/dctr"] = LongEvalSciDataset(
        base_path=data_path_train,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="snapshot-1",
        qrels_path=dlc["longeval_sci_training_2026_qrels_dctr"],
        queries_path=queries_path_train,
    )

    # 2026 test
    queries_path_test = dlc["longeval_sci_testing_2026_queries"]

    ### 03-05
    subsets["snapshot-1"] = LongEvalSciDataset(
        base_path=data_path_train,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="snapshot-1",
        queries_path=queries_path_test,
    )

    subsets["snapshot-1/raw"] = LongEvalSciDataset(
        base_path=data_path_train,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="snapshot-1",
        qrels_path=dlc["test_qrels_snapshot_1_raw"],
        queries_path=queries_path_test,
    )
    subsets["snapshot-1/dctr"] = LongEvalSciDataset(
        base_path=data_path_train,
        timestamp="2025-03",
        prior_datasets=[],
        snapshot="snapshot-1",
        qrels_path=dlc["test_qrels_snapshot_1_dctr"],
        queries_path=queries_path_test,
    )

    ### 06-08
    data_path = (
        ZipExtractCache(
            dlc["longeval_sci_06_08_2026_documents"],
            base_path / "longeval_sci_06_08_2026_documents",
        ).path()
        / "data"
        / "processed"
        / "doc_collection_09032026_abstract_2"
        / "snapshot-2"
        / "longeval_sci_test-06-08_2026_abstract"
        / "documents"
    )

    subsets["snapshot-2"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-06",
        prior_datasets=[subsets["snapshot-1"]],
        snapshot="snapshot-2",
        queries_path=queries_path_test,
    )
    subsets["snapshot-2/raw"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-06",
        prior_datasets=[subsets["snapshot-1/raw"]],
        snapshot="snapshot-2",
        qrels_path=dlc["test_qrels_snapshot_2_raw"],
        queries_path=queries_path_test,
    )
    subsets["snapshot-2/dctr"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-06",
        prior_datasets=[subsets["snapshot-1/dctr"]],
        snapshot="snapshot-2",
        qrels_path=dlc["test_qrels_snapshot_2_dctr"],
        queries_path=queries_path_test,
    )

    ### 09-11
    data_path = (
        ZipExtractCache(
            dlc["longeval_sci_09_11_2026_documents"],
            base_path / "longeval_sci_09_11_2026_documents",
        ).path()
        / "data"
        / "processed"
        / "doc_collection_09032026_abstract_2"
        / "snapshot-3"
        / "longeval_sci_test-09-11_2026_abstract"
        / "documents"
    )

    subsets["snapshot-3"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-09",
        prior_datasets=[subsets["snapshot-2"], subsets["snapshot-1"]],
        snapshot="snapshot-3",
        queries_path=queries_path_test,
    )
    subsets["snapshot-3/raw"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-09",
        prior_datasets=[subsets["snapshot-2/raw"], subsets["snapshot-1/raw"]],
        snapshot="snapshot-3",
        qrels_path=dlc["test_qrels_snapshot_3_raw"],
        queries_path=queries_path_test,
    )
    subsets["snapshot-3/dctr"] = LongEvalSciDataset(
        base_path=data_path,
        timestamp="2025-09",
        prior_datasets=[subsets["snapshot-2/dctr"], subsets["snapshot-1/dctr"]],
        snapshot="snapshot-3",
        qrels_path=dlc["test_qrels_snapshot_3_dctr"],
        queries_path=queries_path_test,
    )

    for s in sorted(subsets):
        registry.register(f"{NAME}/{s}", subsets[s])

    if f"{NAME}/*" in registry:
        return

    registry.register(
        f"{NAME}/*",
        MetaDataset(
            [subsets["snapshot-1"], subsets["snapshot-2"], subsets["snapshot-3"]]
        ),
    )
    registry.register(
        f"{NAME}/*/raw",
        MetaDataset(
            [
                subsets["snapshot-1/raw"],
                subsets["snapshot-2/raw"],
                subsets["snapshot-3/raw"],
            ]
        ),
    )
    registry.register(
        f"{NAME}/*/dctr",
        MetaDataset(
            [
                subsets["snapshot-1/dctr"],
                subsets["snapshot-2/dctr"],
                subsets["snapshot-3/dctr"],
            ]
        ),
    )

    ### Meta datasets
    if f"{NAME}/clef-2026/sci" in registry:
        return
    registry.register(
        f"{NAME}/clef-2026/sci",
        MetaDataset(
            [subsets["snapshot-1"], subsets["snapshot-2"], subsets["snapshot-3"]]
        ),
    )
    registry.register(
        f"{NAME}/clef-2026/sci/raw",
        MetaDataset(
            [
                subsets["snapshot-1/raw"],
                subsets["snapshot-2/raw"],
                subsets["snapshot-3/raw"],
            ]
        ),
    )
    registry.register(
        f"{NAME}/clef-2026/sci/dctr",
        MetaDataset(
            [
                subsets["snapshot-1/dctr"],
                subsets["snapshot-2/dctr"],
                subsets["snapshot-3/dctr"],
            ]
        ),
    )
