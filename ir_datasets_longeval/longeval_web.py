import contextlib
import json
import os
import pickle  # nosec
import sqlite3
from datetime import datetime
from pathlib import Path
from pkgutil import get_data
from typing import List, NamedTuple, Optional

import ir_datasets
import lz4.frame
from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import TrecDocs, TrecQrels, TsvQueries
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import ZipExtractCache, home_path

from ir_datasets_longeval.formats import MetaDataset
from ir_datasets_longeval.util import DownloadConfig, YamlDocumentation

logger = ir_datasets.log.easy()

NAME = "longeval-web"
QREL_DEFS = {
    2: "highly relevant",
    1: "relevant",
    0: "not relevant",
}
SUB_COLLECTIONS_TRAIN = [
    "2022-06",
    "2022-07",
    "2022-08",
    "2022-09",
    "2022-10",
    "2022-11",
    "2022-12",
    "2023-01",
    "2023-02",
]
SUB_COLLECTIONS_TEST = [
    "2023-03",
    "2023-04",
    "2023-05",
    "2023-06",
    "2023-07",
    "2023-08",
]
DUA = "Please confirm you agree to the TREC data usage agreement found at " "<TBD>"


class LongEvalMetadataItem(NamedTuple):
    id: str
    url: str
    last_updated_at: List[int]
    date: List[str]


class LongEvalWebMetadata:
    def __init__(self, dlc, cache_file=None):
        self._dlc = dlc
        self._cache_file = cache_file or f"{self._dlc}/metadata.pklz4"
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            if os.path.exists(self._cache_file):
                try:
                    with lz4.frame.open(self._cache_file, "rb") as f:
                        self._metadata = pickle.load(f)  # nosec
                    logger.info(f"Loaded metadata from cache file {self._cache_file}")

                except Exception as e:
                    logger.warn(f"Failed to load cache file {self._cache_file}: {e}")
                    self._metadata = None

            if self._metadata is None:
                with sqlite3.connect(self._dlc / "collection_db.db") as connection:
                    cursor = connection.cursor()
                    cursor.execute("SELECT id, url, last_updated_at, date FROM mapping")
                    rows = cursor.fetchall()
                    self._metadata = {
                        str(row[0]): LongEvalMetadataItem(
                            str(row[0]),
                            row[1],
                            json.loads(row[2]) if isinstance(row[2], str) else row[2],
                            json.loads(row[3]) if isinstance(row[3], str) else row[3],
                        )
                        for row in rows
                    }

                try:
                    with lz4.frame.open(self._cache_file, "wb") as f:
                        pickle.dump(self._metadata, f)

                except Exception as e:
                    logger.warn(f"Failed to save cache file {self._cache_file}: {e}")

        return self._metadata

    def get_metadata(self, id):
        return self.metadata.get(str(id))


class LongEvalDocument(NamedTuple):
    doc_id: str
    url: str
    last_updated_at: List[int]
    date: List[str]
    text: str

    def default_text(self):
        return self.text


class LongEvalDocs(TrecDocs):
    def __init__(self, dlc, meta=None):
        self._dlc = dlc
        self._meta = meta
        super().__init__(self._dlc)

    @ir_datasets.util.use_docstore
    def docs_iter(self):
        for doc in super().docs_iter():
            if isinstance(doc, LongEvalDocument):
                yield doc
            else:
                docid = doc.doc_id.strip("doc")
                text = doc.text
                if self._meta:
                    metadata = self._meta.get_metadata(docid)
                    url = metadata.url
                    last_updated_at = metadata.last_updated_at
                    date = metadata.date
                else:
                    url = ""
                    last_updated_at = []
                    date = []

                yield LongEvalDocument(docid, url, last_updated_at, date, text)

    # Bug:
    # Document parts in the sub-collection 2023-02 have
    # the wrong file extension jsonl.gz instead of trec.
    # This causes the parser to fail. To fix this, the
    # the docs_iter method is overridden and no extensions
    # are checked.
    def _docs_iter(self, path):
        if Path(path).is_file():
            with open(path, "rb") as f:
                yield from self._parser(f)
        elif Path(path).is_dir():
            for child in path.iterdir():
                yield from self._docs_iter(child)

    def docs_store(self):
        return PickleLz4FullStore(
            path=f"{self._dlc.path()}/docstore.pklz4",
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field="doc_id",
            index_fields=["doc_id"],
        )

    def docs_cls(self):
        return LongEvalDocument


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


class LongEvalWebDataset(Dataset):
    def __init__(
        self,
        base_path: Path,
        meta: Optional[LongEvalWebMetadata] = None,
        yaml_documentation: str = "longeval_web.yaml",
        prior_datasets: Optional[List[str]] = None,
    ):
        """LongEval Web Dataset
        This class needs a metadata file to be available in the base_path for not official datasets.
        The basepath provides the general directory structure of the dataset. From the base path the snapshot and language are inferred. Every non official dataset needs to have follow this structure:
        ```
        <dataset_root>
        └── French
            └── LongEval Train Collection
                ├── qrels
                │   └── <snapshot>_<language>
                │       └── qrels_processed.txt
                ├── Trec
                │   ├── <snapshot>_<language>
                │       ├── <documents split>.trec
                │       ├── ...
                │       └── metadata.json
                └── queries
                    └── <snapshot>_queries.txt
        ```

        Args:
            base_path (Path): Path to the document collection root dir.
            meta (Optional[LongEvalWebMetadata], optional): Path to the metadata dir if available. Defaults to None.
            yaml_documentation (str, optional): Documentation file. Defaults to "longeval_web.yaml".
        """
        documentation = YamlDocumentation(yaml_documentation)
        self.base_path = base_path
        self.meta = meta

        if not base_path or not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(
                f"I expected that the directory {base_path} exists. But the directory does not exist."
            )

        self.snapshot = "_".join(self.base_path.name.split("_")[:-1])
        self.language = self.base_path.name.split("_")[-1]

        timestamp = self.read_property_from_metadata("timestamp")
        self.timestamp = datetime.strptime(timestamp, "%Y-%m")

        if prior_datasets is None:
            prior_datasets = self.read_property_from_metadata("prior-datasets")
        self.prior_datasets = prior_datasets

        docs = LongEvalDocs(ExtractedPath(base_path), meta)

        queries_path = base_path.parents[1] / "queries" / f"{self.snapshot}_queries.txt"
        if not queries_path.exists() or not queries_path.is_file():
            raise FileNotFoundError(
                f"I expected that the file {queries_path} exists. But the directory does not exist."
            )
        queries = TsvQueries(ExtractedPath(queries_path), lang="fr")

        qrels = None
        qrels_path = (
            base_path.parents[1]
            / "qrels"
            / f"{self.snapshot}_{self.language}/qrels_processed.txt"
        )
        if qrels_path.exists() and qrels_path.is_file():
            qrels = TrecQrels(ExtractedPath(qrels_path), QREL_DEFS)

        super().__init__(docs, queries, qrels, documentation)

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
            return [
                LongEvalWebDataset(
                    base_path=self.base_path.parent / f"{i}_{self.language}",
                    meta=self.meta,
                )
                for i in self.prior_datasets
            ]
        else:
            return self.prior_datasets

    def read_property_from_metadata(self, property):
        try:
            return json.load(open(self.base_path / "metadata.json", "r"))[property]
        except FileNotFoundError:
            metadata = json.loads(get_data("ir_datasets_longeval", "metadata.json"))
            return metadata[f"longeval-web/{self.snapshot}"][property]


def register():
    base_path = home_path() / NAME

    dlc = DownloadConfig.context(NAME, base_path)

    data_path = (
        ZipExtractCache(
            dlc["longeval_2025_train_collection"], base_path / "release_2025_p1"
        ).path()
        / "release_2025_p1"
    )

    meta = LongEvalWebMetadata(data_path / "French")

    base_path_train = data_path / "French" / "LongEval Train Collection" / "Trec"

    subsets = {}
    for snapshot in SUB_COLLECTIONS_TRAIN:
        if f"{NAME}/{snapshot}" in registry:
            # Already registered.
            continue

        subsets[snapshot] = LongEvalWebDataset(
            base_path=base_path_train / f"{snapshot}_fr",
            meta=meta,
            prior_datasets=list(subsets.values())[::-1],
        )

    data_path_test = (
        ZipExtractCache(
            dlc["longeval_2025_test_collection"],
            base_path / "LongEval Test Collection",
        ).path()
        / "LongEval Test Collection"
    )
    base_path_test = data_path_test / "Trec"
    for snapshot in SUB_COLLECTIONS_TEST:
        if f"{NAME}/{snapshot}" in registry:
            # Already registered.
            continue

        subsets[snapshot] = LongEvalWebDataset(
            base_path=base_path_test / f"{snapshot}_fr",
            prior_datasets=list(subsets.values())[::-1],
        )

    for s in sorted(subsets):
        registry.register(f"{NAME}/{s}", subsets[s])

    if f"{NAME}/*" in registry:
        return

    registry.register(f"{NAME}/*", MetaDataset(list(subsets.values())))

    if f"{NAME}/clef-2025-test" in registry:
        return
    registry.register(
        f"{NAME}/clef-2025-test",
        MetaDataset([subsets[s] for s in SUB_COLLECTIONS_TEST]),
    )
