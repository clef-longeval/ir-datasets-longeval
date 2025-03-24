import contextlib
import json
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import ir_datasets
import lz4.frame
from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import TrecDocs, TrecQrels, TsvQueries
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import LocalDownload, RelativePath, ZipExtractCache, home_path

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
                        self._metadata = pickle.load(f)
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
    def __init__(self, dlc, meta):
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
                metadata = self._meta.get_metadata(docid)
                url = metadata.url
                last_updated_at = metadata.last_updated_at
                date = metadata.date
                text = doc.text

                yield LongEvalDocument(docid, url, last_updated_at, date, text)

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
        meta: LongEvalWebMetadata,
        yaml_documentation: str = "longeval_web.yaml",
        timestamp: Optional[str] = None,
        prior_datasets: Optional[List[str]] = None,
    ):
        documentation = YamlDocumentation(yaml_documentation)
        self.base_path = base_path
        self.meta = meta

        if not base_path or not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(
                f"I expected that the directory {base_path} exists. But the directory does not exist."
            )
        if not timestamp:
            timestamp = self.read_property_from_metadata("timestamp")

        self.timestamp = datetime.strptime(timestamp, "%Y-%m")

        if prior_datasets is None:
            prior_datasets = self.read_property_from_metadata("prior-datasets")

        self.prior_datasets = prior_datasets

        docs_path = base_path / f"French/LongEval Train Collection/Trec/{timestamp}_fr"
        docs = LongEvalDocs(ExtractedPath(docs_path), meta)

        queries_path = base_path / "French/queries.txt"
        if not queries_path.exists() or not queries_path.is_file():
            raise FileNotFoundError(
                f"I expected that the file {queries_path} exists. But the directory does not exist."
            )
        queries = TsvQueries(ExtractedPath(queries_path), lang="fr")

        qrels = None
        qrels_path = (
            base_path
            / f"French/LongEval Train Collection/qrels/{timestamp}_fr/qrels_processed.txt"
        )
        if qrels_path.exists() and qrels_path.is_file():
            qrels = TrecQrels(ExtractedPath(qrels_path), QREL_DEFS)

        super().__init__(docs, queries, qrels, documentation)

    def get_timestamp(self):
        return self.timestamp

    def get_past_datasets(self):
        return [
            LongEvalWebDataset(
                base_path=self.base_path,
                meta=self.meta,
                timestamp=i,
                prior_datasets=self.prior_datasets[: self.prior_datasets.index(i)],
            )
            for i in self.prior_datasets
        ]

    def read_property_from_metadata(self, property):
        return json.load(open(self.base_path / "metadata.json", "r"))[property]


def register():
    base_path = home_path() / NAME

    dlc = DownloadConfig.context(NAME, base_path)
    base_path = home_path() / NAME

    data_path = (
        ZipExtractCache(
            dlc["longeval_2025_train_collection"], base_path / "release_2025_p1"
        ).path()
        / "release_2025_p1"
    )

    meta = LongEvalWebMetadata(data_path / "French")

    subsets = {}

    for timestamp in SUB_COLLECTIONS_TRAIN:
        if f"{NAME}/{timestamp}" in registry:
            # Already registered.
            continue
        subsets[timestamp] = LongEvalWebDataset(
            base_path=data_path,
            meta=meta,
            yaml_documentation="longeval_web.yaml",
            timestamp=timestamp,
            prior_datasets=SUB_COLLECTIONS_TRAIN[
                : SUB_COLLECTIONS_TRAIN.index(timestamp)
            ],
        )

    for s in sorted(subsets):
        registry.register(f"{NAME}/{s}", subsets[s])
