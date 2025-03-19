import io
import json
from os import PathLike
from pathlib import Path
from typing import List, NamedTuple, Tuple

import ir_datasets
from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocs, TrecDocs, TrecQrels, TrecQueries, TsvQueries
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.log import easy
from ir_datasets.util import (
    Cache,
    Download,
    IterStream,
    LocalDownload,
    RelativePath,
    TarExtract,
    TarExtractAll,
    apply_sub_slice,
    home_path,
    slice_idx,
)

from ir_datasets_longeval.util import DownloadConfig, YamlDocumentation

NAME = "longeval-web"
QREL_DEFS = {
    2: "highly relevant",
    1: "relevant",
    0: "not relevant",
}
SUB_COLLECTIONS = [
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

import sqlite3


class LongEvalMetadataItem(NamedTuple):
    id: str
    url: str
    last_updated_at: List[int]
    date: List[str]


class LongEvalMetadata:
    def __init__(self, dlc):
        self._dlc = dlc
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            connection = sqlite3.connect(self._dlc.path())
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
            connection.close()
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
        self._dlc = LocalDownload(dlc.path())
        self._meta = meta
        # self._dlc = dlc
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
            path=f"{self._dlc.path(force=False)}.pklz4",
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field="doc_id",
            index_fields=["doc_id"],
        )

    def docs_cls(self):
        return LongEvalDocument


def register(sub_collection: List[str] = SUB_COLLECTIONS):
    if "longeval" in registry:
        # Already registered.
        return
    documentation = YamlDocumentation("longeval-web.yaml")
    base_path = home_path() / NAME
    print(base_path)
    dlc = DownloadConfig.context(NAME, base_path)

    base = Dataset(documentation("_"))

    queries = TsvQueries(dlc["queries"], lang="fr")

    meta = LongEvalMetadata(RelativePath(dlc["meta"], "collection_db.db"))

    subsets = {}
    for sub_collection in SUB_COLLECTIONS:
        subsets[sub_collection] = Dataset(
            LongEvalDocs(RelativePath(dlc["docs"], f"{sub_collection}_fr"), meta),
            queries,
            TrecQrels(
                RelativePath(dlc["qrels"], f"{sub_collection}_fr/qrels_processed.txt"),
                QREL_DEFS,
            ),
        )

    registry.register(NAME, base)
    for s in sorted(subsets):
        registry.register(f"{NAME}/{s}", subsets[s])
