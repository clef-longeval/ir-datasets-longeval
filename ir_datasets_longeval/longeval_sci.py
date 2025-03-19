import os
from typing import Dict, List, NamedTuple, Optional

from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import JsonlDocs, TrecQrels, TsvQueries
from ir_datasets.util import RelativePath, ZipExtractCache, home_path

from ir_datasets_longeval.util import DownloadConfig, YamlDocumentation

NAME = "longeval-sci"
QREL_DEFS = {
    2: "highly relevant",
    1: "relevant",
    0: "not relevant",
}
SUB_COLLECTIONS = ["2024-11"]
MAPPING = (
    {
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
    },
)


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
        return self.title + self.abstract


def register():
    if NAME in registry:
        # Already registered.
        return
    documentation = YamlDocumentation("longeval_sci.yaml")
    base_path = home_path() / NAME
    dlc = DownloadConfig.context(NAME, base_path)

    base = Dataset(documentation("_"))

    training_2025_data_cache = ZipExtractCache(
        dlc["longeval_sci_training_2025"], base_path / "longeval_sci_training_2025"
    )
    docs_path = training_2025_data_cache.path() / "longeval_sci_training_2025/documents"
    collection_2024_11 = JsonlDocs(
        [
            RelativePath(
                training_2025_data_cache,
                f"longeval_sci_training_2025/documents/{split}",
            )
            for split in os.listdir(docs_path)
        ],
        doc_cls=LongEvalSciDoc,
        docstore_path=f"{docs_path}.pklz4",
        mapping=MAPPING,
    )

    subsets = {}
    subsets["2024-11"] = Dataset(collection_2024_11, documentation("2024-11"))

    subsets["2024-11/train"] = Dataset(
        collection_2024_11,
        TsvQueries(
            RelativePath(
                training_2025_data_cache, "longeval_sci_training_2025/queries.txt"
            )
        ),
        TrecQrels(
            RelativePath(
                training_2025_data_cache, "longeval_sci_training_2025/qrels.txt"
            ),
            QREL_DEFS,
        ),
        documentation("2024-11/train"),
    )

    registry.register(NAME, base)
    for s in sorted(subsets):
        registry.register(f"{NAME}/{s}", subsets[s])
