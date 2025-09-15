import contextlib
import json
from datetime import datetime
from pathlib import Path
from pkgutil import get_data
from typing import List, NamedTuple, Optional

import ir_datasets
from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import TrecDocs, TrecQrels, TsvQueries
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import ZipExtractCache, home_path

from ir_datasets_longeval.formats import MetaDataset
from ir_datasets_longeval.util import DownloadConfig, YamlDocumentation

logger = ir_datasets.log.easy()

NAME = "longeval-2023"
QREL_DEFS = {
    2: "highly relevant",
    1: "relevant",
    0: "not relevant",
}
SUB_COLLECTIONS_TRAIN = [
    "2022-06-train",
]
SUB_COLLECTIONS_TEST = [
    "2022-06",
    "2022-07",
    "2022-09",
]
DUA = (
    "Please confirm you agree to the Qwant LongEval Attribution-NonCommercial-ShareAlike License found at "
    "<https://lindat.mff.cuni.cz/repository/static/Qwant_LongEval_BY-NC-SA_License.html>"
)


class LongEvalMetadataItem(NamedTuple):
    id: str
    url: str


class LongEvalMetadata:
    def __init__(self, dlc):
        self._dlc = dlc
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
            with open(self._dlc, "r") as f:
                for line in f.readlines():
                    doc_id, url = line.strip().split("\t")
                    self._metadata[doc_id] = LongEvalMetadataItem(doc_id, url)
        return self._metadata

    def get_metadata(self, id):
        return self.metadata.get(str(id))


class LongEvalDocument(NamedTuple):
    doc_id: str
    # original_doc_id: str
    url: str
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
                docid = doc.doc_id
                text = doc.text
                if self._meta:
                    metadata = self._meta.get_metadata(docid)
                    url = metadata.url
                else:
                    url = ""

                yield LongEvalDocument(docid, url, text)

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


class LongEvalDataset(Dataset):
    def __init__(
        self,
        base_path: Path,
        meta: Optional[LongEvalMetadata] = None,
        yaml_documentation: str = "longeval_2023.yaml",
        prior_datasets: Optional[List[str]] = None,
        snapshot: Optional[str] = None,
        lang: str = "en",
    ):
        """LongEval 2023 Dataset"""
        documentation = YamlDocumentation(yaml_documentation)
        self.base_path = base_path
        self.meta = meta

        if not base_path or not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(
                f"I expected that the directory {base_path} exists. But the directory does not exist."
            )
        if snapshot:
            self.snapshot = snapshot
        else:
            self.snapshot = "_".join(self.base_path.name.split("_")[:-1])
        self.lang = lang

        timestamp = self.read_property_from_metadata("timestamp")
        self.timestamp = datetime.strptime(timestamp, "%Y-%m")

        if prior_datasets is None:
            prior_datasets = self.read_property_from_metadata("prior-datasets")
        self.prior_datasets = prior_datasets

        docs = LongEvalDocs(ExtractedPath(base_path / "Documents" / "Trec"), meta)
        query_name_map = {
            "2022-06-train": "train.tsv",
            "2022-06": "heldout.tsv",
            "2022-07": "test07.tsv",
            "2022-09": "test09.tsv",
        }
        query_name = query_name_map.get(self.snapshot, "queries.tsv")
        queries_path = base_path / "Queries" / query_name
        if not queries_path.exists() or not queries_path.is_file():
            raise FileNotFoundError(
                f"I expected that the file {queries_path} exists. But the directory does not exist."
            )
        queries = TsvQueries(ExtractedPath(queries_path), lang=self.lang)

        qrels_path_map = {
            "2022-06-train": base_path.parent / "French" / "Qrels" / "train.txt",
            "2022-06": base_path.parents[2]
            / "2023_test"
            / "longeval-relevance-judgements"
            / "heldout-test.txt",
            "2022-07": base_path.parents[2]
            / "longeval-relevance-judgements"
            / "a-short-july.txt",
            "2022-09": base_path.parents[2]
            / "longeval-relevance-judgements"
            / "b-long-september.txt",
        }
        qrels_path = qrels_path_map.get(self.snapshot)
        qrels = None
        if qrels_path.exists() and qrels_path.is_file():
            qrels = TrecQrels(ExtractedPath(qrels_path), QREL_DEFS)
        else:
            print("Missing qrels_path:", qrels_path)

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
                LongEvalDataset(
                    base_path=self.base_path.parent / f"{i}_{self.language}",
                    meta=self.meta,
                )
                for i in self.prior_datasets
            ]
        else:
            return self.prior_datasets

    def read_property_from_metadata(self, property):
        try:
            return json.load(open(self.base_path / "etc" / "metadata.json", "r"))[
                property
            ]
        except FileNotFoundError:
            metadata = json.loads(get_data("ir_datasets_longeval", "etc/metadata.json"))
            return metadata[f"longeval-2023/{self.snapshot}"][property]


def register():
    base_path = home_path() / NAME

    dlc = DownloadConfig.context(NAME, base_path, dua=DUA)

    # train
    data_path = ZipExtractCache(dlc["train"], base_path).path()

    for language in ["English", "French"]:
        lang = "en" if language == "English" else "fr"

        meta = LongEvalMetadata(
            data_path / "2023_train" / "publish" / "French" / "urls.txt"
        )

        base_path_train = data_path / "2023_train" / "publish" / language

        # Desired structure: longeval/2023-07/en/

        subsets = {}
        subsets[f"2022-06-train/{lang}"] = LongEvalDataset(
            prior_datasets=list(subsets.values())[::-1],
            base_path=base_path_train,
            snapshot="2022-06-train",
            meta=meta,
            lang=lang,
        )
        subsets[f"2022-06/{lang}"] = LongEvalDataset(
            prior_datasets=list(subsets.values())[::-1],
            base_path=base_path_train,
            snapshot="2022-06",
            meta=meta,
            lang=lang,
        )

        # test data
        base_path_test = data_path / "2023_test" / "test-collection"

        meta = LongEvalMetadata(
            base_path_test / "A-Short-July" / "French" / "Documents" / "urls.txt"
        )

        subsets[f"2022-07/{lang}"] = LongEvalDataset(
            prior_datasets=list(subsets.values())[::-1],
            base_path=base_path_test / "A-Short-July" / "English",
            snapshot="2022-07",
            meta=meta,
            lang=lang,
        )

        meta = LongEvalMetadata(
            base_path_test / "B-Long-September" / "French" / "Documents" / "urls.txt"
        )
        subsets[f"2022-09/{lang}"] = LongEvalDataset(
            prior_datasets=list(subsets.values())[::-1],
            base_path=base_path_test / "B-Long-September" / "English",
            snapshot="2022-09",
            meta=meta,
            lang=lang,
        )

        if f"{NAME}/*/{lang}" in registry:
            return

        for s in sorted(subsets):
            registry.register(f"{NAME}/{s}", subsets[s])

        registry.register(f"{NAME}/*/{lang}", MetaDataset(list(subsets.values())))

        if f"{NAME}/clef-2023-test/{lang}" in registry:
            return

        registry.register(
            f"{NAME}/clef-2023-test/{lang}",
            MetaDataset([subsets[f"{s}/{lang}"] for s in SUB_COLLECTIONS_TEST]),
        )

        registry.register(
            f"{NAME}/clef-2023-train/{lang}",
            MetaDataset([subsets[f"{s}/{lang}"] for s in SUB_COLLECTIONS_TRAIN]),
        )
