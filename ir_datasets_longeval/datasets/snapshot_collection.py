"""Loader for snapshot-based dynamic test collections (see docs/snapshot-format.md).

A collection is a folder of snapshot folders, each with a ``manifest.json``.
It is registered under two timelines:

- ``<name>/time/<snapshot>[/<qrels-set>]``: one snapshot as released.
- ``<name>/age/<k>[/<qrels-set>]``: every query ``k`` periods after its first
  appearance. Documents of all snapshots share one store with versioned IDs
  ``<snapshot>/<doc_id>``; qrels and runs are rewritten to match.

The loader assumes a valid collection and does not check it.
"""

from __future__ import annotations

import calendar
import contextlib
import gzip
import json
import keyword
import re
from collections import namedtuple
from datetime import date, datetime, timedelta
from functools import cached_property
from itertools import count
from pathlib import Path

import ir_datasets
from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import (
    BaseDocs,
    BaseQrels,
    BaseQueries,
    BaseScoredDocs,
    GenericScoredDoc,
    TrecQrel,
    TrecQrels,
    TrecScoredDocs,
)
from ir_datasets.indices import PickleLz4FullStore

from ir_datasets_longeval.formats import MetaDataset

RESULT_FILE = re.compile(r"(qrels|run)(?:-([a-z0-9-]+))?\.trec(?:\.gz)?")
# Attribute names set by the loader, per record kind; producer fields with these names get a trailing "_".
RESERVED = {
    "documents": {"doc_id", "snapshot"},
    "queries": {"query_id", "query", "topic", "snapshot"},
    "topics": {"topic_id"},
}
TUPLE_NAMES = {"count", "index", "default_text"}


class ExtractedPath:
    def __init__(self, path: Path):
        self._path = path

    def path(self, force=True):
        return self._path

    @contextlib.contextmanager
    def stream(self):
        opener = gzip.open if self._path.suffix == ".gz" else open
        with opener(self._path, "rb") as f:
            yield f


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def read_jsonl(path: Path):
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_queries(path: Path):
    if ".tsv" in path.name:
        with open_text(path) as f:
            next(f)  # header: query_id<TAB>query
            for line in f:
                if line.strip():
                    query_id, query = line.rstrip("\n").split("\t", 1)
                    yield {"query_id": query_id, "query": query}
    else:
        yield from read_jsonl(path)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_period(text: str) -> tuple[int, int]:
    """ISO 8601 duration of years, months, weeks and days -> (months, days)."""
    match = re.fullmatch(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?", text)
    years, months, weeks, days = (int(g or 0) for g in match.groups())  # type: ignore[union-attr]
    return years * 12 + months, weeks * 7 + days


def add_period(start: datetime, period: tuple[int, int], n: int) -> datetime:
    """start + n × period, in calendar arithmetic (month ends are clamped)."""
    months, days = period
    i = start.month - 1 + months * n
    year, month = start.year + i // 12, i % 12 + 1
    start = start.replace(year=year, month=month, day=min(start.day, calendar.monthrange(year, month)[1]))
    return start + timedelta(days=days * n)


def convert(value, typ):
    """Parse a field declared as "date" or "datetime" in the manifest; None if it does not parse."""
    if value is None or typ is None:
        return value
    try:
        return date.fromisoformat(value) if typ == "date" else parse_time(value)
    except (TypeError, ValueError):
        return None


def attribute(field: str, kind: str) -> str:
    """Attribute name for a producer field of the given record kind."""
    name = re.sub(r"\W", "_", field)
    if name[0].isdigit() or name[0] == "_":
        name = "f" + name
    return name + "_" if keyword.iskeyword(name) or name in TUPLE_NAMES | RESERVED[kind] else name


def record_class(name: str, fields: list[str], text_fields: list[str]) -> type:
    """A NamedTuple type with the given fields; default_text() joins the text fields."""

    def default_text(self):
        return "\n".join(str(v) for v in (getattr(self, f) for f in text_fields) if v not in (None, ""))

    return type(name, (namedtuple(name, fields),), {"__slots__": (), "default_text": default_text})


def union_fields(records, skip) -> list[str]:
    return list(dict.fromkeys(k for r in records for k in r if k not in skip))


class Snapshot:
    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.snapshot_id = self.manifest["snapshot"]
        self.start = parse_time(self.manifest["from"])
        self.slot = 0  # position on the time grid, set by the collection
        self.doc_files = sorted((path / "documents").glob("*.jsonl*"))
        self.queries = list(read_queries(next(path.glob("queries.*"))))
        topics_file = next(path.glob("topics.jsonl*"), None)
        self.topics = {t["topic_id"]: t for t in read_jsonl(topics_file)} if topics_file else {}
        self.qrels: dict[str, Path] = {}
        self.runs: dict[str, Path] = {}
        for f in path.iterdir():
            match = RESULT_FILE.fullmatch(f.name)
            if match:
                (self.qrels if match[1] == "qrels" else self.runs)[match[2] or ""] = f


class SnapshotCollection:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.snapshots = sorted(
            (Snapshot(p) for p in self.root.iterdir() if (p / "manifest.json").is_file()), key=lambda s: s.start
        )
        manifest = self.snapshots[0].manifest
        self.name = manifest.get("collection", self.root.name)
        self.period = parse_period(manifest["period"])

        t0 = self.snapshots[0].start
        for s in self.snapshots:
            s.slot = next(n for n in count() if add_period(t0, self.period, n) >= s.start)
        self.first_slot: dict[str, int] = {}
        for s in self.snapshots:
            for q in s.queries:
                self.first_slot.setdefault(q["query_id"], s.slot)
        self.max_age = max(self.age_of(s, q["query_id"]) for s in self.snapshots for q in s.queries)
        self.qrels_sets = sorted({n for s in self.snapshots for n in s.qrels})
        self.run_names = sorted({n for s in self.snapshots for n in s.runs})

        documents = manifest.get("documents", {})
        self.id_field = documents.get("id_field", "doc_id")
        self.text_fields = documents.get("text_fields", ["title", "text"])
        self.declared_doc_fields = documents.get("fields")
        self.types = {kind: manifest.get(kind, {}).get("types", {}) for kind in ("documents", "queries", "topics")}

        self.topic_fields = union_fields((t for s in self.snapshots for t in s.topics.values()), {"topic_id"})
        self.query_fields = union_fields((q for s in self.snapshots for q in s.queries), {"query_id", "query"})
        topic_text = ["title"] if "title" in self.topic_fields else []
        topic_attrs = [attribute(f, "topics") for f in self.topic_fields]
        self.topic_cls = record_class("SnapshotTopic", ["topic_id", *topic_attrs], topic_text)
        query_attrs = ["query_id", "query", *(attribute(f, "queries") for f in self.query_fields), "topic"]
        self.query_cls = record_class("SnapshotQuery", query_attrs, ["query"])
        self.age_query_cls = record_class("SnapshotAgeQuery", [*query_attrs, "snapshot"], ["query"])

    def age_of(self, s: Snapshot, query_id: str) -> int:
        return s.slot - self.first_slot[query_id]

    # Document fields need a full scan unless declared, so they are discovered on first use.
    @cached_property
    def doc_fields(self) -> list[str]:
        if self.declared_doc_fields is not None:
            return [f for f in self.declared_doc_fields if f != self.id_field]
        return union_fields((d for s in self.snapshots for f in s.doc_files for d in read_jsonl(f)), {self.id_field})

    @cached_property
    def doc_attrs(self) -> list[str]:
        return [attribute(f, "documents") for f in self.doc_fields]

    @cached_property
    def doc_text_attrs(self) -> list[str]:
        return [attribute(f, "documents") for f in self.text_fields if f in self.doc_fields]

    @cached_property
    def doc_cls(self) -> type:
        return record_class("SnapshotDoc", ["doc_id", *self.doc_attrs], self.doc_text_attrs)

    @cached_property
    def age_doc_cls(self) -> type:
        return record_class("SnapshotAgeDoc", ["doc_id", *self.doc_attrs, "snapshot"], self.doc_text_attrs)

    def iter_docs(self, s: Snapshot, age: bool = False):
        types = self.types["documents"]
        for f in s.doc_files:
            for d in read_jsonl(f):
                values = [convert(d.get(k), types.get(k)) for k in self.doc_fields]
                if age:
                    yield self.age_doc_cls(f"{s.snapshot_id}/{d[self.id_field]}", *values, s.snapshot_id)
                else:
                    yield self.doc_cls(d[self.id_field], *values)

    def iter_all_docs(self):
        for s in self.snapshots:
            yield from self.iter_docs(s, age=True)

    def iter_queries(self, s: Snapshot, age: bool = False):
        for q in s.queries:
            topic = s.topics.get(q.get("topic_id"))
            if topic is not None:
                topic = self.topic_cls(
                    topic["topic_id"], *(convert(topic.get(k), self.types["topics"].get(k)) for k in self.topic_fields)
                )
            fields = (convert(q.get(k), self.types["queries"].get(k)) for k in self.query_fields)
            values = [q["query_id"], q["query"], *fields, topic]
            yield self.age_query_cls(*values, s.snapshot_id) if age else self.query_cls(*values)


class SnapshotDocs(BaseDocs):
    def __init__(self, iter_fn, cls_fn, store_path: Path):
        self._iter_fn = iter_fn
        self._cls_fn = cls_fn
        self._store_path = store_path

    @ir_datasets.util.use_docstore
    def docs_iter(self):
        return self._iter_fn()

    def docs_store(self):
        return PickleLz4FullStore(
            path=str(self._store_path),
            init_iter_fn=self._iter_fn,
            data_cls=self.docs_cls(),
            lookup_field="doc_id",
            index_fields=["doc_id"],
        )

    def docs_count(self):
        return self.docs_store().count()

    def docs_cls(self):
        return self._cls_fn()


class SnapshotQueries(BaseQueries):
    def __init__(self, iter_fn, cls):
        self._iter_fn = iter_fn
        self._cls = cls

    def queries_iter(self):
        return self._iter_fn()

    def queries_cls(self):
        return self._cls


class GeneratorQrels(BaseQrels):
    def __init__(self, iter_fn):
        self._iter_fn = iter_fn

    def qrels_iter(self):
        return self._iter_fn()

    def qrels_defs(self):
        return {}

    def qrels_cls(self):
        return TrecQrel


class GeneratorScoredDocs(BaseScoredDocs):
    def __init__(self, iter_fn):
        self._iter_fn = iter_fn

    def scoreddocs_iter(self):
        return self._iter_fn()

    def scoreddocs_cls(self):
        return GenericScoredDoc


class TimeDataset(Dataset):
    def __init__(self, collection: SnapshotCollection, s: Snapshot, qrels_set: str, prior_datasets=()):
        self.collection = collection
        self.snap = s
        self.prior_datasets = prior_datasets
        docs = SnapshotDocs(
            lambda: collection.iter_docs(s), lambda: collection.doc_cls, s.path / "documents" / "docstore.pklz4"
        )
        queries = SnapshotQueries(lambda: collection.iter_queries(s), collection.query_cls)
        qrels = TrecQrels(ExtractedPath(s.qrels[qrels_set]), {}) if qrels_set in s.qrels else None
        scoreddocs = TrecScoredDocs(ExtractedPath(s.runs[""])) if "" in s.runs else None
        super().__init__(docs, queries, qrels, scoreddocs)

    def runs_iter(self, name: str = ""):
        return TrecScoredDocs(ExtractedPath(self.snap.runs[name])).scoreddocs_iter()

    def get_runs(self):
        return self.collection.run_names

    def get_timestamp(self):
        return self.snap.start

    def get_snapshot(self):
        return self.snap.snapshot_id

    def get_datasets(self):
        return None

    def get_prior_datasets(self):
        return list(self.prior_datasets)


class AgeDataset(Dataset):
    def __init__(self, collection: SnapshotCollection, k: int, qrels_set: str, docs: SnapshotDocs, prior_datasets=()):
        self.collection = collection
        self.k = k
        self.prior_datasets = prior_datasets
        # The snapshots contributing to age k, each with the IDs of the queries taken from it.
        self.parts = []
        for s in collection.snapshots:
            qids = {q["query_id"] for q in s.queries if collection.age_of(s, q["query_id"]) == k}
            if qids:
                self.parts.append((s, qids))
        queries = SnapshotQueries(self._queries, collection.age_query_cls)
        has_qrels = any(qrels_set in s.qrels for s, _ in self.parts)
        has_run = any("" in s.runs for s, _ in self.parts)
        qrels = GeneratorQrels(lambda: self._results("qrels", qrels_set)) if has_qrels else None
        scoreddocs = GeneratorScoredDocs(lambda: self._results("runs", "")) if has_run else None
        super().__init__(docs, queries, qrels, scoreddocs)

    def _queries(self):
        for s, qids in self.parts:
            yield from (q for q in self.collection.iter_queries(s, age=True) if q.query_id in qids)

    def _results(self, kind: str, name: str):
        for s, qids in self.parts:
            path = getattr(s, kind).get(name)
            if path is None:
                continue
            if kind == "qrels":
                rows = TrecQrels(ExtractedPath(path), {}).qrels_iter()
            else:
                rows = TrecScoredDocs(ExtractedPath(path)).scoreddocs_iter()
            for r in rows:
                if r.query_id in qids:
                    yield r._replace(doc_id=f"{s.snapshot_id}/{r.doc_id}")

    def runs_iter(self, name: str = ""):
        return self._results("runs", name)

    def get_runs(self):
        return self.collection.run_names

    def get_query_snapshots(self):
        return {qid: s.snapshot_id for s, qids in self.parts for qid in qids}

    def get_timestamp(self):
        return None

    def get_snapshot(self):
        return self.k

    def get_datasets(self):
        return None

    def get_prior_datasets(self):
        return list(self.prior_datasets)


def register(root: Path) -> MetaDataset:
    """Register all time and age datasets of the collection at ``root``; returns ``<name>/time``."""
    c = SnapshotCollection(root)
    age_docs = SnapshotDocs(c.iter_all_docs, lambda: c.age_doc_cls, c.root / "all-documents.pklz4")

    def add(tag, dataset):
        if tag not in registry:
            registry.register(tag, dataset)

    for qrels_set in ["", *[n for n in c.qrels_sets if n]]:
        suffix = f"/{qrels_set}" if qrels_set else ""
        time_datasets: list[TimeDataset] = []
        for s in c.snapshots:
            dataset = TimeDataset(c, s, qrels_set, tuple(reversed(time_datasets)))
            time_datasets.append(dataset)
            add(f"{c.name}/time/{s.snapshot_id}{suffix}", dataset)
        add(f"{c.name}/time{suffix}", MetaDataset(time_datasets))

        age_datasets: list[AgeDataset] = []
        for k in range(c.max_age + 1):
            dataset = AgeDataset(c, k, qrels_set, age_docs, tuple(reversed(age_datasets)))
            age_datasets.append(dataset)
            add(f"{c.name}/age/{k}{suffix}", dataset)
        add(f"{c.name}/age{suffix}", MetaDataset(age_datasets))

    return registry[f"{c.name}/time"]


def is_snapshot_collection(path) -> bool:
    path = Path(path)
    return path.is_dir() and any((p / "manifest.json").is_file() for p in path.iterdir())
