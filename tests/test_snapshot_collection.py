import gzip
import json
import shutil
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ir_datasets_longeval import load

RESOURCES = Path(__file__).parent / "resources"


def write_gz(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)


def make_snapshot(root, snapshot_id, start, end, docs, queries, period="P1D", qrels=None, runs=None, **manifest):
    folder = root / snapshot_id
    folder.mkdir(parents=True)
    manifest = {"snapshot": snapshot_id, "from": start, "to": end, "period": period, **manifest}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    write_gz(folder / "documents" / "documents_000000.jsonl.gz", [json.dumps(d) for d in docs])
    write_gz(folder / "queries.jsonl.gz", [json.dumps(q) for q in queries])
    for name, rows in (qrels or {}).items():
        write_gz(folder / (f"qrels-{name}.trec.gz" if name else "qrels.trec.gz"), [" ".join(map(str, r)) for r in rows])
    for name, rows in (runs or {}).items():
        write_gz(folder / (f"run-{name}.trec.gz" if name else "run.trec.gz"), [" ".join(map(str, r)) for r in rows])


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    # Docstores are written next to the data, so work on copies.
    root = tmp_path_factory.mktemp("examples")
    for name in ["example-minimal-snapshot"]:
        shutil.copytree(RESOURCES / name, root / name)
        load(root / name)
    return root


@pytest.fixture(scope="module")
def timeline(tmp_path_factory):
    """Daily snapshots in slots 0 (11-01), 1 (11-02) and 3 (11-04); 11-03 is a hole.

    q1 appears on all days, q2 from 11-02, q3 only on 11-04. d1 exists every day with changing text.
    """
    root = tmp_path_factory.mktemp("timeline")
    name = f"test-{uuid.uuid4().hex[:8]}"
    days = {"2026-11-01": ["q1"], "2026-11-02": ["q1", "q2"], "2026-11-04": ["q1", "q2", "q3"]}
    for day, qids in days.items():
        end = (date.fromisoformat(day).replace(day=int(day[-2:]) + 1)).isoformat()
        make_snapshot(
            root, day, day, end,
            docs=[{"doc_id": "d1", "title": f"d1 on {day}", "text": f"version {day}"}, {"doc_id": f"d-{day}", "title": day}],
            queries=[{"query_id": q, "query": f"query {q}"} for q in qids],
            qrels={"": [(q, 0, "d1", 1) for q in qids], "human": [(q, 0, f"d-{day}", 2) for q in qids]},
            runs={"": [(q, "Q0", "d1", 1, 1.0, "bm25") for q in qids], "dense": [(q, "Q0", f"d-{day}", 1, 0.5, "x") for q in qids]},
            collection=name,
        )
    load(root)
    return name


def test_sci_example(examples):
    ds = load("example-minimal-snapshot/time/snapshot-1")
    docs = list(ds.docs_iter())
    assert len(docs) == 180
    assert docs[0]._fields == ("doc_id", "title", "abstract", "doi", "fullText")
    assert docs[0].default_text() == docs[0].title + "\n" + docs[0].abstract
    assert len(list(ds.queries_iter())) == 12
    assert len(list(ds.qrels_iter())) == 180
    assert ds.get_timestamp() == datetime(2024, 11, 4, tzinfo=timezone.utc)


def test_time_view(timeline):
    ds = load(f"{timeline}/time/2026-11-04")
    assert [p.get_snapshot() for p in ds.get_prior_datasets()] == ["2026-11-02", "2026-11-01"]
    assert ds.docs_store().get("d1").text == "version 2026-11-04"
    assert {q.doc_id for q in load(f"{timeline}/time/2026-11-02/human").qrels_iter()} == {"d-2026-11-02"}
    assert {r.doc_id for r in ds.runs_iter("dense")} == {"d-2026-11-04"}
    snapshots = [d.get_snapshot() for d in load(f"{timeline}/time").get_datasets()]
    assert snapshots == ["2026-11-01", "2026-11-02", "2026-11-04"]  # the hole is not a dataset


def test_age_alignment(timeline):
    def at(k):
        return {q.query_id: q.snapshot for q in load(f"{timeline}/age/{k}").queries_iter()}

    assert at(0) == {"q1": "2026-11-01", "q2": "2026-11-02", "q3": "2026-11-04"}
    assert at(1) == {"q1": "2026-11-02"}  # q2's age 1 is the hole
    assert at(2) == {"q2": "2026-11-04"}  # q1's age 2 is the hole
    assert at(3) == {"q1": "2026-11-04"}
    assert len(load(f"{timeline}/age").get_datasets()) == 4


def test_age_versioned_documents(timeline):
    ds = load(f"{timeline}/age/0")
    docs = {d.doc_id: d for d in ds.docs_iter()}
    assert len(docs) == 6
    assert docs["2026-11-02/d1"].text == "version 2026-11-02"
    assert ds.docs_store().get("2026-11-04/d1").snapshot == "2026-11-04"
    qrels = {(q.query_id, q.doc_id) for q in ds.qrels_iter()}
    assert qrels == {("q1", "2026-11-01/d1"), ("q2", "2026-11-02/d1"), ("q3", "2026-11-04/d1")}
    assert {(r.query_id, r.doc_id) for r in ds.scoreddocs_iter()} == qrels


def test_monthly_period(tmp_path):
    name = f"test-{uuid.uuid4().hex[:8]}"
    kwargs = dict(period="P3M", docs=[{"doc_id": "d1", "title": "x"}], queries=[{"query_id": "q1", "query": "x"}], collection=name)
    make_snapshot(tmp_path, "s1", "2025-03-01T00:00:00+00:00", "2025-06-01T00:00:00+00:00", **kwargs)
    make_snapshot(tmp_path, "s3", "2025-09-01T00:00:00+00:00", "2025-12-01T00:00:00+00:00", **kwargs)
    load(tmp_path)
    assert [q.snapshot for q in load(f"{name}/age/2").queries_iter()] == ["s3"]
    assert list(load(f"{name}/age/1").queries_iter()) == []


def test_fields(tmp_path):
    name = f"test-{uuid.uuid4().hex[:8]}"
    types = {"documents": {"types": {"published": "date"}}}
    make_snapshot(
        tmp_path, "a", "2026-01-01", "2026-01-02", collection=name, **types,
        docs=[{"doc_id": "d1", "publish-date": "x", "class": "c", "snapshot": "s", "published": "2026-01-01"}],
        queries=[{"query_id": "q1", "query": "x"}],
    )
    make_snapshot(
        tmp_path, "b", "2026-01-02", "2026-01-03", collection=name, **types,
        docs=[{"doc_id": "d1", "later": 1, "published": "not a date"}], queries=[{"query_id": "q1", "query": "x"}],
    )
    load(tmp_path)
    a = next(iter(load(f"{name}/time/a").docs_iter()))
    b = next(iter(load(f"{name}/time/b").docs_iter()))
    assert (a.publish_date, a.class_, a.snapshot_, a.later) == ("x", "c", "s", None)
    assert (a.published, b.published, b.later) == (date(2026, 1, 1), None, 1)
