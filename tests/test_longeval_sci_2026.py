from datetime import datetime

import pytest

from ir_datasets_longeval import load

UNJUDGED_QUERIES = {
    "03-05-train/raw": {},
    "03-05-train/dctr": {"a17857d1a641623051cc8b231abd516b"},
}


class TestLongEval2023Snapshot:
    @pytest.fixture(
        scope="class",
        params=[
            # train
            {
                "snapshot": "03-05-train",
                "n_queries": 100,
                "expected_queries": {"e54f68f74633d43b86d0247af6197544": "mark twain"},
                "n_qrels": 0,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": [],
            },
            {
                "snapshot": "03-05-train/raw",
                "n_queries": 100,
                "expected_queries": {"e54f68f74633d43b86d0247af6197544": "mark twain"},
                "n_qrels": 1336,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": [],
            },
            {
                "snapshot": "03-05-train/dctr",
                "n_queries": 100,
                "expected_queries": {"e54f68f74633d43b86d0247af6197544": "mark twain"},
                "n_qrels": 10406,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": [],
            },
            ## 03-05
            {
                "snapshot": "03-05",
                "n_queries": 1525,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": [],
            },
            {
                "snapshot": "03-05/judged",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": [],
            },
            {
                "snapshot": "03-05/judged/raw",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 3037,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": [],
            },
            {
                "snapshot": "03-05/judged/dctr",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 24080,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": [],
            },
            ## 06-09
            {
                "snapshot": "06-08",
                "n_queries": 1525,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["03-05"],
            },
            {
                "snapshot": "06-08/judged",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["03-05"],
            },
            {
                "snapshot": "06-08/judged/raw",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 3371,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["03-05"],
            },
            {
                "snapshot": "06-08/judged/dctr",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 18832,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["03-05"],
            },
            ## 09-11
            {
                "snapshot": "09-11",
                "n_queries": 1525,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["06-08", "03-05"],
            },
            {
                "snapshot": "09-11/judged",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["06-08", "03-05"],
            },
            {
                "snapshot": "09-11/judged/raw",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 2244,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["06-08", "03-05"],
            },
            {
                "snapshot": "09-11/judged/dctr",
                "n_queries": 338,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 11338,
                "expected_docs": {"10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"},
                "prior_snapshots": ["06-08", "03-05"],
            },
        ],
    )
    def snapshot_data(self, request):
        snapshot = request.param["snapshot"]

        loaded_datasets = load(f"longeval-sci-2026/{snapshot}")
        yield loaded_datasets, request.param

    def test_snapshot_exists(self, snapshot_data):
        dataset, setting = snapshot_data
        assert dataset is not None
        assert dataset.get_snapshot() == setting["snapshot"].split("/")[0]

    def test_queries(self, snapshot_data):
        dataset, setting = snapshot_data

        assert dataset.has_queries()

        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        assert len(actual_queries) == setting["n_queries"]
        for k, v in setting["expected_queries"].items():
            assert v == actual_queries[k]

    def test_docs(self, snapshot_data):
        dataset, setting = snapshot_data

        assert dataset.has_docs()

        example_doc = dataset.docs_iter().__next__()
        assert example_doc is not None

    def test_docstore(self, snapshot_data):
        dataset, setting = snapshot_data

        docs_store = dataset.docs_store()

        for docid, title in setting["expected_docs"].items():
            assert docs_store.get(docid).doc_id == docid
            assert docs_store.get(docid).title == title

    def test_qrels(self, snapshot_data):
        dataset, setting = snapshot_data

        if setting["n_qrels"] == 0:
            assert not dataset.has_qrels()
            return
        assert dataset.has_qrels()

        qrels = list(dataset.qrels_iter())
        assert len(qrels) == setting["n_qrels"]

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        # qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        # qids_in_qrels = qids_in_qrels.union(
        #     UNJUDGED_QUERIES.get(setting["snapshot"], set())
        # )
        # qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        # assert qids_in_qrels == qids_in_queries

        # # all qids have relevant docs
        # qids_in_qrels = {
        #     qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        # }
        # qids_in_qrels = qids_in_qrels.union(
        #     UNJUDGED_QUERIES.get(setting["snapshot"], set())
        # )

        # assert qids_in_qrels == qids_in_queries

        ## TODO: Check with corpus
        # # all docids in qrels are in docs
        # docs_store = dataset.docs_store()
        # docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        # for doc_id in docids_in_qrels:
        #     assert docs_store.get(doc_id) is not None

    def test_timestamp(self, snapshot_data):
        dataset, setting = snapshot_data
        assert dataset.get_timestamp() == datetime.strptime(
            ("2025-" + setting["snapshot"])[:7], "%Y-%m"
        )  # ignore "-test"

    def test_prior_datasets(self, snapshot_data):
        dataset, setting = snapshot_data

        prior_datasets = dataset.get_prior_datasets()
        assert len(prior_datasets) == len(setting["prior_snapshots"])

        for i, prior_snapshot in enumerate(prior_datasets):
            assert prior_snapshot.get_snapshot() == setting["prior_snapshots"][i]
            assert len(prior_snapshot.get_prior_datasets()) == len(
                setting["prior_snapshots"][i + 1 :]
            )
