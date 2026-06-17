from datetime import datetime

import pytest

from ir_datasets_longeval import load

UNJUDGED_QUERIES = {
    "03-05-train/raw": {},
    "03-05-train/dctr": {"a17857d1a641623051cc8b231abd516b"},
}


class TestLongEval2026Snapshot:
    @pytest.fixture(
        scope="class",
        params=[
            # train
            {
                "snapshot": "snapshot-1/train",
                "n_queries": 100,
                "expected_queries": {"e54f68f74633d43b86d0247af6197544": "mark twain"},
                "n_qrels": 0,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
            {
                "snapshot": "snapshot-1/train/raw",
                "n_queries": 100,
                "expected_queries": {"e54f68f74633d43b86d0247af6197544": "mark twain"},
                "n_qrels": 1183,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
            {
                "snapshot": "snapshot-1/train/dctr",
                "n_queries": 100,
                "expected_queries": {"e54f68f74633d43b86d0247af6197544": "mark twain"},
                "n_qrels": 8772,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
            # 03-05
            {
                "snapshot": "snapshot-1",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
             {
                 "snapshot": "snapshot-1/raw",
                 "n_queries": 381,
                 "expected_queries": {
                     "a2a64265721427605ec182827171265b": "cybersecurity"
                 },
                 "n_qrels": 2026,
                 "expected_docs": {
                     "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                 },
                 "timestamp": datetime(2025, 3, 1),
                 "prior_snapshots": [],
             },
             {
                 "snapshot": "snapshot-1/dctr",
                 "n_queries": 381,
                 "expected_queries": {
                     "a2a64265721427605ec182827171265b": "cybersecurity"
                 },
                 "n_qrels": 21247,
                 "expected_docs": {
                     "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                 },
                 "timestamp": datetime(2025, 3, 1),
                 "prior_snapshots": [],
             },
            
            
            {
                "snapshot": "snapshot-1/llama3-1-8b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 19675,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
            {
                "snapshot": "snapshot-1/gpt-oss-20b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 19675,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
            {
                "snapshot": "snapshot-1/gpt-oss-120b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 19675,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
            {
                "snapshot": "snapshot-1/qwen3-32b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 19675,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 3, 1),
                "prior_snapshots": [],
            },
            
            ## 06-09
            {
                "snapshot": "snapshot-2",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 6, 1),
                "prior_snapshots": ["snapshot-1"],
            },
              {
                  "snapshot": "snapshot-2/raw",
                  "n_queries": 381,
                  "expected_queries": {
                      "a2a64265721427605ec182827171265b": "cybersecurity"
                  },
                  "n_qrels": 2169,
                  "expected_docs": {
                      "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                  },
                  "timestamp": datetime(2025, 6, 1),
                  "prior_snapshots": ["snapshot-1/raw"],
              },
              {
                  "snapshot": "snapshot-2/dctr",
                  "n_queries": 381,
                  "expected_queries": {
                      "a2a64265721427605ec182827171265b": "cybersecurity"
                  },
                  "n_qrels": 15941,
                  "expected_docs": {
                      "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                  },
                  "timestamp": datetime(2025, 6, 1),
                  "prior_snapshots": ["snapshot-1/dctr"],
              },
            
            
            {
                "snapshot": "snapshot-2/llama3-1-8b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20489,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 6, 1),
                "prior_snapshots": ["snapshot-1/dctr"],
            },
            {
                "snapshot": "snapshot-2/gpt-oss-20b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20489,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 6, 1),
                "prior_snapshots": ["snapshot-1/dctr"],
            },
            {
                "snapshot": "snapshot-2/gpt-oss-120b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20489,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 6, 1),
                "prior_snapshots": ["snapshot-1/dctr"],
            },
            {
                "snapshot": "snapshot-2/qwen3-32b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20489,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 6, 1),
                "prior_snapshots": ["snapshot-1/dctr"],
            },
            
            ## 09-11
            {
                "snapshot": "snapshot-3",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 0,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 9, 1),
                "prior_snapshots": ["snapshot-2", "snapshot-1"],
            },
             
             {
                 "snapshot": "snapshot-3/raw",
                 "n_queries": 381,
                 "expected_queries": {
                     "a2a64265721427605ec182827171265b": "cybersecurity"
                 },
                 "n_qrels": 1449,
                 "expected_docs": {
                     "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                 },
                 "timestamp": datetime(2025, 9, 1),
                 "prior_snapshots": ["snapshot-2/raw", "snapshot-1/raw"],
             },
             {
                 "snapshot": "snapshot-3/dctr",
                 "n_queries": 381,
                 "expected_queries": {
                     "a2a64265721427605ec182827171265b": "cybersecurity"
                 },
                 "n_qrels": 9878,
                 "expected_docs": {
                     "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                 },
                 "timestamp": datetime(2025, 9, 1),
                 "prior_snapshots": ["snapshot-2/dctr", "snapshot-1/dctr"],
             },
            {
                "snapshot": "snapshot-3/llama3-1-8b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20961,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 9, 1),
                "prior_snapshots": ["snapshot-2/dctr", "snapshot-1/dctr"],
            },
            {
                "snapshot": "snapshot-3/gpt-oss-20b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20961,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 9, 1),
                "prior_snapshots": ["snapshot-2/dctr", "snapshot-1/dctr"],
            },
            {
                "snapshot": "snapshot-3/gpt-oss-120b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20961,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 9, 1),
                "prior_snapshots": ["snapshot-2/dctr", "snapshot-1/dctr"],
            },
            {
                "snapshot": "snapshot-3/qwen3-32b",
                "n_queries": 381,
                "expected_queries": {
                    "a2a64265721427605ec182827171265b": "cybersecurity"
                },
                "n_qrels": 20961,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 9, 1),
                "prior_snapshots": ["snapshot-2/dctr", "snapshot-1/dctr"],
            },
            {
                "snapshot": "snapshot-3/rag",
                "n_queries": 47,
                "expected_queries": {
                    "804711f54939af9622c3c7614da85298": "How well do trial-specific early composite benefits predict broader pooled-event reductions?"
                },
                "n_qrels": 0,
                "expected_docs": {
                    "10145682": "BETWEEN THE MEMORY OF HERITAGE AND THE HERITAGE OF MEMORY. THE SEARCH FOR CONCEPTUAL SIMILARITIES"
                },
                "timestamp": datetime(2025, 9, 1),
                "prior_snapshots": ["snapshot-2", "snapshot-1"],
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

        # # test qrels
        # # all queries have judgments
        # # all qids in qrels are in queries
        # qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        # qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        # assert qids_in_qrels == qids_in_queries

        # # all qids have relevant docs
        # qids_in_qrels = {
        #     qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        # }
        # assert qids_in_qrels == qids_in_queries

        # all docids in qrels are in docs
        docs_store = dataset.docs_store()
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            assert docs_store.get(doc_id) is not None

    def test_timestamp(self, snapshot_data):
        dataset, setting = snapshot_data
        assert dataset.get_timestamp() == setting["timestamp"]

    def test_prior_datasets(self, snapshot_data):
        dataset, setting = snapshot_data

        prior_datasets = dataset.get_prior_datasets()
        assert len(prior_datasets) == len(setting["prior_snapshots"])

        for i, prior_snapshot in enumerate(prior_datasets):
            assert (
                prior_snapshot.get_snapshot()
                == setting["prior_snapshots"][i].split("/")[0]
            )
            assert len(prior_snapshot.get_prior_datasets()) == len(
                setting["prior_snapshots"][i + 1 :]
            )
