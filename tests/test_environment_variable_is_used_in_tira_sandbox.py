import os
import unittest
from pathlib import Path

from ir_datasets_longeval import load


class TestEnvironmentVariableIsUsedInTiraSandbox(unittest.TestCase):

    def test_local_dataset_without_prior_datasets(self):
        os.environ["TIRA_INPUT_DATASET"] = str(
            (
                Path(__file__).parent
                / "resources"
                / "example-local-dataset-no-prior-datasets"
            )
            .absolute()
            .resolve()
        )

        expected_doc_ids = sorted(["77444382", "140120179", "44934830", "34195138"])
        expected_queries = {"1234-1234-1234-1234-1234": "selection bias"}
        expected_qrels = [
            {"doc_id": "140120179", "query_id": "1234-1234-1234-1234-1234", "rel": 2}
        ]
        dataset = load("THIS-DOES-NOT-EXIST")

        self.assertIsNotNone(dataset)
        self.assertEqual(
            expected_doc_ids, sorted([i.doc_id for i in dataset.docs_iter()])
        )
        self.assertEqual(
            expected_queries,
            {i.query_id: i.default_text() for i in dataset.queries_iter()},
        )
        self.assertEqual(
            expected_qrels,
            [
                {"query_id": i.query_id, "doc_id": i.doc_id, "rel": i.relevance}
                for i in dataset.qrels_iter()
            ],
        )
        self.assertEqual(2024, dataset.get_timestamp().year)
        self.assertEqual([], dataset.get_prior_datasets())
        docs_store = dataset.docs_store()

        for doc in expected_doc_ids:
            self.assertEqual(doc, docs_store.get(doc).doc_id)
        del os.environ["TIRA_INPUT_DATASET"]
