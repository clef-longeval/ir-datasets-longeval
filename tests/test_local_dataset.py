import unittest
from ir_datasets_longeval import load
from pathlib import Path

class TestLocalDataset(unittest.TestCase):
    def test_fails_on_non_existing_directory(self):
        dataset_id = str((Path(__file__).parent).absolute().resolve())
        with self.assertRaises(FileNotFoundError):
            load(dataset_id)

    def test_local_dataset_without_prior_datasets(self):
        expected_doc_ids = sorted(["77444382", "140120179", "44934830", "34195138"])
        expected_queries = {"1234-1234-1234-1234-1234": "selection bias"}
        expected_qrels = [{'doc_id': '140120179', 'query_id': '1234-1234-1234-1234-1234', 'rel': 2}]
        dataset_id = str((Path(__file__).parent / "resources" / "example-local-dataset-no-prior-datasets").absolute().resolve())
        dataset = load(dataset_id)

        self.assertIsNotNone(dataset)
        self.assertEqual(expected_doc_ids, sorted([i.doc_id for i in dataset.docs_iter()]))
        self.assertEqual(expected_queries, {i.query_id: i.default_text() for i in dataset.queries_iter()})
        self.assertEqual(expected_qrels, [{"query_id": i.query_id, "doc_id": i.doc_id, "rel": i.relevance} for i in dataset.qrels_iter()])
        self.assertEqual(2024, dataset.get_timestamp().year)
        self.assertEqual([], dataset.get_past_datasets())
        docs_store = dataset.docs_store()

        for doc in expected_doc_ids:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

    def test_local_dataset_with_two_prior_datasets(self):
        expected_doc_ids = sorted(["77444382", "140120179", "44934830", "34195138"])
        expected_queries = {"1": "some cool query"}
        dataset_id = str((Path(__file__).parent / "resources" / "example-local-dataset-two-prior-datasets").absolute().resolve())
        dataset = load(dataset_id)

        self.assertIsNotNone(dataset)
        self.assertEqual(expected_doc_ids, sorted([i.doc_id for i in dataset.docs_iter()]))
        self.assertEqual(expected_queries, {i.query_id: i.default_text() for i in dataset.queries_iter()})
        with self.assertRaises(AttributeError):
            print(dataset.qrels_iter())
        self.assertEqual(2025, dataset.get_timestamp().year)
        docs_store = dataset.docs_store()

        for doc in expected_doc_ids:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

        past_datasets = dataset.get_past_datasets()
        self.assertEqual(2, len(past_datasets))
        for past_dataset in past_datasets:
            self.assertEqual(2024, past_dataset.get_timestamp().year)
            self.assertEqual(0, len(past_dataset.get_past_datasets()))