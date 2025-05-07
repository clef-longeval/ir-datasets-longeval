import unittest
from pathlib import Path

from ir_datasets_longeval import load


class TestLocalDataset(unittest.TestCase):
    def test_fails_on_non_existing_directory(self):
        dataset_id = str((Path(__file__).parent).absolute().resolve())
        with self.assertRaises(FileNotFoundError):
            load(dataset_id)

    def test_local_dataset_without_prior_datasets(self):
        expected_doc_ids = sorted(["77444382", "140120179", "44934830", "34195138"])
        expected_queries = {"1234-1234-1234-1234-1234": "selection bias"}
        expected_qrels = [
            {"doc_id": "140120179", "query_id": "1234-1234-1234-1234-1234", "rel": 2}
        ]
        dataset_id = str(
            (
                Path(__file__).parent
                / "resources"
                / "example-local-dataset-no-prior-datasets"
            )
            .absolute()
            .resolve()
        )
        dataset = load(dataset_id)

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
        self.assertEqual("s1", dataset.get_snapshot())
        self.assertEqual(None, dataset.get_datasets())
        docs_store = dataset.docs_store()

        for doc in expected_doc_ids:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

    def test_local_dataset_with_two_prior_datasets(self):
        expected_doc_ids = sorted(["77444382", "140120179", "44934830", "34195138"])
        expected_queries = {"1": "some cool query"}
        dataset_id = str(
            (
                Path(__file__).parent
                / "resources"
                / "example-local-dataset-two-prior-datasets"
            )
            .absolute()
            .resolve()
        )
        dataset = load(dataset_id)

        self.assertIsNotNone(dataset)
        self.assertEqual(
            expected_doc_ids, sorted([i.doc_id for i in dataset.docs_iter()])
        )
        self.assertEqual(
            expected_queries,
            {i.query_id: i.default_text() for i in dataset.queries_iter()},
        )
        with self.assertRaises(AttributeError):
            print(dataset.qrels_iter())
        self.assertEqual(2025, dataset.get_timestamp().year)
        docs_store = dataset.docs_store()

        for doc in expected_doc_ids:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

        self.assertEqual("s3", dataset.get_snapshot())
        past_datasets = dataset.get_prior_datasets()
        self.assertEqual(2, len(past_datasets))
        for past_dataset in past_datasets:
            self.assertEqual(2024, past_dataset.get_timestamp().year)
            self.assertEqual(0, len(past_dataset.get_prior_datasets()))

    def test_local_dataset_web_without_prior_datasets(self):
        expected_doc_ids = sorted(["16961", "25648", "19467"])
        expected_queries = {"4772": "loi de modernisation de la fonction publique"}
        expected_qrels = [
            {"doc_id": "16961", "query_id": "4772", "rel": 1},
            {"doc_id": "25648", "query_id": "4772", "rel": 2},
            {"doc_id": "19467", "query_id": "4772", "rel": 0},
        ]
        dataset_id = str(
            (
                Path(__file__).parent
                / "resources"
                / "example-local-dataset-web-no-prior-datasets"
                / "French"
                / "LongEval Train Collection"
                / "Trec"
                / "2022-06_fr"
            )
            .absolute()
            .resolve()
        )
        dataset = load(dataset_id)

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
        self.assertEqual(2022, dataset.get_timestamp().year)
        self.assertEqual([], dataset.get_prior_datasets())
        self.assertEqual("2022-06", dataset.get_snapshot())
        self.assertEqual(None, dataset.get_datasets())
        docs_store = dataset.docs_store()

        for doc in expected_doc_ids:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

    def test_local_dataset_web_with_two_prior_datasets(self):
        expected_doc_ids = sorted(["16961", "25648", "19467"])
        expected_queries = {"4772": "loi de modernisation de la fonction publique"}

        dataset_id = str(
            (
                Path(__file__).parent
                / "resources"
                / "example-local-dataset-web-two-prior-datasets"
                / "French"
                / "LongEval Train Collection"
                / "Trec"
                / "2022-06_fr"
            )
            .absolute()
            .resolve()
        )
        dataset = load(dataset_id)

        self.assertIsNotNone(dataset)
        self.assertEqual(
            expected_doc_ids, sorted([i.doc_id for i in dataset.docs_iter()])
        )
        self.assertEqual(
            expected_queries,
            {i.query_id: i.default_text() for i in dataset.queries_iter()},
        )

        self.assertEqual(2022, dataset.get_timestamp().year)

        docs_store = dataset.docs_store()
        for doc in expected_doc_ids:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

        self.assertEqual("2022-06", dataset.get_snapshot())
        self.assertEqual(None, dataset.get_datasets())
        
        past_datasets = dataset.get_prior_datasets()
        expected_prior_datasets = ["prior-dataset-01", "prior-dataset-02"]
        self.assertEqual(len(expected_prior_datasets), len(past_datasets))
        for past_dataset in past_datasets:
            self.assertEqual(expected_prior_datasets[0], past_dataset.get_snapshot())
            self.assertEqual(2022, past_dataset.get_timestamp().year)
            expected_prior_datasets.pop(0)
            self.assertEqual(len(expected_prior_datasets), len(past_dataset.get_prior_datasets()))
