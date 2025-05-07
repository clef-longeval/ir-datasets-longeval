import unittest

from ir_datasets_longeval import load


class TestOfficialDatasets(unittest.TestCase):
    def test_longeval_sci_2024_11_train(self):
        dataset = load("longeval-sci/2024-11/train")

        expected_queries = {"ce5bfacf-8652-4bc1-a5b0-6144a917fb1c": "streptomyces"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(393, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(4262, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("68859258", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("68859258", docs_store.get("68859258").doc_id)

        # Timestamp
        self.assertEqual(2024, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual([], dataset.get_prior_datasets())

        # Lag
        self.assertEqual("2024-11-train", dataset.get_snapshot())
        dataset.qrels_path()

    def test_longeval_sci_2024_11(self):
        dataset = load("longeval-sci/2024-11")

        expected_queries = {"d636a980-15ab-4354-a008-71b4241f2ada": "accounting"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(393, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("68859258", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("42999748", docs_store.get("42999748").doc_id)

        # Timestamp
        self.assertEqual(2024, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(1, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2024-11", dataset.get_snapshot())

    def test_longeval_sci_2025_02(self):
        dataset = load("longeval-sci/2025-02")

        expected_queries = {"92ef8a97-8933-46bc-8c2e-e2d4f27bc4dc": "mpra paper"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(492, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("155554680", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("42999748", docs_store.get("42999748").doc_id)

        # Timestamp
        self.assertEqual(2025, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(2, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2025-02", dataset.get_snapshot())

    def test_web_dataset(self):
        dataset = load("longeval-web/2022-06")

        expected_queries = {"8": "4 mariages 1 enterrement"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(75427, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(85776, len(list(dataset.qrels_iter())))

        # Docs
        self.assertEqual("44971", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("44971", docs_store.get("44971").doc_id)

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual([], dataset.get_prior_datasets())

        # Lag
        self.assertEqual("2022-06", dataset.get_snapshot())

    def test_all_sci_datasets(self):
        dataset_id = "longeval-sci/*"
        meta_dataset = load(dataset_id)

        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        lags = meta_dataset.get_datasets()
        self.assertEqual(3, len(lags))
        self.assertEqual("2024-11-train", lags[0].get_snapshot())

    def test_all_web_datasets(self):
        dataset_id = "longeval-web/*"
        meta_dataset = load(dataset_id)

        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        lags = meta_dataset.get_datasets()
        self.assertEqual(9, len(lags))
        self.assertEqual("2022-06", lags[0].get_snapshot())
