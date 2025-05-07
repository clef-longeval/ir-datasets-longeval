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

        expected_queries = {"51c0e5f8-f270-4996-8a04-cbd9a52b3406": "deus"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(99, len(actual_queries))
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
        dataset = load("longeval-sci/2025-01")

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
        self.assertEqual("2025-01", dataset.get_snapshot())

    def test_web_dataset(self):
        dataset = load("longeval-web/2022-06")

        expected_queries = {"8": "4 mariages 1 enterrement"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(24651, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(85776, len(list(dataset.qrels_iter())))

        # Docs
        self.assertEqual("118070", example_doc.doc_id)

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

        datasets = meta_dataset.get_datasets()
        self.assertEqual(3, len(datasets))
        self.assertEqual("2024-11-train", datasets[0].get_snapshot())

    def test_all_web_datasets(self):
        dataset_id = "longeval-web/*"
        meta_dataset = load(dataset_id)

        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        datasets = meta_dataset.get_datasets()
        self.assertEqual(15, len(datasets))
        self.assertEqual("2022-06", datasets[0].get_snapshot())

        prior_datasets = 0
        for dataset in datasets:
            dataset_prior_datasets = dataset.get_prior_datasets()
            self.assertEqual(prior_datasets, len(dataset_prior_datasets))
            prior_datasets += 1
            self.assertTrue(dataset.has_queries())
            self.assertTrue(dataset.has_docs())

        prior_datasets = datasets[1].get_prior_datasets()
        self.assertEqual(1, len(prior_datasets))
        self.assertTrue(prior_datasets[0].has_queries())
        self.assertTrue(prior_datasets[0].has_docs())

    def test_clef_2025_sci_tag(self):
        datasets = load("longeval-sci/clef-2025-test")

        expected_tags = ["2024-11", "2025-01"]
        tags = []
        for dataset in datasets.get_datasets():
            tags.append(dataset.get_snapshot())

        self.assertEqual(sorted(expected_tags), sorted(tags))

    def test_clef_2025_web_tag(self):
        datasets = load("longeval-web/clef-2025-test")

        expected_tags = [
            "2023-03",
            "2023-04",
            "2023-05",
            "2023-06",
            "2023-07",
            "2023-08",
        ]
        tags = []
        for dataset in datasets.get_datasets():
            tags.append(dataset.get_snapshot())

        self.assertEqual(sorted(expected_tags), sorted(tags))
