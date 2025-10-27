import unittest

from ir_datasets_longeval import load


class TestOfficialDatasets(unittest.TestCase):
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
