import unittest

from ir_datasets_longeval import load


class TestLongEvalSci(unittest.TestCase):

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

        # Docs
        self.assertIsNotNone(example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("68859258", docs_store.get("68859258").doc_id)

        # Qrels
        self.assertEqual(4262, len(list(dataset.qrels_iter())))

        query_ids = [query.query_id for query in dataset.queries_iter()]
        for qrel in dataset.qrels_iter():
            doc = docs_store.get(qrel.doc_id)
            assert doc.title, "Referenced document has no title"
            assert (
                qrel.query_id in query_ids
            ), f"Referenced query_id {qrel.query_id} does not exist"

        # Timestamp
        self.assertEqual(2024, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual([], dataset.get_prior_datasets())

        # Lag
        self.assertEqual("2024-11-train", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

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

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("42999748", docs_store.get("42999748").doc_id)

        # Qrels
        self.assertEqual(947, len(list(dataset.qrels_iter())))

        query_ids = [query.query_id for query in dataset.queries_iter()]
        for qrel in dataset.qrels_iter():
            doc = docs_store.get(qrel.doc_id)
            assert doc.title, "Referenced document has no title"
            assert (
                qrel.query_id in query_ids
            ), f"Referenced query_id {qrel.query_id} does not exist"

        # Docs
        self.assertIsNotNone(example_doc.doc_id)

        # Timestamp
        self.assertEqual(2024, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(1, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2024-11", dataset.get_snapshot())

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_longeval_sci_2025_01(self):
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

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("42999748", docs_store.get("42999748").doc_id)

        # Qrels
        self.assertEqual(5017, len(list(dataset.qrels_iter())))

        query_ids = [query.query_id for query in dataset.queries_iter()]
        for qrel in dataset.qrels_iter():
            doc = docs_store.get(qrel.doc_id)
            assert doc.title, "Referenced document has no title"
            assert (
                qrel.query_id in query_ids
            ), f"Referenced query_id {qrel.query_id} does not exist"

        # Timestamp
        self.assertEqual(2025, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(2, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2025-01", dataset.get_snapshot())

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

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

    def test_clef_2025_sci_tag(self):
        datasets = load("longeval-sci/clef-2025-test")

        expected_tags = ["2024-11", "2025-01"]
        tags = []
        for dataset in datasets.get_datasets():
            tags.append(dataset.get_snapshot())

        self.assertEqual(sorted(expected_tags), sorted(tags))
