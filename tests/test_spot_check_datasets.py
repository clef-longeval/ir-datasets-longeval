import unittest
from ir_datasets_longeval import load
from pathlib import Path

class TestSpotCheckDatasets(unittest.TestCase):
    def test_local_dataset_without_prior_datasets(self):
        expected_queries = {'1901ce75-b35a-4dde-b331-e5e5366dd188': 'ransomware detection', '2f85a909-c4bf-49b6-b7a6-819f7d45f44c': 'chatgpt', 'e6195d15-a4b6-4ce7-9b4f-cf39acdd37e3': 'eye movement'}
        dataset_id = "longeval-sci/spot-check/no-prior-data"
        dataset = load(dataset_id)

        self.assertIsNotNone(dataset)
        self.assertEqual(135, len([i.doc_id for i in dataset.docs_iter()]))
        self.assertEqual(expected_queries, {i.query_id: i.default_text() for i in dataset.queries_iter()})
        self.assertEqual(63, len(list(dataset.qrels_iter())))
        self.assertEqual(2024, dataset.get_timestamp().year)
        self.assertEqual([], dataset.get_past_datasets())
        docs_store = dataset.docs_store()

        for doc in ['159473507', '137793115']:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

    def test_local_dataset_with_prior_datasets(self):
        expected_queries = {'1901ce75-b35a-4dde-b331-e5e5366dd188': 'ransomware detection', '2f85a909-c4bf-49b6-b7a6-819f7d45f44c': 'chatgpt', 'e6195d15-a4b6-4ce7-9b4f-cf39acdd37e3': 'eye movement'}
        dataset_id = "longeval-sci/spot-check/with-prior-data"
        dataset = load(dataset_id)

        self.assertIsNotNone(dataset)
        self.assertEqual(135, len([i.doc_id for i in dataset.docs_iter()]))
        self.assertEqual(expected_queries, {i.query_id: i.default_text() for i in dataset.queries_iter()})
        self.assertEqual(63, len(list(dataset.qrels_iter())))

        self.assertEqual(2024, dataset.get_timestamp().year)
        docs_store = dataset.docs_store()

        for doc in ['159473507', '137793115']:
            self.assertEqual(doc, docs_store.get(doc).doc_id)

        self.assertEqual(2, len(dataset.get_past_datasets()))
        for past_dataset in dataset.get_past_datasets():
            self.assertEqual(135, len([i.doc_id for i in past_dataset.docs_iter()]))
            self.assertEqual(expected_queries, {i.query_id: i.default_text() for i in past_dataset.queries_iter()})
            docs_store = past_dataset.docs_store()
            self.assertTrue(len(list(past_dataset.qrels_iter())) >= 10)

            for doc in ['159473507', '137793115']:
                self.assertEqual(doc, docs_store.get(doc).doc_id)

    def test_all_spot_check_datasets(self):
        dataset_id = "longeval-sci/spot-check/*"
        meta_dataset = load(dataset_id)
        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        lags = meta_dataset.get_lags()
        self.assertEqual(2, len(lags))
        self.assertEqual("lag-3", lags[0].get_lag())
        self.assertEqual("lag-4", lags[1].get_lag())

