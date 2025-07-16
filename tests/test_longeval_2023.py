import unittest

from ir_datasets_longeval import load


class TestLongEval2023(unittest.TestCase):
    def test_longeval_2023_train(self):
        dataset = load("longeval-2023/2022-06-train/en")

        expected_queries = {"q06223196": "Car shelter"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(672, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(9656, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc062206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc062206000001", docs_store.get("doc062206000001").doc_id)
        self.assertEqual(
            "http://balliuexport.com/fr/politique-confidentialite.html",
            docs_store.get("doc062206000001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual([], dataset.get_prior_datasets())

        # Lag
        self.assertEqual("2022-06-train", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_longeval_2023_06(self):

        dataset = load("longeval-2023/2022-06/en")
        expected_queries = {"q062211962": "potato salad"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(98, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(1420, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc062206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc062206000001", docs_store.get("doc062206000001").doc_id)
        self.assertEqual(
            "http://balliuexport.com/fr/politique-confidentialite.html",
            docs_store.get("doc062206000001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(1, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2022-06", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_longeval_2023_07(self):
        dataset = load("longeval-2023/2022-07/en")

        expected_queries = {"q0722281": "silver jewellery"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(882, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(12217, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc072206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc072206000001", docs_store.get("doc072206000001").doc_id)
        self.assertEqual(
            "https://ventes-actifs.cnajmj.fr/actif/104183-bar-ambiance-tall-bar",
            docs_store.get("doc072206000001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(2, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2022-07", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_longeval_2023_09(self):
        dataset = load("longeval-2023/2022-09/en")

        expected_queries = {"q09221257": "distance land moon"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(923, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(13467, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc092206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc092206000001", docs_store.get("doc092206000001").doc_id)
        self.assertEqual(
            "http://boutdegomme.fr/cartes-a-pinces-des-nombres-de-0-a-79",
            docs_store.get("doc092200600001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(3, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2022-09", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_clef_2023_train_tag(self):
        datasets = load("longeval-2023/clef-2023-train/en")

        expected_tags = ["2022-06-train"]
        tags = []
        for dataset in datasets.get_datasets():
            tags.append(dataset.get_snapshot())

        self.assertEqual(sorted(expected_tags), sorted(tags))

    def test_clef_2023_tag(self):
        datasets = load("longeval-2023/clef-2023-test/en")

        expected_tags = ["2022-06", "2022-07", "2022-09"]
        tags = []
        for dataset in datasets.get_datasets():
            tags.append(dataset.get_snapshot())

        self.assertEqual(sorted(expected_tags), sorted(tags))

    def test_all_datasets(self):
        dataset_id = "longeval-2023/*/en"
        meta_dataset = load(dataset_id)

        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        datasets = meta_dataset.get_datasets()
        self.assertEqual(4, len(datasets))
        self.assertEqual("2022-06-train", datasets[0].get_snapshot())

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


class TestLongEval2023Fr(unittest.TestCase):
    def test_longeval_2023_train(self):
        dataset = load("longeval-2023/2022-06-train/fr")

        expected_queries = {"q06223196": "abri de voiture"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(672, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(9656, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc062206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc062206000001", docs_store.get("doc062206000001").doc_id)
        self.assertEqual(
            "http://balliuexport.com/fr/politique-confidentialite.html",
            docs_store.get("doc062206000001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual([], dataset.get_prior_datasets())

        # Lag
        self.assertEqual("2022-06-train", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_longeval_2023_06(self):

        dataset = load("longeval-2023/2022-06/fr")
        expected_queries = {"q062211962": "salade de pomme de terre"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(98, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(1420, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc062206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc062206000001", docs_store.get("doc062206000001").doc_id)
        self.assertEqual(
            "http://balliuexport.com/fr/politique-confidentialite.html",
            docs_store.get("doc062206000001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(1, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2022-06", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_longeval_2023_07(self):
        dataset = load("longeval-2023/2022-07/fr")

        expected_queries = {"q0722281": "silver jewellery"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(882, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(12217, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc072206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc072206000001", docs_store.get("doc072206000001").doc_id)
        self.assertEqual(
            "https://ventes-actifs.cnajmj.fr/actif/104183-bar-ambiance-tall-bar",
            docs_store.get("doc072206000001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(2, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2022-07", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_longeval_2023_09(self):
        dataset = load("longeval-2023/2022-09/fr")

        expected_queries = {"q09221257": "distance land moon"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(923, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(13467, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("doc092206000001", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("doc092206000001", docs_store.get("doc092206000001").doc_id)
        self.assertEqual(
            "http://boutdegomme.fr/cartes-a-pinces-des-nombres-de-0-a-79",
            docs_store.get("doc092200600001").url,
        )

        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual(3, len(dataset.get_prior_datasets()))

        # Lag
        self.assertEqual("2022-09", dataset.get_snapshot())
        dataset.qrels_path()

        # components
        self.assertEqual(dataset.has_qrels(), True)
        self.assertEqual(dataset.has_queries(), True)
        self.assertEqual(dataset.has_docs(), True)

    def test_clef_2023_train_tag(self):
        datasets = load("longeval-2023/clef-2023-train/fr")

        expected_tags = ["2022-06-train"]
        tags = []
        for dataset in datasets.get_datasets():
            tags.append(dataset.get_snapshot())

        self.assertEqual(sorted(expected_tags), sorted(tags))

    def test_clef_2023_tag(self):
        datasets = load("longeval-2023/clef-2023-test/fr")

        expected_tags = ["2022-06", "2022-07", "2022-09"]
        tags = []
        for dataset in datasets.get_datasets():
            tags.append(dataset.get_snapshot())

        self.assertEqual(sorted(expected_tags), sorted(tags))

    def test_all_datasets(self):
        dataset_id = "longeval-2023/*/fr"
        meta_dataset = load(dataset_id)

        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        datasets = meta_dataset.get_datasets()
        self.assertEqual(4, len(datasets))
        self.assertEqual("2022-06-train", datasets[0].get_snapshot())

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
