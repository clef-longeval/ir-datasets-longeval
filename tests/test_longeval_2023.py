import unittest

from ir_datasets_longeval import load

QRELS_WITHOUT_REL_DOCS_TRAIN = {
    "q062223622",
    "q06229033",
    "q062214218",
    "q062219081",
    "q062212442",
    "q062222125",
    "q0622724",
    "q062222134",
    "q062218529",
    "q06223898",
    "q062216451",
    "q062215377",
    "q062220780",
    "q06223848",
    "q062223487",
    "q062210081",
}

QRELS_WITHOUT_REL_DOCS_06 = {"q062210670", "q062211678"}

QRELS_WITHOUT_REL_DOCS_07 = {
    "q072225489",
    "q072218431",
    "q07228384",
    "q072230061",
    "q072218166",
    "q072222808",
    "q072229184",
    "q072213055",
    "q07229243",
    "q072223034",
    "q07225568",
    "q0722830",
    "q07227743",
    "q07225832",
    "q072218365",
    "q07225582",
    "q07228597",
    "q072212233",
    "q072214484",
    "q0722351",
    "q072227978",
    "q072222467",
    "q0722806",
    "q072228421",
    "q07228502",
    "q07225054",
    "q072214898",
    "q07226955",
    "q07228983",
    "q07221244",
    "q072210371",
    "q07224489",
}


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

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }
        qids_in_qrels.update(
            QRELS_WITHOUT_REL_DOCS_TRAIN
        )  # add queries without relevant docs back in
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))

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

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }
        qids_in_qrels.update(
            QRELS_WITHOUT_REL_DOCS_06
        )  # add queries without relevant docs back in
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))

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

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }
        qids_in_qrels.update(
            QRELS_WITHOUT_REL_DOCS_07
        )  # add queries without relevant docs back in
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))

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

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))

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

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }
        qids_in_qrels.update(
            QRELS_WITHOUT_REL_DOCS_TRAIN
        )  # add queries without relevant docs back in
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))

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
        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }
        qids_in_qrels.update(
            QRELS_WITHOUT_REL_DOCS_06
        )  # add queries without relevant docs back in
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))

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

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }
        qids_in_qrels.update(
            QRELS_WITHOUT_REL_DOCS_07
        )  # add queries without relevant docs back in
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))

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

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        self.assertEqual(qids_in_qrels, qids_in_queries)

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }

        # all docids in qrels are in docs
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            self.assertIsNotNone(docs_store.get(doc_id))


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
