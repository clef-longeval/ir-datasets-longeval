import unittest
from datetime import datetime

import pytest
from parameterized import parameterized, parameterized_class

from ir_datasets_longeval import load

QRELS_WITHOUT_REL_DOCS = {
    "": {
        "2022-06-train": {
            "0313c4bc-46df-4018-9562-3c7adaac109b",
            "0c0d9df4-3cc3-42f0-9ad0-fd3bb64fec05",
            "387fe6bd-893c-4332-a3be-525c2c353256",
            "5d945f5b-e118-4dbf-bdfa-6843add85978",
            "6e9a523b-0033-47f8-982b-024c16b42733",
            "6eaaaee2-3f33-4db7-b25d-f050697d8d3c",
            "6fdeec6f-7366-4f50-8f82-898f20415da9",
            "8e90d8c6-41b7-498e-a210-2ec459f712bc",
            "910a7482-8918-4150-b9b0-6474c16a19d4",
            "a252db98-6283-4672-8771-94dc06483653",
            "a2b098b2-d579-46fd-b94c-91ba8dd6ef7f",
            "b28c4326-03d6-4096-b42f-d39bfc38a34e",
            "b2c2da43-d97e-4e09-bb11-5146c2bb5aee",
            "cd121863-b380-4d01-acf1-5c3fd2ce0cc3",
            "e36c7192-8e15-4f91-b1ac-07c722aa6c6d",
            "fc6c4f89-d7bc-4986-a2db-a0ff0f563336",
        },
        "2022-06": {
            "2843d5ef-dec3-4153-b7c4-8e029e3b6cee",
            "78ca6af5-8ca7-4ef5-b530-907d35c82fa5",
        },
        "2022-07": {
            "0c8b7394-7517-4626-a735-8ecfc6c00fb8",
            "112b5304-b93c-4733-b113-e882f8f28175",
            "14980762-3ff1-41e1-91bb-080c7dc4e9fc",
            "195677a2-fb70-40ea-af22-eb6bb0deb578",
            "1c20b084-6168-47ce-b88e-f1b660cf4932",
            "269d922d-896c-4fd5-bc18-8466a979615c",
            "3a7f920e-b1f0-465a-8c97-fe3e39858db8",
            "4844dbdc-551e-479f-be79-2909abbc0489",
            "49dccaae-ea64-47a6-a393-06c1e0665a02",
            "4b6b3063-6b77-4e5b-bf37-820698592f23",
            "50b396c4-9b6f-46ac-b891-38a1b6d8af1c",
            "57331e31-b539-427e-90b6-d6fff519658e",
            "632d140a-cbee-4cb0-a573-1e4c752793df",
            "652b547f-a6fd-434b-b38f-58f6827d4b37",
            "8208b098-9c53-464a-b0ac-bde56e8cc03e",
            "82969a08-acd8-4713-900a-2928c1581e34",
            "85d355ae-ee8d-4dfc-9996-7ca00accf391",
            "8fbc5341-0747-4995-bafa-3bf3cd9529d0",
            "9984e8b5-752b-4d90-b84d-3b17bff41128",
            "a105ac90-5aa0-4251-8ba0-cf05bcd79136",
            "a99e98d7-cc51-4717-924b-d26249125f78",
            "baf7631a-f2df-4bf5-b545-c12c4f2bf7ee",
            "be4a4e90-55b1-4f3b-a9fe-b49adbc80008",
            "cd3fa9af-aad7-47a0-9b37-713a788cf955",
            "d5d92e38-b3df-44ff-81d3-f1c0cfeadbb5",
            "d8ffaf4d-2ca1-4119-9f79-de4c6fa54c90",
            "dfc19b62-69b5-40dc-8871-4f539d5731e9",
            "e53dc6c3-cdf6-4d94-975c-4c6f560fed70",
            "edc7a39d-cd7b-47a4-ad21-c255287ed275",
            "f671c7d2-dbb4-4bdd-ba57-7fc02a3764b7",
            "f9e6f436-4968-44c4-8575-beec526f6672",
            "ffbd20f5-15a5-49b4-a79e-98179ea915be",
        },
        "2022-09": {
            "102a7129-45fa-4b23-8b4f-250681e30b5f",
            "18eb5291-821f-4ba1-8bc5-4938dba850b2",
            "2809dd94-538d-4756-8901-3e32a1a77a7f",
            "2ab31806-4290-4ff0-9b7b-0a7919096d1d",
            "2c2dba09-ed94-4f86-a0ed-e55b29e37010",
            "303655ce-9644-4903-9561-c8c955e877ff",
            "3108a899-cbf0-4525-9a9b-80e49653b66f",
            "39e01973-f5f6-4890-9077-7b2df245855a",
            "3f4515eb-ecf6-4c4e-821d-e5d537ea3d44",
            "432fd9fb-72e0-4a05-9b51-f358cda31ede",
            "4493205e-1022-4a34-8953-c01b7852b956",
            "46b224b3-9589-4660-ad6a-3e1064f769b3",
            "4a1bc66d-010a-4655-874e-bc6398a6c5ea",
            "4b90ac73-a02f-4d34-882c-31e003c0060a",
            "6f6f2277-13f2-4a5c-9713-5e907afc160c",
            "6f7edfef-e3ac-4d18-94a3-04945565e679",
            "7190fc73-932d-42f8-9820-867457253ce1",
            "77c96207-adbe-44f0-8a26-28a8f32d3573",
            "7c5b0f7f-d6e8-410a-9650-b42aa703a37e",
            "8e75a096-1dda-49e4-a3e7-ef9d6c0beac8",
            "9057d21a-0893-4fe7-9ae1-af842f70df9e",
            "94ccc048-1430-426c-9318-081716f83979",
            "99c0e671-f7db-4a60-85ec-e434cfae6b45",
            "9ac32fd3-cb66-423d-ade6-34437b0686bd",
            "a123918f-1a1c-43d8-9775-7d6bc23ac8c5",
            "a5097e4d-e26d-44d8-a436-c28a02264f51",
            "ac475b68-e657-435c-9b6a-fcd97d3839b9",
            "b611092b-7cff-467f-8b80-7cd518c8d64c",
            "b6194edb-59cb-4bc9-b310-ec94372c9026",
            "c81860fc-cffe-44d1-b702-8d6b8f1dbe17",
            "d1a0867f-01a5-4e03-9895-acbcf46ce6b6",
            "eec20116-b4f1-4e2d-960a-619098fd45cb",
            "f177fdc5-4cef-4c90-bef5-d1738ab24c9c",
            "f225ab85-2b02-4e83-b90e-a373d07aa62c",
            "f55e3861-b7f2-4342-84ab-00af59d11837",
        },
    },
    "/non-unified": {
        "2022-06-train": {
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
        },
        "2022-06": {"q062210670", "q062211678"},
        "2022-07": {
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
        },
        "2022-09": {
            "q09221028",
            "q092211529",
            "q092211859",
            "q092212269",
            "q092212770",
            "q092212935",
            "q092213323",
            "q092213466",
            "q092214092",
            "q092214164",
            "q09221676",
            "q092217900",
            "q092218877",
            "q092220928",
            "q09222104",
            "q092222066",
            "q092222694",
            "q092223550",
            "q092230063",
            "q092231346",
            "q092231533",
            "q092231564",
            "q092231615",
            "q092232219",
            "q09223255",
            "q092235137",
            "q092236086",
            "q09223832",
            "q09224178",
            "q09226768",
            "q09227549",
            "q09228100",
            "q0922895",
            "q09229220",
            "q09229565",
            "q09229591",
        },
    },
}


@pytest.mark.parametrize(
    "language,unified_mode",
    [("en", "/non-unified"), ("fr", "/non-unified"), ("en", ""), ("fr", "")],
)
class TestLongEval2023Meta:
    """
    A parameterized test class that runs all "meta" tests (tag checks, all_datasets check) for
    both languages and both modes (unified and non-unified).

    The 'self.language' and self.unified_mode attributes are injected by parameterized_class.
    """

    def test_clef_2023_train_tag(self, language, unified_mode):
        datasets = load(f"longeval-2023/clef-2023-train/{language}{unified_mode}")
        expected_tags = ["2022-06-train"]
        tags = [dataset.get_snapshot() for dataset in datasets.get_datasets()]
        assert sorted(expected_tags) == sorted(tags)

    def test_clef_2023_tag(self, language, unified_mode):
        datasets = load(f"longeval-2023/clef-2023/{language}{unified_mode}")
        expected_tags = ["2022-06", "2022-07", "2022-09"]
        tags = [dataset.get_snapshot() for dataset in datasets.get_datasets()]
        assert sorted(expected_tags) == sorted(tags)

    def test_longeval_2023_all_datasets(self, language, unified_mode):
        dataset_id = f"longeval-2023/*/{language}{unified_mode}"
        meta_dataset = load(dataset_id)
        with pytest.raises(AttributeError):
            meta_dataset.queries_iter()

        with pytest.raises(AttributeError):
            meta_dataset.docs_iter()

        datasets = meta_dataset.get_datasets()
        assert 4 == len(datasets)
        assert "2022-06-train" == datasets[0].get_snapshot()

        prior_datasets_count = 0
        for dataset in datasets:
            dataset_prior_datasets = dataset.get_prior_datasets()
            assert prior_datasets_count == len(dataset_prior_datasets)
            prior_datasets_count += 1
            assert dataset.has_queries()
            assert dataset.has_docs()

        prior_datasets_list = datasets[1].get_prior_datasets()
        assert 1 == len(prior_datasets_list)
        assert prior_datasets_list[0].has_queries()
        assert prior_datasets_list[0].has_docs()


class TestLongEval2023Snapshot:
    @pytest.fixture(
        scope="class",
        params=[
            # {
            #     "snapshot": "2022-06-train",
            #     "language": "en",
            #     "unified_mode": "/non-unified",
            #     "n_queries": 672,
            #     "expected_queries": {"q06223196": "Car shelter"},
            #     "n_qrels": 9656,
            #     "expected_docs": {
            #         "doc062206000001": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": [],
            # },
            # {
            #     "snapshot": "2022-06-train",
            #     "language": "fr",
            #     "unified_mode": "/non-unified",
            #     "n_queries": 672,
            #     "expected_queries": {"q06223196": "abri de voiture"},
            #     "n_qrels": 9656,
            #     "expected_docs": {
            #         "doc062206000001": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": [],
            # },
            # {
            #     "snapshot": "2022-06-train",
            #     "language": "en",
            #     "unified_mode": "",
            #     "n_queries": 672,
            #     "expected_queries": {
            #         "fe0682b1-d3b8-40f1-b059-39e26b257895": "Car shelter"
            #     },
            #     "n_qrels": 9656,
            #     "expected_docs": {
            #         "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": [],
            # },
            # {
            #     "snapshot": "2022-06-train",
            #     "language": "fr",
            #     "unified_mode": "",
            #     "n_queries": 672,
            #     "expected_queries": {
            #         "fe0682b1-d3b8-40f1-b059-39e26b257895": "abri de voiture"
            #     },
            #     "n_qrels": 9656,
            #     "expected_docs": {
            #         "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": [],
            # },
            # # 2022-06
            # {
            #     "snapshot": "2022-06",
            #     "language": "en",
            #     "unified_mode": "/non-unified",
            #     "n_queries": 98,
            #     "expected_queries": {"q06227021": "anti-virus"},
            #     "n_qrels": 1420,
            #     "expected_docs": {
            #         "doc062206000001": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06-train"],
            # },
            # {
            #     "snapshot": "2022-06",
            #     "language": "fr",
            #     "unified_mode": "/non-unified",
            #     "n_queries": 98,
            #     "expected_queries": {"q06227021": "comparatif antivirus"},
            #     "n_qrels": 1420,
            #     "expected_docs": {
            #         "doc062206000001": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06-train"],
            # },
            # {
            #     "snapshot": "2022-06",
            #     "language": "en",
            #     "unified_mode": "",
            #     "n_queries": 98,
            #     "expected_queries": {
            #         "bffdfe4f-e593-4e41-935f-ff0392924ef8": "anti-virus"
            #     },
            #     "n_qrels": 1420,
            #     "expected_docs": {
            #         "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06-train"],
            # },
            # {
            #     "snapshot": "2022-06",
            #     "language": "fr",
            #     "unified_mode": "",
            #     "n_queries": 98,
            #     "expected_queries": {
            #         "bffdfe4f-e593-4e41-935f-ff0392924ef8": "comparatif antivirus"
            #     },
            #     "n_qrels": 1420,
            #     "expected_docs": {
            #         "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06-train"],
            # },
            # # 2022-07
            # {
            #     "snapshot": "2022-07",
            #     "language": "en",
            #     "unified_mode": "/non-unified",
            #     "n_queries": 882,
            #     "expected_queries": {"q07226468": "recipe potato"},
            #     "n_qrels": 12217,
            #     "expected_docs": {
            #         "doc072211604226": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06", "2022-06-train"],
            # },
            # {
            #     "snapshot": "2022-07",
            #     "language": "fr",
            #     "unified_mode": "/non-unified",
            #     "n_queries": 882,
            #     "expected_queries": {"q07226468": "recette pomme de terre"},
            #     "n_qrels": 12217,
            #     "expected_docs": {
            #         "doc072211604226": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06", "2022-06-train"],
            # },
            # {
            #     "snapshot": "2022-07",
            #     "language": "en",
            #     "unified_mode": "",
            #     "n_queries": 855,
            #     "expected_queries": {
            #         "7395cc64-4827-40f0-af2f-a138794c0cb1": "recipe potato"
            #     },
            #     "n_qrels": 12217,
            #     "expected_docs": {
            #         "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06", "2022-06-train"],
            # },
            # {
            #     "snapshot": "2022-07",
            #     "language": "fr",
            #     "unified_mode": "",
            #     "n_queries": 855,
            #     "expected_queries": {
            #         "7395cc64-4827-40f0-af2f-a138794c0cb1": "recette pomme de terre"
            #     },
            #     "n_qrels": 12217,
            #     "expected_docs": {
            #         "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
            #     },
            #     "prior_snapshots": ["2022-06", "2022-06-train"],
            # },
            # 2022-09
            {
                "snapshot": "2022-09",
                "language": "en",
                "unified_mode": "/non-unified",
                "n_queries": 923,
                "expected_queries": {"q092223595": "Car shelter"},
                "n_qrels": 13467,
                "expected_docs": {
                    "doc092208510333": "http://balliuexport.com/fr/politique-confidentialite.html"
                },
                "prior_snapshots": ["2022-07", "2022-06", "2022-06-train"],
            },
            {
                "snapshot": "2022-09",
                "language": "fr",
                "unified_mode": "/non-unified",
                "n_queries": 923,
                "expected_queries": {"q092223595": "abri de voiture"},
                "n_qrels": 13467,
                "expected_docs": {
                    "doc092208510333": "http://balliuexport.com/fr/politique-confidentialite.html"
                },
                "prior_snapshots": ["2022-07", "2022-06", "2022-06-train"],
            },
            {
                "snapshot": "2022-09",
                "language": "en",
                "unified_mode": "",
                "n_queries": 904,  # fewer queries due to unification (e.g. voitures électriques, voitures electriques, voitures électrique)
                "expected_queries": {
                    "a62207e8-c126-4fc1-bddd-cb707672a25c": "Car shelter"
                },
                "n_qrels": 13467,
                "expected_docs": {
                    "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
                },
                "prior_snapshots": ["2022-07", "2022-06", "2022-06-train"],
            },
            {
                "snapshot": "2022-09",
                "language": "fr",
                "unified_mode": "",
                "n_queries": 904,  # fewer queries due to unification (e.g. voitures électriques, voitures electriques, voitures électrique)
                "expected_queries": {
                    "a62207e8-c126-4fc1-bddd-cb707672a25c": "abri de voiture"
                },
                "n_qrels": 13467,
                "expected_docs": {
                    "c404f769-3788-4462-b72a-a03dfeb9e864": "http://balliuexport.com/fr/politique-confidentialite.html"
                },
                "prior_snapshots": ["2022-07", "2022-06", "2022-06-train"],
            },
        ],
    )
    def snapshot_data(self, request):
        snapshot, language, unified_mode = (
            request.param["snapshot"],
            request.param["language"],
            request.param["unified_mode"],
        )
        loaded_datasets = load(f"longeval-2023/{snapshot}/{language}{unified_mode}")
        yield loaded_datasets, request.param

    def test_snapshot_exists(self, snapshot_data):
        dataset, setting = snapshot_data
        assert dataset is not None
        assert dataset.get_snapshot() == setting["snapshot"]

    def test_queries(self, snapshot_data):
        dataset, setting = snapshot_data

        assert dataset.has_queries()

        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        assert len(actual_queries) == setting["n_queries"]
        for k, v in setting["expected_queries"].items():
            assert v == actual_queries[k]

    def test_docs(self, snapshot_data):
        dataset, setting = snapshot_data

        assert dataset.has_docs()

        example_doc = dataset.docs_iter().__next__()
        assert example_doc is not None

    def test_docstore(self, snapshot_data):
        dataset, setting = snapshot_data

        docs_store = dataset.docs_store()

        for docid, url in setting["expected_docs"].items():
            assert docs_store.get(docid).doc_id == docid
            assert docs_store.get(docid).url == url

    def test_qrels(self, snapshot_data):
        dataset, setting = snapshot_data

        assert dataset.has_qrels()

        qrels = list(dataset.qrels_iter())
        assert len(qrels) == setting["n_qrels"]

        # test qrels
        # all queries have judgments
        # all qids in qrels are in queries
        qids_in_qrels = {qrel.query_id for qrel in dataset.qrels_iter()}
        qids_in_queries = {query.query_id for query in dataset.queries_iter()}
        assert qids_in_qrels == qids_in_queries

        # all qids have relevant docs
        qids_in_qrels = {
            qrel.query_id for qrel in dataset.qrels_iter() if qrel.relevance > 0
        }

        # add queries without relevant docs back in
        qids_in_qrels.update(
            QRELS_WITHOUT_REL_DOCS[setting["unified_mode"]][setting["snapshot"]]
        )

        assert qids_in_qrels == qids_in_queries

        # all docids in qrels are in docs
        docs_store = dataset.docs_store()
        docids_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}
        for doc_id in docids_in_qrels:
            assert docs_store.get(doc_id) is not None

    def test_timestamp(self, snapshot_data):
        dataset, setting = snapshot_data
        assert dataset.get_timestamp() == datetime.strptime(
            setting["snapshot"][:7], "%Y-%m"
        )  # ignore "-test"

    def test_prior_datasets(self, snapshot_data):
        dataset, setting = snapshot_data

        prior_datasets = dataset.get_prior_datasets()
        assert len(prior_datasets) == len(setting["prior_snapshots"])

        for i, prior_snapshot in enumerate(prior_datasets):
            assert prior_snapshot.get_snapshot() == setting["prior_snapshots"][i]
            assert len(prior_snapshot.get_prior_datasets()) == len(
                setting["prior_snapshots"][i + 1 :]
            )
