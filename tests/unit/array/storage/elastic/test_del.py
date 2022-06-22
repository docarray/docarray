import docarray.array.storage.elastic.backend as elastic_backend
from docarray.array.storage.base.helper import Offset2ID

MOCK_ES_OFFSET_INDEX = {
    0: {"_id": 0, "_source": {"blob": "r0"}},
    1: {"_id": 1, "_source": {"blob": "r1"}},
    2: {"_id": 2, "_source": {"blob": "r2"}},
    3: {"_id": 3, "_source": {"blob": "r3"}},
    4: {"_id": 4, "_source": {"blob": "r4"}},
    5: {"_id": 5, "_source": {"blob": "r5"}},
    6: {"_id": 6, "_source": {"blob": "r6"}},
    7: {"_id": 7, "_source": {"blob": "r7"}},
    8: {"_id": 8, "_source": {"blob": "r8"}},
}

BUG_OFFSET_INDEX_AFTER_DELETION = {
    0: {"_id": 0, "_source": {"blob": "r3"}},
    1: {"_id": 1, "_source": {"blob": "r4"}},
    2: {"_id": 2, "_source": {"blob": "r5"}},
    3: {"_id": 3, "_source": {"blob": "r6"}},
    4: {"_id": 4, "_source": {"blob": "r7"}},
    5: {"_id": 5, "_source": {"blob": "r8"}},
    6: {"_id": 6, "_source": {"blob": "r6"}},
    7: {"_id": 7, "_source": {"blob": "r7"}},
    8: {"_id": 8, "_source": {"blob": "r8"}},
}

EXPECTED_INDEX_AFTER_DELETION = {
    0: {"_id": 0, "_source": {"blob": "r3"}},
    1: {"_id": 1, "_source": {"blob": "r4"}},
    2: {"_id": 2, "_source": {"blob": "r5"}},
    3: {"_id": 3, "_source": {"blob": "r6"}},
    4: {"_id": 4, "_source": {"blob": "r7"}},
    5: {"_id": 5, "_source": {"blob": "r8"}},
}

ES_BULK_COUNT = 0


class MockElasticIndices:
    def __init__(self, **kwargs):
        pass

    def exists(self, **kwargs):
        return True

    def refresh(self, **kwargs):
        pass


class MockElasticClient:
    bulk_call_count = 0

    def __init__(self, **kwargs):
        self.indices = MockElasticIndices()

    def count(self, **kwargs):
        return {"count": len(MOCK_ES_OFFSET_INDEX)}

    @classmethod
    def bulk(cls, *args, **kwargs):
        cls.bulk_call_count += 1

        requests = args[1]

        for request in requests:
            if request["_op_type"] == "index":
                MOCK_ES_OFFSET_INDEX[request["_id"]] = {
                    "_id": request["_id"],
                    "_source": {"blob": request["blob"]},
                }
            elif request["_op_type"] == "delete":
                del MOCK_ES_OFFSET_INDEX[request["_id"]]

        if ES_BULK_COUNT == 1:  # 1st bulk api call, update offset index
            assert MOCK_ES_OFFSET_INDEX == BUG_OFFSET_INDEX_AFTER_DELETION


def test_delete_offset_success_sync_es_offset_index(monkeypatch):
    monkeypatch.setattr(elastic_backend, "Elasticsearch", MockElasticClient)
    monkeypatch.setattr(elastic_backend, "bulk", MockElasticClient.bulk)

    es_backend = elastic_backend.BackendMixin()
    es_backend._load_offset2ids = lambda **kwargs: None

    # del docs with id ["r0","r1","r2"]
    modified_ids_after_delete = ["r3", "r4", "r5", "r6", "r7", "r8"]
    es_backend._offset2ids = Offset2ID(ids=modified_ids_after_delete)
    es_backend._init_storage(
        config={
            "n_dim": 3,
            "columns": [("price", "int")],
            "distance": "l2_norm",
            "index_name": "old_stuff",
        }
    )
    es_backend._update_offset2ids_meta()
    assert MOCK_ES_OFFSET_INDEX == EXPECTED_INDEX_AFTER_DELETION
