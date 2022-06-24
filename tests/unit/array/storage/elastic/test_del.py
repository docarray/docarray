from docarray import Document, DocumentArray
import pytest


@pytest.mark.parametrize('deleted_elmnts', [[0, 1], ['r0', 'r1']])
def test_delete_offset_success_sync_es_offset_index(deleted_elmnts, start_storage):
    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'columns': [('price', 'int')],
            'distance': 'l2_norm',
            'index_name': 'test_delete',
        },
    )

    with elastic_doc:
        elastic_doc.extend(
            [
                Document(id='r0', embedding=[0, 0, 0]),
                Document(id='r1', embedding=[1, 1, 1]),
                Document(id='r2', embedding=[2, 2, 2]),
                Document(id='r3', embedding=[3, 3, 3]),
                Document(id='r4', embedding=[4, 4, 4]),
                Document(id='r5', embedding=[5, 5, 5]),
                Document(id='r6', embedding=[6, 6, 6]),
                Document(id='r7', embedding=[7, 7, 7]),
            ]
        )

    expected_offset_after_del = ['r2', 'r3', 'r4', 'r5', 'r6', 'r7']

    with elastic_doc:
        del elastic_doc[deleted_elmnts]

    indexed_offset_count = elastic_doc._client.count(
        index=elastic_doc._index_name_offset2id
    )['count']

    assert len(elastic_doc._offset2ids.ids) == indexed_offset_count

    for id in expected_offset_after_del:
        expected_offset = str(expected_offset_after_del.index(id))
        actual_offset_index = elastic_doc._client.search(
            index=elastic_doc._index_name_offset2id, query={'match': {'blob': id}}
        )['hits']['hits'][0]['_id']
        assert actual_offset_index == expected_offset
