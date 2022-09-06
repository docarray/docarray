from docarray import Document, DocumentArray
import pytest


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('deleted_elmnts', [[0, 1], ['r0', 'r1']])
@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_delete_offset_success_sync_es_offset_index(
    deleted_elmnts, start_storage, columns
):
    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'columns': columns,
            'distance': 'l2_norm',
            'index_name': 'test_delete_offset_success_sync_es_offset_index',
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
    assert len(elastic_doc._offset2ids.ids) == 6
    assert len(elastic_doc[:, 'embedding']) == 6

    for id in expected_offset_after_del:
        expected_offset = str(expected_offset_after_del.index(id))
        actual_offset_index = elastic_doc._client.search(
            index=elastic_doc._index_name_offset2id, query={'match': {'blob': id}}
        )['hits']['hits'][0]['_id']
        assert actual_offset_index == expected_offset


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_success_handle_bulk_delete_not_found(start_storage, columns):
    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'columns': columns,
            'distance': 'l2_norm',
            'index_name': 'test_bulk_delete_not_found',
        },
    )
    with elastic_doc:
        elastic_doc.extend(
            [
                Document(id='r0', embedding=[0, 0, 0]),
                Document(id='r1', embedding=[1, 1, 1]),
            ]
        )

    offset_index = elastic_doc._index_name_offset2id

    expected_to_be_fail_del_data = [
        {
            '_op_type': 'delete',
            '_id': 0,  # offset data exist
            '_index': offset_index,
        },
        {
            '_op_type': 'delete',
            '_id': 2,  # offset data not exist, expect to fail
            '_index': offset_index,
        },
    ]

    info = elastic_doc._send_requests(expected_to_be_fail_del_data)

    assert len(info) == 1
    assert 'delete' in info[0].keys()
