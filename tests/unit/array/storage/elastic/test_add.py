from docarray import Document, DocumentArray


def test_add_ignore_existing_doc_id(start_storage):
    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'columns': [('price', 'int')],
            'distance': 'l2_norm',
            'index_name': 'test_add_ignore_existing_doc_id',
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
            ]
        )

    with elastic_doc:
        elastic_doc.extend(
            [
                Document(id='r0', embedding=[0, 0, 0]),
                Document(id='r2', embedding=[2, 2, 2]),
                Document(id='r4', embedding=[4, 4, 4]),
                Document(id='r5', embedding=[2, 2, 2]),
                Document(id='r6', embedding=[4, 4, 4]),
            ]
        )

    indexed_offset_count = elastic_doc._client.count(
        index=elastic_doc._index_name_offset2id
    )['count']

    assert len(elastic_doc) == len(elastic_doc[:, 'embedding'])
    assert len(elastic_doc) == indexed_offset_count
    assert len(elastic_doc[:, 'embedding']) == 7


def test_add_skip_wrong_data_type_and_fix_offset():
    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'columns': [('price', 'int')],
            'index_name': 'test_add_skip_wrong_data_type_and_fix_offset',
        },
    )

    with elastic_doc:
        elastic_doc.extend(
            [
                Document(id='0', price=1000),
                Document(id='1', price=20000),
                Document(id='2', price=103000),
            ]
        )

    with elastic_doc:
        elastic_doc.extend(
            [
                Document(id='0', price=100000000000),
                Document(id='1', price=20000),
            ]
        )
