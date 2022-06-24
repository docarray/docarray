from docarray import Document, DocumentArray


def test_add_ignore_existing_doc_id():
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

    elastic_doc._client.indices.delete(
        index=elastic_doc._index_name_offset2id, ignore=[400, 404]
    )
    elastic_doc._client.indices.delete(
        index=elastic_doc._config.index_name, ignore=[400, 404]
    )
