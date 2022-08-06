from docarray import Document, DocumentArray
import numpy as np


def test_get_bulk_data(start_storage):
    nrof_docs = 20000

    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'columns': [('price', 'int')],
            'distance': 'l2_norm',
            'index_name': 'test_get_bulk_data',
        },
    )

    with elastic_doc:
        elastic_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ]
        )

    assert len(elastic_doc[:, "id"]) == nrof_docs
