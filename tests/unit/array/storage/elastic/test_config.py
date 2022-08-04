from docarray import Document, DocumentArray
import numpy as np
import pytest


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bulk_config(start_storage):
    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'columns': [
                ('is_true', 'bool'),
                ('test_long', 'long'),
                ('test_double', 'double'),
            ],
            'distance': 'l2_norm',
            'index_name': 'test_data_type',
            'bulk_config': {
                'thread_count': 4,
                'chunk_size': 100,
                'max_chunk_bytes': 104857600,
                'queue_size': 4,
            },
        },
    )

    with elastic_doc:
        elastic_doc.extend(
            [Document(id=f'r{i}', embedding=np.ones((3,)) * i) for i in range(100)]
        )
