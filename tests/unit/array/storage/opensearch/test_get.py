import numpy as np
import pytest

from docarray import Document, DocumentArray


@pytest.mark.parametrize('nrof_docs', [10, 100, 10_000, 10_100, 20_000, 20_100])
@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_success_get_bulk_data(start_storage, nrof_docs, columns):
    opensearch_doc = DocumentArray(
        storage='opensearch',
        config={
            'n_dim': 3,
            'columns': columns,
            'distance': 'l2',
        },
    )

    with opensearch_doc:
        opensearch_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ]
        )

    assert len(opensearch_doc[:, 'id']) == nrof_docs


@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_error_get_bulk_data_id_not_exist(start_storage, columns):
    nrof_docs = 10

    opensearch_doc = DocumentArray(
        storage='opensearch',
        config={
            'n_dim': 3,
            'columns': columns,
            'distance': 'l2',
        },
    )

    with opensearch_doc:
        opensearch_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ]
        )

    with pytest.raises(KeyError) as e:
        opensearch_doc[['r1', 'r11', 'r21'], 'id']

    assert e.value.args[0] == ['r11', 'r21']
    assert len(e.value.args[1]) == 1
