import pytest

from docarray import DocumentArray, Document
from docarray.array.weaviate import DocumentArrayWeaviate
import numpy as np


@pytest.fixture()
def docs():
    return DocumentArray([Document(id=f'{i}') for i in range(1, 10)])


@pytest.mark.parametrize(
    'to_delete',
    [
        0,
        1,
        4,
        -1,
        list(range(1, 4)),
        [2, 4, 7, 1, 1],
        slice(0, 2),
        slice(2, 4),
        slice(4, -1),
        [True, True, False],
        ...,
    ],
)
def test_del_all(docs, to_delete):
    doc_to_delete = docs[to_delete]
    del docs[to_delete]
    assert doc_to_delete not in docs


@pytest.mark.parametrize(
    'to_delete, missing_id',
    [
        ([True, False], ['1']),
        ([True, True, False], ['1', '2']),
        ([False, True], ['2']),
        ([False, False, True, True], ['3', '4']),
    ],
)
def test_del_boolean_mask(docs, to_delete, missing_id):
    all_ids = docs[:, 'id']
    # assert each missing_id is present before deleting
    for m_id in missing_id:
        assert m_id in docs[:, 'id']

    del docs[to_delete]

    # assert each missing_id is NOT present AFTER deleting
    for m_id in missing_id:
        assert m_id not in docs[:, 'id']
    for m_id in filter(lambda id: id not in missing_id, all_ids):
        assert m_id in docs[:, 'id']


@pytest.mark.parametrize(
    ['deleted_ids', 'expected_ids'],
    [
        (['1', '2', '3', '4'], ['5', '6', '7', '8', '9']),
        (['2', '4', '7', '1'], ['3', '5', '6', '8', '9']),
    ],
)
def test_del_by_multiple_idx(docs, deleted_ids, expected_ids):
    del docs[deleted_ids]
    assert docs[:, 'id'] == expected_ids


@pytest.mark.parametrize(
    'da_cls,config,persist',
    [
        (DocumentArrayWeaviate, {'n_dim': 10}, False),
        (DocumentArrayWeaviate, {'name': 'Storage', 'n_dim': 10}, True),
    ],
)
def test_del_da_persist(da_cls, config, persist, docs, start_storage):
    da = da_cls(docs, config=config)
    del da

    da2 = da_cls(config=config)
    if persist:
        assert len(da2) == len(docs)
    else:
        assert len(da2) == 0


def test_del_da_attribute():

    da = DocumentArray(
        [
            Document(embedding=np.array([1, 2, 3]), text='d1'),
            Document(embedding=np.array([1, 2, 3]), text='d2'),
        ]
    )

    q = DocumentArray(
        [
            Document(embedding=np.array([4, 5, 6]), text='q1'),
            Document(embedding=np.array([2, 3, 4]), text='q1'),
        ]
    )

    da.match(q)
    del da[...][:, 'embedding']

    for d in da:
        assert d.embedding is None


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2'}),
        ('milvus', {'n_dim': 3, 'distance': 'L2'}),
    ],
)
def test_del_subindex(storage, config, start_storage):

    n_dim = 3
    subindex_configs = (
        {'@c': dict()} if storage in ['sqlite', 'memory'] else {'@c': {'n_dim': 2}}
    )
    da = DocumentArray(
        storage=storage,
        config=config,
        subindex_configs=subindex_configs,
    )

    with da:
        da.extend(
            [
                Document(
                    id=str(i),
                    embedding=i * np.ones(n_dim),
                    chunks=[
                        Document(id=str(i) + '_0', embedding=np.array([i, i])),
                        Document(id=str(i) + '_1', embedding=np.array([i, i])),
                    ],
                )
                for i in range(10)
            ]
        )

    del da['0']
    assert len(da) == 9
    assert len(da._subindices['@c']) == 18

    del da[-2:]
    assert len(da) == 7
    assert len(da._subindices['@c']) == 14


def test_del_subindex_annlite_multimodal():
    from docarray import dataclass
    from docarray.typing import Text

    @dataclass
    class MMDoc:
        my_text: Text
        my_other_text: Text

    n_dim = 3
    da = DocumentArray(
        storage='annlite',
        config={'n_dim': n_dim, 'metric': 'Euclidean'},
        subindex_configs={'@.[my_text, my_other_text]': {'n_dim': 2}},
    )

    num_docs = 10
    docs_to_add = DocumentArray(
        [
            Document(MMDoc(my_text='hello', my_other_text='world'))
            for _ in range(num_docs)
        ]
    )
    for i, d in enumerate(docs_to_add):
        d.id = str(i)
        d.embedding = i * np.ones(n_dim)
        d.my_text.id = str(i) + '_0'
        d.my_text.embedding = [i, i]
        d.my_other_text.id = str(i) + '_1'
        d.my_other_text.embedding = [i, i]

    with da:
        da.extend(docs_to_add)

    del da['0']
    assert len(da) == 9
    assert len(da._subindices['@.[my_text, my_other_text]']) == 18
