import pytest

from docarray import DocumentArray, Document


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
    ['deleted_ids', 'expected_ids'],
    [
        (['1', '2', '3', '4'], ['5', '6', '7', '8', '9']),
        (['2', '4', '7', '1'], ['3', '5', '6', '8', '9']),
    ],
)
def test_del_by_multiple_idx(docs, deleted_ids, expected_ids):
    del docs[deleted_ids]
    assert docs[:, 'id'] == expected_ids
