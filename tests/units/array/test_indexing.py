import numpy as np
import pytest
import torch

from docarray import DocumentArray
from docarray.documents import Text
from docarray.typing import TorchTensor


@pytest.fixture()
def da():
    texts = [f'hello {i}' for i in range(10)]
    tensors = [torch.ones((4,)) * i for i in range(10)]
    return DocumentArray[Text](
        [Text(text=text, embedding=tens) for text, tens in zip(texts, tensors)],
        tensor_type=TorchTensor,
    )


@pytest.fixture()
def da_to_set():
    texts = [f'hello {2*i}' for i in range(5)]
    tensors = [torch.ones((4,)) * i * 2 for i in range(5)]
    return DocumentArray[Text](
        [Text(text=text, embedding=tens) for text, tens in zip(texts, tensors)],
        tensor_type=TorchTensor,
    )


###########
# getitem
###########


@pytest.mark.parametrize('stack', [True, False])
def test_simple_getitem(stack, da):
    if stack:
        da = da.stack()

    assert torch.all(da[0].embedding == torch.zeros((4,)))
    assert da[0].text == 'hello 0'


@pytest.mark.parametrize('stack', [True, False])
def test_get_none(stack, da):
    if stack:
        da = da.stack()

    assert da[None] is da


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('index', [(1, 2, 3, 4, 6), [1, 2, 3, 4, 6]])
def test_iterable_getitem(stack, da, index):
    if stack:
        da = da.stack()

    indexed_da = da[index]

    for pos, d in zip(index, indexed_da):
        assert d.text == f'hello {pos}'
        assert torch.all(d.embedding == torch.ones((4,)) * pos)


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('index_dtype', [torch.int64])
def test_torchtensor_getitem(stack, da, index_dtype):
    if stack:
        da = da.stack()

    index = torch.tensor([1, 2, 3, 4, 6], dtype=index_dtype)

    indexed_da = da[index]

    for pos, d in zip(index, indexed_da):
        assert d.text == f'hello {pos}'
        assert torch.all(d.embedding == torch.ones((4,)) * pos)


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('index_dtype', [int, np.int_, np.int32, np.int64])
def test_nparray_getitem(stack, da, index_dtype):
    if stack:
        da = da.stack()

    index = np.array([1, 2, 3, 4, 6], dtype=index_dtype)

    indexed_da = da[index]
    for pos, d in zip(index, indexed_da):
        assert d.text == f'hello {pos}'
        assert torch.all(d.embedding == torch.ones((4,)) * pos)


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize(
    'index',
    [
        [False, True, True, True, True, False, True, False, False, False],
        (False, True, True, True, True, False, True, False, False, False),
        torch.tensor([0, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool),
        np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool),
    ],
)
def test_boolmask_getitem(stack, da, index):
    if stack:
        da = da.stack()

    indexed_da = da[index]

    mask_true_idx = [1, 2, 3, 4, 6]

    for pos, d in zip(mask_true_idx, indexed_da):
        assert d.text == f'hello {pos}'
        assert torch.all(d.embedding == torch.ones((4,)) * pos)


###########
# setitem
###########


@pytest.mark.parametrize('stack_left', [True, False])
def test_simple_setitem(stack_left, da, da_to_set):
    if stack_left:
        da = da.stack()

    da[0] = da_to_set[0]

    assert torch.all(da[0].embedding == da_to_set[0].embedding)
    assert da[0].text == da_to_set[0].text


@pytest.mark.parametrize('stack_left', [True, False])
@pytest.mark.parametrize('stack_right', [True, False])
@pytest.mark.parametrize('index', [(1, 2, 3, 4, 6), [1, 2, 3, 4, 6]])
def test_iterable_setitem(stack_left, stack_right, da, da_to_set, index):
    if stack_left:
        da = da.stack()
    if stack_right:
        da_to_set = da_to_set.stack()

    da[index] = da_to_set

    i_da_to_set = 0
    for i, d in enumerate(da):
        if i in index:
            d_reference = da_to_set[i_da_to_set]
            assert d.text == d_reference.text
            assert torch.all(d.embedding == d_reference.embedding)
            i_da_to_set += 1
        else:
            assert d.text == f'hello {i}'
            assert torch.all(d.embedding == torch.ones((4,)) * i)


@pytest.mark.parametrize('stack_left', [True, False])
@pytest.mark.parametrize('stack_right', [True, False])
@pytest.mark.parametrize('index_dtype', [torch.int64])
def test_torchtensor_setitem(stack_left, stack_right, da, da_to_set, index_dtype):
    if stack_left:
        da = da.stack()
    if stack_right:
        da_to_set = da_to_set.stack()

    index = torch.tensor([1, 2, 3, 4, 6], dtype=index_dtype)

    da[index] = da_to_set

    i_da_to_set = 0
    for i, d in enumerate(da):
        if i in index:
            d_reference = da_to_set[i_da_to_set]
            assert d.text == d_reference.text
            assert torch.all(d.embedding == d_reference.embedding)
            i_da_to_set += 1
        else:
            assert d.text == f'hello {i}'
            assert torch.all(d.embedding == torch.ones((4,)) * i)


@pytest.mark.parametrize('stack_left', [True, False])
@pytest.mark.parametrize('stack_right', [True, False])
@pytest.mark.parametrize('index_dtype', [int, np.int_, np.int32, np.int64])
def test_nparray_setitem(stack_left, stack_right, da, da_to_set, index_dtype):
    if stack_left:
        da = da.stack()
    if stack_right:
        da_to_set = da_to_set.stack()

    index = np.array([1, 2, 3, 4, 6], dtype=index_dtype)

    da[index] = da_to_set

    i_da_to_set = 0
    for i, d in enumerate(da):
        if i in index:
            d_reference = da_to_set[i_da_to_set]
            assert d.text == d_reference.text
            assert torch.all(d.embedding == d_reference.embedding)
            i_da_to_set += 1
        else:
            assert d.text == f'hello {i}'
            assert torch.all(d.embedding == torch.ones((4,)) * i)


@pytest.mark.parametrize('stack_left', [True, False])
@pytest.mark.parametrize('stack_right', [True, False])
@pytest.mark.parametrize(
    'index',
    [
        [False, True, True, True, True, False, True, False, False, False],
        (False, True, True, True, True, False, True, False, False, False),
        torch.tensor([0, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool),
        np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool),
    ],
)
def test_boolmask_setitem(stack_left, stack_right, da, da_to_set, index):
    if stack_left:
        da = da.stack()
    if stack_right:
        da_to_set = da_to_set.stack()

    da[index] = da_to_set

    mask_true_idx = [1, 2, 3, 4, 6]
    i_da_to_set = 0
    for i, d in enumerate(da):
        if i in mask_true_idx:
            d_reference = da_to_set[i_da_to_set]
            assert d.text == d_reference.text
            assert torch.all(d.embedding == d_reference.embedding)
            i_da_to_set += 1
        else:
            assert d.text == f'hello {i}'
            assert torch.all(d.embedding == torch.ones((4,)) * i)


def test_setitiem_update_column():
    texts = [f'hello {i}' for i in range(10)]
    tensors = [torch.ones((4,)) * (i + 1) for i in range(10)]
    da = DocumentArray[Text](
        [Text(text=text, embedding=tens) for text, tens in zip(texts, tensors)],
        tensor_type=TorchTensor,
    ).stack()

    da[0] = Text(text='hello', embedding=torch.zeros((4,)))

    assert da[0].text == 'hello'
    assert (da[0].embedding == torch.zeros((4,))).all()
    assert (da.embedding[0] == torch.zeros((4,))).all()
