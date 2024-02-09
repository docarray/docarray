// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import numpy as np
import pytest
import torch

from docarray import DocList, DocVec
from docarray.documents import TextDoc
from docarray.typing import TorchTensor


@pytest.fixture()
def da():
    texts = [f'hello {i}' for i in range(10)]
    tensors = [torch.ones((4,)) * i for i in range(10)]
    return DocList[TextDoc](
        [TextDoc(text=text, embedding=tens) for text, tens in zip(texts, tensors)],
    )


@pytest.fixture()
def da_to_set():
    texts = [f'hello {2*i}' for i in range(5)]
    tensors = [torch.ones((4,)) * i * 2 for i in range(5)]
    return DocList[TextDoc](
        [TextDoc(text=text, embedding=tens) for text, tens in zip(texts, tensors)],
    )


###########
# getitem
###########


@pytest.mark.parametrize('stack', [True, False])
def test_simple_getitem(stack, da):
    if stack:
        da = da.to_doc_vec(tensor_type=TorchTensor)

    assert torch.all(da[0].embedding == torch.zeros((4,)))
    assert da[0].text == 'hello 0'


@pytest.mark.parametrize('stack', [True, False])
def test_get_none(stack, da):
    if stack:
        da = da.to_doc_vec(tensor_type=TorchTensor)

    assert da[None] is da


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('index', [(1, 2, 3, 4, 6), [1, 2, 3, 4, 6]])
def test_iterable_getitem(stack, da, index):
    if stack:
        da = da.to_doc_vec(tensor_type=TorchTensor)

    indexed_da = da[index]

    for pos, d in zip(index, indexed_da):
        assert d.text == f'hello {pos}'
        assert torch.all(d.embedding == torch.ones((4,)) * pos)


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('index_dtype', [torch.int64])
def test_torchtensor_getitem(stack, da, index_dtype):
    if stack:
        da = da.to_doc_vec(tensor_type=TorchTensor)

    index = torch.tensor([1, 2, 3, 4, 6], dtype=index_dtype)

    indexed_da = da[index]

    for pos, d in zip(index, indexed_da):
        assert d.text == f'hello {pos}'
        assert torch.all(d.embedding == torch.ones((4,)) * pos)


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('index_dtype', [int, np.int_, np.int32, np.int64])
def test_nparray_getitem(stack, da, index_dtype):
    if stack:
        da = da.to_doc_vec(tensor_type=TorchTensor)

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
        da = da.to_doc_vec(tensor_type=TorchTensor)

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
        da = da.to_doc_vec(tensor_type=TorchTensor)

    da[0] = da_to_set[0]

    assert torch.all(da[0].embedding == da_to_set[0].embedding)
    assert da[0].text == da_to_set[0].text


@pytest.mark.parametrize('stack_left', [True, False])
@pytest.mark.parametrize('stack_right', [True, False])
@pytest.mark.parametrize('index', [(1, 2, 3, 4, 6), [1, 2, 3, 4, 6]])
def test_iterable_setitem(stack_left, stack_right, da, da_to_set, index):
    if stack_left:
        da = da.to_doc_vec(tensor_type=TorchTensor)
    if stack_right:
        da_to_set = da_to_set.to_doc_vec(tensor_type=TorchTensor)

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
        da = da.to_doc_vec(tensor_type=TorchTensor)
    if stack_right:
        da_to_set = da_to_set.to_doc_vec(tensor_type=TorchTensor)

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
        da = da.to_doc_vec(tensor_type=TorchTensor)
    if stack_right:
        da_to_set = da_to_set.to_doc_vec(tensor_type=TorchTensor)

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
        da = da.to_doc_vec(tensor_type=TorchTensor)
    if stack_right:
        da_to_set = da_to_set.to_doc_vec(tensor_type=TorchTensor)

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


def test_setitem_update_column():
    texts = [f'hello {i}' for i in range(10)]
    tensors = [torch.ones((4,)) * (i + 1) for i in range(10)]
    da = DocVec[TextDoc](
        [TextDoc(text=text, embedding=tens) for text, tens in zip(texts, tensors)],
        tensor_type=TorchTensor,
    )

    da[0] = TextDoc(text='hello', embedding=torch.zeros((4,)))

    assert da[0].text == 'hello'
    assert (da[0].embedding == torch.zeros((4,))).all()
    assert (da.embedding[0] == torch.zeros((4,))).all()

    assert da._storage.any_columns['text'][0] == 'hello'
    assert (da._storage.tensor_columns['embedding'][0] == torch.zeros((4,))).all()
    assert (da._storage.tensor_columns['embedding'][0] == torch.zeros((4,))).all()


@pytest.mark.parametrize(
    'index',
    [
        [False, True, True, True, True, False, True, False, False, False],
        (False, True, True, True, True, False, True, False, False, False),
        torch.tensor([0, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool),
        np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool),
    ],
)
def test_del_getitem(da, index):
    del da[index]
