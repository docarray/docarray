# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

import numpy as np
import pytest
import torch

from docarray import BaseDoc, DocList
from docarray.typing import NdArray, TorchTensor
from docarray.utils.find import find, find_batched


class TorchDoc(BaseDoc):
    tensor: TorchTensor


class NdDoc(BaseDoc):
    tensor: NdArray


@pytest.fixture()
def random_torch_query():
    return TorchDoc(tensor=torch.rand(128))


@pytest.fixture()
def random_torch_batch_query():
    return DocList[TorchDoc]([TorchDoc(tensor=torch.rand(128)) for _ in range(5)])


@pytest.fixture()
def random_nd_query():
    return NdDoc(tensor=np.random.rand(128))


@pytest.fixture()
def random_nd_batch_query():
    return DocList[NdDoc]([NdDoc(tensor=np.random.rand(128)) for _ in range(5)])


@pytest.fixture()
def random_torch_index():
    return DocList[TorchDoc](TorchDoc(tensor=torch.rand(128)) for _ in range(10))


@pytest.fixture()
def random_nd_index():
    return DocList[NdDoc](NdDoc(tensor=np.random.rand(128)) for _ in range(10))


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_torch(random_torch_query, random_torch_index, metric):
    top_k, scores = find(
        random_torch_index,
        random_torch_query,
        search_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert top_k.doc_type == random_torch_index.doc_type

    if metric.endswith('_dist'):
        assert (torch.stack(sorted(scores)) == scores).all()
    else:
        assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_torch_tensor_query(random_torch_query, random_torch_index):
    query = random_torch_query.tensor
    top_k, scores = find(
        random_torch_index,
        query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_torch_stacked(random_torch_query, random_torch_index):
    random_torch_index = random_torch_index.to_doc_vec()
    top_k, scores = find(
        random_torch_index,
        random_torch_query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_np(random_nd_query, random_nd_index, metric):
    top_k, scores = find(
        random_nd_index,
        random_nd_query,
        search_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    if metric.endswith('_dist'):
        assert (sorted(scores) == scores).all()
    else:
        assert (sorted(scores, reverse=True) == scores).all()


def test_find_np_tensor_query(random_nd_query, random_nd_index):
    query = random_nd_query.tensor
    top_k, scores = find(
        random_nd_index,
        query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (sorted(scores, reverse=True) == scores).all()


def test_find_np_stacked(random_nd_query, random_nd_index):
    random_nd_index = random_nd_index.to_doc_vec()
    top_k, scores = find(
        random_nd_index,
        random_nd_query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (sorted(scores, reverse=True) == scores).all()


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_batched_torch(random_torch_batch_query, random_torch_index, metric):
    documents, scores = find_batched(
        random_torch_index,
        random_torch_batch_query,
        search_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(documents) == len(random_torch_batch_query)
    assert len(scores) == len(random_torch_batch_query)
    for top_k, top_scores in zip(documents, scores):
        assert len(top_k) == 7
        assert len(top_scores) == 7
        assert top_k.doc_type == random_torch_index.doc_type

    for sc in scores:
        if metric.endswith('_dist'):
            assert (torch.stack(sorted(sc)) == sc).all()
        else:
            assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


def test_find_batched_torch_tensor_query(random_torch_batch_query, random_torch_index):
    query = torch.stack(random_torch_batch_query.tensor)
    documents, scores = find_batched(
        random_torch_index,
        query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(documents) == len(random_torch_batch_query)
    assert len(scores) == len(random_torch_batch_query)
    for top_k, top_scores in zip(documents, scores):
        assert len(top_k) == 7
        assert len(top_scores) == 7
    for sc in scores:
        assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


@pytest.mark.parametrize('stack_what', ['index', 'query', 'both'])
def test_find_batched_torch_stacked(
    random_torch_batch_query, random_torch_index, stack_what
):
    if stack_what in ('index', 'both'):
        random_torch_index = random_torch_index.to_doc_vec()
    if stack_what in ('query', 'both'):
        random_torch_batch_query = random_torch_batch_query.to_doc_vec()

    documents, scores = find_batched(
        random_torch_index,
        random_torch_batch_query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(documents) == len(random_torch_batch_query)
    assert len(scores) == len(random_torch_batch_query)
    for top_k, top_scores in zip(documents, scores):
        assert len(top_k) == 7
        assert len(top_scores) == 7
    for sc in scores:
        assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_batched_np(random_nd_batch_query, random_nd_index, metric):
    documents, scores = find_batched(
        random_nd_index,
        random_nd_batch_query,
        search_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(documents) == len(random_nd_batch_query)
    assert len(scores) == len(random_nd_batch_query)
    for top_k, top_scores in zip(documents, scores):
        assert len(top_k) == 7
        assert len(top_scores) == 7
    for sc in scores:
        if metric.endswith('_dist'):
            assert (sorted(sc) == sc).all()
        else:
            assert (sorted(sc, reverse=True) == sc).all()


def test_find_batched_np_tensor_query(random_nd_batch_query, random_nd_index):
    query = np.stack(random_nd_batch_query.tensor)
    documents, scores = find_batched(
        random_nd_index,
        query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(documents) == len(random_nd_batch_query)
    assert len(scores) == len(random_nd_batch_query)
    for top_k, top_scores in zip(documents, scores):
        assert len(top_k) == 7
        assert len(top_scores) == 7
    for sc in scores:
        assert (sorted(sc, reverse=True) == sc).all()


@pytest.mark.parametrize('stack_what', ['index', 'query', 'both'])
def test_find_batched_np_stacked(random_nd_batch_query, random_nd_index, stack_what):
    if stack_what in ('index', 'both'):
        random_nd_index = random_nd_index.to_doc_vec()
    if stack_what in ('query', 'both'):
        random_nd_batch_query = random_nd_batch_query.to_doc_vec()
    documents, scores = find_batched(
        random_nd_index,
        random_nd_batch_query,
        search_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(documents) == len(random_nd_batch_query)
    assert len(scores) == len(random_nd_batch_query)
    for top_k, top_scores in zip(documents, scores):
        assert len(top_k) == 7
        assert len(top_scores) == 7
    for sc in scores:
        assert (sorted(sc, reverse=True) == sc).all()


def test_find_optional():
    class MyDoc(BaseDoc):
        embedding: Optional[TorchTensor]

    query = MyDoc(embedding=torch.rand(10))
    index = DocList[MyDoc]([MyDoc(embedding=torch.rand(10)) for _ in range(10)])

    top_k, scores = find(
        index,
        query,
        search_field='embedding',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_union():
    class MyDoc(BaseDoc):
        embedding: Union[TorchTensor, NdArray]

    query = MyDoc(embedding=torch.rand(10))
    index = DocList[MyDoc]([MyDoc(embedding=torch.rand(10)) for _ in range(10)])

    top_k, scores = find(
        index,
        query,
        search_field='embedding',
        limit=7.0,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


@pytest.mark.parametrize('stack', [False, True])
def test_find_nested(stack):
    class InnerDoc(BaseDoc):
        title: str
        embedding: TorchTensor

    class MyDoc(BaseDoc):
        inner: InnerDoc

    query = MyDoc(inner=InnerDoc(title='query', embedding=torch.rand(2)))
    index = DocList[MyDoc](
        [
            MyDoc(inner=InnerDoc(title=f'doc {i}', embedding=torch.rand(2)))
            for i in range(10)
        ]
    )
    if stack:
        index = index.to_doc_vec()

    top_k, scores = find(
        index,
        query,
        search_field='inner__embedding',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_nested_union_optional():
    class MyDoc(BaseDoc):
        embedding: Union[Optional[TorchTensor], Optional[NdArray]]
        embedding2: Optional[Union[TorchTensor, NdArray]]
        embedding3: Optional[Optional[TorchTensor]]
        embedding4: Union[Optional[Union[TorchTensor, NdArray]], TorchTensor]

    query = MyDoc(
        embedding=torch.rand(10),
        embedding2=torch.rand(10),
        embedding3=torch.rand(10),
        embedding4=torch.rand(10),
    )
    index = DocList[MyDoc](
        [
            MyDoc(
                embedding=torch.rand(10),
                embedding2=torch.rand(10),
                embedding3=torch.rand(10),
                embedding4=torch.rand(10),
            )
            for _ in range(10)
        ]
    )

    top_k, scores = find(
        index,
        query,
        search_field='embedding',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()

    top_k, scores = find(
        index,
        query,
        search_field='embedding2',
        limit=7.0,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()

    top_k, scores = find(
        index,
        query,
        search_field='embedding3',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()

    top_k, scores = find(
        index,
        query,
        search_field='embedding4',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()
