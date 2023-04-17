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


N_QUERIES = 5


@pytest.fixture()
def random_torch_query():
    return TorchDoc(tensor=torch.rand(128))


@pytest.fixture()
def random_torch_batch_query():
    return DocList[TorchDoc](
        [TorchDoc(tensor=torch.rand(128)) for _ in range(N_QUERIES)]
    )


@pytest.fixture()
def random_nd_query():
    return NdDoc(tensor=np.random.rand(128))


@pytest.fixture()
def random_nd_batch_query():
    return DocList[NdDoc]([NdDoc(tensor=np.random.rand(128)) for _ in range(N_QUERIES)])


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
        embedding_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    if metric.endswith('_dist'):
        assert (torch.stack(sorted(scores)) == scores).all()
    else:
        assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_torch_tensor_query(random_torch_query, random_torch_index):
    query = random_torch_query.tensor
    top_k, scores = find(
        random_torch_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_torch_stacked(random_torch_query, random_torch_index):
    random_torch_index = random_torch_index.stack()
    top_k, scores = find(
        random_torch_index,
        random_torch_query,
        embedding_field='tensor',
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
        embedding_field='tensor',
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
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (sorted(scores, reverse=True) == scores).all()


def test_find_np_stacked(random_nd_query, random_nd_index):
    random_nd_index = random_nd_index.stack()
    top_k, scores = find(
        random_nd_index,
        random_nd_query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (sorted(scores, reverse=True) == scores).all()


@pytest.mark.slow
@pytest.mark.parametrize('n_queries,batch_size', [(20, 5), (5, 2), (5, 1)])
@pytest.mark.parametrize('backend', ['thread', 'process'])
@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_batched_torch(n_queries, random_torch_index, batch_size, backend, metric):
    query = DocList[TorchDoc](
        [TorchDoc(tensor=torch.rand(128)) for _ in range(n_queries)]
    )
    results = find_batched(
        random_torch_index,
        query,
        embedding_field='tensor',
        limit=7,
        batch_size=batch_size,
        backend=backend,
        metric=metric,
    )
    # without parallel processing
    expected_result = find_batched(
        random_torch_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(results) == len(query)

    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        if metric.endswith('_dist'):
            assert (torch.stack(sorted(sc)) == sc).all()
        else:
            assert (torch.stack(sorted(sc, reverse=True)) == sc).all()
    for (top_k, scores), (exp_top_k, exp_scores) in zip(results, expected_result):
        for i, j in zip(top_k, exp_top_k):
            assert i == j
        for i, j in zip(scores, exp_scores):
            assert torch.allclose(i, j)


def test_find_batched_torch_tensor_query(random_torch_batch_query, random_torch_index):
    query = torch.stack(random_torch_batch_query.tensor)
    results = find_batched(
        random_torch_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_torch_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


@pytest.mark.parametrize('stack_what', ['index', 'query', 'both'])
def test_find_batched_torch_stacked(
    random_torch_batch_query, random_torch_index, stack_what
):
    if stack_what in ('index', 'both'):
        random_torch_index = random_torch_index.stack()
    if stack_what in ('query', 'both'):
        random_torch_batch_query = random_torch_batch_query.stack()

    results = find_batched(
        random_torch_index,
        random_torch_batch_query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_torch_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


@pytest.mark.slow
@pytest.mark.parametrize('n_queries,batch_size', [(20, 5), (5, 2), (5, 1)])
@pytest.mark.parametrize('backend', ['thread', 'process'])
@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_batched_np(n_queries, random_nd_index, batch_size, backend, metric):
    query = DocList[NdDoc](
        [NdDoc(tensor=np.random.rand(128)) for _ in range(n_queries)]
    )
    results = find_batched(
        random_nd_index,
        query,
        embedding_field='tensor',
        batch_size=batch_size,
        backend=backend,
        limit=7,
        metric=metric,
    )
    # without parallel processing
    expected_result = find_batched(
        random_nd_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(results) == len(query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        if metric.endswith('_dist'):
            assert (sorted(sc) == sc).all()
        else:
            assert (sorted(sc, reverse=True) == sc).all()

    for (top_k, scores), (exp_top_k, exp_scores) in zip(results, expected_result):
        for i, j in zip(top_k, exp_top_k):
            assert i == j
        for i, j in zip(scores, exp_scores):
            assert i == pytest.approx(j)


def test_find_batched_np_tensor_query(random_nd_batch_query, random_nd_index):
    query = np.stack(random_nd_batch_query.tensor)
    results = find_batched(
        random_nd_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_nd_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (sorted(sc, reverse=True) == sc).all()


@pytest.mark.parametrize('stack_what', ['index', 'query', 'both'])
def test_find_batched_np_stacked(random_nd_batch_query, random_nd_index, stack_what):
    if stack_what in ('index', 'both'):
        random_nd_index = random_nd_index.stack()
    if stack_what in ('query', 'both'):
        random_nd_batch_query = random_nd_batch_query.stack()
    results = find_batched(
        random_nd_index,
        random_nd_batch_query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_nd_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (sorted(sc, reverse=True) == sc).all()


def test_find_optional():
    class MyDoc(BaseDoc):
        embedding: Optional[TorchTensor]

    query = MyDoc(embedding=torch.rand(10))
    index = DocList[MyDoc]([MyDoc(embedding=torch.rand(10)) for _ in range(10)])

    top_k, scores = find(
        index,
        query,
        embedding_field='embedding',
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
        embedding_field='embedding',
        limit=7,
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
        index = index.stack()

    top_k, scores = find(
        index,
        query,
        embedding_field='inner__embedding',
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
        embedding_field='embedding',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()

    top_k, scores = find(
        index,
        query,
        embedding_field='embedding2',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()

    top_k, scores = find(
        index,
        query,
        embedding_field='embedding3',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()

    top_k, scores = find(
        index,
        query,
        embedding_field='embedding4',
        limit=7,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()
