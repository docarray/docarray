from typing import Any, Dict, List, NamedTuple, Optional, Type, Union, cast, Tuple, Callable, Generator, TypeVar

from typing_inspect import is_union_type
from contextlib import nullcontext
from math import ceil
from multiprocessing.pool import Pool, ThreadPool

from rich.progress import track
from docarray.array.abstract_array import AnyDocArray
from docarray.array.array.array import DocArray
from docarray.array.stacked.array_stacked import DocArrayStacked
from docarray.base_doc import BaseDoc
from docarray.helper import _get_field_type_by_access_path
from docarray.typing import AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils.map import map_docs_batch

class FindResult(NamedTuple):
    documents: DocArray
    scores: AnyTensor

class _FindResult(NamedTuple):
    documents: Union[DocArray, List[Dict[str, Any]]]
    scores: AnyTensor

class Doc(BaseDoc):
    embedding: AnyTensor=None

def find(
    index: AnyDocArray,
    query: Union[AnyTensor, BaseDoc],
    embedding_field: str = 'embedding',
    metric: str = 'cosine_sim',
    limit: int = 10,
    device: Optional[str] = None,
    descending: Optional[bool] = None,
) -> FindResult:
    """
    Find the closest Documents in the index to the query.
    Supports PyTorch and NumPy embeddings.

    .. note::
        This utility function is likely to be removed once
        Document Stores are available.
        At that point, and in-memory Document Store will serve the same purpose
        by exposing a .find() method.

    .. note::
        This is a simple implementation that assumes the same embedding field name for
        both query and index, does not support nested search, and does not support
        hybrid (multi-vector) search. These shortcoming will be addressed in future
        versions.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import DocArray, BaseDoc
        from docarray.typing import TorchTensor
        from docarray.utils.find import find


        class MyDocument(BaseDoc):
            embedding: TorchTensor


        index = DocArray[MyDocument](
            [MyDocument(embedding=torch.rand(128)) for _ in range(100)]
        )

        # use Document as query
        query = MyDocument(embedding=torch.rand(128))
        top_matches, scores = find(
            index=index,
            query=query,
            metric='cosine_sim',
        )

        # use tensor as query
        query = torch.rand(128)
        top_matches, scores = find(
            index=index,
            query=query,
            metric='cosine_sim',
        )

    :param index: the index of Documents to search in
    :param query: the query to search for
    :param embedding_field: the tensor-like field in the index to use
        for the similarity computation
    :param metric: the distance metric to use for the similarity computation.
        Can be one of the following strings:
        'cosine_sim' for cosine similarity, 'euclidean_dist' for euclidean distance,
        'sqeuclidean_dist' for squared euclidean distance
    :param limit: return the top `limit` results
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :param descending: sort the results in descending order.
        Per default, this is chosen based on the `metric` argument.
    :return: A named tuple of the form (DocArray, AnyTensor),
        where the first element contains the closes matches for the query,
        and the second element contains the corresponding scores.
    """
    query = _extract_embedding_single(query, embedding_field)
    return find_batched(
        index=index,
        query=query,
        embedding_field=embedding_field,
        batch_size=None,
        metric=metric,
        limit=limit,
        device=device,
        descending=descending,
    )[0]


def find_batched(
    index: AnyDocArray,
    query: Union[AnyTensor, DocArray],
    embedding_field: str = 'embedding',
    batch_size: int=1,
    metric: str = 'cosine_sim',
    limit: int = 10,
    device: Optional[str] = None,
    descending: Optional[bool] = None,
    shuffle: bool = False,
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> List[FindResult]:
    """
    Find the closest Documents in the index to the queries.
    Supports PyTorch and NumPy embeddings.

    .. note::
        This utility function is likely to be removed once
        Document Stores are available.
        At that point, and in-memory Document Store will serve the same purpose
        by exposing a .find() method.

    .. note::
        This is a simple implementation that assumes the same embedding field name for
        both query and index, does not support nested search, and does not support
        hybrid (multi-vector) search. These shortcoming will be addressed in future
        versions.

        EXAMPLE USAGE

    .. code-block:: python

        from docarray import DocArray, BaseDoc
        from docarray.typing import TorchTensor
        from docarray.utils.find import find_batched
        import torch

        class MyDocument(BaseDoc):
            embedding: TorchTensor


        index = DocArray[MyDocument](
            [MyDocument(embedding=torch.rand(128)) for _ in range(100)]
        )

        # use DocArray as query
        query = DocArray[MyDocument]([MyDocument(embedding=torch.rand(128)) for _ in range(10)])
        results = find_batched(
            index=index,
            query=query,
            batch_size=5,
            metric='cosine_sim',
        )
        top_matches, scores = results[0]

        # use tensor as query
        query = torch.rand(10, 128)
        results= find_batched(
            index=index,
            query=query,
            batch_size=5,
            metric='cosine_sim',
        )
        top_matches, scores = results[0]

    :param index: the index of Documents to search in
    :param query: the query to search for
    :param embedding_field: the tensor-like field in the index to use
        for the similarity computation
    :param batch_size: Size of each generated batch (except the last one, which might
        be smaller).
    :param metric: the distance metric to use for the similarity computation.
        Can be one of the following strings:
        'cosine_sim' for cosine similarity, 'euclidean_dist' for euclidean distance,
        'sqeuclidean_dist' for squared euclidean distance
    :param limit: return the top `limit` results
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :param descending: sort the results in descending order.
        Per default, this is chosen based on the `metric` argument.
    :param shuffle: If set, shuffle the Documents before dividing into minibatches.
    :param backend: `thread` for multithreading and `process` for multiprocessing.
        Defaults to `thread`.
        In general, if `func` is IO-bound then `thread` is a good choice.
        On the other hand, if `func` is CPU-bound, then you may use `process`.
        In practice, you should try yourselves to figure out the best value.
        However, if you wish to modify the elements in-place, regardless of IO/CPU-bound,
        you should always use `thread` backend.
        Note that computation that is offloaded to non-python code (e.g. through np/torch/tf)
        falls under the "IO-bound" category.
        .. warning::
            When using `process` backend, your `func` should not modify elements in-place.
            This is because the multiprocessing backend passes the variable via pickle
            and works in another process.
            The passed object and the original object do **not** share the same memory.
    :param num_worker: the number of parallel workers. If not given, then the number of CPUs
        in the system will be used.
    :param show_progress: show a progress bar
    :param pool: use an existing/external pool. If given, `backend` is ignored and you will
        be responsible for closing the pool.
    :return: a list of named tuples of the form (DocArray, AnyTensor),
        where the first element contains the closes matches for each query,
        and the second element contains the corresponding scores.
    """
        
    if descending is None:
        descending = metric.endswith('_sim')  # similarity metrics are descending

    embedding_type = _da_attr_type(index, embedding_field)
    comp_backend = embedding_type.get_comp_backend()
    # extract embeddings from index
    index_embeddings = _extract_embeddings(index, embedding_field, embedding_type)
    metric_fn = getattr(comp_backend.Metrics, metric)

    results = []

    def _get_result(query: Union[AnyDocArray, BaseDoc, AnyTensor]):
        q_embed = _extract_embeddings(query, embedding_field, embedding_type)
        dists=metric_fn(q_embed, index_embeddings, device=device)
        top_scores, top_indices = comp_backend.Retrieval.top_k(
            dists, k=limit, device=device, descending=descending
        )
        return top_indices,top_scores
    
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError(
                f'`batch_size` must be larger than 0, receiving {batch_size}'
            )
        else:
            batch_size = int(batch_size)
    else :
        top_indices,top_scores= _get_result(query)
        res = []
        for indices_per_query, scores_per_query in zip(top_indices, top_scores):
            docs_per_query: DocArray = DocArray([])
            for idx in indices_per_query:  # workaround until #930 is fixed
                docs_per_query.append(index[idx])
            docs_per_query = DocArray(docs_per_query)
            res.append(FindResult(scores=scores_per_query, documents=docs_per_query))
        return res
    
    if not(isinstance(query, DocArray) or isinstance(query, (DocArrayStacked, BaseDoc))):
        query = cast(AnyTensor, query)
        d = DocArray[Doc](Doc() for _ in range(comp_backend.shape(tensor=query)[0]))
        d.embedding = query
        query=d


    it = map_docs_batch(da=query,func=_get_result, batch_size=batch_size,
                                        backend=backend,
                                        num_worker=num_worker,
                                        shuffle=shuffle,
                                        pool=pool,
                                        show_progress=show_progress)
    
    for (indices_per_query, scores_per_query) in it:
        docs_per_query: DocArray = DocArray([])
        for idx,scores in zip(indices_per_query,scores_per_query):  # workaround until #930 is fixed
            docs_per_query=(index[idx])
            results.append(FindResult(scores=scores, documents=docs_per_query))

    return results


def _extract_embedding_single(
    data: Union[DocArray, BaseDoc, AnyTensor],
    embedding_field: str,
) -> AnyTensor:
    """Extract the embeddings from a single query,
    and return it in a batched representation.

    :param data: the data
    :param embedding_field: the embedding field
    :param embedding_type: type of the embedding: torch.Tensor, numpy.ndarray etc.
    :return: the embeddings
    """
    if isinstance(data, BaseDoc):
        emb = next(AnyDocArray._traverse(data, embedding_field))
    else:  # treat data as tensor
        emb = data
    if len(emb.shape) == 1:
        # all currently supported frameworks provide `.reshape()`. Onc this is not true
        # anymore, we need to add a `.reshape()` method to the computational backend
        emb = emb.reshape(1, -1)
    return emb


def _extract_embeddings(
    data: Union[AnyDocArray, BaseDoc, AnyTensor],
    embedding_field: str,
    embedding_type: Type,
) -> AnyTensor:
    """Extract the embeddings from the data.

    :param data: the data
    :param embedding_field: the embedding field
    :param embedding_type: type of the embedding: torch.Tensor, numpy.ndarray etc.
    :return: the embeddings
    """
    emb: AnyTensor
    if isinstance(data, DocArray):
        emb_list = list(AnyDocArray._traverse(data, embedding_field))
        emb = embedding_type._docarray_stack(emb_list)
    elif isinstance(data, (DocArrayStacked, BaseDoc)):
        emb = next(AnyDocArray._traverse(data, embedding_field))
    else:  # treat data as tensor
        emb = cast(AnyTensor, data)

    if len(emb.shape) == 1:
        emb = emb.get_comp_backend().reshape(array=emb, shape=(1, -1))
    return emb


def _da_attr_type(da: AnyDocArray, access_path: str) -> Type[AnyTensor]:
    """Get the type of the attribute according to the Document type
    (schema) of the DocArray.

    :param da: the DocArray
    :param access_path: the "__"-separated access path
    :return: the type of the attribute
    """
    field_type: Optional[Type] = _get_field_type_by_access_path(
        da.document_type, access_path
    )
    if field_type is None:
        raise ValueError(f"Access path is not valid: {access_path}")

    if is_union_type(field_type):
        # determine type based on the fist element
        field_type = type(next(AnyDocArray._traverse(da[0], access_path)))

    if not issubclass(field_type, AbstractTensor):
        raise ValueError(
            f'attribute {access_path} is not a tensor-like type, '
            f'but {field_type.__class__.__name__}'
        )

    return cast(Type[AnyTensor], field_type)
