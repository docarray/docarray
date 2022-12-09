import importlib
from typing import Callable, NamedTuple, Optional, Type, Union

from docarray import Document, DocumentArray
from docarray.typing import Tensor
from docarray.typing.tensor import type_to_framework


class FindResult(NamedTuple):
    documents: DocumentArray
    scores: Tensor


def find(
    index: DocumentArray,
    query: Union[Tensor, Document, DocumentArray],
    embedding_field: Optional[str] = 'embedding',
    metric: Union[str, Callable[['Tensor', 'Tensor'], 'Tensor']] = 'cosine',
    limit: int = 10,
    device: Optional[str] = None,
) -> FindResult:
    """
    Find the closest Documents in the index to the query.

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

    :param index: the index of Documents to search in
    :param query: the query to search for
    :param embedding_field: the tensor-like field in the index to use
        for the similarity computation
    :param metric: the distance metric to use for the similarity computation.
        Can be a string specifying a predefined metric
        (TODO(johannes) specify) or a callable that takes two tensors and returns
        a tensor of distances. TODO(johannes) implement passing a callable
    :param limit: return the top `limit` results
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :return: the closest Documents in the index to the query
    """
    embedding_type = _da_attr_type(index, embedding_field)

    # get framework-specific distance and top_k function
    distance_fn = _get_distance_fn(embedding_type, metric)
    top_k_fn = _get_topk_fn(embedding_type)

    index_embeddings = getattr(index, embedding_field)
    if not index.is_stacked():
        index_embeddings = embedding_type.__docarray_stack__(index_embeddings)

    dists = distance_fn(
        index_embeddings, getattr(query, embedding_field), device=device
    )
    top_scores, top_indices = top_k_fn(dists, k=limit, device=device)
    results_docs = DocumentArray(
        index[i] for i in top_indices
    )  # workaround until #930 is fixed
    return FindResult(documents=results_docs, scores=top_scores)


def _da_attr_type(da: DocumentArray, attr: str) -> Type:
    """Get the type of the attribute according to the Document type
    (schema) of the DocumentArray.

    :param da: the DocumentArray
    :param attr: the attribute name
    :return: the type of the attribute
    """
    return da.document_type.__fields__[attr].type_


def _get_topk_fn(embedding_type: Type) -> Callable:
    """Dynamically import the distance function from the framework-specific module.

    :param embedding_type: the type of the embedding
    :param distance_name: the name of the distance function
    :return: the framework-specific distance function
    """
    framework = type_to_framework[embedding_type]
    return getattr(
        importlib.import_module(f'docarray.utility.helper.{framework}'),
        'top_k',
    )


def _get_distance_fn(embedding_type: Type, distance_name: str) -> Callable:
    """Dynamically import the distance function from the framework-specific module.

    :param embedding_type: the type of the embedding
    :param distance_name: the name of the distance function
    :return: the framework-specific distance function
    """
    framework = type_to_framework[embedding_type]
    return getattr(
        importlib.import_module(f'docarray.utility.math.distance.{framework}'),
        f'{distance_name}',
    )
