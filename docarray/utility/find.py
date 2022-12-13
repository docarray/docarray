import importlib
from typing import Callable, List, NamedTuple, Optional, Type, Union

import torch

from docarray import Document, DocumentArray
from docarray.typing import Tensor
from docarray.typing.tensor import type_to_framework


class FindResult(NamedTuple):
    #             for single query | for multiple queries
    documents: Union[DocumentArray, List[DocumentArray]]
    scores: Tensor


def find(
    index: DocumentArray,
    query: Union[Tensor, Document, DocumentArray],
    embedding_field: Optional[str] = 'embedding',
    metric: Union[str, Callable[['Tensor', 'Tensor'], 'Tensor']] = 'cosine_sim',
    limit: int = 10,
    device: Optional[str] = None,
    descending: Optional[bool] = None,
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
    :param descending: sort the results in descending order.
        Per default, this is chosen based on the `metric` argument.
    :return: the closest Documents in the index to the query
    """
    if descending is None:
        descending = metric.endswith('_sim')  # similarity metrics are descending

    embedding_type = _da_attr_type(index, embedding_field)

    # get framework-specific distance and top_k function
    distance_fn = _get_distance_fn(embedding_type, metric)
    top_k_fn = _get_topk_fn(embedding_type)

    # extract embeddings from query and index
    index_embeddings = _extraxt_embeddings(index, embedding_field, embedding_type)
    query_embeddings = _extraxt_embeddings(query, embedding_field, embedding_type)

    # compute distances and return top results
    dists = distance_fn(query_embeddings, index_embeddings, device=device)
    top_scores, top_indices = top_k_fn(
        dists, k=limit, device=device, descending=descending
    )

    result_docs = []
    to_da = True
    for top_idx in top_indices:  # workaround until #930 is fixed
        if top_idx.shape == () or len(top_idx) == 0:  # single result for single query
            result_docs.append(index[top_idx])
        else:  # there were multiple queries, so multiple results
            inner_result_docs = []
            to_da = False
            for inner_top_idx in top_idx:
                inner_result_docs.append(index[inner_top_idx])
    if to_da:
        result_docs = DocumentArray(result_docs)

    return FindResult(documents=result_docs, scores=top_scores)


def _extraxt_embeddings(
    data: Union[DocumentArray, Document, Tensor],
    embedding_field: str,
    embedding_type: Type,
) -> Tensor:
    """Extract the embeddings from the data.

    :param data: the data
    :param embedding_field: the embedding field
    :param embedding_type: type of the embedding: torch.Tensor, numpy.ndarray etc.
    :return: the embeddings
    """
    # TODO(johannes) put docarray stack in the computational backend
    if isinstance(data, DocumentArray):
        emb = getattr(data, embedding_field)
        if not data.is_stacked():
            emb = embedding_type.__docarray_stack__(emb)
    elif isinstance(data, Document):
        emb = getattr(data, embedding_field)
    else:  # treat data as tensor
        emb = data

    if len(emb.shape) == 1:
        # TODO(johannes) solve this with computational backend,
        #  this is ugly hack for now
        if isinstance(emb, torch.Tensor):
            emb = emb.unsqueeze(0)
        else:
            import numpy as np

            if isinstance(emb, np.ndarray):
                emb = np.expand_dims(emb, axis=0)
    return emb


def _to_documentarray_query(
    query: Union[Tensor, Document, DocumentArray], embedding_field: str
) -> DocumentArray:
    """Convert the query to a DocumentArray.

    :param query: the query
    :return: the query as DocumentArray
    """
    if isinstance(query, DocumentArray):
        return query
    elif isinstance(query, Document):
        return DocumentArray([query])
    else:
        return DocumentArray([Document(**{embedding_field: query})])


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
        importlib.import_module(f'docarray.utility.math.metrics.{framework}'),
        f'{distance_name}',
    )
