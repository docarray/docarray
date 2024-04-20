from typing import Any, Dict, List, Tuple, Type, cast

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
from docarray.utils.filter import filter_docs
from docarray.utils.find import FindResult


def _collect_query_args(method_name: str):  # TODO: use partialmethod instead
    def inner(self, *args, **kwargs):
        if args:
            raise ValueError(
                f'Positional arguments are not supported for '
                f'`{type(self)}.{method_name}`.'
                f' Use keyword arguments instead.'
            )
        updated_query = self._queries + [(method_name, kwargs)]
        return type(self)(updated_query)

    return inner


def _execute_find_and_filter_query(
    doc_index: BaseDocIndex, query: List[Tuple[str, Dict]], reverse_order: bool = False
) -> FindResult:
    """
    Executes all find calls from query first using `doc_index.find()`,
    and filtering queries after that using DocArray's `filter_docs()`.

    Text search is not supported.

    :param doc_index: Document index instance.
        Either InMemoryExactNNIndex or HnswDocumentIndex.
    :param query: Dictionary containing search and filtering configuration.
    :param reverse_order: Flag indicating whether to sort in descending order.
        If set to False (default), the sorting will be in ascending order.
        This option is necessary because, depending on the index, lower scores
        can correspond to better matches, and vice versa.
    :return: Sorted documents and their corresponding scores.
    """
    docs_found = DocList.__class_getitem__(cast(Type[BaseDoc], doc_index._schema))([])
    filter_conditions = []
    filter_limit = None
    doc_to_score: Dict[BaseDoc, Any] = {}
    for op, op_kwargs in query:
        if op == 'find':
            docs, scores = doc_index.find(**op_kwargs)
            docs_found.extend(docs)
            doc_to_score.update(zip(docs.__getattribute__('id'), scores))
        elif op == 'filter':
            filter_conditions.append(op_kwargs['filter_query'])
            filter_limit = op_kwargs.get('limit')
        else:
            raise ValueError(f'Query operation is not supported: {op}')

    doc_index._logger.debug(f'Executing query {query}')
    docs_filtered = docs_found
    for cond in filter_conditions:
        docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], doc_index._schema))
        docs_filtered = docs_cls(filter_docs(docs_filtered, cond))

    if filter_limit:
        docs_filtered = docs_filtered[:filter_limit]

    doc_index._logger.debug(f'{len(docs_filtered)} results found')
    docs_and_scores = zip(
        docs_filtered, (doc_to_score[doc.id] for doc in docs_filtered)
    )
    docs_sorted = sorted(docs_and_scores, key=lambda x: x[1], reverse=reverse_order)
    out_docs, out_scores = zip(*docs_sorted)

    return FindResult(documents=out_docs, scores=out_scores)
