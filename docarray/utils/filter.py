import json

from typing import Union, Dict, List


from docarray.array.abstract_array import AnyDocumentArray
from docarray.array.array import DocumentArray


def filter(
    index: AnyDocumentArray,
    query: Union[str, Dict, List[Dict]],
) -> AnyDocumentArray:
    """
    Filter the Documents in the index according to the given filter query.


    EXAMPLE USAGE

    .. code-block:: python

        TODO: fill it

    :param index: the index of Documents to filter in
    :param query: the query to filter by
    :return: A DocumentArray containing the Documents
    inside DocumentArray that fullfil the filter conditions
    """
    from docarray.utils.query_parser import QueryParser

    if query:
        query = query if not isinstance(query, str) else json.loads(query)
        parser = QueryParser(query)
        return DocumentArray(d for d in index if parser.evaluate(d))
    else:
        return index
