__all__ = ['filter_docs']

import json
from typing import Dict, List, Union

from docarray.array.any_array import AnyDocArray
from docarray.array.doc_list.doc_list import DocList


def filter_docs(
    docs: AnyDocArray,
    query: Union[str, Dict, List[Dict]],
) -> AnyDocArray:
    """
    Filter the Documents in the index according to the given filter query.



    ---

    ```python
    from docarray import DocList, BaseDoc
    from docarray.documents import TextDoc, ImageDoc
    from docarray.utils.filter import filter_docs


    class MyDocument(BaseDoc):
        caption: TextDoc
        ImageDoc: ImageDoc
        price: int


    docs = DocList[MyDocument](
        [
            MyDocument(
                caption='A tiger in the jungle',
                ImageDoc=ImageDoc(url='tigerphoto.png'),
                price=100,
            ),
            MyDocument(
                caption='A swimming turtle',
                ImageDoc=ImageDoc(url='turtlepic.png'),
                price=50,
            ),
            MyDocument(
                caption='A couple birdwatching with binoculars',
                ImageDoc=ImageDoc(url='binocularsphoto.png'),
                price=30,
            ),
        ]
    )
    query = {
        '$and': {
            'ImageDoc__url': {'$regex': 'photo'},
            'price': {'$lte': 50},
        }
    }

    results = filter_docs(docs, query)
    assert len(results) == 1
    assert results[0].price == 30
    assert results[0].caption == 'A couple birdwatching with binoculars'
    assert results[0].ImageDoc.url == 'binocularsphoto.png'
    ```

    ---

    :param docs: the DocList where to apply the filter
    :param query: the query to filter by
    :return: A DocList containing the Documents
    in `docs` that fulfill the filter conditions in the `query`
    """
    from docarray.utils._internal.query_language.query_parser import QueryParser

    if query:
        query = query if not isinstance(query, str) else json.loads(query)
        parser = QueryParser(query)
        return DocList.__class_getitem__(docs.doc_type)(
            d for d in docs if parser.evaluate(d)
        )
    else:
        return docs
